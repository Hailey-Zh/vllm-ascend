#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# C_TEMPLATE isolation test (diagnostic for the IS_DENSE dense-path bug).
#
# Why this exists
# ---------------
# test_dense.py compares dense (sparse_indices=None) vs sparse-full (block_size=1). Those are NOT the
# same kernel binary:
#   - block_size <= 4  -> tiling picks V_TEMPLATE  (MergeKv path)
#   - dense / None     -> tiling forces C_TEMPLATE (direct-read path)
# Every passing test today (the 5 `test_run.sh single` cases, and Run A of test_dense.py) uses
# block_size=1 -> V_TEMPLATE. The C_TEMPLATE cube path has zero passing coverage. So when test_dense
# fails by ~46x, we cannot tell whether the bug is in the four `if constexpr(SFAT::isDense)` offset
# branches, or in the (untested) C_TEMPLATE base path itself.
#
# This test runs THREE configs on identical inputs:
#   A) sparse, block_size=1, full coverage  -> V_TEMPLATE          (golden-validated reference)
#   B) sparse, block_size=8, full coverage  -> C_TEMPLATE, isDense=false   (C_TEMPLATE BASE)
#   C) dense, sparse_indices=None           -> C_TEMPLATE, isDense=true    (the failing path)
#
# Interpretation:
#   - if B != A  -> the C_TEMPLATE base path is broken; the isDense branches are innocent.
#   - if B == A but C != A -> bug is isolated to the isDense branches; instrument them next.
#
# IMPORTANT — no rebuild needed for the decisive signal:
# Runs A and B pass explicit sparse_indices, so B (block_size=8) hits C_TEMPLATE regardless of the
# torch_adpt arange dummy. So B-vs-A is meaningful on CURRENT main (dummy present), no rebuild.
# Run C only reaches the isDense path AFTER the dummy is removed (re-apply 898c895b); with the dummy
# present, None -> arange -> block_size=1 -> V_TEMPLATE, i.e. C effectively equals A. Read diff_ca /
# diff_cb accordingly.
#
# Run manually:
#   pytest test_ctemplate_isolation.py -s -v

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


SCALE = 1.0 / (576 ** 0.5)


def _call_op(query, key, value, sparse_indices, sparse_block_size, *,
             block_table, actual_seq_q, actual_seq_kv,
             query_rope, key_rope, layout_query, layout_kv):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE,
        sparse_block_size=sparse_block_size,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query=layout_query, layout_kv=layout_kv,
        sparse_mode=0,
        return_softmax_lse=False,
    )


def _full_indices(B, S1, N2, block_count, device):
    # 全选：每 [b, s1, n2, :] = [0, 1, ..., block_count-1]，选中所有 block。
    return (
        torch.arange(block_count, dtype=torch.int32, device=device)
        .view(1, 1, 1, block_count)
        .expand(B, S1, N2, block_count)
        .contiguous()
    )


def _make_bsnd_case():
    torch.manual_seed(42)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16
    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    value = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)
    return dict(
        query=query, key=key, value=value,
        B=B, S1=S1, S2=S2, N2=N2,
        block_table=None,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
    )


def test_ctemplate_base_matches_vtemplate():
    case = _make_bsnd_case()
    common = {k: case[k] for k in (
        "query", "key", "value", "block_table",
        "actual_seq_q", "actual_seq_kv",
        "query_rope", "key_rope", "layout_query", "layout_kv")}
    B, S1, S2, N2 = case["B"], case["S1"], case["S2"], case["N2"]
    device = "npu"

    # A) block_size=1 全选 -> V_TEMPLATE（参考基准，已被 single 5 用例对 golden 校验）
    idx_bs1 = _full_indices(B, S1, N2, S2, device)
    out_a = _call_op(sparse_indices=idx_bs1, sparse_block_size=1, **common)[0]
    torch.npu.synchronize()

    # B) block_size=8 全选 -> C_TEMPLATE, isDenseMode=false（C_TEMPLATE 基线路径）
    assert S2 % 8 == 0, "S2 must be divisible by sparse_block_size for full coverage"
    idx_bs8 = _full_indices(B, S1, N2, S2 // 8, device)
    out_b = _call_op(sparse_indices=idx_bs8, sparse_block_size=8, **common)[0]
    torch.npu.synchronize()

    # C) dense -> C_TEMPLATE, isDenseMode=true（当前失败的路径，供对比，不作断言）
    out_c = _call_op(sparse_indices=None, sparse_block_size=1, **common)[0]
    torch.npu.synchronize()

    a = out_a.float().cpu()
    b = out_b.float().cpu()
    c = out_c.float().cpu()

    diff_ba = (b - a).abs().max().item()   # C_TEMPLATE base vs V_TEMPLATE
    diff_ca = (c - a).abs().max().item()   # dense vs V_TEMPLATE (已知应失败)
    diff_cb = (c - b).abs().max().item()   # dense vs C_TEMPLATE base
    print(f"[isolation] max_abs_diff(C_TEMPLATE base bs=8, V_TEMPLATE bs=1) = {diff_ba:.6e}")
    print(f"[isolation] max_abs_diff(dense, V_TEMPLATE bs=1)               = {diff_ca:.6e}")
    print(f"[isolation] max_abs_diff(dense, C_TEMPLATE base bs=8)          = {diff_cb:.6e}")

    # 主断言：C_TEMPLATE 基线（无 isDense）是否正确。
    # 失败 => 锅在 C_TEMPLATE 公共路径，不在 isDense 分支。
    assert torch.allclose(b, a, rtol=1e-3, atol=1e-3), \
        f"C_TEMPLATE base path diverges from V_TEMPLATE: max_abs_diff={diff_ba:.6e}"


def _cpu_mla_dense_golden(query, key, value, query_rope, key_rope, scale):
    # BSND, GQA(N2=1). 标准 dense attention 的 CPU 参考（fp32 累加）。
    # score[b,s1,h,s2] = (q.k_nope + qr.kr_rope)*scale；softmax over s2；out = P @ V。
    q = query.float().cpu()        # [B,S1,N1,D]
    k = key.float().cpu()          # [B,S2,N2,D]
    v = value.float().cpu()        # [B,S2,N2,D]
    qr = query_rope.float().cpu()  # [B,S1,N1,R]
    kr = key_rope.float().cpu()    # [B,S2,N2,R]
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    k0 = k[:, :, 0, :]             # [B,S2,D]   (N2=1，所有 query head 共享)
    v0 = v[:, :, 0, :]             # [B,S2,D]
    kr0 = kr[:, :, 0, :]           # [B,S2,R]
    out = torch.empty(B, S1, N1, D, dtype=torch.float32)
    for b in range(B):
        for h in range(N1):
            s_nope = q[b, :, h, :] @ k0[b].transpose(0, 1)    # [S1,S2]
            s_rope = qr[b, :, h, :] @ kr0[b].transpose(0, 1)  # [S1,S2]
            score = (s_nope + s_rope) * scale
            p = torch.softmax(score, dim=-1)                  # [S1,S2]
            out[b, :, h, :] = p @ v0[b]                       # [S1,D]
    return out


def test_ctemplate_vs_cpu_golden():
    # 不依赖 V_TEMPLATE 的铁证：block_size=8 (C_TEMPLATE) 直接对 CPU dense golden。
    # 同时报告 block_size=1 (V_TEMPLATE) 对 golden，作为"CPU 参考可信"的 sanity。
    case = _make_bsnd_case()
    common = {k: case[k] for k in (
        "query", "key", "value", "block_table",
        "actual_seq_q", "actual_seq_kv",
        "query_rope", "key_rope", "layout_query", "layout_kv")}
    B, S1, S2, N2 = case["B"], case["S1"], case["S2"], case["N2"]
    device = "npu"

    golden = _cpu_mla_dense_golden(
        case["query"], case["key"], case["value"],
        case["query_rope"], case["key_rope"], SCALE)

    idx_bs1 = _full_indices(B, S1, N2, S2, device)
    out_v = _call_op(sparse_indices=idx_bs1, sparse_block_size=1, **common)[0].float().cpu()
    torch.npu.synchronize()

    idx_bs8 = _full_indices(B, S1, N2, S2 // 8, device)
    out_c = _call_op(sparse_indices=idx_bs8, sparse_block_size=8, **common)[0].float().cpu()
    torch.npu.synchronize()

    diff_v = (out_v - golden).abs().max().item()
    diff_c = (out_c - golden).abs().max().item()
    print(f"[golden] max_abs_diff(V_TEMPLATE bs=1, CPU golden) = {diff_v:.6e}")
    print(f"[golden] max_abs_diff(C_TEMPLATE bs=8, CPU golden) = {diff_c:.6e}")

    # sanity：V_TEMPLATE 应当接近 golden（证明 CPU 参考可信）。fp16 kernel，放宽到 3e-3。
    assert diff_v < 3e-3, f"CPU golden 与已验证的 V_TEMPLATE 都对不上，参考实现可疑: {diff_v:.6e}"
    # 主断言：C_TEMPLATE 直接对解析 golden。失败 => C_TEMPLATE 本身错，与对拍对象无关。
    assert diff_c < 3e-3, f"C_TEMPLATE 对 CPU golden 偏离: {diff_c:.6e}"


def _cpu_mla_golden_subset(query, key, value, query_rope, key_rope, scale, sel):
    # 同 _cpu_mla_dense_golden，但只在 sel（token 下标列表）这些 KV token 上做 attention。
    q = query.float().cpu()
    qr = query_rope.float().cpu()
    k0 = key.float().cpu()[:, sel, 0, :]
    v0 = value.float().cpu()[:, sel, 0, :]
    kr0 = key_rope.float().cpu()[:, sel, 0, :]
    B, S1, N1, D = q.shape
    out = torch.empty(B, S1, N1, D, dtype=torch.float32)
    for b in range(B):
        for h in range(N1):
            s_nope = q[b, :, h, :] @ k0[b].transpose(0, 1)
            s_rope = qr[b, :, h, :] @ kr0[b].transpose(0, 1)
            p = torch.softmax((s_nope + s_rope) * scale, dim=-1)
            out[b, :, h, :] = p @ v0[b]
    return out


def test_vtemplate_trigger_scan():
    # 触发条件扫描：固定 shape，只变"选中的 token 数 K"（indices=[0..K-1]），
    # 对比 V_TEMPLATE(block_size=1) 与同一子集上的 CPU golden。看 diff 从哪个 K 开始变大。
    torch.manual_seed(42)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16
    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    value = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    print("\n[scan] V_TEMPLATE(bs=1) vs CPU golden，按选中 token 数 K：")
    for K in [1, 2, 7, 8, 16, 31, 32, 48, 64, 96, 120, 127, 128]:
        idx = (torch.arange(K, dtype=torch.int32, device=device)
               .view(1, 1, 1, K).expand(B, S1, N2, K).contiguous())
        out = _call_op(
            query=query, key=key, value=value, sparse_indices=idx, sparse_block_size=1,
            block_table=None, actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
            query_rope=query_rope, key_rope=key_rope, layout_query="BSND", layout_kv="BSND",
        )[0].float().cpu()
        torch.npu.synchronize()
        golden = _cpu_mla_golden_subset(query, key, value, query_rope, key_rope, SCALE, list(range(K)))
        diff = (out - golden).abs().max().item()
        flag = "  <-- 错" if diff > 3e-3 else ""
        print(f"[scan] K={K:4d}  max_abs_diff={diff:.6e}{flag}")


def test_vtemplate_ignores_indices():
    # 一锤定音：选单个"非前缀"token t=64（indices=[[64]], K=1）。
    #   - 正确稀疏  => out == value[64]
    #   - H3(无视 indices，全量 attend [0,128)) => out == 全量 attention，且 != value[64]
    # 同时对照 C_TEMPLATE 无法在此直接验证（K=1 凑不出 block_size>4 的单 token），故只测 V。
    torch.manual_seed(42)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16
    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    value = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    t = 64
    idx = torch.tensor([t], dtype=torch.int32, device=device).view(1, 1, 1, 1).expand(B, S1, N2, 1).contiguous()
    out = _call_op(
        query=query, key=key, value=value, sparse_indices=idx, sparse_block_size=1,
        block_table=None, actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope, layout_query="BSND", layout_kv="BSND",
    )[0].float().cpu()
    torch.npu.synchronize()

    golden_indexed = _cpu_mla_golden_subset(query, key, value, query_rope, key_rope, SCALE, [t])  # 正确稀疏
    golden_full = _cpu_mla_golden_subset(query, key, value, query_rope, key_rope, SCALE, list(range(S2)))  # 全量
    golden_prefix = _cpu_mla_golden_subset(query, key, value, query_rope, key_rope, SCALE, [0])  # 选前缀(token0)

    d_idx = (out - golden_indexed).abs().max().item()
    d_full = (out - golden_full).abs().max().item()
    d_pref = (out - golden_prefix).abs().max().item()
    print(f"\n[uses-indices] 选 token {t}：")
    print(f"[uses-indices]  vs golden_indexed(value[{t}]) = {d_idx:.6e}  (≈0 则 V 正确按 indices 取数)")
    print(f"[uses-indices]  vs golden_full([0,128))       = {d_full:.6e}  (≈0 则 V 无视 indices 做全量 attend)")
    print(f"[uses-indices]  vs golden_prefix(value[0])    = {d_pref:.6e}  (≈0 则 V 取了前缀 token0)")


def test_vtemplate_token_id_probe():
    # 探针：value[b,j,0,:] = 标量 j。则 out = Σ_j w_j * j = V 实际 attend 的 token 重心。
    # 单 token 选择时 softmax 权重=1，out 应精确等于被选 token 的 id。直接读出 V 取了哪些 token。
    torch.manual_seed(42)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16
    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)
    # value[b,j,0,:] = j
    ids = torch.arange(S2, dtype=dtype, device=device).view(1, S2, 1, 1)
    value = ids.expand(B, S2, N2, D).contiguous()
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    def run(idx_list):
        K = len(idx_list)
        idx = (torch.tensor(idx_list, dtype=torch.int32, device=device)
               .view(1, 1, 1, K).expand(B, S1, N2, K).contiguous())
        out = _call_op(
            query=query, key=key, value=value, sparse_indices=idx, sparse_block_size=1,
            block_table=None, actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
            query_rope=query_rope, key_rope=key_rope, layout_query="BSND", layout_kv="BSND",
        )[0].float().cpu()
        torch.npu.synchronize()
        return out

    print("\n[probe] value[j]=j，out=被 attend token 的加权重心：")
    for idx_list, note in [
        ([0], "单选 token0 -> 期望 out≈0"),
        ([64], "单选 token64 -> 期望 out≈64"),
        ([127], "单选 token127 -> 期望 out≈127"),
        ([10, 20], "选 {10,20} -> 期望 out∈[10,20]"),
        (list(range(128)), "全选 -> 期望 out 为某加权均值"),
    ]:
        out = run(idx_list)
        print(f"[probe] idx={str(idx_list)[:24]:24s} mean={out.mean():.3f} min={out.min():.3f} max={out.max():.3f}  ({note})")


def test_ctemplate_mm2_constant_value():
    # mm2 / softmax 归一化隔离：把所有 V 行设成 per-batch 常量 v0。
    # 任意正确的 attention 都应输出 out == v0（softmax 权重和为 1，与 QK 分数无关）。
    #   - C_TEMPLATE 返回 v0  => V 读取 + softmax 归一化正确 => 锅在 mm1（K/rope/scale）。
    #   - C_TEMPLATE != v0    => 锅在 mm2（V 读取）或 softmax 归一化。
    # block_size=8 -> C_TEMPLATE，isDenseMode=false；与 dummy 无关，无需 rebuild。
    torch.manual_seed(7)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16

    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)

    # 每个 batch 一个常量行 v0[b]，沿 S2 广播 => value[b, :, 0, :] 全等于 v0[b]
    v0 = (torch.randn(B, N2, D, dtype=dtype, device=device) * 0.1)
    value = v0.view(B, 1, N2, D).expand(B, S2, N2, D).contiguous()

    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    assert S2 % 8 == 0
    idx_bs8 = _full_indices(B, S1, N2, S2 // 8, device)
    out = _call_op(
        query=query, key=key, value=value, sparse_indices=idx_bs8, sparse_block_size=8,
        block_table=None, actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope, layout_query="BSND", layout_kv="BSND",
    )[0]
    torch.npu.synchronize()

    out_cpu = out.float().cpu()
    # 期望：out[b, s1, n1, :] == v0[b, 0, :]（N2=1，GQA 下所有 query head 共享同一 KV head）
    expected = v0.float().cpu().view(B, 1, 1, D).expand(B, S1, N1, D)
    max_abs = (out_cpu - expected).abs().max().item()
    print(f"[mm2-isolation] max_abs_diff(C_TEMPLATE bs=8, constant-V golden) = {max_abs:.6e}")
    assert torch.allclose(out_cpu, expected, rtol=1e-3, atol=1e-3), \
        f"C_TEMPLATE mm2/softmax wrong: out != constant V, max_abs_diff={max_abs:.6e}"


def test_key_equals_value():
    """key==value 时 C_TEMPLATE 和 V_TEMPLATE 都应该接近 CPU golden。
    验证框架 single 用例的设计不是巧合：当 K==V 时 V_TEMPLATE 的"读 K 当 V"bug 被掩盖。"""
    torch.manual_seed(42)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16

    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    kv = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)  # key==value
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    kv_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)  # key_rope==? 无关，保持独立
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    common = dict(query=query, key=kv, value=kv,
                  block_table=None, actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
                  query_rope=query_rope, key_rope=kv_rope,
                  layout_query="BSND", layout_kv="BSND")

    # CPU golden（key==value）
    golden = _cpu_mla_dense_golden(query, kv, kv, query_rope, kv_rope, SCALE)

    # V_TEMPLATE: bs=1 全选
    idx1 = _full_indices(B, S1, N2, S2, device)
    out_v = _call_op(sparse_indices=idx1, sparse_block_size=1, **common)[0].float().cpu()
    torch.npu.synchronize()

    # C_TEMPLATE: bs=8 全选
    idx8 = _full_indices(B, S1, N2, S2 // 8, device)
    out_c = _call_op(sparse_indices=idx8, sparse_block_size=8, **common)[0].float().cpu()
    torch.npu.synchronize()

    dv = (out_v - golden).abs().max().item()
    dc = (out_c - golden).abs().max().item()
    dvc = (out_v - out_c).abs().max().item()
    print(f"[key==value] V_TEMPLATE vs CPU golden = {dv:.6e}")
    print(f"[key==value] C_TEMPLATE vs CPU golden = {dc:.6e}")
    print(f"[key==value] V_TEMPLATE vs C_TEMPLATE  = {dvc:.6e}")

    assert torch.allclose(out_v, golden, rtol=1e-3, atol=1e-3), \
        f"key==value: V_TEMPLATE wrong vs CPU golden, diff={dv:.6e}"
    assert torch.allclose(out_c, golden, rtol=1e-3, atol=1e-3), \
        f"key==value: C_TEMPLATE wrong vs CPU golden, diff={dc:.6e}"
    assert torch.allclose(out_v, out_c, rtol=1e-3, atol=1e-3), \
        f"key==value: V vs C mismatch, diff={dvc:.6e}"
