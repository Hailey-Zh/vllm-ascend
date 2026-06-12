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
# Step 3c dense-mode correctness test.
#
# Compare the dense path (sparse_indices=None) against an all-blocks-selected sparse baseline.
# IMPORTANT: the baseline uses sparse_block_size=8 so it runs on C_TEMPLATE (same template family
# as dense). It must NOT use sparse_block_size=1: that runs V_TEMPLATE, whose mm2 reads K from the
# merge workspace instead of V (P×K bug, see STEP3_DENSE_KERNEL_FIX.md). With key != value the
# block_size=1 baseline is itself wrong, so it cannot validate dense.
#
# Both runs walk tokens 0..S2-1 and read valueGm; attn_out must match up to fp16 noise.
#
# PRECONDITION: dense (None) only reaches C_TEMPLATE after the torch_adpt arange dummy is removed
# (re-apply 898c895b). With the dummy present, None -> arange -> V_TEMPLATE and this test fails by
# design (proving dense currently does NOT take the real IS_DENSE path).
#
# Covers three layouts:
#   - BSND / BSND        : exercises the BSND+BSND key/value offset path (no block table)
#   - BSND / PA_BSND     : exercises DataCopyPA with isDense
#   - TND  / TND         : variable-length packed layout (cumulative actual_seq), per-batch golden
#
# Run manually:
#   pytest test_dense.py -m step3c_dense -s -v

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


SCALE = 1.0 / (576 ** 0.5)


def _call_op(query, key, value, sparse_indices, *,
             block_table, actual_seq_q, actual_seq_kv,
             query_rope, key_rope, layout_query, layout_kv,
             sparse_block_size=1):
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


def _cpu_golden(query, key, value, query_rope, key_rope, scale):
    # 连续 BSND 形式的 CPU dense attention 参考（fp32），GQA N2=1。
    # key/value/key_rope: [B,S2,N2,D] 连续（PA case 需先还原）。
    q = query.float().cpu()
    qr = query_rope.float().cpu()
    k0 = key.float().cpu()[:, :, 0, :]
    v0 = value.float().cpu()[:, :, 0, :]
    kr0 = key_rope.float().cpu()[:, :, 0, :]
    B, S1, N1, D = q.shape
    out = torch.empty(B, S1, N1, D, dtype=torch.float32)
    for b in range(B):
        for h in range(N1):
            s = q[b, :, h, :] @ k0[b].transpose(0, 1) + qr[b, :, h, :] @ kr0[b].transpose(0, 1)
            p = torch.softmax(s * scale, dim=-1)
            out[b, :, h, :] = p @ v0[b]
    return out


def _cpu_golden_tnd(query, key, value, query_rope, key_rope, scale, cum_q, cum_kv):
    # TND 变长打包：query [T1,N1,D]，KV [T2,N2,D]，cum_q/cum_kv 是 cumulative 边界。
    # 每个 batch 的 query 段只 attend 该 batch 的 KV 段。GQA N2=1。
    q = query.float().cpu()
    qr = query_rope.float().cpu()
    k0 = key.float().cpu()[:, 0, :]
    v0 = value.float().cpu()[:, 0, :]
    kr0 = key_rope.float().cpu()[:, 0, :]
    T1, N1, D = q.shape
    out = torch.empty(T1, N1, D, dtype=torch.float32)
    qs, ks = 0, 0
    for b in range(len(cum_q)):
        qe, ke = cum_q[b], cum_kv[b]
        for h in range(N1):
            s = (q[qs:qe, h, :] @ k0[ks:ke].transpose(0, 1)
                 + qr[qs:qe, h, :] @ kr0[ks:ke].transpose(0, 1))
            p = torch.softmax(s * scale, dim=-1)
            out[qs:qe, h, :] = p @ v0[ks:ke]
        qs, ks = qe, ke
    return out


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
    # baseline 全选 sparse_indices：block_size=8 走 C_TEMPLATE（读 valueGm，算 P×V，正确）。
    # 不能用 block_size=1（走 V_TEMPLATE，mm2 误读 K 当 V，key≠value 时算成 P×K）。
    # S2=128 可被 8 整除：16 个 block 全选覆盖 token 0..127。
    sparse_indices_full = (
        torch.arange(S2 // 8, dtype=torch.int32, device=device)
        .view(1, 1, 1, S2 // 8)
        .expand(B, S1, N2, S2 // 8)
        .contiguous()
    )
    return dict(
        query=query, key=key, value=value,
        sparse_indices_full=sparse_indices_full,
        block_table=None,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
        # golden 用连续 BSND（BSND case 即原张量）
        golden_key=key, golden_value=value, golden_key_rope=key_rope,
    )


def _make_pa_bsnd_case():
    torch.manual_seed(43)
    B, S1, N1, N2, D, ROPE = 2, 4, 8, 1, 512, 64
    BLOCK_SIZE = 64
    BLOCKS_PER_BATCH = 2
    BLOCK_NUM = B * BLOCKS_PER_BATCH  # 4
    S2 = BLOCKS_PER_BATCH * BLOCK_SIZE  # 128
    device = "npu"
    dtype = torch.float16

    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    # PA: key/value 是 [block_num, block_size, N2, D] 池子；逻辑上每个 batch 占连续 BLOCKS_PER_BATCH 个块
    key_cache = (torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D, dtype=dtype, device=device) * 0.1)
    value_cache = (torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope_cache = (torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, ROPE, dtype=dtype, device=device) * 0.1)

    # block_table[b] = [b*2, b*2+1]：batch 0→物理块 0,1；batch 1→物理块 2,3
    block_table = torch.arange(BLOCK_NUM, dtype=torch.int32, device=device).view(B, BLOCKS_PER_BATCH)
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    # baseline 用 block_size=8 走 C_TEMPLATE（PA block_size=64 % 8 == 0，合法）。
    sparse_indices_full = (
        torch.arange(S2 // 8, dtype=torch.int32, device=device)
        .view(1, 1, 1, S2 // 8)
        .expand(B, S1, N2, S2 // 8)
        .contiguous()
    )
    # 把 paged KV 按 block_table 还原成连续 [B,S2,N2,*]，供 CPU golden 使用
    gk = torch.empty(B, S2, N2, D, dtype=dtype, device=device)
    gv = torch.empty(B, S2, N2, D, dtype=dtype, device=device)
    gkr = torch.empty(B, S2, N2, ROPE, dtype=dtype, device=device)
    for b in range(B):
        for blk in range(BLOCKS_PER_BATCH):
            phys = int(block_table[b, blk].item())
            sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
            gk[b, sl] = key_cache[phys]
            gv[b, sl] = value_cache[phys]
            gkr[b, sl] = key_rope_cache[phys]

    return dict(
        query=query, key=key_cache, value=value_cache,
        sparse_indices_full=sparse_indices_full,
        block_table=block_table,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope_cache,
        layout_query="BSND", layout_kv="PA_BSND",
        golden_key=gk, golden_value=gv, golden_key_rope=gkr,
    )


def _make_tnd_case():
    torch.manual_seed(44)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    T1, T2 = B * S1, B * S2  # 8, 256
    device = "npu"
    dtype = torch.float16
    query = (torch.randn(T1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(T2, N2, D, dtype=dtype, device=device) * 0.1)
    value = (torch.randn(T2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(T1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(T2, N2, ROPE, dtype=dtype, device=device) * 0.1)
    # TND actual_seq 是 cumulative 前缀和
    cum_q = [S1, 2 * S1]      # [4, 8]
    cum_kv = [S2, 2 * S2]     # [128, 256]
    actual_seq_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor(cum_kv, dtype=torch.int32, device=device)
    # baseline bs=8：TND sparse_indices [T1, N2, K]，每 batch 内全选 128 token = 16 个 block
    sparse_indices_full = (
        torch.arange(S2 // 8, dtype=torch.int32, device=device)
        .view(1, 1, S2 // 8)
        .expand(T1, N2, S2 // 8)
        .contiguous()
    )
    return dict(
        query=query, key=key, value=value,
        sparse_indices_full=sparse_indices_full,
        block_table=None,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="TND", layout_kv="TND",
        golden_key=key, golden_value=value, golden_key_rope=key_rope,
        cum_q=cum_q, cum_kv=cum_kv,
    )


def _run_pair_and_compare(case, tag):
    common_kwargs = {k: case[k] for k in (
        "query", "key", "value", "block_table",
        "actual_seq_q", "actual_seq_kv",
        "query_rope", "key_rope", "layout_query", "layout_kv")}

    # Run A: baseline = block_size=8 全选 -> C_TEMPLATE（isDense=false，读 valueGm，正确 P×V）
    out_sparse = _call_op(sparse_indices=case["sparse_indices_full"],
                          sparse_block_size=8, **common_kwargs)[0]
    torch.npu.synchronize()

    # Run B: dense 路径（None）-> C_TEMPLATE（isDense=true）。需先去掉 torch_adpt 的 arange dummy，
    # 否则 None 被 arange 拦截走 V_TEMPLATE。两者都走 C_TEMPLATE 读 valueGm，key≠value 也应一致。
    out_dense = _call_op(sparse_indices=None, **common_kwargs)[0]
    torch.npu.synchronize()

    out_sparse_cpu = out_sparse.float().cpu()
    out_dense_cpu = out_dense.float().cpu()
    max_abs = (out_dense_cpu - out_sparse_cpu).abs().max().item()
    print(f"[{tag}] max_abs_diff(dense, C_TEMPLATE bs=8) = {max_abs:.6e}")
    assert torch.allclose(out_dense_cpu, out_sparse_cpu, rtol=1e-3, atol=1e-3), \
        f"[{tag}] dense vs C_TEMPLATE bs=8 mismatch: max_abs_diff={max_abs:.6e}"

    # 直接对 CPU dense golden（不靠"dense==bs8"传递），把链条钉死
    if case["layout_query"] == "TND":
        golden = _cpu_golden_tnd(case["query"], case["golden_key"], case["golden_value"],
                                 case["query_rope"], case["golden_key_rope"], SCALE,
                                 case["cum_q"], case["cum_kv"])
    else:
        golden = _cpu_golden(case["query"], case["golden_key"], case["golden_value"],
                             case["query_rope"], case["golden_key_rope"], SCALE)
    diff_golden = (out_dense_cpu - golden).abs().max().item()
    print(f"[{tag}] max_abs_diff(dense, CPU golden)     = {diff_golden:.6e}")
    assert torch.allclose(out_dense_cpu, golden, rtol=1e-3, atol=3e-3), \
        f"[{tag}] dense vs CPU golden mismatch: max_abs_diff={diff_golden:.6e}"


@pytest.mark.step3c_dense
def test_dense_matches_sparse_full_bsnd():
    _run_pair_and_compare(_make_bsnd_case(), tag="BSND/BSND")


@pytest.mark.step3c_dense
def test_dense_matches_sparse_full_pa_bsnd():
    _run_pair_and_compare(_make_pa_bsnd_case(), tag="BSND/PA_BSND")


@pytest.mark.step3c_dense
def test_dense_matches_sparse_full_tnd():
    _run_pair_and_compare(_make_tnd_case(), tag="TND/TND")
