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
