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
# Compare the dense path (sparse_indices=None) against the sparse path with all-blocks-selected
# sparse_indices on the same kernel binary. With sparse_block_size=1 and K=S2, the sparse path
# walks tokens 0..S2-1 in order; the dense path walks the same range. Their attn_out must match
# up to fp16 rounding noise.
#
# Covers two layouts:
#   - BSND / BSND        : exercises the BSND+BSND key/value offset path (no block table)
#   - BSND / PA_BSND     : exercises DataCopyPA with isDense
#
# Run manually:
#   pytest test_sparse_flash_attention_dense_correctness.py -m step3c_dense -s -v

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


SCALE = 1.0 / (576 ** 0.5)


def _call_op(query, key, value, sparse_indices, *,
             block_table, actual_seq_q, actual_seq_kv,
             query_rope, key_rope, layout_query, layout_kv):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query=layout_query, layout_kv=layout_kv,
        sparse_mode=0,
        return_softmax_lse=False,
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
    # 全选 sparse_indices：每 [b, s1, n2, :] = [0,1,...,S2-1]
    sparse_indices_full = (
        torch.arange(S2, dtype=torch.int32, device=device)
        .view(1, 1, 1, S2)
        .expand(B, S1, N2, S2)
        .contiguous()
    )
    return dict(
        query=query, key=key, value=value,
        sparse_indices_full=sparse_indices_full,
        block_table=None,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
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

    sparse_indices_full = (
        torch.arange(S2, dtype=torch.int32, device=device)
        .view(1, 1, 1, S2)
        .expand(B, S1, N2, S2)
        .contiguous()
    )
    return dict(
        query=query, key=key_cache, value=value_cache,
        sparse_indices_full=sparse_indices_full,
        block_table=block_table,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope_cache,
        layout_query="BSND", layout_kv="PA_BSND",
    )


def _run_pair_and_compare(case, tag):
    common_kwargs = {k: case[k] for k in (
        "query", "key", "value", "block_table",
        "actual_seq_q", "actual_seq_kv",
        "query_rope", "key_rope", "layout_query", "layout_kv")}

    # Run A: sparse 路径 + 全选 indices
    out_sparse = _call_op(sparse_indices=case["sparse_indices_full"], **common_kwargs)[0]
    torch.npu.synchronize()

    # Run B: dense 路径
    out_dense = _call_op(sparse_indices=None, **common_kwargs)[0]
    torch.npu.synchronize()

    out_sparse_cpu = out_sparse.float().cpu()
    out_dense_cpu = out_dense.float().cpu()
    max_abs = (out_dense_cpu - out_sparse_cpu).abs().max().item()
    print(f"[{tag}] max_abs_diff(dense, sparse-full) = {max_abs:.6e}")
    assert torch.allclose(out_dense_cpu, out_sparse_cpu, rtol=1e-3, atol=1e-3), \
        f"[{tag}] dense vs sparse-full mismatch: max_abs_diff={max_abs:.6e}"


@pytest.mark.step3c_dense
def test_dense_matches_sparse_full_bsnd():
    _run_pair_and_compare(_make_bsnd_case(), tag="BSND/BSND")


@pytest.mark.step3c_dense
def test_dense_matches_sparse_full_pa_bsnd():
    _run_pair_and_compare(_make_pa_bsnd_case(), tag="BSND/PA_BSND")
