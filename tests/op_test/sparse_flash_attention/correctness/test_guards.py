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
# Step 5 guard / boundary tests.
#
# Verify the two host-side guards added for packed_kv (sparse_flash_attention_torch_adpt.h):
#   - return_packed_kv=True requires sparse mode (sparse_indices must be provided)
#   - return_packed_kv=True requires sparse_block_size <= 4 (MergeKv hard limit)
# Plus the sparse_block_size=4 boundary, which must run and gather correctly.
#
# Run manually (must source set_env.bash first):
#   pytest test_guards.py -m guards -s -v

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SCALE = 1.0 / (576 ** 0.5)


def _make_bsnd_inputs(B=2, S1=4, S2=128, N1=8, N2=1, D=512, ROPE=64,
                      seed=401, dtype=torch.float16):
    torch.manual_seed(seed)
    device = "npu"
    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([S1] * B, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2] * B, dtype=torch.int32, device=device)
    return dict(query=query, key=key, value=key,
                query_rope=query_rope, key_rope=key_rope,
                actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
                B=B, S1=S1, S2=S2, N2=N2, D=D, ROPE=ROPE)


def _gather_ref_block_n(key, key_rope, sparse_indices, tokens_per_block):
    """Reference packed gather where each block index expands to
    `tokens_per_block` consecutive tokens. -1 terminates the row.

    key: [B,S2,N2,D]  key_rope: [B,S2,N2,ROPE]  sparse_indices: [B,S1,N2,K]
    Returns (exp_key [B,S1,N2,K*tpb,D], exp_rope [B,S1,N2,K*tpb,ROPE]).
    """
    B, S1, N2, K = sparse_indices.shape
    D = key.shape[-1]
    ROPE = key_rope.shape[-1]
    NT = K * tokens_per_block
    idx = sparse_indices.long().cpu()
    k_c = key.float().cpu()
    r_c = key_rope.float().cpu()
    ek = torch.zeros(B, S1, N2, NT, D)
    er = torch.zeros(B, S1, N2, NT, ROPE)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                for i in range(K):
                    blk = int(idx[b, s1, n2, i])
                    if blk == -1:
                        break
                    for t in range(tokens_per_block):
                        out_pos = i * tokens_per_block + t
                        src_pos = blk * tokens_per_block + t
                        ek[b, s1, n2, out_pos] = k_c[b, src_pos, n2, :]
                        er[b, s1, n2, out_pos] = r_c[b, src_pos, n2, :]
    return ek, er


# ---------------------------------------------------------------------------
# Guard #1: return_packed_kv=True requires sparse_indices
# ---------------------------------------------------------------------------

@pytest.mark.guards
def test_dense_with_packed_kv_raises():
    """sparse_indices=None + return_packed_kv=True must raise (dense has no MergeKv)."""
    inp = _make_bsnd_inputs(seed=401)
    with pytest.raises(RuntimeError, match="return_packed_kv only supports sparse"):
        torch.ops._C_ascend.npu_sparse_flash_attention(
            query=inp["query"], key=inp["key"], value=inp["value"],
            sparse_indices=None,
            scale_value=SCALE, sparse_block_size=1,
            block_table=None,
            actual_seq_lengths_query=inp["actual_seq_q"],
            actual_seq_lengths_kv=inp["actual_seq_kv"],
            query_rope=inp["query_rope"], key_rope=inp["key_rope"],
            layout_query="BSND", layout_kv="BSND",
            sparse_mode=0,
            return_softmax_lse=False,
            return_packed_kv=True,
        )


# ---------------------------------------------------------------------------
# Guard #2: return_packed_kv=True requires sparse_block_size <= 4
# ---------------------------------------------------------------------------

@pytest.mark.guards
def test_packed_kv_block_size_over_4_raises():
    """sparse_block_size=8 + return_packed_kv=True must raise (MergeKv limit)."""
    inp = _make_bsnd_inputs(seed=402)
    B, S1, N2 = inp["B"], inp["S1"], inp["N2"]
    K = 4
    # valid block ids for block_size=8: max id = S2/8 - 1 = 15
    sel = torch.randint(0, inp["S2"] // 8, (B, S1, N2, K), dtype=torch.int32).to("npu")
    with pytest.raises(RuntimeError, match="sparse_block_size <= 4"):
        torch.ops._C_ascend.npu_sparse_flash_attention(
            query=inp["query"], key=inp["key"], value=inp["value"],
            sparse_indices=sel,
            scale_value=SCALE, sparse_block_size=8,
            block_table=None,
            actual_seq_lengths_query=inp["actual_seq_q"],
            actual_seq_lengths_kv=inp["actual_seq_kv"],
            query_rope=inp["query_rope"], key_rope=inp["key_rope"],
            layout_query="BSND", layout_kv="BSND",
            sparse_mode=0,
            return_softmax_lse=False,
            return_packed_kv=True,
        )


# ---------------------------------------------------------------------------
# Boundary: sparse_block_size=4 must run and gather correctly
# ---------------------------------------------------------------------------

@pytest.mark.guards
def test_packed_kv_block_size_4_ok():
    """sparse_block_size=4 (the documented max for packed_kv) runs and gathers."""
    inp = _make_bsnd_inputs(seed=403)
    B, S1, S2, N2, D, ROPE = inp["B"], inp["S1"], inp["S2"], inp["N2"], inp["D"], inp["ROPE"]
    BLOCK_SIZE = 4
    K = 4                       # block indices
    NT = K * BLOCK_SIZE         # total packed tokens = 16
    MAX_BLK = S2 // BLOCK_SIZE - 1   # 31

    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                sel[b, s1, n2] = torch.randperm(MAX_BLK + 1)[:K].to(torch.int32)
    sparse_indices = sel.to("npu")

    out, _, _, packed_key, packed_key_rope = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=inp["query"], key=inp["key"], value=inp["value"],
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=BLOCK_SIZE,
        block_table=None,
        actual_seq_lengths_query=inp["actual_seq_q"],
        actual_seq_lengths_kv=inp["actual_seq_kv"],
        query_rope=inp["query_rope"], key_rope=inp["key_rope"],
        layout_query="BSND", layout_kv="BSND",
        sparse_mode=0,
        return_softmax_lse=False,
        return_packed_kv=True,
    )
    torch.npu.synchronize()

    assert packed_key is not None and packed_key_rope is not None
    assert tuple(packed_key.shape) == (B, S1, N2, NT, D)
    assert tuple(packed_key_rope.shape) == (B, S1, N2, NT, ROPE)

    ek, er = _gather_ref_block_n(inp["key"], inp["key_rope"], sparse_indices, BLOCK_SIZE)
    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()
    max_k = (pk - ek).abs().max().item()
    max_r = (pkr - er).abs().max().item()
    print(f"[packed block4] max_abs_diff key={max_k:.3e} rope={max_r:.3e}")
    assert torch.allclose(pk, ek, rtol=0, atol=1e-3), f"packed_key block4 mismatch max={max_k:.3e}"
    assert torch.allclose(pkr, er, rtol=0, atol=1e-3), f"packed_key_rope block4 mismatch max={max_r:.3e}"
