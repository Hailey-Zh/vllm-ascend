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
# Step 3 dense-mode extended tests.
#
# Fills coverage gaps left by the basic dense correctness test:
#   - TND/TND dense vs sparse-full
#   - TND/PA_BSND dense vs sparse-full
#   - Dense + LSE combined (BSND/BSND)
#
# Run manually (must source set_env.bash first):
#   pytest test_sparse_flash_attention_dense_extended.py -m dense_ext -s -v

import math

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SCALE = 1.0 / (576 ** 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t1_to_batch_s1(t1, cum_q):
    """Map global T1 index to (batch_idx, s1_within_batch)."""
    for b, c in enumerate(cum_q):
        if t1 < c:
            prev = cum_q[b - 1] if b > 0 else 0
            return b, t1 - prev
    raise IndexError(f"t1={t1} out of range for cum_q={cum_q}")


def _call_op(query, key, value, sparse_indices, *,
             block_table, actual_seq_q, actual_seq_kv,
             query_rope, key_rope, layout_query, layout_kv,
             return_softmax_lse=False):
    """Thin wrapper around the SFA op call."""
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
        return_softmax_lse=return_softmax_lse,
    )


def _make_full_sparse_indices_tnd(T1, N2, kv_len, cum_q, device):
    """Create sparse_indices [T1, N2, max_kv] that select ALL kv tokens per batch.
    Unused positions are -1.
    """
    max_kv = max(kv_len)
    sel = torch.full((T1, N2, max_kv), -1, dtype=torch.int32, device=device)
    for t1 in range(T1):
        b, _ = _t1_to_batch_s1(t1, cum_q)
        n_tokens = kv_len[b]
        for n2 in range(N2):
            sel[t1, n2, :n_tokens] = torch.arange(n_tokens, dtype=torch.int32, device=device)
    return sel


# ---------------------------------------------------------------------------
# TND/TND dense vs sparse-full
# ---------------------------------------------------------------------------

@pytest.mark.dense_ext
def test_dense_matches_sparse_full_tnd_tnd():
    """Dense path (sparse_indices=None) matches sparse-full for TND/TND layout."""
    torch.manual_seed(201)
    N1, N2, D, ROPE = 8, 1, 512, 64
    device, dtype = "npu", torch.float16

    q_len = [2, 4]          # batch 0: 2 query tokens, batch 1: 4
    kv_len = [64, 128]      # batch 0: 64 kv tokens, batch 1: 128
    T1 = sum(q_len)         # 6
    T2 = sum(kv_len)        # 192
    B = 2
    cum_q = [2, 6]
    cum_kv = [64, 192]

    actual_seq_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor(cum_kv, dtype=torch.int32, device=device)

    query = torch.randn(T1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(T2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(T1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(T2, N2, ROPE, dtype=dtype, device=device) * 0.1

    sparse_indices_full = _make_full_sparse_indices_tnd(
        T1, N2, kv_len, cum_q, device)

    common_kwargs = dict(
        query=query, key=key, value=value,
        block_table=None,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="TND", layout_kv="TND",
    )

    out_sparse = _call_op(sparse_indices=sparse_indices_full, **common_kwargs)[0]
    torch.npu.synchronize()

    out_dense = _call_op(sparse_indices=None, **common_kwargs)[0]
    torch.npu.synchronize()

    out_s = out_sparse.float().cpu()
    out_d = out_dense.float().cpu()
    max_abs = (out_d - out_s).abs().max().item()
    print(f"[TND/TND dense] max_abs_diff(dense, sparse-full) = {max_abs:.6e}")
    assert torch.allclose(out_d, out_s, rtol=1e-3, atol=1e-3), \
        f"TND/TND dense vs sparse-full mismatch: max_abs_diff={max_abs:.6e}"


# ---------------------------------------------------------------------------
# TND/PA_BSND dense vs sparse-full
# ---------------------------------------------------------------------------

@pytest.mark.dense_ext
def test_dense_matches_sparse_full_tnd_pa_bsnd():
    """Dense path matches sparse-full for TND/PA_BSND (production layout)."""
    torch.manual_seed(202)
    N1, N2, D, ROPE = 8, 1, 512, 64
    BLOCK_SIZE = 64
    BLOCKS_PER_BATCH = 2
    B = 2
    BLOCK_NUM = B * BLOCKS_PER_BATCH
    device, dtype = "npu", torch.float16

    q_len = [3, 2]
    kv_len = [128, 128]  # 2 blocks each, same kv length for convenience
    T1 = sum(q_len)  # 5
    cum_q = [3, 5]

    actual_seq_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor(kv_len, dtype=torch.int32, device=device)

    query = torch.randn(T1, N1, D, dtype=dtype, device=device) * 0.1
    query_rope = torch.randn(T1, N1, ROPE, dtype=dtype, device=device) * 0.1

    key_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D,
                            dtype=dtype, device=device) * 0.1
    key_rope_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, ROPE,
                                 dtype=dtype, device=device) * 0.1
    block_table = (
        torch.arange(BLOCK_NUM, dtype=torch.int32, device=device)
        .view(B, BLOCKS_PER_BATCH)
    )

    sparse_indices_full = _make_full_sparse_indices_tnd(
        T1, N2, kv_len, cum_q, device)

    common_kwargs = dict(
        query=query, key=key_cache, value=key_cache,
        block_table=block_table,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope_cache,
        layout_query="TND", layout_kv="PA_BSND",
    )

    out_sparse = _call_op(sparse_indices=sparse_indices_full, **common_kwargs)[0]
    torch.npu.synchronize()

    out_dense = _call_op(sparse_indices=None, **common_kwargs)[0]
    torch.npu.synchronize()

    out_s = out_sparse.float().cpu()
    out_d = out_dense.float().cpu()
    max_abs = (out_d - out_s).abs().max().item()
    print(f"[TND/PA_BSND dense] max_abs_diff(dense, sparse-full) = {max_abs:.6e}")
    assert torch.allclose(out_d, out_s, rtol=1e-3, atol=1e-3), \
        f"TND/PA_BSND dense vs sparse-full mismatch: max_abs_diff={max_abs:.6e}"


# ---------------------------------------------------------------------------
# Dense + LSE combined (BSND/BSND)
# ---------------------------------------------------------------------------

def _dense_lse_ref_bsnd(query, key, scale, actual_seq_q, actual_seq_kv):
    """Reference dense attention with LSE for BSND layout.

    query: [B, S1, N1, D]     key: [B, S2, N2, D]
    Returns (attn_out, softmax_max, softmax_sum) — all fp32 on cpu.
      attn_out:    [B, S1, N1, D]
      softmax_max: [B, N2, S1, g]   g = N1/N2
      softmax_sum: [B, N2, S1, g]
    """
    B, S1, N1, D = query.shape
    _, _, N2, _ = key.shape
    g = N1 // N2

    q = query.float().cpu()
    k = key.float().cpu()
    aq = actual_seq_q.cpu()
    akv = actual_seq_kv.cpu()

    out = torch.zeros(B, S1, N1, D)
    lse_max = torch.zeros(B, N2, S1, g)
    lse_sum = torch.zeros(B, N2, S1, g)

    for b in range(B):
        act_q = int(aq[b])
        act_kv = int(akv[b])
        for s1 in range(act_q):
            for n2 in range(N2):
                qh = q[b, s1, n2 * g:(n2 + 1) * g, :]           # [g, D]
                kh = k[b, :act_kv, n2, :]                         # [act_kv, D]
                scores = torch.matmul(qh, kh.T) * scale            # [g, act_kv]
                smax = scores.max(dim=-1).values                  # [g]
                ssub = scores - smax.unsqueeze(-1)
                sexp_sum = ssub.exp().sum(dim=-1)                 # [g]
                attn_w = (ssub.exp() / sexp_sum.unsqueeze(-1)).half()
                attn_o = torch.matmul(attn_w.float(), kh)          # [g, D]
                out[b, s1, n2 * g:(n2 + 1) * g, :] = attn_o
                lse_max[b, n2, s1, :] = smax
                lse_sum[b, n2, s1, :] = sexp_sum

    return out, lse_max, lse_sum


@pytest.mark.dense_ext
def test_dense_lse_bsnd_bsnd():
    """Dense mode with return_softmax_lse=True for BSND/BSND."""
    torch.manual_seed(203)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=None,
        scale_value=SCALE, sparse_block_size=1,
        block_table=None,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
        sparse_mode=0,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    ref_out, ref_max, ref_sum = _dense_lse_ref_bsnd(
        query, key, SCALE, actual_seq_q, actual_seq_kv)

    # attn_out: fp16 output, standard tolerance
    out_diff = (attn_out.float().cpu() - ref_out).abs().max().item()
    print(f"[dense+LSE BSND] attn_out max_diff={out_diff:.6e}")
    assert torch.allclose(attn_out.float().cpu(), ref_out, rtol=1e-3, atol=1e-3), \
        f"dense+LSE attn_out mismatch: max_diff={out_diff:.6e}"

    # LSE: fp32 output, tighter tolerance
    max_diff = (lse_max.float().cpu() - ref_max).abs().max().item()
    sum_diff = (lse_sum.float().cpu() - ref_sum).abs().max().item()
    print(f"[dense+LSE BSND] softmax_max max_diff={max_diff:.6e}  "
          f"softmax_sum max_diff={sum_diff:.6e}")
    assert torch.allclose(lse_max.float().cpu(), ref_max, rtol=1e-3, atol=1e-3), \
        f"dense+LSE softmax_max mismatch: max_diff={max_diff:.6e}"
    assert torch.allclose(lse_sum.float().cpu(), ref_sum, rtol=1e-3, atol=1e-3), \
        f"dense+LSE softmax_sum mismatch: max_diff={sum_diff:.6e}"
