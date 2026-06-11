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
# Step 4 packed_kv extended tests.
#
# Covers layouts / modes absent from the basic packed_kv test:
#   - TND/TND
#   - BSND/PA_BSND
#   - sparse_mode=3 (causal truncation — tests zero-fill of truncated slots)
#   - sparse_block_size=2
#   - packed_kv + LSE combined
#
# Run manually (must source set_env.bash first):
#   pytest test_sparse_flash_attention_packed_kv_extended.py -m packed_ext -s -v

import math

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SCALE = 1.0 / (576 ** 0.5)


# ---------------------------------------------------------------------------
# Gather helpers
# ---------------------------------------------------------------------------

def _t1_to_batch_s1(t1, cum_q):
    """Map global T1 index to (batch_idx, s1_within_batch)."""
    for b, c in enumerate(cum_q):
        if t1 < c:
            prev = cum_q[b - 1] if b > 0 else 0
            return b, t1 - prev
    raise IndexError(f"t1={t1} out of range for cum_q={cum_q}")


def _gather_ref_bsnd(key, key_rope, sparse_indices):
    """Reference packed gather for BSND query + BSND key.

    key: [B, S2, N2, D]      key_rope: [B, S2, N2, ROPE]
    sparse_indices: [B, S1, N2, K]
    Returns (exp_key, exp_rope): ([B,S1,N2,K,D], [B,S1,N2,K,ROPE])
    """
    B, S1, N2, K = sparse_indices.shape
    D = key.shape[-1]
    ROPE = key_rope.shape[-1]
    idx = sparse_indices.long().cpu()
    k_c = key.float().cpu()
    r_c = key_rope.float().cpu()
    ek = torch.zeros(B, S1, N2, K, D)
    er = torch.zeros(B, S1, N2, K, ROPE)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                ids = idx[b, s1, n2]
                ek[b, s1, n2] = k_c[b, ids, n2, :]
                er[b, s1, n2] = r_c[b, ids, n2, :]
    return ek, er


def _gather_ref_tnd(key_bsnd, key_rope_bsnd, sparse_indices_tnd, cum_q):
    """Reference packed gather for TND query + BSND key.

    sparse_indices_tnd: [T1, N2, K]
    Returns (exp_key, exp_rope): ([T1,N2,K,D], [T1,N2,K,ROPE])
    """
    T1, N2, K = sparse_indices_tnd.shape
    D = key_bsnd.shape[-1]
    ROPE = key_rope_bsnd.shape[-1]
    idx = sparse_indices_tnd.long().cpu()
    k_c = key_bsnd.float().cpu()
    r_c = key_rope_bsnd.float().cpu()
    ek = torch.zeros(T1, N2, K, D)
    er = torch.zeros(T1, N2, K, ROPE)
    for t1 in range(T1):
        b, _ = _t1_to_batch_s1(t1, cum_q)
        for n2 in range(N2):
            ids = idx[t1, n2]
            ek[t1, n2] = k_c[b, ids, n2, :]
            er[t1, n2] = r_c[b, ids, n2, :]
    return ek, er


def _gather_ref_bsnd_block2(key, key_rope, sparse_indices):
    """Reference packed gather with sparse_block_size=2.

    Each sparse index selects 2 consecutive tokens.
    sparse_indices: [B, S1, N2, K]  (K block indices)
    Returns (exp_key, exp_rope): ([B,S1,N2,K*2,D], [B,S1,N2,K*2,ROPE])
    """
    B, S1, N2, K = sparse_indices.shape
    D = key.shape[-1]
    ROPE = key_rope.shape[-1]
    TOKENS_PER_BLOCK = 2
    NT = K * TOKENS_PER_BLOCK  # total tokens in packed output
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
                        break  # -1 terminates; remaining slots stay zero
                    for t in range(TOKENS_PER_BLOCK):
                        out_pos = i * TOKENS_PER_BLOCK + t
                        src_pos = blk * TOKENS_PER_BLOCK + t
                        ek[b, s1, n2, out_pos] = k_c[b, src_pos, n2, :]
                        er[b, s1, n2, out_pos] = r_c[b, src_pos, n2, :]
    return ek, er


def _resolve_pa_to_logical(key_cache, block_table, B, blocks_per_batch, block_size, N2, D):
    """Convert PA key cache to logical BSND for reference."""
    key_cpu = key_cache.cpu()
    bt_cpu = block_table.cpu()
    S2 = blocks_per_batch * block_size
    result = torch.zeros(B, S2, N2, D, dtype=key_cpu.dtype)
    for b in range(B):
        for blk_i in range(blocks_per_batch):
            phys = int(bt_cpu[b, blk_i])
            if phys == -1:
                continue
            start = blk_i * block_size
            end = start + block_size
            result[b, start:end] = key_cpu[phys, :, :, :]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.packed_ext
def test_packed_kv_tnd_tnd():
    """Packed KV for TND/TND layout (non-PA)."""
    torch.manual_seed(301)
    N1, N2, D, ROPE, K = 8, 1, 512, 64, 8
    device, dtype = "npu", torch.float16

    q_len = [2, 3]
    kv_len = [64, 128]
    T1 = sum(q_len)   # 5
    T2 = sum(kv_len)  # 192
    B = 2
    cum_q = [2, 5]
    cum_kv = [64, 192]
    actual_seq_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor(cum_kv, dtype=torch.int32, device=device)

    query = torch.randn(T1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(T2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(T1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(T2, N2, ROPE, dtype=dtype, device=device) * 0.1

    sel = torch.empty(T1, N2, K, dtype=torch.int32)
    for t1 in range(T1):
        b, _ = _t1_to_batch_s1(t1, cum_q)
        for n2 in range(N2):
            sel[t1, n2] = torch.randperm(kv_len[b])[:K].to(torch.int32)
    sparse_indices = sel.to(device)

    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=1,
        block_table=None,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="TND", layout_kv="TND",
        sparse_mode=0,
        return_softmax_lse=False,
        return_packed_kv=True,
    )
    torch.npu.synchronize()
    _, _, _, packed_key, packed_key_rope = output
    assert packed_key is not None and packed_key_rope is not None
    # TND packed shape: [T1, N2, K, D]
    assert tuple(packed_key.shape) == (T1, N2, K, D)
    assert tuple(packed_key_rope.shape) == (T1, N2, K, ROPE)

    # Resolve TND key to BSND for reference
    S2 = int(max(kv_len))
    key_bsnd = torch.zeros(B, S2, N2, D, dtype=dtype)
    key_rope_bsnd = torch.zeros(B, S2, N2, ROPE, dtype=dtype)
    off = 0
    for b in range(B):
        act_kv = kv_len[b]
        key_bsnd[b, :act_kv] = key[off:off + act_kv].unsqueeze(1)
        key_rope_bsnd[b, :act_kv] = key_rope[off:off + act_kv].unsqueeze(1)
        off += act_kv

    ek, er = _gather_ref_tnd(key_bsnd, key_rope_bsnd, sparse_indices, cum_q)
    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()

    max_k = (pk - ek).abs().max().item()
    max_r = (pkr - er).abs().max().item()
    print(f"[packed TND/TND] max_abs_diff key={max_k:.3e} rope={max_r:.3e}")
    assert torch.allclose(pk, ek, rtol=0, atol=1e-3), \
        f"packed_key mismatch max={max_k:.3e}"
    assert torch.allclose(pkr, er, rtol=0, atol=1e-3), \
        f"packed_key_rope mismatch max={max_r:.3e}"


@pytest.mark.packed_ext
def test_packed_kv_bsnd_pa_bsnd():
    """Packed KV for BSND/PA_BSND layout."""
    torch.manual_seed(302)
    B, S1, N1, N2, D, ROPE, K = 2, 4, 8, 1, 512, 64, 8
    BLOCK_SIZE = 64
    BLOCKS_PER_BATCH = 2
    BLOCK_NUM = B * BLOCKS_PER_BATCH
    S2 = BLOCKS_PER_BATCH * BLOCK_SIZE  # 128
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D,
                            dtype=dtype, device=device) * 0.1
    key_rope_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, ROPE,
                                 dtype=dtype, device=device) * 0.1
    block_table = (
        torch.arange(BLOCK_NUM, dtype=torch.int32, device=device)
        .view(B, BLOCKS_PER_BATCH)
    )
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                sel[b, s1, n2] = torch.randperm(S2)[:K].to(torch.int32)
    sparse_indices = sel.to(device)

    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key_cache, value=key_cache,
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope_cache,
        layout_query="BSND", layout_kv="PA_BSND",
        sparse_mode=0,
        return_softmax_lse=False,
        return_packed_kv=True,
    )
    torch.npu.synchronize()
    _, _, _, packed_key, packed_key_rope = output
    assert packed_key is not None and packed_key_rope is not None
    assert tuple(packed_key.shape) == (B, S1, N2, K, D)
    assert tuple(packed_key_rope.shape) == (B, S1, N2, K, ROPE)

    key_logical = _resolve_pa_to_logical(key_cache, block_table, B,
                                         BLOCKS_PER_BATCH, BLOCK_SIZE, N2, D)
    key_rope_logical = _resolve_pa_to_logical(key_rope_cache, block_table, B,
                                              BLOCKS_PER_BATCH, BLOCK_SIZE, N2, ROPE)
    ek, er = _gather_ref_bsnd(key_logical, key_rope_logical, sparse_indices)
    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()

    max_k = (pk - ek).abs().max().item()
    max_r = (pkr - er).abs().max().item()
    print(f"[packed BSND/PA_BSND] max_abs_diff key={max_k:.3e} rope={max_r:.3e}")
    assert torch.allclose(pk, ek, rtol=0, atol=1e-3), \
        f"packed_key mismatch max={max_k:.3e}"
    assert torch.allclose(pkr, er, rtol=0, atol=1e-3), \
        f"packed_key_rope mismatch max={max_r:.3e}"


@pytest.mark.packed_ext
def test_packed_kv_mode3():
    """Packed KV with sparse_mode=3 (causal).

    Early s1 positions have smaller causal window → fewer valid tokens.
    Truncated slots should be zero-filled by the kernel.
    """
    torch.manual_seed(303)
    B, S1, S2, N1, N2, D, ROPE, K = 2, 5, 256, 8, 1, 512, 64, 16
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([3, 5], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([128, 256], dtype=torch.int32, device=device)

    # For mode 3, ensure early s1 positions have small causal window.
    # Use valid block indices (no -1), but threshold limits how many are used.
    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        kv_len = int(actual_seq_kv[b])
        for s1 in range(int(actual_seq_q[b])):
            for n2 in range(N2):
                # All valid indices, -1 for beyond K
                n_valid = min(K, kv_len)
                sel[b, s1, n2, :n_valid] = torch.arange(n_valid, dtype=torch.int32)
                sel[b, s1, n2, n_valid:] = -1
    sparse_indices = sel.to(device)

    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=1,
        block_table=None,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
        sparse_mode=3,
        return_softmax_lse=False,
        return_packed_kv=True,
    )
    torch.npu.synchronize()
    _, _, _, packed_key, packed_key_rope = output
    assert packed_key is not None and packed_key_rope is not None
    assert tuple(packed_key.shape) == (B, S1, N2, K, D)
    assert tuple(packed_key_rope.shape) == (B, S1, N2, K, ROPE)

    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()
    key_c = key.float().cpu()
    rope_c = key_rope.float().cpu()
    sel_c = sel.cpu()

    # Verify: for each (b,s1,n2,i), if i < threshold(s1) → must match gather;
    # otherwise → must be zero.
    errors = 0
    max_err = 0.0
    for b in range(B):
        act_q = int(actual_seq_q[b])
        act_kv = int(actual_seq_kv[b])
        for s1 in range(act_q):
            # mode 3 threshold: tokens 0 .. (act_kv - act_q + s1)
            threshold = act_kv - act_q + s1 + 1  # exclusive upper bound
            for n2 in range(N2):
                for i in range(K):
                    blk = int(sel_c[b, s1, n2, i])
                    actual, expected_k, expected_r = None, None, None
                    if blk >= 0 and blk * 1 < threshold:
                        # within valid range: must match gather
                        actual = pk[b, s1, n2, i, :]
                        expected_k = key_c[b, blk, n2, :]
                        expected_r = rope_c[b, blk, n2, :]
                    else:
                        # truncated: must be zero
                        actual = pk[b, s1, n2, i, :]
                        expected_k = torch.zeros(D)
                        expected_r = torch.zeros(ROPE)
                    diff_k = (actual - expected_k).abs().max().item()
                    diff_r = (pkr[b, s1, n2, i, :] - expected_r).abs().max().item()
                    max_err = max(max_err, diff_k, diff_r)
                    if diff_k > 1e-3 or diff_r > 1e-3:
                        errors += 1

    print(f"[packed mode3] max_err={max_err:.3e} errors={errors}")
    assert errors == 0, f"Packed KV mode3 mismatch: {errors} slots wrong, max_err={max_err:.3e}"


@pytest.mark.packed_ext
def test_packed_kv_block2():
    """Packed KV with sparse_block_size=2."""
    torch.manual_seed(304)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    BLOCK_SIZE = 2   # sparse_block_size
    K = 4            # number of block indices
    NT = K * BLOCK_SIZE  # total tokens in packed output = 8
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    # Select K blocks, each covering 2 tokens. So max block id = S2/2 - 1 = 63
    MAX_BLK = S2 // BLOCK_SIZE - 1  # 63
    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                sel[b, s1, n2] = torch.randperm(MAX_BLK + 1)[:K].to(torch.int32)
    sparse_indices = sel.to(device)

    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=BLOCK_SIZE,
        block_table=None,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
        sparse_mode=0,
        return_softmax_lse=False,
        return_packed_kv=True,
    )
    torch.npu.synchronize()
    _, _, _, packed_key, packed_key_rope = output
    assert packed_key is not None and packed_key_rope is not None
    # Each selected block gives BLOCK_SIZE=2 tokens → NT = K * 2
    assert tuple(packed_key.shape) == (B, S1, N2, NT, D)
    assert tuple(packed_key_rope.shape) == (B, S1, N2, NT, ROPE)

    ek, er = _gather_ref_bsnd_block2(key, key_rope, sparse_indices)
    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()

    max_k = (pk - ek).abs().max().item()
    max_r = (pkr - er).abs().max().item()
    print(f"[packed block2] max_abs_diff key={max_k:.3e} rope={max_r:.3e}")
    assert torch.allclose(pk, ek, rtol=0, atol=1e-3), \
        f"packed_key block2 mismatch max={max_k:.3e}"
    assert torch.allclose(pkr, er, rtol=0, atol=1e-3), \
        f"packed_key_rope block2 mismatch max={max_r:.3e}"


@pytest.mark.packed_ext
def test_packed_kv_with_lse():
    """Packed KV + LSE combined (both return_packed_kv and return_softmax_lse)."""
    torch.manual_seed(305)
    B, S1, S2, N1, N2, D, ROPE, K = 2, 4, 128, 8, 1, 512, 64, 8
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                sel[b, s1, n2] = torch.randperm(S2)[:K].to(torch.int32)
    sparse_indices = sel.to(device)

    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=1,
        block_table=None,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
        sparse_mode=0,
        return_softmax_lse=True,
        return_packed_kv=True,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, packed_key, packed_key_rope = output

    # 1. Verify packed KV (element-wise gather)
    ek, er = _gather_ref_bsnd(key, key_rope, sparse_indices)
    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()
    max_k = (pk - ek).abs().max().item()
    max_r = (pkr - er).abs().max().item()
    print(f"[packed+LSE] packed max_abs_diff key={max_k:.3e} rope={max_r:.3e}")
    assert torch.allclose(pk, ek, rtol=0, atol=1e-3), \
        f"packed+LSE: packed_key mismatch max={max_k:.3e}"
    assert torch.allclose(pkr, er, rtol=0, atol=1e-3), \
        f"packed+LSE: packed_key_rope mismatch max={max_r:.3e}"

    # 2. Verify LSE outputs against reference (from lse test file)
    from test_sparse_flash_attention_lse import _lse_ref_bsnd
    ref_out, ref_max, ref_sum = _lse_ref_bsnd(
        query, key, sparse_indices, SCALE,
        actual_seq_q, actual_seq_kv, sparse_mode=0, sparse_block_size=1)

    out_d = (attn_out.float().cpu() - ref_out).abs().max().item()
    max_d = (lse_max.float().cpu() - ref_max).abs().max().item()
    sum_d = (lse_sum.float().cpu() - ref_sum).abs().max().item()
    print(f"[packed+LSE] attn_out diff={out_d:.6e}  "
          f"softmax_max diff={max_d:.6e}  softmax_sum diff={sum_d:.6e}")
    assert torch.allclose(attn_out.float().cpu(), ref_out, rtol=1e-3, atol=1e-3), \
        f"packed+LSE: attn_out mismatch"
    assert torch.allclose(lse_max.float().cpu(), ref_max, rtol=1e-3, atol=1e-3), \
        f"packed+LSE: softmax_max mismatch"
    assert torch.allclose(lse_sum.float().cpu(), ref_sum, rtol=1e-3, atol=1e-3), \
        f"packed+LSE: softmax_sum mismatch"
