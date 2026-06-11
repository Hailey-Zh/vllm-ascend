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
# Step 2 LSE correctness test.
#
# Verify softmax_max and softmax_sum outputs from the SFA kernel when
# return_softmax_lse=True. Compares NPU output against a pure-PyTorch
# reference that replicates the kernel's sparse-selection + softmax logic.
#
# Run manually (must source set_env.bash first):
#   pytest test_lse.py -m lse -s -v

import math

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SCALE = 1.0 / (576 ** 0.5)


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def _gather_token_ids(sparse_indices_row, sparse_block_size, threshold):
    """Replicate kernel gatherVGatherIndices logic (kernel_mla.h:294-309).
    Expands block indices to individual token indices, clamping to threshold.

    sparse_indices_row: 1-D tensor of block ids, length K
    Returns list of token ids in [0, threshold)
    """
    K = len(sparse_indices_row)
    if threshold <= 0:
        return []
    valid_count = min(K, math.ceil(threshold / sparse_block_size))
    tokens = []
    for i in range(valid_count):
        blk_id = int(sparse_indices_row[i])
        if blk_id == -1:
            break
        begin = blk_id * sparse_block_size
        if begin >= threshold:
            continue
        end = min(begin + sparse_block_size, threshold)
        tokens.extend(range(begin, end))
    return tokens


def _lse_ref_bsnd(query_bsnd, key_bsnd, query_rope_bsnd, key_rope_bsnd,
                  sparse_indices_bsnd, scale,
                  actual_seq_q, actual_seq_kv, sparse_mode, sparse_block_size):
    """Compute reference attention + LSE for BSND query and BSND key.

    MLA score uses the FULL 576-dim vector = NoPE(D=512) concat RoPE(64):
        score = (q_nope . k_nope + q_rope . k_rope) * scale
    The value (bmm2) uses ONLY the NoPE part (matches golden:
    v = k_bnsd[..., :512]).

    query_bsnd:      [B, S1, N1, D]      key_bsnd:      [B, S2, N2, D]
    query_rope_bsnd: [B, S1, N1, ROPE]   key_rope_bsnd: [B, S2, N2, ROPE]
    sparse_indices_bsnd: [B, S1, N2, K] int32
    Returns (attn_out, softmax_max, softmax_sum):
      attn_out:    [B, S1, N1, D]  fp32
      softmax_max: [B, N2, S1, g]  fp32  (g = N1/N2)
      softmax_sum: [B, N2, S1, g]  fp32
    """
    B, S1, N1, D = query_bsnd.shape
    _, _, N2, _ = key_bsnd.shape
    g = N1 // N2

    q = query_bsnd.float().cpu()
    k = key_bsnd.float().cpu()
    qr = query_rope_bsnd.float().cpu()
    kr = key_rope_bsnd.float().cpu()
    si = sparse_indices_bsnd.cpu()
    aq = actual_seq_q.cpu()
    akv = actual_seq_kv.cpu()

    out = torch.zeros(B, S1, N1, D)
    lse_max = torch.zeros(B, N2, S1, g)
    lse_sum = torch.zeros(B, N2, S1, g)

    for b in range(B):
        act_q = int(aq[b])
        act_kv = int(akv[b])
        for s1 in range(act_q):
            threshold = _threshold(sparse_mode, act_kv, act_q, s1)
            if threshold <= 0:
                continue
            for n2 in range(N2):
                tokens = _gather_token_ids(si[b, s1, n2], sparse_block_size, threshold)
                if not tokens:
                    continue
                qh = q[b, s1, n2 * g:(n2 + 1) * g, :]         # [g, D]
                kh = k[b, tokens, n2, :]                       # [T, D]
                qh_r = qr[b, s1, n2 * g:(n2 + 1) * g, :]       # [g, ROPE]
                kh_r = kr[b, tokens, n2, :]                    # [T, ROPE]
                # full QK = NoPE.NoPE + RoPE.RoPE
                qh_full = torch.cat([qh, qh_r], dim=-1)        # [g, D+ROPE]
                kh_full = torch.cat([kh, kh_r], dim=-1)        # [T, D+ROPE]

                scores = torch.matmul(qh_full, kh_full.T) * scale  # [g, T] fp32
                smax = scores.max(dim=-1).values              # [g]
                ssub = scores - smax.unsqueeze(-1)
                sexp_sum = ssub.exp().sum(dim=-1)             # [g]
                # bmm2: kernel casts softmax to fp16 before matmul; value = NoPE only
                attn_w = (ssub.exp() / sexp_sum.unsqueeze(-1)).half()
                attn_o = torch.matmul(attn_w.float(), kh)      # [g, D]
                out[b, s1, n2 * g:(n2 + 1) * g, :] = attn_o
                lse_max[b, n2, s1, :] = smax
                lse_sum[b, n2, s1, :] = sexp_sum

    return out, lse_max, lse_sum


def _lse_ref_tnd(query_tnd, key_bsnd, query_rope_tnd, key_rope_bsnd,
                 sparse_indices_tnd, scale,
                 actual_seq_q_cum, actual_seq_kv_len, sparse_mode, sparse_block_size):
    """Compute reference for TND query + BSND key (MLA full-576 QK score).

    query_tnd:           [T1, N1, D]       query_rope_tnd: [T1, N1, ROPE]
    key_bsnd:            [B, S2, N2, D]     key_rope_bsnd:  [B, S2, N2, ROPE]
    sparse_indices_tnd:  [T1, N2, K]
    actual_seq_q_cum:    [B]  — TND-style cumulative prefix sums
    actual_seq_kv_len:   [B]  — per-batch KV lengths (NOT cumsum)
    Returns (attn_out, softmax_max, softmax_sum):
      attn_out:    [T1, N1, D]  fp32
      softmax_max: [N2, T1, g]  fp32
      softmax_sum: [N2, T1, g]  fp32
    """
    T1, N1, D = query_tnd.shape
    _, _, N2, _ = key_bsnd.shape
    g = N1 // N2
    B = len(actual_seq_q_cum)

    q = query_tnd.float().cpu()
    k = key_bsnd.float().cpu()
    qr = query_rope_tnd.float().cpu()
    kr = key_rope_bsnd.float().cpu()
    si = sparse_indices_tnd.cpu()
    aq_cum = actual_seq_q_cum.cpu()
    akv = actual_seq_kv_len.cpu()  # per-batch KV lengths (already converted)

    # per-batch query lengths from cumsum
    aq_len = [int(aq_cum[0])]
    for i in range(1, B):
        aq_len.append(int(aq_cum[i] - aq_cum[i - 1]))
    assert sum(aq_len) == T1

    out = torch.zeros(T1, N1, D)
    lse_max = torch.zeros(N2, T1, g)
    lse_sum = torch.zeros(N2, T1, g)

    for t1 in range(T1):
        # map global T1 position to (batch, s1)
        b, s1 = _t1_to_batch_s1(t1, aq_cum)
        act_kv = int(akv[b])
        act_q = aq_len[b]
        threshold = _threshold(sparse_mode, act_kv, act_q, s1)
        if threshold <= 0:
            continue
        for n2 in range(N2):
            tokens = _gather_token_ids(si[t1, n2], sparse_block_size, threshold)
            if not tokens:
                continue
            qh = q[t1, n2 * g:(n2 + 1) * g, :]
            kh = k[b, tokens, n2, :]
            qh_r = qr[t1, n2 * g:(n2 + 1) * g, :]
            kh_r = kr[b, tokens, n2, :]
            qh_full = torch.cat([qh, qh_r], dim=-1)
            kh_full = torch.cat([kh, kh_r], dim=-1)

            scores = torch.matmul(qh_full, kh_full.T) * scale
            smax = scores.max(dim=-1).values
            ssub = scores - smax.unsqueeze(-1)
            sexp_sum = ssub.exp().sum(dim=-1)
            attn_w = (ssub.exp() / sexp_sum.unsqueeze(-1)).half()
            attn_o = torch.matmul(attn_w.float(), kh)  # value = NoPE only
            out[t1, n2 * g:(n2 + 1) * g, :] = attn_o
            lse_max[n2, t1, :] = smax
            lse_sum[n2, t1, :] = sexp_sum

    return out, lse_max, lse_sum


def _threshold(sparse_mode, act_kv, act_q, s1):
    if sparse_mode == 0:
        return act_kv
    elif sparse_mode == 3:
        return act_kv - act_q + s1 + 1
    return act_kv


def _t1_to_batch_s1(t1, cum_q):
    """Map global T1 index to (batch_idx, s1_within_batch)."""
    for b, c in enumerate(cum_q):
        if t1 < c:
            prev = cum_q[b - 1] if b > 0 else 0
            return b, t1 - prev
    raise IndexError(f"t1={t1} out of range for cum_q={cum_q}")


def _resolve_pa_to_logical(key_cache, block_table, B, blocks_per_batch, block_size, N2, D):
    """Convert PA_BSND key cache to logical BSND for reference computation.
    Uses sequential block allocation: batch b → physical blocks [b*blocks_per_batch, (b+1)*blocks_per_batch).
    """
    key_cpu = key_cache.cpu()  # single bulk transfer, avoid many small D2H copies
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
# Test helpers
# ---------------------------------------------------------------------------

def _assert_close(actual, expected, name, rtol=1e-3, atol=1e-3):
    actual_cpu = actual.float().cpu()
    expected_cpu = expected.float().cpu()
    max_diff = (actual_cpu - expected_cpu).abs().max().item()
    ok = torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol)
    if not ok:
        # print a few worst elements to aid debugging
        diff = (actual_cpu - expected_cpu).abs()
        flat_diff = diff.flatten()
        top_k = min(5, flat_diff.numel())
        top_idx = flat_diff.topk(top_k).indices
        print(f"  [{name}] top {top_k} diffs:", {i: flat_diff[i].item() for i in top_idx})
    assert ok, f"{name} mismatch: max_abs_diff={max_diff:.6e}"


def _assert_lse_close(actual, expected, name):
    """LSE values are fp32 in kernel; remaining diff is fp32 matmul
    accumulation-order noise over the 576-dim QK. Use a modest tolerance."""
    _assert_close(actual, expected, name, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests: BSND/BSND
# ---------------------------------------------------------------------------

@pytest.mark.lse
def test_lse_bsnd_bsnd_mode0():
    """LSE for BSND/BSND, sparse_mode=0, fp16."""
    torch.manual_seed(101)
    B, S1, S2, N1, N2, D, ROPE, K = 2, 4, 128, 8, 1, 512, 64, 8
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([S1] * B, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2] * B, dtype=torch.int32, device=device)

    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                perm = torch.randperm(S2)[:K].to(torch.int32)
                sel[b, s1, n2] = perm
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
        return_packed_kv=False,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    ref_out, ref_max, ref_sum = _lse_ref_bsnd(
        query, key, query_rope, key_rope, sparse_indices, SCALE,
        actual_seq_q, actual_seq_kv, sparse_mode=0, sparse_block_size=1)

    _assert_close(attn_out, ref_out, "attn_out BSND/BSND mode0")
    _assert_lse_close(lse_max, ref_max, "softmax_max BSND/BSND mode0")
    _assert_lse_close(lse_sum, ref_sum, "softmax_sum BSND/BSND mode0")


@pytest.mark.lse
def test_lse_bsnd_bsnd_mode3():
    """LSE for BSND/BSND, sparse_mode=3 (causal), fp16."""
    torch.manual_seed(102)
    B, S1, S2, N1, N2, D, ROPE, K = 2, 5, 256, 8, 1, 512, 64, 16
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([3, 5], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([128, 256], dtype=torch.int32, device=device)

    # sparse_indices: all valid (no -1), within each batch's kv range
    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        max_idx = int(actual_seq_kv[b]) - 1
        for s1 in range(int(actual_seq_q[b])):
            for n2 in range(N2):
                sel[b, s1, n2, :K] = torch.randint(0, max_idx + 1, (K,), dtype=torch.int32)
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
        return_softmax_lse=True,
        return_packed_kv=False,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    ref_out, ref_max, ref_sum = _lse_ref_bsnd(
        query, key, query_rope, key_rope, sparse_indices, SCALE,
        actual_seq_q, actual_seq_kv, sparse_mode=3, sparse_block_size=1)

    _assert_close(attn_out, ref_out, "attn_out BSND/BSND mode3")
    _assert_lse_close(lse_max, ref_max, "softmax_max BSND/BSND mode3")
    _assert_lse_close(lse_sum, ref_sum, "softmax_sum BSND/BSND mode3")


# ---------------------------------------------------------------------------
# Tests: BSND/PA_BSND
# ---------------------------------------------------------------------------

@pytest.mark.lse
def test_lse_bsnd_pa_bsnd_mode3():
    """LSE for BSND/PA_BSND, sparse_mode=3, fp16."""
    torch.manual_seed(103)
    B, S1, N1, N2, D, ROPE, K = 2, 4, 8, 1, 512, 64, 8
    BLOCK_SIZE = 64
    BLOCKS_PER_BATCH = 2
    BLOCK_NUM = B * BLOCKS_PER_BATCH
    S2 = BLOCKS_PER_BATCH * BLOCK_SIZE  # 128
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1

    # PA key cache: sequential physical block assignment
    key_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D, dtype=dtype, device=device) * 0.1
    key_rope_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, ROPE, dtype=dtype, device=device) * 0.1
    block_table = (
        torch.arange(BLOCK_NUM, dtype=torch.int32, device=device)
        .view(B, BLOCKS_PER_BATCH)
    )
    actual_seq_q = torch.tensor([S1, 3], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([128, 128], dtype=torch.int32, device=device)

    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        kv_len = int(actual_seq_kv[b])
        for s1 in range(int(actual_seq_q[b])):
            for n2 in range(N2):
                sel[b, s1, n2] = torch.randperm(kv_len)[:K].to(torch.int32)
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
        sparse_mode=3,
        return_softmax_lse=True,
        return_packed_kv=False,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    # Resolve PA → logical BSND for reference (key NoPE + key RoPE)
    key_logical = _resolve_pa_to_logical(key_cache, block_table, B,
                                         BLOCKS_PER_BATCH, BLOCK_SIZE, N2, D)
    key_rope_logical = _resolve_pa_to_logical(key_rope_cache, block_table, B,
                                              BLOCKS_PER_BATCH, BLOCK_SIZE, N2, ROPE)
    ref_out, ref_max, ref_sum = _lse_ref_bsnd(
        query, key_logical, query_rope, key_rope_logical, sparse_indices, SCALE,
        actual_seq_q, actual_seq_kv, sparse_mode=3, sparse_block_size=1)

    _assert_close(attn_out, ref_out, "attn_out BSND/PA_BSND mode3")
    _assert_lse_close(lse_max, ref_max, "softmax_max BSND/PA_BSND mode3")
    _assert_lse_close(lse_sum, ref_sum, "softmax_sum BSND/PA_BSND mode3")


# ---------------------------------------------------------------------------
# Tests: TND/TND
# ---------------------------------------------------------------------------

@pytest.mark.lse
def test_lse_tnd_tnd_mode0():
    """LSE for TND/TND, sparse_mode=0, fp16."""
    torch.manual_seed(104)
    N1, N2, D, ROPE, K = 8, 1, 512, 64, 8
    device, dtype = "npu", torch.float16

    # Two batches: query [2, 3] (cum -> [2, 5]), kv [64, 128]
    q_len = [2, 3]
    kv_len = [64, 128]
    T1 = sum(q_len)  # 5
    T2 = sum(kv_len)  # 192
    B = 2
    S1 = max(q_len)
    S2 = max(kv_len)

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
        return_softmax_lse=True,
        return_packed_kv=False,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    # Convert TND key/key_rope ([T2, N2, *]) to BSND for reference
    key_bsnd = torch.zeros(B, S2, N2, D, dtype=dtype)
    key_rope_bsnd = torch.zeros(B, S2, N2, ROPE, dtype=dtype)
    key_cpu = key.cpu()
    key_rope_cpu = key_rope.cpu()
    t_start = 0
    for b in range(B):
        act = kv_len[b]
        key_bsnd[b, :act] = key_cpu[t_start:t_start + act]
        key_rope_bsnd[b, :act] = key_rope_cpu[t_start:t_start + act]
        t_start += act

    ref_out, ref_max, ref_sum = _lse_ref_tnd(
        query, key_bsnd, query_rope, key_rope_bsnd, sparse_indices, SCALE,
        actual_seq_q, torch.tensor(kv_len), sparse_mode=0, sparse_block_size=1)

    _assert_close(attn_out, ref_out, "attn_out TND/TND mode0")
    _assert_lse_close(lse_max, ref_max, "softmax_max TND/TND mode0")
    _assert_lse_close(lse_sum, ref_sum, "softmax_sum TND/TND mode0")


# ---------------------------------------------------------------------------
# Tests: TND/PA_BSND  (production layout)
# ---------------------------------------------------------------------------

@pytest.mark.lse
def test_lse_tnd_pa_bsnd_mode3():
    """LSE for TND/PA_BSND (production layout), sparse_mode=3, fp16."""
    torch.manual_seed(105)
    N1, N2, D, ROPE, K = 8, 1, 512, 64, 8
    BLOCK_SIZE = 64
    BLOCKS_PER_BATCH = 2
    B = 2
    BLOCK_NUM = B * BLOCKS_PER_BATCH
    device, dtype = "npu", torch.float16

    q_len = [2, 3]
    T1 = sum(q_len)  # 5
    kv_len = [128, 128]  # 2 blocks each
    S2 = BLOCKS_PER_BATCH * BLOCK_SIZE
    cum_q = [2, 5]
    actual_seq_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor(kv_len, dtype=torch.int32, device=device)

    query = torch.randn(T1, N1, D, dtype=dtype, device=device) * 0.1
    query_rope = torch.randn(T1, N1, ROPE, dtype=dtype, device=device) * 0.1

    key_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D, dtype=dtype, device=device) * 0.1
    key_rope_cache = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, ROPE, dtype=dtype, device=device) * 0.1
    block_table = (
        torch.arange(BLOCK_NUM, dtype=torch.int32, device=device)
        .view(B, BLOCKS_PER_BATCH)
    )

    sel = torch.empty(T1, N2, K, dtype=torch.int32)
    for t1 in range(T1):
        b, _ = _t1_to_batch_s1(t1, cum_q)
        for n2 in range(N2):
            sel[t1, n2] = torch.randperm(kv_len[b])[:K].to(torch.int32)
    sparse_indices = sel.to(device)

    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key_cache, value=key_cache,
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope_cache,
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3,
        return_softmax_lse=True,
        return_packed_kv=False,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    key_logical = _resolve_pa_to_logical(key_cache, block_table, B,
                                         BLOCKS_PER_BATCH, BLOCK_SIZE, N2, D)
    key_rope_logical = _resolve_pa_to_logical(key_rope_cache, block_table, B,
                                              BLOCKS_PER_BATCH, BLOCK_SIZE, N2, ROPE)
    ref_out, ref_max, ref_sum = _lse_ref_tnd(
        query, key_logical, query_rope, key_rope_logical, sparse_indices, SCALE,
        actual_seq_q, actual_seq_kv, sparse_mode=3, sparse_block_size=1)

    _assert_close(attn_out, ref_out, "attn_out TND/PA_BSND mode3")
    _assert_lse_close(lse_max, ref_max, "softmax_max TND/PA_BSND mode3")
    _assert_lse_close(lse_sum, ref_sum, "softmax_sum TND/PA_BSND mode3")


# ---------------------------------------------------------------------------
# Edge case: LSE with varying actual_seq_q (some batches shorter than S1)
# ---------------------------------------------------------------------------

@pytest.mark.lse
def test_lse_varying_actual_seq():
    """LSE with varying actual_seq_q across batches (tests padding zero-init)."""
    torch.manual_seed(106)
    B, S1, S2, N1, N2, D, ROPE, K = 3, 6, 128, 8, 1, 512, 64, 8
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([4, 0, 6], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([128, 64, 128], dtype=torch.int32, device=device)

    sel = torch.full((B, S1, N2, K), -1, dtype=torch.int32)
    for b in range(B):
        for s1 in range(int(actual_seq_q[b])):
            for n2 in range(N2):
                max_idx = int(actual_seq_kv[b]) - 1
                sel[b, s1, n2] = torch.randint(0, max(1, max_idx + 1), (K,), dtype=torch.int32)
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
        return_packed_kv=False,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    ref_out, ref_max, ref_sum = _lse_ref_bsnd(
        query, key, query_rope, key_rope, sparse_indices, SCALE,
        actual_seq_q, actual_seq_kv, sparse_mode=0, sparse_block_size=1)

    _assert_close(attn_out, ref_out, "attn_out var-act-seq")
    _assert_lse_close(lse_max, ref_max, "softmax_max var-act-seq")
    _assert_lse_close(lse_sum, ref_sum, "softmax_sum var-act-seq")


# ---------------------------------------------------------------------------
# dtype coverage: bf16 LSE (different rounding path from fp16)
# ---------------------------------------------------------------------------

@pytest.mark.lse
def test_lse_bsnd_bsnd_bf16():
    """LSE for BSND/BSND, sparse_mode=0, bf16.

    bf16 has ~3 fewer mantissa bits than fp16; LSE (fp32 output computed from
    bf16 inputs) accumulates more rounding, so use a wider 1e-2 tolerance.
    """
    torch.manual_seed(107)
    B, S1, S2, N1, N2, D, ROPE, K = 2, 4, 128, 8, 1, 512, 64, 8
    device, dtype = "npu", torch.bfloat16

    query = torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key
    query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([S1] * B, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2] * B, dtype=torch.int32, device=device)

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
        return_packed_kv=False,
    )
    torch.npu.synchronize()
    attn_out, lse_max, lse_sum, _, _ = output

    ref_out, ref_max, ref_sum = _lse_ref_bsnd(
        query, key, query_rope, key_rope, sparse_indices, SCALE,
        actual_seq_q, actual_seq_kv, sparse_mode=0, sparse_block_size=1)

    # bf16: wider tolerance for both attn_out and LSE
    _assert_close(attn_out, ref_out, "attn_out BSND/BSND bf16", rtol=1e-2, atol=1e-2)
    _assert_close(lse_max, ref_max, "softmax_max BSND/BSND bf16", rtol=1e-2, atol=1e-2)
    _assert_close(lse_sum, ref_sum, "softmax_sum BSND/BSND bf16", rtol=1e-2, atol=1e-2)
