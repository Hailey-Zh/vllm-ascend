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
# Step 4b packed_kv correctness test.
#
# With return_packed_kv=True the op gathers the sparse-selected K/RoPE-K tokens contiguously into
# packed_key (NoPE 512) + packed_key_rope (64). packed is a pure copy of the selected KV, so it must
# match a PyTorch gather *element-wise* (no compute, so bit-exact up to none).
#
# We use a non-trivial (subset, no -1) selection with sparse_mode=0 (no causal truncation), so every
# selected id is valid -> actual valid count = K = S2 -> the whole packed buffer is real, no padding.
# (len is computed by the framework from sparse_indices; the op does not output it. causal/-1 padding
# cases -> the downstream reads [0, len); validated framework-side.)
#
# Run manually (must source set_env.bash first):
#   pytest test_sparse_flash_attention_packed_kv.py -m step4_packed_kv -s -v

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


SCALE = 1.0 / (576 ** 0.5)


def _gather_ref(key, key_rope, sparse_indices):
    """Reference: packed[b,s1,n2,i] = key[b, sparse_indices[b,s1,n2,i], n2].
    key: [B,S2,N2,D]  key_rope: [B,S2,N2,ROPE]  sparse_indices: [B,S1,N2,K]
    returns (exp_key [B,S1,N2,K,D], exp_rope [B,S1,N2,K,ROPE])
    """
    B, S1, N2, K = sparse_indices.shape
    D = key.shape[-1]
    ROPE = key_rope.shape[-1]
    idx = sparse_indices.long().cpu()
    key_c = key.float().cpu()
    rope_c = key_rope.float().cpu()
    exp_key = torch.zeros(B, S1, N2, K, D)
    exp_rope = torch.zeros(B, S1, N2, K, ROPE)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                ids = idx[b, s1, n2]
                exp_key[b, s1, n2] = key_c[b, ids, n2, :]
                exp_rope[b, s1, n2] = rope_c[b, ids, n2, :]
    return exp_key, exp_rope


@pytest.mark.step4_packed_kv
def test_packed_kv_bsnd_gather():
    torch.manual_seed(7)
    B, S1, N2, D, ROPE = 2, 4, 1, 512, 64
    S2 = 128          # kv length
    K = 8             # selected per (b,s1,n2); sparse_block_size=1 -> packed S2 dim = K
    device, dtype = "npu", torch.float16

    query = torch.randn(B, S1, N2, D, dtype=dtype, device=device) * 0.1
    key = torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1
    value = key  # MLA: value == c_KV == key NoPE; pass same tensor
    query_rope = torch.randn(B, S1, N2, ROPE, dtype=dtype, device=device) * 0.1
    key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1
    actual_seq_q = torch.tensor([S1] * B, dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2] * B, dtype=torch.int32, device=device)

    # 非平凡选择：每个 (b,s1,n2) 选 K 个不同的 id（不同 s1 选得不同），范围 [0, S2)，无 -1
    sel = torch.empty(B, S1, N2, K, dtype=torch.int32)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                perm = torch.randperm(S2)[:K].to(torch.int32)
                sel[b, s1, n2] = perm
    sparse_indices = sel.to(device)

    out, _, _, packed_key, packed_key_rope = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE,
        sparse_block_size=1,
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

    assert packed_key is not None and packed_key_rope is not None, "packed outputs are None"
    # shape: [B, S1, N2, K, D] / [B, S1, N2, K, ROPE]
    assert tuple(packed_key.shape) == (B, S1, N2, K, D), f"packed_key shape {tuple(packed_key.shape)}"
    assert tuple(packed_key_rope.shape) == (B, S1, N2, K, ROPE), f"packed_key_rope shape {tuple(packed_key_rope.shape)}"

    exp_key, exp_rope = _gather_ref(key, key_rope, sparse_indices)
    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()

    max_key = (pk - exp_key).abs().max().item()
    max_rope = (pkr - exp_rope).abs().max().item()
    print(f"[packed_kv BSND] max_abs_diff key={max_key:.3e} rope={max_rope:.3e}")
    # 纯拷贝，应逐元素一致（留极小容差防 dtype 往返）
    assert torch.allclose(pk, exp_key, rtol=0, atol=1e-3), f"packed_key mismatch max={max_key:.3e}"
    assert torch.allclose(pkr, exp_rope, rtol=0, atol=1e-3), f"packed_key_rope mismatch max={max_rope:.3e}"


@pytest.mark.step4_packed_kv
def test_packed_kv_tnd_pa_bsnd_marker():
    """生产路径 TND query + PA_BSND kv，跨 batch（测 TND 前缀和段基址 + PA block_table 间址 + 保序）。
    用 marker 编码：把每个 kv token 的整段向量填成它的全局标号 (b*BASE + pos)，packed 出来直接验标号。
    """
    torch.manual_seed(11)
    device, dtype = "npu", torch.float16
    N1, N2, D, ROPE = 8, 1, 512, 64        # MLA: N2=1, gSize=N1
    BS = 64                                  # PA block size
    BASE = 1000                              # marker = b*BASE + pos（<2048，fp16 精确）

    # 两 batch，query 长度 [2,3] → T1=5；actual_seq_q 用 cumsum（TND 约定）
    q_len = [2, 3]
    T1 = sum(q_len)                          # 5
    cum_q = [2, 5]
    actual_seq_q = torch.tensor(cum_q, dtype=torch.int32, device=device)

    # 每 batch kv 长度 128（2 块）；actual_seq_kv 用 RAW 每 batch（PA 约定）
    kv_len = [128, 128]
    BLOCKS_PER_BATCH = 2                      # 128/64
    B = 2
    BLOCK_NUM = B * BLOCKS_PER_BATCH         # 4
    actual_seq_kv = torch.tensor(kv_len, dtype=torch.int32, device=device)
    # block_table[b] = [b*2, b*2+1]
    block_table = torch.arange(BLOCK_NUM, dtype=torch.int32, device=device).view(B, BLOCKS_PER_BATCH)

    # key/value/key_rope cache：[BLOCK_NUM, BS, N2, *]，每个 token 整段填 marker
    key_cache = torch.zeros(BLOCK_NUM, BS, N2, D, dtype=dtype, device=device)
    key_rope_cache = torch.zeros(BLOCK_NUM, BS, N2, ROPE, dtype=dtype, device=device)
    for b in range(B):
        for p in range(kv_len[b]):
            phys = int(block_table[b, p // BS].item())
            off = p % BS
            marker = float(b * BASE + p)
            key_cache[phys, off, 0, :] = marker
            key_rope_cache[phys, off, 0, :] = marker
    value = key_cache  # MLA: value==c_KV; 不被 V_TEMPLATE 读，传同一份

    query = torch.randn(T1, N1, D, dtype=dtype, device=device) * 0.1
    query_rope = torch.randn(T1, N1, ROPE, dtype=dtype, device=device) * 0.1

    K = 8
    # sparse_indices: [T1, N2, K]，每个 query token 选 K 个不同 kv 位置（其所在 batch 的 [0,kv_len)）
    sel = torch.empty(T1, N2, K, dtype=torch.int32)
    def t1_to_b(t1):
        return 0 if t1 < cum_q[0] else 1
    for t1 in range(T1):
        b = t1_to_b(t1)
        for n2 in range(N2):
            sel[t1, n2] = torch.randperm(kv_len[b])[:K].to(torch.int32)
    sparse_indices = sel.to(device)

    out, _, _, packed_key, packed_key_rope = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key_cache, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope_cache,
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=0,
        return_softmax_lse=False,
        return_packed_kv=True,
    )
    torch.npu.synchronize()

    assert packed_key is not None and packed_key_rope is not None
    assert tuple(packed_key.shape) == (T1, N2, K, D), f"packed_key shape {tuple(packed_key.shape)}"
    assert tuple(packed_key_rope.shape) == (T1, N2, K, ROPE), f"packed_key_rope shape {tuple(packed_key_rope.shape)}"

    pk = packed_key.float().cpu()
    pkr = packed_key_rope.float().cpu()
    # 期望 marker：packed_key[t1,n2,i,:] 全 == b(t1)*BASE + sel[t1,n2,i]
    sel_cpu = sel.cpu()
    bad = 0
    max_diff = 0.0
    for t1 in range(T1):
        b = t1_to_b(t1)
        for n2 in range(N2):
            for i in range(K):
                want = float(b * BASE + int(sel_cpu[t1, n2, i]))
                dk = (pk[t1, n2, i] - want).abs().max().item()
                dr = (pkr[t1, n2, i] - want).abs().max().item()
                max_diff = max(max_diff, dk, dr)
                if dk > 0.5 or dr > 0.5:
                    bad += 1
    print(f"[packed_kv TND/PA_BSND] max_marker_diff={max_diff:.3e} bad_slots={bad}")
    assert bad == 0, f"TND/PA_BSND packed marker mismatch: {bad} slots wrong, max_diff={max_diff:.3e}"
