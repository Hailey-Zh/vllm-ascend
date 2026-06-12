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
# 真稀疏子集正确性测试（覆盖盲区 #5）。
#
# 现有 single 用例和 test_dense 全是 full-coverage（K≈seq，indices 是 [0..K-1] 前缀），
# 从没验证过 sparse 的"选择"逻辑：选一个真正的子集（K≪seq、非前缀、稀疏、batch 间不同、带 -1 哨兵），
# kernel 是否按 sparse_indices 取到正确的 token。indices 用升序（贴近 lightning_indexer 输出，
# 也避开"kernel 是否要求升序"这个未知变量；乱序鲁棒性可后续单独测）。
#
# 两条路径分开测：
#   - C_TEMPLATE (block_size=8)：key≠value（C 真读独立 value），同时验证选择 + value 读取
#   - V_TEMPLATE (block_size=1)：key==value（满足 MLA value==key_nope 不变量），纯验证 token 选择
#
# golden 只在 sparse_indices 选中的 token 子集上做 attention。

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SCALE = 1.0 / (576 ** 0.5)


def _call_op(query, key, value, sparse_indices, sparse_block_size, *,
             actual_seq_q, actual_seq_kv, query_rope, key_rope):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE,
        sparse_block_size=sparse_block_size,
        block_table=None,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND",
        sparse_mode=0,
        return_softmax_lse=False,
    )


def _cpu_golden_subset(query, key, value, query_rope, key_rope, scale, sel_per_batch):
    # 每个 batch 只在 sel_per_batch[b]（token id 列表）这些 KV token 上做 attention。GQA N2=1。
    q = query.float().cpu()
    qr = query_rope.float().cpu()
    kf = key.float().cpu()
    vf = value.float().cpu()
    krf = key_rope.float().cpu()
    B, S1, N1, D = q.shape
    out = torch.empty(B, S1, N1, D, dtype=torch.float32)
    for b in range(B):
        sel = sel_per_batch[b]
        k0 = kf[b, sel, 0, :]
        v0 = vf[b, sel, 0, :]
        kr0 = krf[b, sel, 0, :]
        for h in range(N1):
            s = q[b, :, h, :] @ k0.transpose(0, 1) + qr[b, :, h, :] @ kr0.transpose(0, 1)
            p = torch.softmax(s * scale, dim=-1)
            out[b, :, h, :] = p @ v0
    return out


def _make_indices(ids_per_batch, K, B, S1, N2, device):
    # [B,S1,N2,K]：每 batch 的 ids（block id；bs=1 时即 token id）填前 len 个，其余 -1 哨兵。
    # 每 batch 内所有 query 行用同一组选择（broadcast over S1）。
    t = torch.full((B, S1, N2, K), -1, dtype=torch.int32, device=device)
    for b in range(B):
        for i, v in enumerate(ids_per_batch[b]):
            t[b, :, :, i] = v
    return t


def _common_inputs(seed, key_eq_value):
    torch.manual_seed(seed)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16
    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    value = key if key_eq_value else (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)
    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)
    return dict(B=B, S1=S1, S2=S2, N2=N2, query=query, key=key, value=value,
                query_rope=query_rope, key_rope=key_rope,
                actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv)


def test_sparse_subset_ctemplate():
    # C_TEMPLATE (bs=8)，key≠value。选非前缀、稀疏、batch 间不同数量的 block 子集（含 -1 哨兵）。
    inp = _common_inputs(seed=101, key_eq_value=False)
    B, S1, N2 = inp["B"], inp["S1"], inp["N2"]
    BS = 8
    # batch0 选 block [0,4,9,15] -> token; batch1 选 [1,5,10]（少一个，靠 -1 padding）
    blocks = [[0, 4, 9, 15], [1, 5, 10]]
    K = max(len(b) for b in blocks)  # 4
    sel_tokens = [[blk * BS + j for blk in bl for j in range(BS)] for bl in blocks]
    idx = _make_indices(blocks, K, B, S1, N2, "npu")

    out = _call_op(inp["query"], inp["key"], inp["value"], idx, BS,
                   actual_seq_q=inp["actual_seq_q"], actual_seq_kv=inp["actual_seq_kv"],
                   query_rope=inp["query_rope"], key_rope=inp["key_rope"])[0].float().cpu()
    torch.npu.synchronize()
    golden = _cpu_golden_subset(inp["query"], inp["key"], inp["value"],
                                inp["query_rope"], inp["key_rope"], SCALE, sel_tokens)
    diff = (out - golden).abs().max().item()
    print(f"\n[subset-C] block_size=8 key!=value, max_abs_diff(NPU, golden) = {diff:.6e}")
    assert torch.allclose(out, golden, rtol=1e-3, atol=3e-3), \
        f"C_TEMPLATE sparse-subset selection wrong: max_abs_diff={diff:.6e}"


def test_sparse_subset_vtemplate():
    # V_TEMPLATE (bs=1)，key==value（满足 MLA 不变量）。选非前缀、稀疏、batch 间不同数量的 token 子集。
    inp = _common_inputs(seed=202, key_eq_value=True)
    B, S1, N2 = inp["B"], inp["S1"], inp["N2"]
    # batch0 选 6 个稀疏 token; batch1 选 7 个（靠 -1 padding 对齐 K）
    tokens = [[5, 17, 33, 64, 99, 120], [3, 8, 50, 77, 88, 111, 127]]
    K = max(len(t) for t in tokens)  # 7
    idx = _make_indices(tokens, K, B, S1, N2, "npu")

    out = _call_op(inp["query"], inp["key"], inp["value"], idx, 1,
                   actual_seq_q=inp["actual_seq_q"], actual_seq_kv=inp["actual_seq_kv"],
                   query_rope=inp["query_rope"], key_rope=inp["key_rope"])[0].float().cpu()
    torch.npu.synchronize()
    golden = _cpu_golden_subset(inp["query"], inp["key"], inp["value"],
                                inp["query_rope"], inp["key_rope"], SCALE, tokens)
    diff = (out - golden).abs().max().item()
    print(f"\n[subset-V] block_size=1 key==value, max_abs_diff(NPU, golden) = {diff:.6e}")
    assert torch.allclose(out, golden, rtol=1e-3, atol=3e-3), \
        f"V_TEMPLATE sparse-subset selection wrong: max_abs_diff={diff:.6e}"
