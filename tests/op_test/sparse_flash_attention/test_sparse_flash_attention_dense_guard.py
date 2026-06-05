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
# Step 3a transitional guard test.
#
# After step 3a, sparse_indices is OPTIONAL in the op schema, but the kernel/tiling-key still requires
# it. To prevent silent crashes during the 3a -> 3b transition, the tiling layer rejects
# sparse_indices=None with GRAPH_FAILED. This test pins that behavior so 3b can knowingly remove the
# guard (the test will be replaced by real dense correctness tests then).
#
# Run manually:
#   pytest test_sparse_flash_attention_dense_guard.py -m step3a_guard -s
#
# Not part of the default CI run (no @pytest.mark.ci) — only fires when -m step3a_guard is requested.

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend


def _build_minimal_bsnd_inputs():
    """Minimal valid MLA BSND/BSND tensors. Match the layout/N2/D/rope constraints in
    check_valid_param.py so only the sparse_indices=None signal triggers the failure."""
    B, S1, S2, N1, N2, D, ROPE = 1, 2, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16

    query = torch.zeros((B, S1, N1, D), dtype=dtype, device=device)
    key = torch.zeros((B, S2, N2, D), dtype=dtype, device=device)
    value = torch.zeros((B, S2, N2, D), dtype=dtype, device=device)
    query_rope = torch.zeros((B, S1, N1, ROPE), dtype=dtype, device=device)
    key_rope = torch.zeros((B, S2, N2, ROPE), dtype=dtype, device=device)
    actual_seq_q = torch.tensor([S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2], dtype=torch.int32, device=device)

    return {
        "query": query,
        "key": key,
        "value": value,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "actual_seq_lengths_query": actual_seq_q,
        "actual_seq_lengths_kv": actual_seq_kv,
    }


@pytest.mark.step3a_guard
def test_dense_mode_rejected_by_tiling():
    inputs = _build_minimal_bsnd_inputs()
    with pytest.raises(RuntimeError):
        # sparse_indices=None: tiling 应当通过 OPS_LOG_E + GRAPH_FAILED 拒绝（step 3a temporary guard）。
        # 3b 落地后该用例会改写为真实 dense 正确性比对。
        torch.ops._C_ascend.npu_sparse_flash_attention(
            query=inputs["query"],
            key=inputs["key"],
            value=inputs["value"],
            sparse_indices=None,
            scale_value=1.0 / (576 ** 0.5),
            sparse_block_size=1,
            block_table=None,
            actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
            actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
            query_rope=inputs["query_rope"],
            key_rope=inputs["key_rope"],
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=0,
            return_softmax_lse=False,
        )
        torch.npu.synchronize()
