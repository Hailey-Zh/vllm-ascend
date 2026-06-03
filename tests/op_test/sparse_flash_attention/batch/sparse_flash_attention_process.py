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
# vllm-ascend baseline runner:
# - 调 torch.ops._C_ascend.npu_sparse_flash_attention（vllm-ascend 注册的算子）
# - 暂不透传 pre_tokens / next_tokens / attention_mode / return_softmax_lse，
#   等改造步骤 2/3 完成后再开启。

import torch
import torch_npu  # noqa: F401  # 必须先 import 才能拿到 NPU 设备
import vllm_ascend  # noqa: F401  # 注册 torch.ops._C_ascend


def _load_inputs_to_npu(input_dict):
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value

    return {key: to_npu(value) for key, value in input_dict.items()}


def call_npu_eager(torch_tensor_dict, params):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        query=torch_tensor_dict.get("query"),
        key=torch_tensor_dict.get("key_cache"),
        value=torch_tensor_dict.get("value_cache"),
        sparse_indices=torch_tensor_dict.get("sparse_indices"),
        scale_value=params.get("scalevalue"),
        sparse_block_size=params.get("sparse_blocksize", 1),
        block_table=torch_tensor_dict.get("block_table") if params.get("layout_kv") == "PA_BSND" else None,
        actual_seq_lengths_query=torch.tensor(params["actual_seq_q"], dtype=torch.int32).to("npu"),
        actual_seq_lengths_kv=torch.tensor(params["actual_seq_kv"], dtype=torch.int32).to("npu"),
        query_rope=torch_tensor_dict.get("query_rope"),
        key_rope=torch_tensor_dict.get("key_rope_cache"),
        layout_query=params.get("layout_query"),
        layout_kv=params.get("layout_kv"),
        sparse_mode=params.get("sparsemode", 3),
    )


def call_npu(input_tensor_dict, params):
    torch_npu.npu.set_device(0)
    tensor_keys = ["query", "key_cache", "value_cache", "sparse_indices", "block_table",
                   "query_rope", "key_rope_cache"]
    filtered_input_dict = {k: input_tensor_dict.get(k) for k in tensor_keys if input_tensor_dict.get(k) is not None}
    torch_tensor_dict = _load_inputs_to_npu(filtered_input_dict)
    npu_result = call_npu_eager(torch_tensor_dict, params)
    torch.npu.synchronize()
    return npu_result
