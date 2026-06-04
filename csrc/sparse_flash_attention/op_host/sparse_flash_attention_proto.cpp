/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;

constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

// 属性顺序对齐 def.cpp
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 3;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 5;

constexpr uint32_t OUTPUT_INDEX_0 = 0;  // attention_out
constexpr uint32_t OUTPUT_INDEX_1 = 1;  // softmax_max
constexpr uint32_t OUTPUT_INDEX_2 = 2;  // softmax_sum

ge::graphStatus InferShapeSparseFlashAttention(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED)

    gert::Shape *attentionOutShape = context->GetOutputShape(OUTPUT_INDEX_0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxMaxShape = context->GetOutputShape(OUTPUT_INDEX_1);
    OPS_LOG_E_IF_NULL(context, softmaxMaxShape, return ge::GRAPH_FAILED)
    gert::Shape *softmaxSumShape = context->GetOutputShape(OUTPUT_INDEX_2);
    OPS_LOG_E_IF_NULL(context, softmaxSumShape, return ge::GRAPH_FAILED)

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)
    const char *layoutKvPtr = attrs->GetAttrPointer<char>(LAYOUT_KV_ATTR_INDEX);
    OPS_LOG_E_IF_NULL(context, layoutKvPtr, return ge::GRAPH_FAILED)
    std::string layoutKvStr = std::string(layoutKvPtr);
    // 无论 returnSoftmaxLse 真假都按真实 layout 推 shape：
    // false 时 kernel 不写但仍占合法 NPU 内存（[0] 空 tensor 会让 aclnn dispatch 崩）。
    if (queryShape->GetDimNum() == DIM_NUM_3) {
        // TND: 输出 [N2, T1, G]，key 的 N2 位置：PA_BSND=dim2, BSND/TND=dim1
        int64_t n2 = (layoutKvStr == "PA_BSND")
                         ? keyShape->GetDim(DIM_INDEX_2)
                         : keyShape->GetDim(DIM_INDEX_1);
        int64_t t1 = queryShape->GetDim(DIM_INDEX_0);
        int64_t g = queryShape->GetDim(DIM_INDEX_1) / n2;

        softmaxMaxShape->SetDimNum(DIM_NUM_3);
        softmaxMaxShape->SetDim(DIM_INDEX_0, n2);
        softmaxMaxShape->SetDim(DIM_INDEX_1, t1);
        softmaxMaxShape->SetDim(DIM_INDEX_2, g);

        softmaxSumShape->SetDimNum(DIM_NUM_3);
        softmaxSumShape->SetDim(DIM_INDEX_0, n2);
        softmaxSumShape->SetDim(DIM_INDEX_1, t1);
        softmaxSumShape->SetDim(DIM_INDEX_2, g);
    } else {
        // BSND: 输出 [B, N2, S1, G]
        int64_t b = queryShape->GetDim(DIM_INDEX_0);
        int64_t s1 = queryShape->GetDim(DIM_INDEX_1);
        int64_t n2 = keyShape->GetDim(DIM_INDEX_2);
        int64_t g = queryShape->GetDim(DIM_INDEX_2) / n2;

        softmaxMaxShape->SetDimNum(DIM_NUM_4);
        softmaxMaxShape->SetDim(DIM_INDEX_0, b);
        softmaxMaxShape->SetDim(DIM_INDEX_1, n2);
        softmaxMaxShape->SetDim(DIM_INDEX_2, s1);
        softmaxMaxShape->SetDim(DIM_INDEX_3, g);

        softmaxSumShape->SetDimNum(DIM_NUM_4);
        softmaxSumShape->SetDim(DIM_INDEX_0, b);
        softmaxSumShape->SetDim(DIM_INDEX_1, n2);
        softmaxSumShape->SetDim(DIM_INDEX_2, s1);
        softmaxSumShape->SetDim(DIM_INDEX_3, g);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX_0, inputDataType);
    context->SetOutputDataType(OUTPUT_INDEX_1, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_INDEX_2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFlashAttention).InferShape(InferShapeSparseFlashAttention).InferDataType(InferDataTypeSparseFlashAttention);
} // namespace ops
