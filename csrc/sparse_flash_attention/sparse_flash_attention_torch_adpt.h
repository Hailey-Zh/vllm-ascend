/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &sparse_indices, double scale_value, int64_t sparse_block_size,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    bool return_softmax_lse)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    // attention_out: 与 query 同 shape / dtype
    at::Tensor output = at::empty(query.sizes(), query.options().dtype(query.dtype()));

    // softmax_max / softmax_sum：始终按 layout 推真实 shape（避免 [0] 空 tensor 让 aclnn dispatch 崩）
    //   - TND : [N2, T1, G]
    //   - BSND: [B, N2, S1, G]
    // N2 在 key 的位置：PA_BSND/BSND = dim 2，TND = dim 1
    std::vector<int64_t> lse_shape;
    {
        int64_t n2 = (layout_kv_str == "TND") ? key.size(1) : key.size(2);
        TORCH_CHECK(n2 > 0, "key's N2 dim must be > 0, got ", n2);
        if (layout_query_str == "TND") {
            int64_t t1 = query.size(0);
            int64_t n1 = query.size(1);
            lse_shape = {n2, t1, n1 / n2};
        } else {  // BSND
            int64_t b = query.size(0);
            int64_t s1 = query.size(1);
            int64_t n1 = query.size(2);
            lse_shape = {b, n2, s1, n1 / n2};
        }
    }
    auto lse_options = query.options().dtype(at::kFloat);
    // padding 行的清零由 kernel InitOutputSingleCore / InitAllZeroOutput 完成（见
    // sparse_flash_attention_kernel_mla.h），host 端用 at::empty 不必再清。
    at::Tensor softmax_max = at::empty(lse_shape, lse_options);
    at::Tensor softmax_sum = at::empty(lse_shape, lse_options);

    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // [step 3c workaround — final]
    // For sparse_indices=None (dense mode), instead of fighting CANN's runtime nullptr-checks
    // and StorageShape::GetDimNum semantic surprises (see earlier commits), we substitute a
    // SHAPE-COMPLIANT tensor whose content selects every token: a full-coverage arange.
    // With sparse_block_size forced to 1 and indices = [0, 1, ..., S2-1], the existing sparse
    // path mathematically computes dense FA. No special host/kernel dense branch is exercised
    // for this call path; CompareShape passes naturally because the shape matches what tiling
    // expects.
    at::Tensor sparse_indices_passthrough;
    int64_t effective_sparse_block_size = sparse_block_size;
    if (sparse_indices.has_value()) {
        sparse_indices_passthrough = sparse_indices.value();
    } else {
        // Compute S2 (max KV seq length) and N2 per layout_kv
        int64_t S2;
        int64_t N2;
        if (layout_kv_str == "PA_BSND") {
            TORCH_CHECK(block_table.has_value(),
                        "sparse_flash_attention: PA_BSND mode requires block_table");
            int64_t block_size = key.size(1);              // [block_num, block_size, N2, D]
            int64_t max_blocks_per_batch = block_table.value().size(1);
            S2 = max_blocks_per_batch * block_size;
            N2 = key.size(2);
        } else if (layout_kv_str == "TND") {
            S2 = key.size(0);                              // [T2, N2, D]
            N2 = key.size(1);
        } else {                                            // BSND
            S2 = key.size(1);                              // [B, S2, N2, D]
            N2 = key.size(2);
        }
        TORCH_CHECK(S2 > 0 && N2 > 0,
                    "sparse_flash_attention dense substitute: S2=", S2, " N2=", N2);
        effective_sparse_block_size = 1;
        int64_t K = S2;

        auto arange = at::arange(K, query.options().dtype(at::kInt));
        if (layout_query_str == "TND") {
            int64_t T1 = query.size(0);
            sparse_indices_passthrough =
                arange.view({1, 1, K}).expand({T1, N2, K}).contiguous();
        } else {                                            // BSND
            int64_t B = query.size(0);
            int64_t S1 = query.size(1);
            sparse_indices_passthrough =
                arange.view({1, 1, 1, K}).expand({B, S1, N2, K}).contiguous();
        }
    }

    EXEC_NPU_CMD(
        aclnnSparseFlashAttention,
        query,
        key,
        value,
        sparse_indices_passthrough,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        effective_sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        return_softmax_lse,
        output,
        softmax_max,
        softmax_sum);
    return std::make_tuple(output, softmax_max, softmax_sum);
}
}
#endif
