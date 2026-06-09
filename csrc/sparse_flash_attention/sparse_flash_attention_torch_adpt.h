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

std::tuple<at::Tensor, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>>
npu_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &sparse_indices, double scale_value, int64_t sparse_block_size,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    bool return_softmax_lse,
    bool return_packed_kv)
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

    // [step 4] packed KV 输出。
    // 重要：CANN aclnn 即使元数据标了 OPTIONAL，运行期对输出传 nullopt 仍会让 executor
    // 匹配不到 binary（报 "binary bin not found" / NnopbaseExecutorMatchCache failed），
    // 与历史上 OPTIONAL 输入拒绝 null 同源（见 dense 的 sparse_indices dummy 替换）。
    // 因此始终分配真实 shape 的张量；return_packed_kv=false 时 kernel 不写、只占位。
    if (return_packed_kv) {
        // 仅支持稀疏 + sparseBlockSize<=4（MergeKv 硬约束，tiling 侧也会校验）
        TORCH_CHECK(sparse_indices.has_value(),
            "return_packed_kv only supports sparse mode (sparse_indices must be provided)");
        TORCH_CHECK(sparse_block_size <= 4,
            "return_packed_kv only supports sparse_block_size <= 4, got ", sparse_block_size);
    }
    int64_t pkv_n2 = (layout_kv_str == "TND") ? key.size(1) : key.size(2);
    // S2 = sparse_block_count(sparse_indices 末维) * effective_sparse_block_size，
    // 与 proto.cpp InferShape 一致（dense 替换路径下 effective=1、indices 为全选 arange）。
    int64_t pkv_s2 = sparse_indices_passthrough.size(-1) * effective_sparse_block_size;
    int64_t pkv_head_dim = key.size(-1);
    int64_t pkv_rope_dim = key_rope.has_value() ? key_rope.value().size(-1) : 64;
    std::vector<int64_t> pk_shape, pkr_shape, len_shape;
    if (layout_query_str == "TND") {
        int64_t t1 = query.size(0);
        pk_shape  = {t1, pkv_n2, pkv_s2, pkv_head_dim};
        pkr_shape = {t1, pkv_n2, pkv_s2, pkv_rope_dim};
        len_shape = {t1, pkv_n2};
    } else {  // BSND
        int64_t b = query.size(0);
        int64_t s1 = query.size(1);
        pk_shape  = {b, s1, pkv_n2, pkv_s2, pkv_head_dim};
        pkr_shape = {b, s1, pkv_n2, pkv_s2, pkv_rope_dim};
        len_shape = {b, s1, pkv_n2};
    }
    // padding/尾部清零由 kernel 负责，host 端 at::empty 不必清。
    c10::optional<at::Tensor> packed_key =
        at::empty(pk_shape, query.options().dtype(query.dtype()));
    c10::optional<at::Tensor> packed_key_rope =
        at::empty(pkr_shape, query.options().dtype(query.dtype()));
    c10::optional<at::Tensor> actual_packed_len =
        at::empty(len_shape, query.options().dtype(at::kInt));

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
        return_packed_kv,
        output,
        softmax_max,
        softmax_sum,
        packed_key,
        packed_key_rope,
        actual_packed_len);
    return std::make_tuple(output, softmax_max, softmax_sum,
                           packed_key, packed_key_rope, actual_packed_len);
}
}
#endif
