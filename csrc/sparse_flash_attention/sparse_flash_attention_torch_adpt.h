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
           c10::optional<at::Tensor>, c10::optional<at::Tensor>>
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

    // dense(sparse_indices=None) 时按 OPTIONAL 直接透传 null：
    //   tiling 端 isDenseMode=(tensor==nullptr) 走 IS_DENSE / C_TEMPLATE 路径，
    //   sparseBlockSize 由 host 强制为 1、sparseBlockCount=s2Size（见 SFAInfoParser::Parse）。
    // 之前的 arange full-coverage dummy 是漏 source set_env.bash 时的误修，且每次现场
    //   at::arange().expand().contiguous() 造 [B,S1,N2,S2] int32 张量（4~8MB/call），现去除。
    // C_TEMPLATE/dense 路径正确性见 STEP3_DENSE_KERNEL_FIX.md（vs CPU golden 1.7e-5）。

    // [step 4] packed KV 输出。OPTIONAL：return_packed_kv=false 时传 nullopt、不分配显存。
    // actual_packed_len 不由算子输出，框架用 sparse_indices + causal 自算。
    c10::optional<at::Tensor> packed_key;
    c10::optional<at::Tensor> packed_key_rope;
    if (return_packed_kv) {
        // 仅支持稀疏 + sparseBlockSize<=4（MergeKv 硬约束，tiling 侧也会校验）
        TORCH_CHECK(sparse_indices.has_value(),
            "return_packed_kv only supports sparse mode (sparse_indices must be provided)");
        TORCH_CHECK(sparse_block_size <= 4,
            "return_packed_kv only supports sparse_block_size <= 4, got ", sparse_block_size);

        int64_t n2 = (layout_kv_str == "TND") ? key.size(1) : key.size(2);
        // S2 = sparse_block_count(sparse_indices 末维) * sparse_block_size
        int64_t sparse_block_count = sparse_indices.value().size(-1);
        int64_t s2 = sparse_block_count * sparse_block_size;
        int64_t head_dim = key.size(-1);
        int64_t rope_dim = key_rope.has_value() ? key_rope.value().size(-1) : 64;

        std::vector<int64_t> pk_shape, pkr_shape;
        if (layout_query_str == "TND") {
            int64_t t1 = query.size(0);
            pk_shape  = {t1, n2, s2, head_dim};
            pkr_shape = {t1, n2, s2, rope_dim};
        } else {  // BSND
            int64_t b = query.size(0);
            int64_t s1 = query.size(1);
            pk_shape  = {b, s1, n2, s2, head_dim};
            pkr_shape = {b, s1, n2, s2, rope_dim};
        }
        // 有效区由 kernel 写，尾部/len 交给框架（下游按自算的 len 切片读）。
        packed_key       = at::empty(pk_shape,  query.options().dtype(query.dtype()));
        packed_key_rope  = at::empty(pkr_shape, query.options().dtype(query.dtype()));
    }

    EXEC_NPU_CMD(
        aclnnSparseFlashAttention,
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        return_softmax_lse,
        return_packed_kv,
        output,
        softmax_max,
        softmax_sum,
        packed_key,
        packed_key_rope);
    return std::make_tuple(output, softmax_max, softmax_sum, packed_key, packed_key_rope);
}
}
#endif
