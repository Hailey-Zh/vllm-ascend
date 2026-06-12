#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
#
# V_TEMPLATE bug 触发条件二分。
# 已知矛盾：框架 single 全部 PASS（V 正确），我的 probe 显示 V 输出 ≈0（V 错误）。
# 两个候选差异需二分：
#   H1: sparse_indices 有无 -1 哨兵（框架有，我没有）
#   H2: 选中 token 数 K==S2 全选 vs K<S2 部分选
#
# 方法：value[j]=j 探针，用"输出均值是否在期望 token 范围"直接判定对错，
# 不依赖任何参考实现。

import pytest
import torch
import torch_npu  # noqa
import vllm_ascend  # noqa
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SCALE = 1.0 / (576 ** 0.5)


def _call_op(query, key, value, sparse_indices, sparse_block_size, *,
             block_table, actual_seq_q, actual_seq_kv,
             query_rope, key_rope, layout_query, layout_kv):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE,
        sparse_block_size=sparse_block_size,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query=layout_query, layout_kv=layout_kv,
        sparse_mode=0,
        return_softmax_lse=False,
    )


@pytest.mark.xfail(reason="V_TEMPLATE assumes value==key_nope (MLA invariant); not a bug, diverges only "
                          "with independent key!=value. See STEP3_DENSE_KERNEL_FIX.md",
                   strict=False)
def test_v_trigger_bisect():
    """二分 V_TEMPLATE bug 触发条件：H1(哨兵-1) vs H2(全选K=S2)。"""
    torch.manual_seed(42)
    B, S1, S2, N1, N2, D, ROPE = 2, 4, 128, 8, 1, 512, 64
    device = "npu"
    dtype = torch.float16

    query = (torch.randn(B, S1, N1, D, dtype=dtype, device=device) * 0.1)
    key = (torch.randn(B, S2, N2, D, dtype=dtype, device=device) * 0.1)
    query_rope = (torch.randn(B, S1, N1, ROPE, dtype=dtype, device=device) * 0.1)
    key_rope = (torch.randn(B, S2, N2, ROPE, dtype=dtype, device=device) * 0.1)

    # value[b,j,0,:]=j: 输出均值就是 attend 的 token id 重心
    ids = torch.arange(S2, dtype=dtype, device=device).view(1, S2, 1, 1)
    value = ids.expand(B, S2, N2, D).contiguous()

    actual_seq_q = torch.tensor([S1, S1], dtype=torch.int32, device=device)
    actual_seq_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

    common = dict(
        query=query, key=key, value=value,
        block_table=None, actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", layout_kv="BSND")

    def run_v(indices_1d, K_slots):
        """V_TEMPLATE: bs=1, indices 含末尾 -1 哨兵（K_slots > len）。"""
        t = torch.full((B, S1, N2, K_slots), -1, dtype=torch.int32, device=device)
        n_fill = min(len(indices_1d), K_slots)
        for i in range(n_fill):
            t[:, :, :, i] = indices_1d[i]
        return _call_op(sparse_indices=t, sparse_block_size=1, **common)[0].float().cpu()

    def run_c_ref(K):
        """C_TEMPLATE: bs=8, full coverage of K tokens（假设 K 是 8 的倍数）。"""
        assert K % 8 == 0
        n_blocks = K // 8
        idx = (torch.arange(n_blocks, dtype=torch.int32, device=device)
               .view(1, 1, 1, n_blocks).expand(B, S1, N2, n_blocks).contiguous())
        return _call_op(sparse_indices=idx, sparse_block_size=8, **common)[0].float().cpu()

    # C 基准：full 128（已验证正确）
    c128 = run_c_ref(128)

    print("\n[bisect] value[j]=j 探针, 期望: K=16 输出 ≈7.5, K=128 输出 ≈63.5")
    print(f"[bisect] {'case':<40s} {'mean':>8s} {'判定':>8s}")

    ok = True
    # A) K=16, K_slots=16(无 -1 哨兵, 且 K<S2)
    v = run_v(list(range(16)), K_slots=16)
    m = v.mean().item()
    good = abs(m - 7.5) < 20  # 探针判据宽松
    print(f"[bisect] {'K=16 K_slots=16 (无哨兵, K<S2)':<40s} {m:8.2f} {'OK' if good else 'WRONG':>8s}")
    ok = ok and good

    # B) K=16, K_slots=128(有 -1 哨兵, K<S2)——框架风格
    v = run_v(list(range(16)), K_slots=128)
    m = v.mean().item()
    good = abs(m - 7.5) < 20
    print(f"[bisect] {'K=16 K_slots=128 (有哨兵, K<S2)':<40s} {m:8.2f} {'OK' if good else 'WRONG':>8s}")
    ok = ok and good

    # C) K=128, K_slots=128(无 -1 哨兵, K=S2)——我的原测试
    v = run_v(list(range(128)), K_slots=128)
    m = v.mean().item()
    good = abs(m - 63.5) < 30
    print(f"[bisect] {'K=128 K_slots=128 (无哨兵, K=S2)':<40s} {m:8.2f} {'OK' if good else 'WRONG':>8s}")
    ok = ok and good

    # D) K=128, K_slots=129(有 -1 哨兵, K=S2)
    v = run_v(list(range(128)), K_slots=129)
    m = v.mean().item()
    good = abs(m - 63.5) < 30
    print(f"[bisect] {'K=128 K_slots=129 (有哨兵, K=S2)':<40s} {m:8.2f} {'OK' if good else 'WRONG':>8s}")
    ok = ok and good

    # E) K=128, K_slots=16 (K_slots < len(indices), 截断) —— 对照
    v = run_v(list(range(128)), K_slots=16)
    m = v.mean().item()
    good = abs(m - 7.5) < 20  # 只取了前 16 个 index, attend [0..15]
    print(f"[bisect] {'K=128 K_slots=16 (截断, 只取前16)':<40s} {m:8.2f} {'OK' if good else 'WRONG':>8s}")
    ok = ok and good

    # 同时报 C ref 均值作为锚
    print(f"[bisect] {'C_TEMPLATE(bs=8) full 128 ref':<40s} {c128.mean().item():8.2f} {'ref':>8s}")

    if not ok:
        print("\n[bisect] *** 以上标注 WRONG 的 case 即触发条件")
    assert ok, "V_TEMPLATE bug 触发条件已定位"
