#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Op-level tests for the DISCRETE sparse-index layout of SparseFlashAttention.

The kernel historically assumes the ``sparse_indices`` (-1)-padding only trails
the valid token ids, e.g. ``[3, 2, -1, -1]`` (COMPACTED). With the new
``sparse_indices_discrete=True`` attribute the -1 holes may sit anywhere, e.g.
``[-1, 3, -1, 2]`` (DISCRETE); the kernel must skip interior holes instead of
stopping at the first -1.

These tests call ``torch.ops._C_ascend.npu_sparse_flash_attention`` directly so
they can flip the new attribute, and they run on NPU (decode shapes only, so
every selected token is causally valid for sparse_mode=3).
"""

import math
import sys
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

# MLA combined-KV dims are fixed in the kernel: 512 (nope) + 64 (rope) = 576.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
BLOCK_SIZE = 128
NUM_HEADS = 8  # query heads; MQA -> 1 kv head, group size = NUM_HEADS
SPARSE_COUNT = 256  # topk width (>= seq_len used below)


def _build_paged_kv_cache(
    seq_lens: list[int],
    dtype: torch.dtype,
    device: torch.device,
):
    """Per-batch contiguous K + a paged cache/block_table selecting the same data.

    Returns ``(k_nope_cache, k_rope_cache, block_table, k_nope_flat, k_rope_flat)``
    where the ``*_flat`` lists hold the dense per-batch tensors used by the CPU
    golden, and the caches/block_table feed the kernel (PA_BSND layout).
    """
    blocks_per_seq = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens]
    total_blocks = sum(blocks_per_seq) + 1  # block 0 reserved as padding
    max_blocks = max(blocks_per_seq)

    k_nope_cache = torch.zeros(total_blocks, BLOCK_SIZE, 1, KV_LORA_RANK, dtype=dtype, device=device)
    k_rope_cache = torch.zeros(total_blocks, BLOCK_SIZE, 1, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32, device=device)

    k_nope_flat: list[torch.Tensor] = []
    k_rope_flat: list[torch.Tensor] = []

    next_block_id = 1
    for b, s_len in enumerate(seq_lens):
        k_nope = torch.randn(s_len, KV_LORA_RANK, dtype=dtype, device=device)
        k_rope = torch.randn(s_len, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
        k_nope_flat.append(k_nope)
        k_rope_flat.append(k_rope)
        for i in range(blocks_per_seq[b]):
            block_id = next_block_id
            block_table[b, i] = block_id
            tok_start = i * BLOCK_SIZE
            tok_end = min(tok_start + BLOCK_SIZE, s_len)
            length = tok_end - tok_start
            k_nope_cache[block_id, :length, 0, :] = k_nope[tok_start:tok_end]
            k_rope_cache[block_id, :length, 0, :] = k_rope[tok_start:tok_end]
            next_block_id += 1

    return k_nope_cache, k_rope_cache, block_table, k_nope_flat, k_rope_flat


def _run_op(
    ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table, topk_indices,
    cum_query_lens, seq_lens_tensor, scale, *, discrete: bool,
):
    """Direct kernel call; only ``sparse_indices_discrete`` differs across runs."""
    attn_output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope,
        key=k_nope_cache,
        value=k_nope_cache,
        sparse_indices=topk_indices,
        scale_value=scale,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=cum_query_lens,
        actual_seq_lengths_kv=seq_lens_tensor,
        query_rope=q_pe,
        key_rope=k_rope_cache,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        sparse_indices_discrete=discrete,
    )
    return attn_output


def _compacted_topk(selected: list[list[int]], device: torch.device) -> torch.Tensor:
    """Front-packed indices + trailing -1 pad: ``[t0, t1, ..., -1, -1]``."""
    num_tokens = len(selected)
    topk = torch.full((num_tokens, 1, SPARSE_COUNT), -1, dtype=torch.int32, device=device)
    for t, sel in enumerate(selected):
        if sel:
            topk[t, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)
    return topk


def _scatter_topk(selected: list[list[int]], span: int, device: torch.device) -> torch.Tensor:
    """Same token ids spread across ``[0, span)`` with interior -1 holes.

    Places id ``sel[k]`` at slot ``2*k`` (a hole at every odd slot), requiring
    ``2*len(sel) <= span``. Everything else in ``[0, SPARSE_COUNT)`` stays -1.
    """
    num_tokens = len(selected)
    topk = torch.full((num_tokens, 1, SPARSE_COUNT), -1, dtype=torch.int32, device=device)
    for t, sel in enumerate(selected):
        assert 2 * len(sel) <= span, "scatter pattern does not fit in scan span"
        for k, tok in enumerate(sel):
            topk[t, 0, 2 * k] = tok
    return topk


def _cpu_golden(
    ql_nope, q_pe, k_nope_flat, k_rope_flat, selected, scale, out_dtype,
):
    """fp32 gather-softmax over exactly the selected (non -1) token ids."""
    outputs = []
    for t, sel in enumerate(selected):
        k_nope = k_nope_flat[t][sel].float()                # (M, 512)
        k_rope = k_rope_flat[t][sel].float()                # (M, 64)
        K = torch.cat([k_nope, k_rope], dim=-1)             # (M, 576)
        V = k_nope                                          # (M, 512)
        head_out = []
        for h in range(ql_nope.shape[1]):
            Q = torch.cat([ql_nope[t, h].float(), q_pe[t, h].float()], dim=-1)  # (576,)
            scores = (Q @ K.transpose(0, 1)) * scale        # (M,)
            attn = torch.softmax(scores, dim=-1)
            head_out.append(attn @ V)                       # (512,)
        outputs.append(torch.stack(head_out, dim=0))        # (H, 512)
    return torch.stack(outputs, dim=0).to(out_dtype)        # (T, H, 512)


def _make_decode_inputs(seq_lens, dtype, device):
    """Decode batch (q_len=1 per request); returns kernel + golden inputs."""
    num_tokens = len(seq_lens)  # one query token per request
    k_nope_cache, k_rope_cache, block_table, k_nope_flat, k_rope_flat = _build_paged_kv_cache(
        seq_lens, dtype, device
    )
    ql_nope = torch.randn(num_tokens, NUM_HEADS, KV_LORA_RANK, dtype=dtype, device=device)
    q_pe = torch.randn(num_tokens, NUM_HEADS, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    cum_query_lens = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    return (ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
            cum_query_lens, seq_lens_tensor, scale, k_nope_flat, k_rope_flat)


def _assert_close(actual, expected, dtype, tag):
    atol = 5e-3 if dtype == torch.float16 else 1e-2
    rtol = 5e-3 if dtype == torch.float16 else 1e-2
    assert actual.shape == expected.shape, f"[{tag}] shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    diff = (actual.float() - expected.float()).abs()
    peak = expected.float().abs().amax().clamp_min(1e-6)
    rel = (diff.amax() / peak).item()
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), (
        f"[{tag}] mismatch: max|err|={diff.amax().item():.3e} peak|ref|={peak.item():.3e} relerr={rel:.3e}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_lens", [[64], [128], [64, 96, 120]])
def test_discrete_matches_compacted(dtype, seq_lens):
    """DISCRETE [-1,t0,-1,t1,...] must equal COMPACTED [t0,t1,...] (same set)."""
    torch.manual_seed(2026)
    device = torch.device("npu")
    (ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
     cum_query_lens, seq_lens_tensor, scale, _, _) = _make_decode_inputs(seq_lens, dtype, device)

    # Select roughly half the (causally valid) tokens for each request.
    selected = [sorted(range(0, s, 2)) for s in seq_lens]

    compacted_topk = _compacted_topk(selected, device)
    discrete_topk = _scatter_topk(selected, span=max(seq_lens), device=device)

    out_compacted = _run_op(ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
                            compacted_topk, cum_query_lens, seq_lens_tensor, scale, discrete=False)
    out_discrete = _run_op(ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
                           discrete_topk, cum_query_lens, seq_lens_tensor, scale, discrete=True)

    _assert_close(out_discrete, out_compacted, dtype, f"discrete-vs-compacted seq={seq_lens}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_lens", [[64], [128]])
def test_discrete_vs_cpu_golden(dtype, seq_lens):
    """DISCRETE selection vs an independent fp32 gather-softmax golden."""
    torch.manual_seed(7)
    device = torch.device("npu")
    (ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
     cum_query_lens, seq_lens_tensor, scale, k_nope_flat, k_rope_flat) = _make_decode_inputs(
        seq_lens, dtype, device)

    selected = [sorted(range(1, s, 3)) for s in seq_lens]  # arbitrary scattered subset
    discrete_topk = _scatter_topk(selected, span=max(seq_lens), device=device)

    out = _run_op(ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
                  discrete_topk, cum_query_lens, seq_lens_tensor, scale, discrete=True)
    golden = _cpu_golden(ql_nope, q_pe, k_nope_flat, k_rope_flat, selected, scale, dtype)

    _assert_close(out, golden, dtype, f"discrete-vs-golden seq={seq_lens}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_discrete_leading_hole(dtype):
    """A -1 in slot 0 (and other interior holes) must not drop later tokens."""
    torch.manual_seed(11)
    device = torch.device("npu")
    seq_lens = [64]
    (ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
     cum_query_lens, seq_lens_tensor, scale, k_nope_flat, k_rope_flat) = _make_decode_inputs(
        seq_lens, dtype, device)

    selected = [[3, 7, 9, 20, 41]]  # _scatter_topk puts a -1 before each id -> slot 0 is a hole
    discrete_topk = _scatter_topk(selected, span=64, device=device)

    out = _run_op(ql_nope, q_pe, k_nope_cache, k_rope_cache, block_table,
                  discrete_topk, cum_query_lens, seq_lens_tensor, scale, discrete=True)
    golden = _cpu_golden(ql_nope, q_pe, k_nope_flat, k_rope_flat, selected, scale, dtype)

    _assert_close(out, golden, dtype, "discrete-leading-hole")
