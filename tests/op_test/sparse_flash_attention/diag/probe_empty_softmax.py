#!/usr/bin/env python3
"""Probe the all-(-1) sparse_indices case: what do softmax_max / softmax_sum hold
when NO kv participates? (return_softmax_lse=True so the op writes real outputs.)

NOTE: return_softmax_lse=True does NOT support PA_BSND (tiling rejects it), so
this uses contiguous BSND layout (no paging). The empty-attention softmax
contract is layout-independent, so BSND is fine for this check.

    query / query_rope : (B, Sq, N, D) / (B, Sq, N, Drope)   D=512, Drope=64
    key / value / krope: (B, Skv, Nkv, D...)                 Nkv=1 (MQA)
    sparse_indices     : (B, Sq, Nkv, K) int32

Run:  python3 probe_empty_softmax.py
"""

import math

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

B, SQ, SKV, N, NKV, D, DROPE, K = 1, 1, 64, 8, 1, 512, 64, 8
DTYPE = torch.bfloat16


def log(*a):
    print("[empty]", *a, flush=True)


def build(device):
    torch.manual_seed(0)
    query = torch.randn(B, SQ, N, D, dtype=DTYPE, device=device)
    qrope = torch.randn(B, SQ, N, DROPE, dtype=DTYPE, device=device)
    key = torch.randn(B, SKV, NKV, D, dtype=DTYPE, device=device)
    krope = torch.randn(B, SKV, NKV, DROPE, dtype=DTYPE, device=device)
    value = key  # MLA: value == key_nope
    return query, qrope, key, value, krope, 1.0 / math.sqrt(D + DROPE)


def run(topk, query, qrope, key, value, krope, scale):
    out, smax, ssum = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query, key=key, value=value, sparse_indices=topk, scale_value=scale,
        sparse_block_size=1, query_rope=qrope, key_rope=krope,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3, attention_mode=2,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()
    return out, smax, ssum


def describe(name, t):
    f = t.float()
    flat = f.flatten()
    log(f"{name}: shape={tuple(t.shape)} min={f.min().item():.4e} max={f.max().item():.4e} "
        f"nan={torch.isnan(f).any().item()} inf={torch.isinf(f).any().item()}")
    log(f"    sample[:8]={[round(x, 4) for x in flat[:8].tolist()]}")


def case(label, topk, *args):
    log(f"==== {label} ====")
    out, smax, ssum = run(topk, *args)
    describe("attention_out", out)
    describe("softmax_max", smax)
    describe("softmax_sum", ssum)
    lse = smax.float() + torch.log(ssum.float().clamp_min(0))
    describe("derived_lse = max + log(sum)", lse)


def main():
    device = torch.device("npu")
    args = build(device)

    topk_a = torch.full((B, SQ, NKV, K), -1, dtype=torch.int32, device=device)
    topk_a[0, 0, 0, :3] = torch.tensor([3, 17, 40], dtype=torch.int32, device=device)
    case("CASE A: 3 valid ids (sanity)", topk_a, *args)

    topk_b = torch.full((B, SQ, NKV, K), -1, dtype=torch.int32, device=device)
    case("CASE B: ALL -1 (no kv participates)", topk_b, *args)


if __name__ == "__main__":
    main()
