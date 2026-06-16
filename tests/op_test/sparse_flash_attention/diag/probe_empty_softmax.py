#!/usr/bin/env python3
"""Probe the all-(-1) sparse_indices case: what do softmax_max / softmax_sum hold
when NO kv participates? (return_softmax_lse=True so the op writes real outputs.)

Same proven TND + PA_BSND / N=8 inputs as probe_discrete_sfa.py. Prints the raw
softmax_max / softmax_sum and the derived lse = max + log(sum), and flags
nan/inf, so you can confirm the empty-row contract empirically instead of
guessing from the masked-softmax degenerate path.

Run:  python3 probe_empty_softmax.py
"""

import math

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

NUM_HEADS, D, DROPE, BLOCK_SIZE, SEQ_LEN, K = 8, 512, 64, 128, 64, 8
DTYPE = torch.bfloat16


def log(*a):
    print("[empty]", *a, flush=True)


def build(device):
    torch.manual_seed(0)
    kn = torch.randn(SEQ_LEN, D, dtype=DTYPE, device=device)
    kr = torch.randn(SEQ_LEN, DROPE, dtype=DTYPE, device=device)
    knc = torch.zeros(2, BLOCK_SIZE, 1, D, dtype=DTYPE, device=device); knc[1, :SEQ_LEN, 0, :] = kn
    krc = torch.zeros(2, BLOCK_SIZE, 1, DROPE, dtype=DTYPE, device=device); krc[1, :SEQ_LEN, 0, :] = kr
    bt = torch.tensor([[1]], dtype=torch.int32, device=device)
    ql = torch.randn(1, NUM_HEADS, D, dtype=DTYPE, device=device)
    qp = torch.randn(1, NUM_HEADS, DROPE, dtype=DTYPE, device=device)
    cq = torch.tensor([1], dtype=torch.int32, device=device)
    sk = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)
    return knc, krc, bt, ql, qp, cq, sk, 1.0 / math.sqrt(D + DROPE)


def run(topk, knc, krc, bt, ql, qp, cq, sk, scale):
    out, smax, ssum = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql, key=knc, value=knc, sparse_indices=topk, scale_value=scale,
        sparse_block_size=1, block_table=bt, actual_seq_lengths_query=cq,
        actual_seq_lengths_kv=sk, query_rope=qp, key_rope=krc,
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()
    return out, smax, ssum


def describe(name, t):
    f = t.float()
    log(f"{name}: shape={tuple(t.shape)} dtype={t.dtype}")
    log(f"    min={f.min().item():.4e} max={f.max().item():.4e} "
        f"has_nan={torch.isnan(f).any().item()} has_inf={torch.isinf(f).any().item()}")
    flat = f.flatten()
    log(f"    sample[:8]={[round(x, 4) for x in flat[:8].tolist()]}")


def main():
    device = torch.device("npu")
    knc, krc, bt, ql, qp, cq, sk, scale = build(device)

    log("==== CASE A: a few valid ids (sanity) ====")
    topk = torch.full((1, 1, K), -1, dtype=torch.int32, device=device)
    topk[0, 0, :3] = torch.tensor([3, 17, 40], dtype=torch.int32, device=device)
    out, smax, ssum = run(topk, knc, krc, bt, ql, qp, cq, sk, scale)
    describe("attention_out", out)
    describe("softmax_max", smax)
    describe("softmax_sum", ssum)

    log("==== CASE B: ALL -1 (no kv participates) ====")
    topk_empty = torch.full((1, 1, K), -1, dtype=torch.int32, device=device)
    out, smax, ssum = run(topk_empty, knc, krc, bt, ql, qp, cq, sk, scale)
    describe("attention_out", out)
    describe("softmax_max", smax)
    describe("softmax_sum", ssum)
    # derived lse = max + log(sum); print to see whether it is 0 / -inf / nan
    lse = smax.float() + torch.log(ssum.float())
    describe("derived_lse = max + log(sum)", lse)


if __name__ == "__main__":
    main()
