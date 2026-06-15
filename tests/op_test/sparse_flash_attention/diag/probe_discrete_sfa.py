#!/usr/bin/env python3
"""Minimal standalone probe for SparseFlashAttention discrete sparse_indices.

No vllm / no HuggingFace / no pytest / no forward_context / no paging. Mirrors
the op author's aclnn example (csrc/.../examples/test_aclnn_sparse_flash_attention.cpp):
contiguous BSND layout, no block_table, no actual_seq_lengths.

    query/key/value : (B, S, N, D)      D = 512 (nope)
    query/key rope  : (B, S, N, Drope)  Drope = 64
    sparse_indices  : (B, Sq, N, K)     int32, token ids into [0, Skv)

Run:  python3 probe_discrete_sfa.py
Prints before/after each op call so a device segfault is attributable. The
FIRST call is COMPACTED (legacy, known-good); if that crashes the inputs/env
are wrong, not the discrete change.
"""

import math
import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

B, N = 1, 1            # batch, heads (example uses N=1)
SQ = 1                 # query tokens (decode)
SKV = 8                # kv sequence length
D = 512                # nope dim
DROPE = 64             # rope dim
K = 8                  # topk width (sparse_indices last dim)
DTYPE = torch.bfloat16


def log(*a):
    print("[probe]", *a, flush=True)


def build(device):
    torch.manual_seed(0)
    query = torch.randn(B, SQ, N, D, dtype=DTYPE, device=device)
    key = torch.randn(B, SKV, N, D, dtype=DTYPE, device=device)
    value = key  # MLA: value == key_nope
    query_rope = torch.randn(B, SQ, N, DROPE, dtype=DTYPE, device=device)
    key_rope = torch.randn(B, SKV, N, DROPE, dtype=DTYPE, device=device)
    scale = 1.0 / math.sqrt(D + DROPE)
    return dict(query=query, key=key, value=value, query_rope=query_rope,
                key_rope=key_rope, scale=scale)


def run_op(t, topk, discrete):
    out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=t["query"], key=t["key"], value=t["value"],
        sparse_indices=topk, scale_value=t["scale"], sparse_block_size=1,
        query_rope=t["query_rope"], key_rope=t["key_rope"],
        layout_query="BSND", layout_kv="BSND",
        sparse_mode=3, attention_mode=2,
        sparse_indices_discrete=discrete,
    )
    torch.npu.synchronize()
    return out


def compacted_topk(sel, device):
    topk = torch.full((B, SQ, N, K), -1, dtype=torch.int32, device=device)
    topk[0, 0, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)
    return topk


def scatter_topk(sel, device):
    topk = torch.full((B, SQ, N, K), -1, dtype=torch.int32, device=device)
    for k, tok in enumerate(sel):
        topk[0, 0, 0, 2 * k] = tok  # hole before/between each id
    return topk


def cpu_golden(t, sel):
    key = t["key"][0, sel, 0, :].float()         # (M, 512)
    key_rope = t["key_rope"][0, sel, 0, :].float()  # (M, 64)
    Kmat = torch.cat([key, key_rope], dim=-1)    # (M, 576)
    V = t["value"][0, sel, 0, :].float()         # (M, 512)
    Q = torch.cat([t["query"][0, 0, 0, :].float(), t["query_rope"][0, 0, 0, :].float()], dim=-1)
    attn = torch.softmax((Q @ Kmat.transpose(0, 1)) * t["scale"], dim=-1)
    return (attn @ V)  # (512,)


def main():
    log("torch", torch.__version__, "| npu:", torch.npu.is_available())
    device = torch.device("npu")
    t = build(device)
    log("inputs built. query", tuple(t["query"].shape), "key", tuple(t["key"].shape))

    sel = [1, 3, 4, 6]  # 4 selected token ids in [0, 8); scattered slots 0,2,4,6 < K
    log("selected ids:", sel)

    log(">>> COMPACTED (discrete=False) ...")
    out_c = run_op(t, compacted_topk(sel, device), discrete=False)
    log("COMPACTED ok. out", tuple(out_c.shape))

    log(">>> DISCRETE (discrete=True) ...")
    out_d = run_op(t, scatter_topk(sel, device), discrete=True)
    log("DISCRETE ok. out", tuple(out_d.shape))

    gold = cpu_golden(t, sel)
    od = out_d.reshape(-1)[:D].float()
    oc = out_c.reshape(-1)[:D].float()
    dc = (od - oc).abs().amax().item()
    dg = (od - gold).abs().amax().item()
    peak = gold.abs().amax().item()
    log(f"max|discrete - compacted| = {dc:.3e}")
    log(f"max|discrete - golden|    = {dg:.3e}  (peak|gold|={peak:.3e})")
    ok = dc < 2e-2 and dg < 3e-2
    log("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
