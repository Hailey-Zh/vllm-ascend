#!/usr/bin/env python3
"""Minimal standalone probe for SparseFlashAttention discrete sparse_indices.

No vllm / no HuggingFace / no pytest / no forward_context. Hand-rolls the
smallest valid decode input (TND query + PA_BSND paged KV, MLA 512+64) and
calls the op directly. Run:  python3 probe_discrete_sfa.py

Prints progress before/after each op call so a device segfault is attributable
to a specific step. The FIRST call uses COMPACTED (the legacy, known-good path):
if that already crashes, the problem is the inputs/env, not the discrete change.
"""

import math
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Fixed MLA dims (kernel hardcodes 512 nope + 64 rope).
KV_LORA = 512
QK_ROPE = 64
BLOCK_SIZE = 128
NUM_HEADS = 8          # query heads; 1 kv head (MQA), group size = 8
SPARSE_COUNT = 128     # topk width
SEQ_LEN = 64           # single decode request
DTYPE = torch.bfloat16


def log(*a):
    print("[probe]", *a, flush=True)


def build_inputs(device):
    torch.manual_seed(0)
    # --- paged KV cache: block 0 reserved, block 1 holds the 64 tokens --------
    total_blocks = 2
    k_nope_dense = torch.randn(SEQ_LEN, KV_LORA, dtype=DTYPE, device=device)
    k_rope_dense = torch.randn(SEQ_LEN, QK_ROPE, dtype=DTYPE, device=device)
    k_nope_cache = torch.zeros(total_blocks, BLOCK_SIZE, 1, KV_LORA, dtype=DTYPE, device=device)
    k_rope_cache = torch.zeros(total_blocks, BLOCK_SIZE, 1, QK_ROPE, dtype=DTYPE, device=device)
    k_nope_cache[1, :SEQ_LEN, 0, :] = k_nope_dense
    k_rope_cache[1, :SEQ_LEN, 0, :] = k_rope_dense
    block_table = torch.tensor([[1]], dtype=torch.int32, device=device)  # (batch=1, 1 block)

    # --- query (one decode token) --------------------------------------------
    ql_nope = torch.randn(1, NUM_HEADS, KV_LORA, dtype=DTYPE, device=device)
    q_pe = torch.randn(1, NUM_HEADS, QK_ROPE, dtype=DTYPE, device=device)

    cum_query_lens = torch.tensor([1], dtype=torch.int32, device=device)   # TND prefix sum
    seq_lens_kv = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)  # raw per-batch
    scale = 1.0 / math.sqrt(KV_LORA + QK_ROPE)
    return dict(
        ql_nope=ql_nope, q_pe=q_pe, k_nope_cache=k_nope_cache, k_rope_cache=k_rope_cache,
        block_table=block_table, cum_query_lens=cum_query_lens, seq_lens_kv=seq_lens_kv,
        scale=scale, k_nope_dense=k_nope_dense, k_rope_dense=k_rope_dense,
    )


def run_op(t, topk, discrete):
    out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=t["ql_nope"], key=t["k_nope_cache"], value=t["k_nope_cache"],
        sparse_indices=topk, scale_value=t["scale"], sparse_block_size=1,
        block_table=t["block_table"], actual_seq_lengths_query=t["cum_query_lens"],
        actual_seq_lengths_kv=t["seq_lens_kv"], query_rope=t["q_pe"], key_rope=t["k_rope_cache"],
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
        sparse_indices_discrete=discrete,
    )
    torch.npu.synchronize()
    return out


def compacted_topk(sel, device):
    topk = torch.full((1, 1, SPARSE_COUNT), -1, dtype=torch.int32, device=device)
    topk[0, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)
    return topk


def scatter_topk(sel, device):
    topk = torch.full((1, 1, SPARSE_COUNT), -1, dtype=torch.int32, device=device)
    for k, tok in enumerate(sel):
        topk[0, 0, 2 * k] = tok  # hole before/between each id
    return topk


def cpu_golden(t, sel):
    K = torch.cat([t["k_nope_dense"][sel].float(), t["k_rope_dense"][sel].float()], dim=-1)
    V = t["k_nope_dense"][sel].float()
    outs = []
    for h in range(NUM_HEADS):
        Q = torch.cat([t["ql_nope"][0, h].float(), t["q_pe"][0, h].float()], dim=-1)
        attn = torch.softmax((Q @ K.transpose(0, 1)) * t["scale"], dim=-1)
        outs.append(attn @ V)
    return torch.stack(outs, dim=0)  # (H, 512)


def main():
    log("torch", torch.__version__, "| npu available:", torch.npu.is_available())
    device = torch.device("npu")
    t = build_inputs(device)
    log("inputs built. q", tuple(t["ql_nope"].shape), "kv_cache", tuple(t["k_nope_cache"].shape))

    sel = list(range(0, SEQ_LEN, 4))  # 16 tokens; scattered slots 0..30 < seq_len
    log("selected ids:", sel)

    log(">>> calling COMPACTED (discrete=False) ...")
    out_c = run_op(t, compacted_topk(sel, device), discrete=False)
    log("COMPACTED ok. out", tuple(out_c.shape))

    log(">>> calling DISCRETE (discrete=True) ...")
    out_d = run_op(t, scatter_topk(sel, device), discrete=True)
    log("DISCRETE ok. out", tuple(out_d.shape))

    gold = cpu_golden(t, sel).to(out_d.dtype).unsqueeze(0)  # (1, H, 512)
    dc = (out_d.float() - out_c.float()).abs().amax().item()
    dg = (out_d.float() - gold.float()).abs().amax().item()
    peak = gold.float().abs().amax().item()
    log(f"max|discrete - compacted| = {dc:.3e}")
    log(f"max|discrete - cpu_golden| = {dg:.3e}  (peak|gold|={peak:.3e})")

    ok = dc < 2e-2 and dg < 3e-2
    log("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
