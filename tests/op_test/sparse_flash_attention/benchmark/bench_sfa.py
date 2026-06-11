#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# SparseFlashAttention latency benchmark.
#
# Measures kernel latency (NOT accuracy) across a shape grid for each feature:
#   - basic   : plain sparse attention (attn_out only)
#   - lse     : return_softmax_lse=True
#   - dense   : sparse_indices=None (dense FA)
#   - packed  : return_packed_kv=True (sparse_block_size <= 4)
# Both decode (S1=1, large S2) and prefill (large S1) shapes are covered, fp16 + bf16.
#
# Output: a stdout summary table + a CSV under benchmark/results/.
#
# Run manually (must source set_env.bash first):
#   pytest bench_sfa.py -m bench -s -v
# or standalone (no pytest):
#   python bench_sfa.py [--warmup N] [--iters N] [--csv path] [--filter substr]

import argparse
import csv
import os
import statistics
import time
from datetime import datetime

import pytest
import torch
import torch_npu  # noqa: F401  # registers npu device
import vllm_ascend  # noqa: F401  # registers torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SCALE = 1.0 / (576 ** 0.5)
DEVICE = "npu"
N2 = 1            # MLA: always 1 KV head
D = 512           # NoPE dim
ROPE = 64         # RoPE dim

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Shape grid.  Each entry: (name, feature, layout_q, layout_kv, dtype, B, S1, S2, N1, K)
#   feature ∈ {basic, lse, dense, packed}
#   For TND, S1 is the per-batch query length (T1 = B * S1 here for simplicity).
#   packed requires sparse_block_size <= 4 -> uses block_size=1, modest K.
# ---------------------------------------------------------------------------

def _build_grid():
    grid = []
    fp16, bf16 = torch.float16, torch.bfloat16

    # --- decode (S1=1, large S2) across the 4 features & layouts ---
    decode_shapes = [
        # (B, S2, N1, K)
        (1, 4096, 16, 2048),
        (8, 4096, 16, 2048),
        (1, 8192, 16, 2048),
    ]
    for (B, S2, N1, K) in decode_shapes:
        grid.append((f"basic_decode_B{B}_S2{S2}", "basic", "TND", "PA_BSND", fp16, B, 1, S2, N1, K))
        grid.append((f"lse_decode_B{B}_S2{S2}", "lse", "TND", "PA_BSND", fp16, B, 1, S2, N1, K))
    # dense decode (no sparse_indices) — K is irrelevant, uses full S2
    grid.append(("dense_decode_B1_S2-4096", "dense", "TND", "PA_BSND", fp16, 1, 1, 4096, 16, 0))
    grid.append(("dense_decode_B8_S2-4096", "dense", "TND", "PA_BSND", fp16, 8, 1, 4096, 16, 0))

    # --- prefill (large S1) ---
    prefill_shapes = [
        # (B, S1, S2, N1, K)
        (1, 512, 4096, 16, 2048),
        (2, 1024, 4096, 16, 2048),
    ]
    for (B, S1, S2, N1, K) in prefill_shapes:
        grid.append((f"basic_prefill_B{B}_S1{S1}", "basic", "BSND", "PA_BSND", fp16, B, S1, S2, N1, K))
        grid.append((f"lse_prefill_B{B}_S1{S1}", "lse", "BSND", "PA_BSND", fp16, B, S1, S2, N1, K))
    grid.append(("dense_prefill_B1_S1-512", "dense", "BSND", "PA_BSND", fp16, 1, 512, 4096, 16, 0))

    # --- packed_kv (sparse_block_size<=4) decode + prefill ---
    grid.append(("packed_decode_B1_S2-2048", "packed", "BSND", "BSND", fp16, 1, 1, 2048, 16, 256))
    grid.append(("packed_decode_B8_S2-2048", "packed", "BSND", "BSND", fp16, 8, 1, 2048, 16, 256))
    grid.append(("packed_prefill_B1_S1-128", "packed", "BSND", "BSND", fp16, 1, 128, 2048, 16, 256))

    # --- bf16 coverage (one per feature) ---
    grid.append(("basic_decode_bf16_B8", "basic", "TND", "PA_BSND", bf16, 8, 1, 4096, 16, 2048))
    grid.append(("lse_decode_bf16_B8", "lse", "TND", "PA_BSND", bf16, 8, 1, 4096, 16, 2048))
    grid.append(("dense_decode_bf16_B8", "dense", "TND", "PA_BSND", bf16, 8, 1, 4096, 16, 0))
    grid.append(("packed_decode_bf16_B8", "packed", "BSND", "BSND", bf16, 8, 1, 2048, 16, 256))

    return grid


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

PA_BLOCK_SIZE = 128


def _make_inputs(feature, layout_q, layout_kv, dtype, B, S1, S2, N1, K):
    """Build a ready-to-call kwargs dict for npu_sparse_flash_attention."""
    torch.manual_seed(2026)
    is_tnd = (layout_q == "TND")
    is_pa = (layout_kv == "PA_BSND")

    # query
    if is_tnd:
        T1 = B * S1
        query = torch.randn(T1, N1, D, dtype=dtype, device=DEVICE) * 0.1
        query_rope = torch.randn(T1, N1, ROPE, dtype=dtype, device=DEVICE) * 0.1
        actual_seq_q = torch.arange(1, B + 1, dtype=torch.int32, device=DEVICE) * S1  # cumsum
    else:
        query = torch.randn(B, S1, N1, D, dtype=dtype, device=DEVICE) * 0.1
        query_rope = torch.randn(B, S1, N1, ROPE, dtype=dtype, device=DEVICE) * 0.1
        actual_seq_q = torch.tensor([S1] * B, dtype=torch.int32, device=DEVICE)

    # key/value (+ rope), block_table
    if is_pa:
        blocks_per_batch = (S2 + PA_BLOCK_SIZE - 1) // PA_BLOCK_SIZE
        block_num = B * blocks_per_batch
        key = torch.randn(block_num, PA_BLOCK_SIZE, N2, D, dtype=dtype, device=DEVICE) * 0.1
        key_rope = torch.randn(block_num, PA_BLOCK_SIZE, N2, ROPE, dtype=dtype, device=DEVICE) * 0.1
        block_table = torch.arange(block_num, dtype=torch.int32, device=DEVICE).view(B, blocks_per_batch)
        actual_seq_kv = torch.tensor([S2] * B, dtype=torch.int32, device=DEVICE)
        kv_len_for_sel = S2
    else:
        key = torch.randn(B, S2, N2, D, dtype=dtype, device=DEVICE) * 0.1
        key_rope = torch.randn(B, S2, N2, ROPE, dtype=dtype, device=DEVICE) * 0.1
        block_table = None
        actual_seq_kv = torch.tensor([S2] * B, dtype=torch.int32, device=DEVICE)
        kv_len_for_sel = S2

    value = key

    # sparse_indices (None for dense)
    sparse_block_size = 1
    if feature == "dense":
        sparse_indices = None
    elif feature == "packed":
        sparse_block_size = 1  # block<=4; keep 1 for simplicity
        if is_tnd:
            sparse_indices = torch.randint(0, kv_len_for_sel, (B * S1, N2, K), dtype=torch.int32, device=DEVICE)
        else:
            sparse_indices = torch.randint(0, kv_len_for_sel, (B, S1, N2, K), dtype=torch.int32, device=DEVICE)
    else:
        if is_tnd:
            sparse_indices = torch.randint(0, kv_len_for_sel, (B * S1, N2, K), dtype=torch.int32, device=DEVICE)
        else:
            sparse_indices = torch.randint(0, kv_len_for_sel, (B, S1, N2, K), dtype=torch.int32, device=DEVICE)

    return dict(
        query=query, key=key, value=value,
        sparse_indices=sparse_indices,
        scale_value=SCALE, sparse_block_size=sparse_block_size,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query=layout_q, layout_kv=layout_kv,
        sparse_mode=0,
        return_softmax_lse=(feature == "lse"),
        return_packed_kv=(feature == "packed"),
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_op(kwargs, warmup, iters):
    """Return (mean_ms, p50_ms, p90_ms) over `iters` timed runs."""
    op = torch.ops._C_ascend.npu_sparse_flash_attention

    for _ in range(warmup):
        op(**kwargs)
    torch.npu.synchronize()

    # Prefer device-side events; fall back to host timer if unavailable.
    use_events = hasattr(torch.npu, "Event")
    samples = []
    if use_events:
        try:
            for _ in range(iters):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                op(**kwargs)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))  # ms
        except (RuntimeError, TypeError):
            use_events = False
            samples = []
    if not use_events:
        for _ in range(iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            op(**kwargs)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    mean = statistics.fmean(samples)
    p50 = statistics.median(samples)
    p90 = samples[min(len(samples) - 1, int(round(0.9 * (len(samples) - 1))))]
    return mean, p50, p90


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(warmup=DEFAULT_WARMUP, iters=DEFAULT_ITERS, csv_path=None, name_filter=None):
    grid = _build_grid()
    rows = []
    print(f"\n{'='*108}")
    print(f"SFA latency benchmark  (warmup={warmup}, iters={iters})")
    print(f"{'='*108}")
    header = f"{'name':<28}{'feat':<8}{'layout':<16}{'dt':<6}{'B':>3}{'S1':>6}{'S2':>7}{'N1':>4}{'K':>6}" \
             f"{'mean':>9}{'p50':>9}{'p90':>9}"
    print(header)
    print("-" * 108)

    for (name, feature, lq, lkv, dtype, B, S1, S2, N1, K) in grid:
        if name_filter and name_filter not in name:
            continue
        dt = "bf16" if dtype == torch.bfloat16 else "fp16"
        try:
            kwargs = _make_inputs(feature, lq, lkv, dtype, B, S1, S2, N1, K)
            mean, p50, p90 = _time_op(kwargs, warmup, iters)
            print(f"{name:<28}{feature:<8}{lq+'/'+lkv:<16}{dt:<6}{B:>3}{S1:>6}{S2:>7}{N1:>4}{K:>6}"
                  f"{mean:>9.3f}{p50:>9.3f}{p90:>9.3f}")
            rows.append(dict(name=name, feature=feature, layout_q=lq, layout_kv=lkv, dtype=dt,
                             B=B, S1=S1, S2=S2, N1=N1, K=K,
                             mean_ms=round(mean, 4), p50_ms=round(p50, 4), p90_ms=round(p90, 4)))
        except Exception as e:  # noqa: BLE001 - bench should not abort on one bad shape
            print(f"{name:<28}{feature:<8}{lq+'/'+lkv:<16}{dt:<6}  ERROR: {type(e).__name__}: {e}")
            rows.append(dict(name=name, feature=feature, layout_q=lq, layout_kv=lkv, dtype=dt,
                             B=B, S1=S1, S2=S2, N1=N1, K=K,
                             mean_ms="ERROR", p50_ms="", p90_ms=""))
    print("-" * 108)

    if csv_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(RESULTS_DIR, f"bench_sfa_{stamp}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[bench] wrote {len(rows)} rows -> {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# pytest entrypoint (single test that runs the whole grid)
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_benchmark_all():
    rows = run_benchmark()
    ok = [r for r in rows if r["mean_ms"] != "ERROR"]
    assert ok, "all benchmark shapes errored — check environment / op build"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFA latency benchmark")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--csv", type=str, default=None, help="output CSV path")
    parser.add_argument("--filter", type=str, default=None, help="only run shapes whose name contains this substring")
    args = parser.parse_args()
    run_benchmark(warmup=args.warmup, iters=args.iters, csv_path=args.csv, name_filter=args.filter)
