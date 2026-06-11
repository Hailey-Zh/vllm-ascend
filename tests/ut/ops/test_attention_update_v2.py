"""
Unit tests for npu_attention_update_v2 custom NPU operator.

Interface:
    out, lse_out = torch.ops._C_ascend.npu_attention_update_v2(lse, local_out, update_type)

Inputs:
    lse        : Tensor [sp, bsh]      float32
    local_out  : Tensor [sp, bsh, hd]  float32 | float16 | bfloat16
    update_type: int  0 = only out;  1 = out + lse_out

Outputs:
    out    : Tensor [bsh, hd]        same dtype as local_out
    lse_out: Tensor [bsh]            float32 (update_type=1) / undefined (update_type=0)
"""

import gc

import pytest
import torch

# Skip all tests if not on NPU hardware
try:
    import torch_npu
    HAS_NPU = torch_npu.npu.is_available()
except (ImportError, RuntimeError):
    HAS_NPU = False

from vllm_ascend.utils import enable_custom_op
enable_custom_op()

SEED = 42


# ---------------------------------------------------------------------------
# CPU golden reference
# ---------------------------------------------------------------------------

def golden_attention_update_v2(lse, local_out, update_type):
    """
    lse       : [sp, bsh]      float32
    local_out : [sp, bsh, hd]  float32/float16/bfloat16

    Formula:
        lse_global = logsumexp(lse, dim=0)                    # [bsh]
        weights    = exp(lse - lse_global)                     # [sp, bsh]
        out        = sum(weights * local_out, dim=0)           # [bsh, hd]
        lse_out    = lse_global  if update_type==1 else None   # [bsh]
    """
    lse_fp32 = lse.to(torch.float32)
    local_fp32 = local_out.to(torch.float32)

    lse_global = torch.logsumexp(lse_fp32, dim=0)
    weights = torch.exp(lse_fp32 - lse_global.unsqueeze(0))
    out = (weights.unsqueeze(-1) * local_fp32).sum(dim=0)

    lse_out = lse_global if update_type == 1 else None
    return out, lse_out


# ---------------------------------------------------------------------------
# NPU runner
# ---------------------------------------------------------------------------

def _run_npu(lse, local_out, update_type):
    """Run on NPU, return CPU tensors with original dtype."""
    out_npu, lse_out_npu = torch.ops._C_ascend.npu_attention_update_v2(
        lse.npu(),
        local_out.npu(),
        update_type,
    )
    out_cpu = out_npu.cpu()
    if lse_out_npu is not None and lse_out_npu.numel() > 0:
        lse_out_cpu = lse_out_npu.cpu()
    else:
        lse_out_cpu = None
    return out_cpu, lse_out_cpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cleanup():
    """Release NPU memory after each test."""
    gc.collect()
    if HAS_NPU:
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()


def _make_inputs(sp, bsh, hd, dtype=torch.float32, seed=SEED):
    """Create consistent lse and local_out tensors."""
    torch.manual_seed(seed)
    lse = torch.randn(sp, bsh, dtype=torch.float32)
    local_out = torch.randn(sp, bsh, hd, dtype=dtype)
    return lse, local_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NPU, reason="Requires NPU hardware")
class TestAttentionUpdateV2:

    # ---- Basic correctness ----

    @pytest.mark.parametrize("sp,bsh,hd", [
        (1, 128, 64),
        (2, 256, 128),
        (3, 512, 256),
        (4, 128, 64),
        (8, 256, 128),
        (16, 512, 128),
    ])
    @pytest.mark.parametrize("update_type", [0, 1])
    def test_random_vs_golden(self, sp, bsh, hd, update_type):
        """Random inputs: NPU result must match CPU golden."""
        lse, local_out = _make_inputs(sp, bsh, hd, dtype=torch.float32)
        ref_out, ref_lse_out = golden_attention_update_v2(lse, local_out, update_type)
        npu_out, npu_lse_out = _run_npu(lse, local_out, update_type)

        torch.testing.assert_close(
            npu_out, ref_out, rtol=1e-3, atol=1e-3,
            msg=f"out mismatch sp={sp} bsh={bsh} hd={hd} update_type={update_type}",
        )
        if update_type == 1:
            assert npu_lse_out is not None
            torch.testing.assert_close(
                npu_lse_out, ref_lse_out, rtol=1e-3, atol=1e-3,
                msg=f"lse_out mismatch sp={sp} bsh={bsh} hd={hd}",
            )
        else:
            assert npu_lse_out is None or npu_lse_out.numel() == 0
        _cleanup()

    # ---- Analytic / exact comparison ----

    def test_analytic_update_type0(self):
        """lse=0, local_out[0]=1 local_out[1]=3 → equal weights → out=2.0"""
        sp, bsh, hd = 2, 128, 64
        lse = torch.zeros(sp, bsh)
        local_out = torch.zeros(sp, bsh, hd)
        local_out[0] = 1.0
        local_out[1] = 3.0

        npu_out, npu_lse_out = _run_npu(lse, local_out, 0)

        torch.testing.assert_close(
            npu_out, torch.full((bsh, hd), 2.0), rtol=1e-3, atol=1e-3,
        )
        assert npu_lse_out is None or npu_lse_out.numel() == 0
        _cleanup()

    def test_analytic_update_type1(self):
        """Same inputs, update_type=1 → lse_out = log(2) ≈ 0.6931"""
        sp, bsh, hd = 2, 128, 64
        lse = torch.zeros(sp, bsh)
        local_out = torch.zeros(sp, bsh, hd)
        local_out[0] = 1.0
        local_out[1] = 3.0

        npu_out, npu_lse_out = _run_npu(lse, local_out, 1)

        torch.testing.assert_close(
            npu_out, torch.full((bsh, hd), 2.0), rtol=1e-3, atol=1e-3,
        )
        assert npu_lse_out is not None and npu_lse_out.shape == torch.Size([bsh])
        torch.testing.assert_close(
            npu_lse_out,
            torch.full((bsh,), float(torch.tensor(2.0).log().item())),
            rtol=1e-3, atol=1e-3,
        )
        _cleanup()

    # ---- sp = 1 (single slice) ----

    def test_sp1_weight_is_one(self):
        """sp=1: weight=1 → out equals local_out[0]"""
        sp, bsh, hd = 1, 256, 128
        lse = torch.randn(sp, bsh)
        local_out = torch.randn(sp, bsh, hd)

        ref_out, _ = golden_attention_update_v2(lse, local_out, 0)
        npu_out, _ = _run_npu(lse, local_out, 0)

        torch.testing.assert_close(npu_out, ref_out, rtol=1e-3, atol=1e-3)
        _cleanup()

    # ---- sp = 16 (max) ----

    @pytest.mark.parametrize("bsh,hd", [(128, 64), (256, 128)])
    @pytest.mark.parametrize("update_type", [0, 1])
    def test_sp16_max(self, bsh, hd, update_type):
        """Max sp slices (16) should work correctly."""
        sp = 16
        lse, local_out = _make_inputs(sp, bsh, hd, dtype=torch.float32)
        ref_out, ref_lse_out = golden_attention_update_v2(lse, local_out, update_type)
        npu_out, npu_lse_out = _run_npu(lse, local_out, update_type)

        torch.testing.assert_close(npu_out, ref_out, rtol=1e-3, atol=1e-3)
        if update_type == 1:
            torch.testing.assert_close(npu_lse_out, ref_lse_out, rtol=1e-3, atol=1e-3)
        _cleanup()

    # ---- Data type tests ----

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("update_type", [0, 1])
    def test_dtypes(self, dtype, update_type):
        """All supported dtypes for local_out."""
        sp, bsh, hd = 3, 256, 128
        lse, local_out = _make_inputs(sp, bsh, hd, dtype=dtype)

        # Golden uses fp32 for comparison
        lse_fp32 = lse.to(torch.float32)
        local_fp32 = local_out.to(torch.float32)
        ref_out, ref_lse_out = golden_attention_update_v2(lse_fp32, local_fp32, update_type)

        npu_out, npu_lse_out = _run_npu(lse, local_out, update_type)

        # For fp16/bf16, use looser tolerance
        rtol = 1e-2 if dtype != torch.float32 else 1e-3
        atol = 1e-2 if dtype != torch.float32 else 1e-3
        torch.testing.assert_close(
            npu_out.to(torch.float32), ref_out, rtol=rtol, atol=atol,
            msg=f"dtype={dtype} update_type={update_type}",
        )
        if update_type == 1:
            torch.testing.assert_close(
                npu_lse_out.to(torch.float32), ref_lse_out, rtol=1e-3, atol=1e-3,
            )
        _cleanup()

    # ---- Large shapes ----

    def test_large_shape(self):
        """Larger head_dim and tokens to stress-test tiling."""
        sp, bsh, hd = 4, 8192, 512
        lse = torch.randn(sp, bsh, dtype=torch.float32)
        local_out = torch.randn(sp, bsh, hd, dtype=torch.float16)

        ref_out, _ = golden_attention_update_v2(lse, local_out, 0)
        npu_out, _ = _run_npu(lse, local_out, 0)

        # Large shapes may have more numerical drift
        torch.testing.assert_close(
            npu_out.to(torch.float32), ref_out, rtol=5e-2, atol=1e-2,
        )
        _cleanup()

    # ---- Equal weights ----

    def test_equal_weights_sp4(self):
        """sp=4, all lse equal → each slice weight = 1/4. local_out[i]=i → out=1.5"""
        sp, bsh, hd = 4, 64, 32
        lse = torch.ones(sp, bsh)
        local_out = (
            torch.arange(sp, dtype=torch.float32)
            .view(sp, 1, 1)
            .expand(sp, bsh, hd)
            .contiguous()
        )

        ref_out, _ = golden_attention_update_v2(lse, local_out, 0)
        npu_out, _ = _run_npu(lse, local_out, 0)

        torch.testing.assert_close(npu_out, ref_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            npu_out, torch.full((bsh, hd), 1.5), rtol=1e-3, atol=1e-3,
        )
        _cleanup()

    # ---- Extreme lse values (numerical stability) ----

    def test_large_lse_diff(self):
        """Large LSE difference between slices: one slice dominates."""
        sp, bsh, hd = 3, 128, 64
        lse = torch.zeros(sp, bsh)
        lse[0] = 100.0   # dominates
        lse[1] = 0.0
        lse[2] = -50.0
        local_out = torch.randn(sp, bsh, hd, dtype=torch.float32)

        ref_out, ref_lse_out = golden_attention_update_v2(lse, local_out, 1)
        npu_out, npu_lse_out = _run_npu(lse, local_out, 1)

        torch.testing.assert_close(npu_out, ref_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(npu_lse_out, ref_lse_out, rtol=1e-3, atol=1e-3)
        _cleanup()

    def test_negative_large_lse(self):
        """All LSE very negative: exp should still work correctly."""
        sp, bsh, hd = 2, 128, 64
        lse = -100.0 * torch.ones(sp, bsh)
        local_out = torch.randn(sp, bsh, hd, dtype=torch.float32)

        ref_out, _ = golden_attention_update_v2(lse, local_out, 0)
        npu_out, _ = _run_npu(lse, local_out, 0)

        torch.testing.assert_close(npu_out, ref_out, rtol=1e-3, atol=1e-3)
        _cleanup()

    # ---- Reproducibility ----

    def test_determinism(self):
        """Same inputs should produce identical outputs across runs."""
        sp, bsh, hd = 3, 256, 128
        lse, local_out = _make_inputs(sp, bsh, hd)

        out1, lse1 = _run_npu(lse, local_out, 1)
        _cleanup()
        out2, lse2 = _run_npu(lse, local_out, 1)
        _cleanup()

        assert torch.equal(out1, out2), "Non-deterministic output"
        assert torch.equal(lse1, lse2), "Non-deterministic lse_out"
