#!/bin/bash
# -----------------------------------------------------------------------------
# vllm-ascend SparseFlashAttention baseline regression
# -----------------------------------------------------------------------------
# 用途：在 NPU 机器上一键跑 5 个 baseline 用例（覆盖 4 种 layout 组合），用于
#       sfa_modify_plan.md 中每一步改造后的回归对比。
#
# 用法：
#   bash run_baseline.sh
#
# 前置：
#   1. 已在 NPU 机器上编译并安装 vllm-ascend（含 csrc 自定义算子）
#   2. torch / torch_npu / vllm_ascend 可正常 import
#   3. NPU device 0 可用
# -----------------------------------------------------------------------------

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- step 1: 环境检查 ----------
echo "===== [baseline] 环境检查 ====="

python3 - <<'PY'
import sys

try:
    import torch
except ImportError:
    print("[FATAL] torch 不可用")
    sys.exit(2)
print(f"  torch          = {torch.__version__}")

try:
    import torch_npu
except ImportError:
    print("[FATAL] torch_npu 不可用，无法在 NPU 上跑")
    sys.exit(2)
print(f"  torch_npu      = {torch_npu.__version__}")

try:
    import vllm_ascend  # noqa: F401
except ImportError:
    print("[FATAL] vllm_ascend 未安装；请先 `pip install -e .` 或 build")
    sys.exit(2)
print(f"  vllm_ascend    = OK")

if not hasattr(torch.ops, "_C_ascend"):
    print("[FATAL] torch.ops._C_ascend 未注册（C++ 算子可能没编进来）")
    sys.exit(2)
if not hasattr(torch.ops._C_ascend, "npu_sparse_flash_attention"):
    print("[FATAL] torch.ops._C_ascend.npu_sparse_flash_attention 未注册")
    sys.exit(2)
print("  custom op      = npu_sparse_flash_attention 已注册")

dev_cnt = torch.npu.device_count() if hasattr(torch, "npu") else 0
if dev_cnt < 1:
    print("[FATAL] 未检测到 NPU device")
    sys.exit(2)
print(f"  npu devices    = {dev_cnt}")
PY

rc=$?
if [ "$rc" != "0" ]; then
    echo "===== [baseline] 环境检查失败，退出 ====="
    exit $rc
fi

echo "===== [baseline] 环境检查通过 ====="
echo

# ---------- step 2: 跑 5 个 baseline 用例 ----------
mkdir -p ./golden ./excel

echo "===== [baseline] 跑 5 个用例（CPU golden + NPU + 精度对比） ====="
bash test_run.sh single
result=$?

echo
if [ "$result" == "0" ]; then
    echo "===== [baseline] 全部 PASS ====="
else
    echo "===== [baseline] 有用例失败，rc=$result ====="
fi

exit $result
