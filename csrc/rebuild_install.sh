#!/bin/bash
# One-key rebuild + install for the SparseFlashAttention work on this branch.
#
# The op spans TWO independent build artifacts; this script does both in order:
#   Layer 1  CANN custom-ops package  (def.cpp / tiling.{h,cpp} / op_kernel/* / common.h)
#            -> installed into vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/
#   Layer 2  vllm-ascend Python ext _C_ascend.so  (torch_binding{,_meta}.cpp / *_torch_adpt.h)
#            -> installed in-tree via editable pip
#
# Usage:
#   bash csrc/rebuild_install.sh [--ops-only|--py-only] [--verify]
#
# Env overrides:
#   SOC=ascend910b   target soc (default ascend910b; 910b is our target)
#
# NOTE: this script CANNOT export the op env vars into your current shell.
# After it finishes you MUST run the printed `source .../set_env.bash` line in
# the SAME shell where you launch the op / pytest, or the kernel will fail with
# `EZ9999 AclNN_Inner_Error: The binary bin not found!`.

set -euo pipefail

SOC="${SOC:-ascend910b}"
DO_OPS=1
DO_PY=1
DO_VERIFY=0

for arg in "$@"; do
    case "$arg" in
        --ops-only) DO_PY=0 ;;
        --py-only)  DO_OPS=0 ;;
        --verify)   DO_VERIFY=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# Repo root: prefer git, fall back to the parent of csrc/ (this script lives in csrc/).
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$REPO_ROOT"

# Logs to relative tmp/ (shared NPU server: never absolute /tmp).
mkdir -p tmp
TS="$(date +%Y%m%d_%H%M%S)"
LOG="tmp/rebuild_install_${TS}.log"

banner() { echo; echo "==================== $* ===================="; }
log()    { echo "[rebuild] $*"; }

banner "config"
log "REPO_ROOT = $REPO_ROOT"
log "SOC       = $SOC"
log "layer1(ops)=$DO_OPS  layer2(py)=$DO_PY  verify=$DO_VERIFY"
log "log file  = $LOG"

# Sanity: layer 2 needs torch importable in the current env (for --no-build-isolation).
if [[ "$DO_PY" == "1" ]]; then
    if ! python3 -c "import torch" >/dev/null 2>&1; then
        echo "[rebuild] ERROR: 'import torch' failed in current env." >&2
        echo "[rebuild] --no-build-isolation needs torch already installed here. Activate the right venv." >&2
        exit 1
    fi
fi

START=$(date +%s)

# ---------------------------------------------------------------- Layer 1
if [[ "$DO_OPS" == "1" ]]; then
    banner "layer 1: CANN custom-ops package ($SOC)"
    log "cleaning csrc/build csrc/output"
    rm -rf csrc/build csrc/output
    log "running build_aclnn.sh (this is the slow part)"
    # build_aclnn.sh picks the op set for the soc (includes sparse_flash_attention)
    # and installs into vllm_ascend/_cann_ops_custom.
    bash csrc/build_aclnn.sh "$REPO_ROOT" "$SOC" 2>&1 | tee -a "$LOG"
    log "layer 1 done"
fi

# ---------------------------------------------------------------- Layer 2
if [[ "$DO_PY" == "1" ]]; then
    banner "layer 2: vllm-ascend Python ext (_C_ascend.so)"
    log "pip install -e . --no-build-isolation --no-deps --force-reinstall"
    pip install -e . --no-build-isolation --no-deps --force-reinstall -v 2>&1 | tee -a "$LOG"
    log "layer 2 done"
fi

# ---------------------------------------------------------------- Verify (optional)
if [[ "$DO_VERIFY" == "1" && "$DO_PY" == "1" ]]; then
    banner "verify: registered op schema"
    # Should show the new trailing `bool sparse_indices_discrete=False` arg.
    python3 - <<'PY' 2>&1 | tee -a "$LOG" || true
import torch, torch_npu, vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
for s in torch._C._jit_get_schemas_for_operator("_C_ascend::npu_sparse_flash_attention"):
    print(s)
PY
fi

END=$(date +%s)
banner "finished in $((END - START))s"

# Vendor dir name varies (e.g. custom_transformer); locate set_env.bash by glob.
VENDORS_DIR="$REPO_ROOT/vllm_ascend/_cann_ops_custom/vendors"
SET_ENV="$(ls "$VENDORS_DIR"/*/bin/set_env.bash 2>/dev/null | head -1 || true)"
if [[ -z "$SET_ENV" ]]; then
    SET_ENV="$VENDORS_DIR/<vendor>/bin/set_env.bash  (not found - check $VENDORS_DIR)"
fi
echo
echo ">>> NEXT STEP (run in the SAME shell you use for the op / pytest):"
echo
echo "    source $SET_ENV"
echo
echo ">>> then e.g.:"
echo "    pytest -q tests/ut/attention/a2/test_sfa_discrete_indices.py"
echo
log "full log: $LOG"
