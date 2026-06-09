#!/usr/bin/env bash
# 干净重建 vllm-ascend SparseFlashAttention 的两层产物，避免"脏环境"段错误。
#
# 背景：本算子有两层独立产物，build_aclnn.sh 只自清 csrc/build 和 csrc/output，
#       已安装的 vendors/ 是覆盖式安装、不会删旧文件。改了 def 接口后若残留旧的
#       aclnn wrapper / 旧 .so，两层不同步 → EXEC_NPU_CMD 调到旧签名 → segfault。
#
# 用法（在 vllm-ascend 仓库根目录跑，或任意目录——脚本会自动 cd 到 git 根）：
#   bash tests/op_test/sparse_flash_attention/diag/diag_clean_rebuild.sh <soc> [mode]
#     <soc>  : ascend910b / ascend910_93 等（层1构建必填）
#     mode   : all (默认) | layer1 | layer2
#              layer1 = 只重建 CANN 算子包（改了 def/proto/tiling/op_kernel 时）
#              layer2 = 只重建 Python 扩展（改了 torch_binding*/torch_adpt 时）
#              all    = 两层都重建（改了 def 接口时必须，如 step 4a）
#
# 示例：
#   bash .../diag/diag_clean_rebuild.sh ascend910b           # 全清重建
#   bash .../diag/diag_clean_rebuild.sh ascend910b layer1    # 只重建算子包
#   bash .../diag/diag_clean_rebuild.sh "" layer2            # 只重建 Python 扩展
#
# 把输出 tee 到日志方便回贴：
#   bash .../diag/diag_clean_rebuild.sh ascend910b 2>&1 | tee /tmp/diag_rebuild.log

set -u

SOC="${1:-}"
MODE="${2:-all}"

# 定位仓库根目录
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$ROOT_DIR" ]]; then
    echo "[FATAL] 不在 git 仓库内，无法定位仓库根目录" >&2
    exit 1
fi
cd "$ROOT_DIR" || exit 1
echo "=== 仓库根: $ROOT_DIR ==="
echo "=== 模式: $MODE   SOC: ${SOC:-<未提供>} ==="
echo

rebuild_layer1() {
    if [[ -z "$SOC" ]]; then
        echo "[FATAL] 重建层1（CANN 算子包）需要 <soc> 参数，例如 ascend910b" >&2
        exit 1
    fi
    echo "=== [层1] 清理 CANN 算子包产物 ==="
    echo "    删除: csrc/build  csrc/output  vllm_ascend/_cann_ops_custom/vendors"
    rm -rf csrc/build csrc/output vllm_ascend/_cann_ops_custom/vendors
    echo

    echo "=== [层1] 构建 + 安装 CANN 算子包（soc=$SOC） ==="
    bash csrc/build_aclnn.sh "$ROOT_DIR" "$SOC"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[FATAL] build_aclnn.sh 失败（rc=$rc）。常见原因：def 改了输出但 kernel 入口签名" >&2
        echo "        没同步 → binary gen 报 template mismatch / no matching function。" >&2
        exit $rc
    fi
    echo "=== [层1] 完成 ==="
    echo "    已安装包内容:"
    ls -d vllm_ascend/_cann_ops_custom/vendors/* 2>/dev/null
    echo
}

rebuild_layer2() {
    echo "=== [层2] 清理 Python 扩展产物 ==="
    echo "    删除: build  vllm_ascend.egg-info  _C_ascend*.so  __pycache__"
    rm -rf build vllm_ascend.egg-info
    find . -name "_C_ascend*.so" -delete 2>/dev/null
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    echo

    echo "=== [层2] 安装 Python 扩展（--no-build-isolation 避免重装 torch 卡死） ==="
    pip install -e . --no-build-isolation --no-deps --force-reinstall
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[FATAL] pip install 失败（rc=$rc）" >&2
        exit $rc
    fi
    echo "=== [层2] 完成 ==="
    find . -name "_C_ascend*.so" 2>/dev/null
    echo
}

case "$MODE" in
    all)
        rebuild_layer1
        rebuild_layer2
        ;;
    layer1)
        rebuild_layer1
        ;;
    layer2)
        rebuild_layer2
        ;;
    *)
        echo "[FATAL] 未知 mode: $MODE （应为 all | layer1 | layer2）" >&2
        exit 1
        ;;
esac

echo "=== 验证：算子是否重新注册 ==="
python -c "import torch, vllm_ascend; print('npu_sparse_flash_attention registered:', hasattr(torch.ops._C_ascend, 'npu_sparse_flash_attention'))"
echo
echo "=== 全部完成。建议接着跑回归： ==="
echo "    cd tests/op_test/sparse_flash_attention && bash test_run.sh single"
