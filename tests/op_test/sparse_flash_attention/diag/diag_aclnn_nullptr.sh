#!/usr/bin/env bash
# 排查 aclnnSparseFlashAttention 里 "Check sparseIndices != nullptr failed" 到底在哪
# 用法（在 vllm-ascend 仓库根目录跑）：
#   bash tests/op_test/sparse_flash_attention/diag/diag_aclnn_nullptr.sh 2>&1 | tee /tmp/diag_aclnn.log
# 然后把 /tmp/diag_aclnn.log 内容粘回对话。

set -u

LIB="vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so"

if [[ ! -f "$LIB" ]]; then
    echo "[FATAL] 找不到 $LIB" >&2
    echo "       请先 bash csrc/build_aclnn.sh \$(pwd) <soc> 装好 CANN 算子包" >&2
    exit 1
fi

echo "=== 0. libcust_opapi.so 基本信息 ==="
ls -la "$LIB"
echo

echo "=== 1. nullptr-检查模板字符串（找 \"Check %s != nullptr\" 或类似） ==="
strings "$LIB" | grep -E "nullptr|NULL|nullopt" | head -30
echo

echo "=== 2. sparse/block/seq/rope 几个 Optional 输入的上下文对比 ==="
echo "    （strings 抓 sparseIndicesOptional vs blockTableOptional 等，看注册时第三个参数 bool 值是否不同） "
strings "$LIB" | grep -E "(sparseIndicesOptional|blockTableOptional|actualSeqLengthsQueryOptional|actualSeqLengthsKvOptional|queryRopeOptional|keyRopeOptional)" | sort -u
echo

echo "=== 3. 含 aclnnSparseFlashAttention 的导出符号 ==="
nm -D --defined-only "$LIB" | grep -i sparseflash
echo

echo "=== 4. 反汇编 aclnnSparseFlashAttentionGetWorkspaceSize 函数体前 200 行 ==="
SYM=$(nm -D --defined-only "$LIB" | grep "aclnnSparseFlashAttentionGetWorkspaceSize" | awk '{print $3}' | head -1)
echo "    symbol = $SYM"
if [[ -n "${SYM:-}" ]]; then
    objdump -d --disassemble="$SYM" "$LIB" 2>/dev/null | head -200
else
    echo "    [WARN] 没找到 GetWorkspaceSize 符号"
fi
echo

echo "=== 5. 对照：apply_top_k_top_p_custom 的 GetWorkspaceSize（手写 wrapper，能正常处理 nullptr） ==="
SYM2=$(nm -D --defined-only "$LIB" | grep "aclnnApplyTopKTopPCustomGetWorkspaceSize" | awk '{print $3}' | head -1)
echo "    symbol = $SYM2"
if [[ -n "${SYM2:-}" ]]; then
    objdump -d --disassemble="$SYM2" "$LIB" 2>/dev/null | head -100
else
    echo "    [WARN] 没找到对照符号"
fi
echo

echo "=== 6. CANN 系统库里是否也有"Check .* != nullptr"模板（说明可能是框架级检查） ==="
for syslib in /usr/local/Ascend/cann-*/aarch64-linux/lib64/libnnopbase.so \
              /usr/local/Ascend/cann-*/aarch64-linux/lib64/libopapi.so; do
    if [[ -f "$syslib" ]]; then
        echo "  --- $syslib ---"
        strings "$syslib" 2>/dev/null | grep -E "Check.*nullptr|!= nullptr" | head -10
    fi
done
echo

echo "[done]"
