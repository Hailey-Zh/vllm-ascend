#!/usr/bin/env bash
# 排查 NnopbaseAddInput 对 sparseIndices 报 nullptr 的原因：
# 怀疑 vllm_ascend/_cann_ops_custom 里有某份 metadata（proto.h / json / yaml）
# 还把 sparse_indices 当 required，与 def.cpp 不一致。
#
# 用法：
#   bash tests/op_test/sparse_flash_attention/diag/diag_sparse_indices_metadata.sh 2>&1 | tee /tmp/diag_meta.log

set -u

ROOT="vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"

if [[ ! -d "$ROOT" ]]; then
    echo "[FATAL] $ROOT 不存在；先 bash csrc/build_aclnn.sh \$(pwd) <soc>" >&2
    exit 1
fi

echo "=== 1. 所有"sparse_indices"出现的位置 ==="
grep -rn "sparse_indices" "$ROOT" 2>/dev/null
echo

echo "=== 2. sparse_flash_attention_proto.h 全文 ==="
cat "$ROOT/op_proto/inc/sparse_flash_attention_proto.h"
echo

echo "=== 3. ascend910b JSON 中 sparse_indices 的 paramType ==="
JSON="$ROOT/op_impl/ai_core/tbe/kernel/config/ascend910b/sparse_flash_attention.json"
if [[ -f "$JSON" ]]; then
    python3 -c "
import json
with open('$JSON') as f:
    data = json.load(f)
for bin_entry in data.get('binList', []):
    for inp in bin_entry.get('inputs', []):
        print(f\"  index={inp.get('index')} name={inp.get('name'):30s} paramType={inp.get('paramType')}\")
    print('---')
    "
fi
echo

echo "=== 4. dynamic 下的 .py impl 看输入声明 ==="
DYN_PY="$ROOT/op_impl/ai_core/tbe/vllm-ascend_impl/dynamic/sparse_flash_attention.py"
if [[ -f "$DYN_PY" ]]; then
    head -80 "$DYN_PY"
fi
echo

echo "=== 5. 找 .ini / .yaml / .toml / .cfg 这类 OP 注册文件 ==="
find "$ROOT" -type f \( -name "*.ini" -o -name "*.yaml" -o -name "*.toml" -o -name "*.cfg" -o -name "*.xml" \) | head -20
echo

echo "=== 6. 对比 block_table 在所有这些文件里的声明（对照组——它能正常传 nullptr） ==="
grep -rn "block_table" "$ROOT" 2>/dev/null | head -30
echo

echo "=== 7. 在所有 .so 里搜"sparse_indices"和"block_table"出现的次数，看差异 ==="
for so in $(find "$ROOT" -name "*.so" 2>/dev/null); do
    sparse_cnt=$(strings "$so" 2>/dev/null | grep -c "sparse_indices")
    block_cnt=$(strings "$so" 2>/dev/null | grep -c "block_table")
    echo "  $(basename "$so"):  sparse_indices=$sparse_cnt   block_table=$block_cnt"
done
echo

echo "=== 8. liboptiling.so 里 sparse_indices 的所有上下文（不止 SFATilingCheck，看有没有框架级的 required 标记） ==="
strings "$ROOT/op_impl/ai_core/tbe/op_tiling/liboptiling.so" 2>/dev/null | grep -E "sparse_indices|sparseIndices" | sort -u
echo

echo "[done]"
