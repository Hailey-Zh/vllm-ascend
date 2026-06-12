# 交接文档：修复 IS_DENSE kernel 连续 dense 路径（延后任务）

> 独立任务,后面单开一个对话查。本文档不提交(仓库文档不进 commit)。
> 读完即可冷启动。相关讨论发生在 step 4a 期间(2026-06-09)。
> **2026-06-11 更新**：根因已定位，不是 IS_DENSE 分支的问题，见下方"真实根因"。

## 一句话目标

让 **IS_DENSE=1 的 kernel 连续 dense 路径算对**(输出 == sparse-full),从而:
1. 去掉 torch_adpt.h 里的 arange dummy;
2. dense 真正走"连续整块读 + 并行",而不是离散 gather。

## 为什么要做(动机)

dense 的本心 = **去除离散访存,连续整块读取并行加速**。

- 当前 dense 走的是 **dummy 方案**:torch_adpt 给 `sparse_indices=None` 时塞一个 full-coverage arange → `isDenseMode=false` → kernel 走 **sparse 分支**,逐 block 读 `topKGm`、按索引离散 gather。**正确但慢**,把 dense 的性能优势全丢了。
- IS_DENSE 分支才是为连续读写的快路,但**有 bug,算错**。

**对最终蓝图很关键**:decode KV 复用里,stage-1 SFA(sparse)吐出连续 packed KV,stage-2 SFA 在其上做 dense。stage-2 要的就是 IS_DENSE 连续快路;若走 dummy(离散 gather),等于白压缩。所以 IS_DENSE 正确+高效是这条 pipeline 的真需求。

## 量化收益(benchmark 实测,2026-06-11)

`benchmark/bench_sfa.py` 加了一组**等算量对照**(dense 与 sparse 都算 2048 个 KV token、同 layout/S2、prefill B1 S1=512),实测当前 dummy 方案的额外开销:

| 用例 | 路径 | p50 (ms) |
|---|---|---|
| `eqc_dense_prefill_2048` | dummy(arange→sparse kernel) | **1.739** |
| `eqc_sparse_prefill_2048` | sparse(K=2048 全选) | **1.540** |

**等算量下 dense 反而比 sparse 慢 ~13%**。两点结论:

1. **离散 gather 本身不是瓶颈**——稀疏随机索引照样比 dummy-dense 快,说明慢的不是访存模式。
2. **慢的真正来源是 dummy 的每调用开销**:`torch_adpt.h` 给 `sparse_indices=None` 时每次都现场
   `at::arange(K).view(...).expand(...).contiguous()` 造一个 `[B,S1,N2,K]` int32 索引张量
   (此例 `[1,512,1,2048]` = **4MB/call**,S2=4096 时 8MB/call),这步 host 分配+拷贝算进了时延。

→ 即使先不谈"连续整块读"的算力收益,**单是去掉 arange dummy 就能省掉这 4~8MB/call 的实例化开销**。
这是做 IS_DENSE 快路的量化依据:不仅更快的访存,还省掉每调用的索引材料化。real dense 快路落地后,
预期至少回收这 ~13%,加上连续读的并行收益应更多。

---

## ★ 真实根因（2026-06-11 定位）

### 原始诊断是错的

文档最初列出的 4 处 `if constexpr(SFAT::isDense)` 分支 **没有问题**。
C_TEMPLATE（dense 走的路）已经正确——它在 `key≠value` 时对 CPU 解析 golden 的误差：
- `max_abs_diff(C_TEMPLATE bs=8, CPU golden) = 1.73e-5` ← 完全正确

测试文件及结论：
- `correctness/test_ctemplate_isolation.py::test_ctemplate_vs_cpu_golden`
- `correctness/test_ctemplate_isolation.py::test_ctemplate_base_matches_vtemplate`

### 真正错的：V_TEMPLATE 的 mm2 抄近道读到了 K

**V_TEMPLATE**（`block_size≤4` 的**稀疏主力路径**）在 `ComputeMm2`（P×V 矩阵乘）
阶段有一个抄近道：直接从 `kvMergeGm_` workspace 连续读 B 矩阵，而不是像 C_TEMPLATE
那样从 `valueGm` 按 token 索引读。

问题在于：`kvMergeGm_` 只有 `MergeKv` 写入的 **K 和 K_rope 数据**，没有任何代码
把 VALUE gather 进去。所以 mm2 实际算的是 **P×K** 而不是 **P×V**。

相关代码位置：

| 文件 | 行号 | 内容 |
|---|---|---|
| `op_kernel/sparse_flash_attention_service_vector_mla.h` | 964 | `MergeKv` — 只 gather K/K_rope |
| 同上 | 834 | `CopyInSingleKv` — 只从 `keyGm_`/`keyRopeGm_` 读 |
| 同上 | 925 | `CopyOutMrgeResult` — 只写 K(942行)和 K_rope(952行)到 `kvMergeGm_`，没有 V |
| `op_kernel/sparse_flash_attention_service_cube_mla.h` | 899-912 | `ComputeMm2` V_TEMPLATE 抄近道 — 从 `kvMergeGm_` 读 B 矩阵 |
| 同上 | 913-955 | `ComputeMm2` else 分支 — 正确从 `valueGm` 直读（C_TEMPLATE 走这条路） |

### 为什么一直没发现（框架测试全 PASS）

框架 golden 生成时把 **value 设成了 key**（同一个 tensor）：

`tests/op_test/sparse_flash_attention/framework/sparse_flash_attention_golden.py:289`
```python
"value": raw["key"],
```

当 key==value 时，P×K == P×V，V_TEMPLATE 的 bug 完全被掩盖。

验证测试：`correctness/test_ctemplate_isolation.py::test_key_equals_value`
```
key==value 时: V vs CPU golden = 1.35e-5  ← 通过
key!=value 时: V vs CPU golden = 4.63e-2  ← 失败
```

### bug 来源

这是**原始代码的 bug**（commit `18b90b50`，Song Mingyang，2025-12-03，第一版 SFA 算子）。
我们的 step3c commit (`7a23e086`) 改的全在 C_TEMPLATE 非 V_TEMPLATE 路径，
与此 bug 无交集。

---

## 影响面

| 路径 | 条件 | key≠value 时 | key==value 时 |
|---|---|---|---|
| C_TEMPLATE（含 IS_DENSE） | `block_size>4` 或 dense | ✓ 正确 | ✓ 正确 |
| V_TEMPLATE（主力稀疏） | `block_size≤4` | ✗ P×K 代替 P×V | ✓ K=V 掩盖 |

V_TEMPLATE 是所有 5 个 single 用例和 dense-via-dummy 的实际路径。对于 DeepSeek MLA
使用场景，压缩 KV cache 本身就是同一个 latent（key==value），所以 bug 不触发。
但如果有人传独立的 key/value（如 test_dense.py），V_TEMPLATE 就算错。

---

## 修复方向

两种思路：

### 方案 A：让 V_TEMPLATE 的 mm2 走 C_TEMPLATE 的 valueGm 直读路径
- 改 `cube_mla.h:899`：去掉 `if constexpr (TEMPLATE_MODE == V_TEMPLATE)` 抄近道，
  让 V_TEMPLATE 也走 913 行的 else 分支（`while(copyFinishRowCnt < kL0Size)` +
  `CalcTopKBlockInfo` + `CopyInMm2BToL1`）。
- 优点：改动小，逻辑复用 C_TEMPLATE 已验证的 value 直读。
- 注意：V_TEMPLATE 的 `topKGm`/`sparseBlockCount` 等模板参数在 else 分支已经可用，
  `CalcTopKBlockInfo` 非 isDense 分支和 C_TEMPLATE 共享。

### 方案 B：加 value merge 步骤
- 在 Queue2 前加一次 MergeKv 风格的 value gather，写入 `kvMergeGm_` 的 V 区域。
- 优点：保持 V_TEMPLATE 连续读的性能优势。
- 注意：需要调整 workspace 布局、同步流水，改动大。

推荐先做方案 A（风险低、验证快），性能如有退化再做方案 B。

---

## 附带发现：C_TEMPLATE 已经可以落地 STEP3

既然 C_TEMPLATE/dense 路径已被证明正确（vs CPU golden 1.73e-5），**去掉 arange dummy
让 dense 走 IS_DENSE 这条目标可以先完成**，不依赖 V_TEMPLATE bug 修复。

做法：
1. 把 `test_dense.py` 的 baseline 从 `sparse-full (V_TEMPLATE)` 改为 CPU golden 或 C_TEMPLATE(bs=8)
2. 重新应用 `898c895b` 去掉 torch_adpt 的 arange dummy
3. 验收：`test_dense.py` 对新的 baseline 通过

这样 dense 快路收益先拿到，V_TEMPLATE bug 单独修。

---

## 复现 / 验证命令（无需 rebuild，dummy 无关）

C_TEMPLATE 验证：
```bash
cd tests/op_test/sparse_flash_attention
pytest correctness/test_ctemplate_isolation.py::test_ctemplate_vs_cpu_golden -s -v
pytest correctness/test_ctemplate_isolation.py::test_ctemplate_mm2_constant_value -s -v
```

V_TEMPLATE bug 验证：
```bash
pytest correctness/test_ctemplate_isolation.py::test_key_equals_value -s -v
pytest correctness/test_ctemplate_isolation.py::test_vtemplate_trigger_scan -s -v
pytest correctness/test_v_trigger.py -s -v
```

V_TEMPLATE 是否使用 sparse_indices 判定：
```bash
pytest correctness/test_ctemplate_isolation.py::test_vtemplate_ignores_indices -s -v
pytest correctness/test_ctemplate_isolation.py::test_vtemplate_token_id_probe -s -v
```

原始 test_dense（需要先去掉 dummy）：
```bash
cd /home/zhy/kv-offload-sfa/vllm-ascend
git show 898c895b          # 看 diff，照着改 torch_adpt.h 去掉 arange dummy
bash tests/op_test/sparse_flash_attention/diag/diag_clean_rebuild.sh ascend910b
source vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash
cd tests/op_test/sparse_flash_attention
pytest correctness/test_dense.py -s -v
```

---

## 注意事项(踩过的坑)

- **build 后必须 `source .../vendors/vllm-ascend/bin/set_env.bash`**,否则 EZ9999 "binary bin not found"。
- 改 op_kernel/* → 重建 layer1;改 torch_adpt → 重建 layer2。
- V_TEMPLATE 的 `sparse_indices` 取数逻辑（`GetRealS2Idx`/`CopyInKv`）本身是正确的，问题只在 mm2 读值源。
- 框架 `check_result` 的对比（`np.isclose atol=2.5e-5`）对 out≈0 的 case 可能虚假通过，golden 输出量级很小时需注意。
