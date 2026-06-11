# 交接文档：修复 IS_DENSE kernel 连续 dense 路径（延后任务）

> 独立任务,后面单开一个对话查。本文档不提交(仓库文档不进 commit)。
> 读完即可冷启动。相关讨论发生在 step 4a 期间(2026-06-09)。

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

## 当前状态(很重要)

- 主分支 `feature/sfa-extend`:**dummy 已恢复**(去 dummy 的实验 commit `898c895b` 已被 revert)。所以现在 dense 正确(走 sparse 路径)、稳定,但慢。
- 复现 IS_DENSE bug 的方法:**重新应用 `898c895b` 的改动**(让 torch_adpt 把 `sparse_indices` 直接按 OPTIONAL 透传 null,而不是造 arange)。`git show 898c895b` 看 diff。

## 症状(已观测)

去掉 dummy 后(走真 IS_DENSE 路径):
- **不再是 "binary not found"**(那是漏 source env,已解决)。dense kernel 能跑。
- 但 `correctness/test_dense.py` 失败:
  - BSND/BSND:`max_abs_diff(dense, sparse-full) = 4.63e-2`
  - BSND/PA_BSND:`3.96e-2`
  - tolerance 是 `rtol=1e-3, atol=1e-3`,差了 ~46 倍。
- 输出是**完全不同的一组数**(不是精度噪声),说明 dense 路径在 token 遍历/取数/累加 某处算错。
- sparse 路径(baseline `test_run.sh single` 5 用例)全过,不受影响。

## 根因区域:4 处 `if constexpr (SFAT::isDense)` 分支

step 3c 写的 dense 快路,跟 sparse 完全不同的取数逻辑:

| 文件:行 | 函数 | dense 干了什么 |
|---|---|---|
| `op_kernel/sparse_flash_attention_kernel_mla.h:948`(函数定义 941) | `CalcSinnerTopKBegin` | `curTopKIdx` 重定义为"已处理 token 数";按 `[startPos, threshold)` 切 `s2BaseSize` 大小的连续段;不读 topKGm |
| `op_kernel/sparse_flash_attention_service_cube_mla.h:513` | `CalcTopKBlockInfo` | `copyRowCnt = threshold - idInTopK`(一把取剩余连续块);`curOffsetInSparseBlock=0` |
| `op_kernel/sparse_flash_attention_service_cube_mla.h:586` | `ComputeMm1` | `idInTopK = curTopKIdx`(连续位置)而非 `topKGm.GetValue(...)` |
| `op_kernel/sparse_flash_attention_service_cube_mla.h:872` | mm2/value 取数 | dense 分支(同理用连续位置取 value) |

对照基准:sparse 分支(同文件的 else 支)逐 block 读 topk、算 `keyOffset = (idInTopK*sparseBlockSize + curOffsetInSparseBlock) * kvHeadNum * headDim`。dense 下 `sparseBlockSize=1`、`curOffset=0`,理论上 `keyOffset = idInTopK * kvHeadNum * headDim`,连续。需要核对的就是这套连续遍历是否和 sparse 全选**逐元素等价**。

## 怀疑点 / 调查方向(下次从这里入手)

1. **cube tiling 边界**:`CalcTopKBlockInfo` dense 把 `copyRowCnt` 设成全部剩余,下游靠 `copyFinishRowCnt + copyRowCnt > nL1Size` 截断;核对跨 `nL1`(N_SPLIT_SIZE=128)/ `kL0` chunk 时 `idInTopK` 的推进是否正确,有没有重复/漏读 token。
2. **s2 inner-loop 切分**:`CalcSinnerTopKBegin` dense 每轮取 `min(threshold-startPos, s2BaseSize)`;核对多轮 inner-loop 拼接是否覆盖完整 [0, threshold) 无洞无叠。
3. **mm2(value)dense 分支(cube:872)** 与 mm1 的 token 对齐:QK 和 PV 用的 token 序是否一致。
4. **threshold 语义**:dense 测试用 `sparse_mode=0`(threshold=curActualSeqLenOri 全长);确认 dense 分支用的 threshold 与 sparse 全选一致。
5. 手段:在 4 个 dense 分支加 `OPS_LOG_E`(OPS_LOG_I 默认不输出)打印 `idInTopK / copyRowCnt / keyOffset / actualSingleProcessSInnerSize`,跟 sparse 全选同 case 对拍。

## 复现步骤(下次)

```bash
cd /home/zhy/kv-offload-sfa/vllm-ascend
git checkout feature/sfa-extend && git pull
# 重新应用去 dummy 的改动以复现（或在 torch_adpt.h 手动把 sparse_indices 直接透传 null）
git show 898c895b        # 看当时的 diff，照着改 torch_adpt.h
# 全量重建（改了 torch_adpt 只需 layer2；若要动 kernel 则 layer1）
bash tests/op_test/sparse_flash_attention/diag/diag_clean_rebuild.sh ascend910b
# ★ 必须 source
source vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash
# 复现
cd tests/op_test/sparse_flash_attention
pytest correctness/test_dense.py -s -v
```

## 验收标准

- `correctness/test_dense.py` 两个用例通过(dense == sparse-full,atol/rtol 1e-3)。
- baseline `test_run.sh single` 5 用例不退化。
- 通过后:删掉 torch_adpt.h 的 arange dummy,dense 走 IS_DENSE 连续快路。

## 注意事项(踩过的坑)

- **build 后必须 `source .../vendors/vllm-ascend/bin/set_env.bash`**,否则 EZ9999 "binary bin not found"(这跟 dense 对错无关,是环境变量)。
- 改 op_kernel/* → 重建 layer1;改 torch_adpt → 重建 layer2。
- 这是个**独立任务**,别和 step 4(packed_kv 输出)混在一起。建议 step 4 的 4b/4c 先做完。
