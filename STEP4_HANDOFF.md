# Step 4 交接文档：packed_key / packed_key_rope 输出

> 写给下一个 Claude Code 会话的 context。读完本文 + `sfa_modify_plan.md` step 4 节即可开始。
>
> **本文档已按 2026-06-09 的设计讨论定稿**，与最初草案（packed_key 576 + packed_value）有重大出入，请以本版为准。关键变化：①输出改为两个独立 tensor packed_key(512)+packed_key_rope(64)，砍掉 packed_value；②实现走 Strategy B（加法 copy-out，cube 零改动）；③明确职责边界——算子只吐连续 KV，复用逻辑全在框架层。

## 进展（2026-06-10）

- **4a（plumbing）✅ 通过**：return_packed_kv=false baseline 0 退化。
- **4b（kernel 写出）✅ 通过**：packed_key/packed_key_rope vs PyTorch gather **bit-exact (max_abs_diff 0)**，BSND；baseline 5/5 不退化。
- **4c（生产路径）✅ 通过**：TND + PA_BSND（B=2 跨 batch，cumsum q / raw kv / block_table 间址）用 marker 编码验证 **bit-exact (marker_diff 0, bad_slots 0)**。TND 前缀和段基址 + PA gather + 保序 全部正确。
- **Step 4 完工。** 测试文件 `test_sparse_flash_attention_packed_kv.py`（BSND gather + TND/PA_BSND marker，marker step4_packed_kv）。
- **最终输出收敛**：`packed_key(512) + packed_key_rope(64)`，**actual_packed_len 砍掉**（框架用 sparse_indices+causal 自算，下游按 len 切片读 `[0,len)`）。维度顺序 **option A `[T1/B, (S1,) N2, S2, dim]`**。
- **踩的两个大坑**：
  1. **build 后必须 `source .../vendors/vllm-ascend/bin/set_env.bash`**，否则 EZ9999 "binary bin not found"（跟 optional/dense 无关，纯环境变量）。所有"binary not found"先查这个。
  2. **MergeKv 不保序**：strided 双块 gather 按内存升序读，打乱 sparse_indices 顺序（注意力无所谓，packed 要保序）。修法：returnPackedKv 时强制走 `CopyInSingleKv` 逐 token 路径（CopyInKv 条件加 `constInfo.returnPackedKv ||`）。
- **OPTIONAL vs REQUIRED 输出**：实测 OPTIONAL 输出可用(之前"不行"是没 source env 的无效测试)，packed 用 OPTIONAL，false 时传 nullopt 不分配。
- **待办 4c**：目前测试只覆盖 **BSND**。生产路径是 **TND + PA_BSND**，kernel 的 offset 有 TND 分支、PA 走 MergeKv 现成 PA 路径，但 **packed 在 TND/PA_BSND 下未经测试**，需补测例验证。

---

## Step 4 要做什么（一句话）

稀疏 FA 当前输出 attention_out（和 LSE）。需求：**额外吐出当前 token 选中的连续 KV**——把稀疏选中的 K / RoPE-K 按 sparse_indices 顺序拼接成连续内存输出，配 `actual_packed_len` 标记每段有效长度。

**核心策略**：复用 V_TEMPLATE 既有的 `MergeKv` 机制（将稀疏 K/RoPE-K gather 后拼接写入 workspace），在它写 workspace 的同时**额外多写一份到 device 持久输出**（Strategy B）。

---

## 应用场景（决定了所有设计取舍，务必先理解）

用途是 **decode 阶段跨 step 的 KV 复用**：

```
 step t:                          step t+1:
 ┌──────────────┐                 indexer 选出 {B,C,E}
 │ SFA (sparse, │                 framework 拿 t 的 packed + sparse_indices[t]:
 │  V_TEMPLATE) │                   ├ 建 id→槽 映射(用 sparse_indices[t])
 └──────┬───────┘                   ├ B,C 命中 → device 本地拷贝(快)
        │ packed_key(512,device)    ├ E 未命中 → host 搬运(慢,但只搬这一个)
        │ packed_key_rope(64)       └ 压实成 {B,C,E} 连续 buffer  ← 路A
        │ actual_packed_len                  │
        ▼                                     ▼
   framework 缓存(device)            SFA (dense, C_TEMPLATE) 跑满长度
                                     key=value=压实buffer[:512]
                                     key_rope=压实buffer 的 rope 那份
```

理论依据：decode 相邻 token 的 top-k 选择高度重叠（时间局部性），且**同层内某 id 的 KV 值固定** → 命中部分复用是**逐元素精确、无损**。收益来源：命中走 device 本地、未命中才走 host，砍掉 decode 的 host 往返带宽（这是 decode 的真瓶颈）。

**两个已拍板的前提**：
1. **路 A（压实）**：framework 把命中块拷到新紧凑 buffer + host 补未命中，拼成连续后再喂给下游 dense。**不需要给算子加散落 mask**（路 B 才需要，已否决）。
2. **内存分层 = device(packed) vs host(全量 KV cache)**：这是收益成立的前提。

### 职责边界（关键）

**算子（第 4 步）只负责：吐出当前 token 选中的连续 KV + 有效长度。**
复用、id 映射、命中判定、压实、host 补齐——**全部在框架层**，不进算子。第 4 步范围因此很干净，别把框架逻辑塞进 kernel。

---

## 输出 schema（定稿）

| 输出 | shape (BSND) | dtype | 说明 |
|---|---|---|---|
| `packed_key` | `[B, S1, N2, S2, 512]` | 同 query | c_KV（NoPE 部分）；下游同时当 key 和 value |
| `packed_key_rope` | `[B, S1, N2, S2, 64]` | 同 query | RoPE-K 部分 |
| `actual_packed_len` | `[B, S1, N2]` | int32 | 每个 (b,s1,n2) 段有效 token 数 |

`S2 = sparse_block_count * sparse_block_size`。TND 同理，前缀 `B,S1` 换成 `T1`。
decode 时 S1=1，自然退化为 `[B, N2, S2, ...]`，无 per-s1 shape 问题。

外加 OPTIONAL attr：`return_packed_kv`（bool，默认 false）。

### 为什么是这个 schema（本次讨论的核心结论）

**① 为什么没有 packed_value？** 这是 MLA：K 和 V 共享同一份压缩 latent `c_KV`。
- 576 维 = `c_KV(512, NoPE)` + `k_rope(64, RoPE)`。
- value == c_KV == packed_key（数学上同一份）。代码佐证：V_TEMPLATE 的 mm2(P·V) 读 merged KV 的 NoPE 部分（`cube_mla.h` L910），**根本不读 `valueGm` 输入**（valueGm 只在 C_TEMPLATE L937 用）。
- 下游 dense SFA 需要独立的 key/value/key_rope，但 value 数据 == c_KV，所以**把 packed_key 同一个 tensor 同时当 key 和 value 两个输入参数传**即可，不必物理复制。
- 结论：packed_value 冗余，砍掉，省 ~47% 额外写。

```
packed_key (c_KV, 512)  ──┬──→ 下游 key
                          └──→ 下游 value   (同一 tensor 传两次)
packed_key_rope (64)    ─────→ 下游 key_rope
```

**② 为什么是两个独立 tensor，不是合并 576？** 下游 dense（C_TEMPLATE）的 key/value/key_rope 是**三个独立、各自连续**的输入（`tensorBOffset` 见 `kernel_mla.h` L672，KV 输入 layout = `[B,S2,N2,512]`，key 和 key_rope 分开读）。
- 若输出一个 row-major `[S2,576]`，`[:,:512]` 和 `[:,512:]` 都是**非连续切片**（行距 576≠512），带 `.AutoContiguous()` 的下游会触发自动拷贝 → 又多一次搬运。
- 拆成两个独立 tensor，各自连续，下游零 reshape。这是"分块布局"推到极致的结果。

```
✅ 两独立 tensor                    ❌ 合并 576 row-major
packed_key      [.., S2, 512] 连续   packed_key [.., S2, 576]
packed_key_rope [.., S2, 64]  连续    下游切 [:,:512]/[:,512:] → 非连续 → 自动拷贝
```

**③ 为什么不输出选中 id？** packed buffer 第 i 槽 == `sparse_indices[i]`（kernel 按 sparse_indices 顺序 gather，截断到 actual_packed_len）。调用方手里有 sparse_indices，**槽↔id 映射自己能重建**，算子不必再吐。

---

## 实现策略：Strategy B（加法 copy-out，cube 零改动）

讨论中对比过两个方案，**选 B**：

```
现状（无 packed 输出时）:
  稀疏K/V ─MergeKv─UB→GM写─▶ workspace环形4槽 ─GM→UB读─▶ cube matmul ─▶ attention_out

Strategy A（redirect，已否决）:
  把 kvMergeGm_ 直接指向用户输出，cube 改成从用户输出读
  → 额外带宽 0，但要改 10 处含 cube 5 处读 + 布局被绑死 → 回归风险中

Strategy B（采用）:
                  ┌─UB→GM写(原样)─▶ workspace环形4槽 ─▶ cube matmul ─▶ attention_out
  稀疏K/V ─MergeKv─┤                                      ↑ cube/流水/布局 全不动
                  └─UB→GM写(★新增)─▶ device持久输出       
                     同一份UB写第二个目的地  packed_key / packed_key_rope (只读不参与计算)
```

| | Strategy A | **Strategy B（采用）** |
|---|---|---|
| 额外写 | 0 | +1 次写（512+64/token） |
| 改 cube? | 要（5 处读） | **不要** |
| 产线 sparse 回归风险 | 中 | **接近 0** |
| packed layout 自由度 | 被 cube 绑死 | **独立** |

**性能澄清**：Strategy B 那次额外写**正是优化本身**——第 1 个 SFA 多写一次连续 KV，换来第 2 个 SFA 跳过大量离散 gather + host 往返。成本被下游收益摊销，不是纯开销。且 `return_packed_kv=false` 时整条分支不进，**产线零损耗**。

---

## 关键约束 / guard

```cpp
// tiling: return_packed_kv=true 时强制 V_TEMPLATE（只有它有 MergeKv）
if (sfaInfo_->returnPackedKv) {
    perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
}
// 但 V_TEMPLATE 的 MergeKv 只在 sparseBlockSize<=4 安全（见下"块大小依赖"），
// 且 dense 不产 packed → 加互斥校验：
if (sfaInfo_->returnPackedKv && (sfaInfo_->sparseBlockSize > 4 || sfaInfo_->isDenseMode)) {
    OPS_LOG_E(..., "return_packed_kv only supports sparse mode with sparseBlockSize<=4");
    return GRAPH_FAILED;
}
```

### V_TEMPLATE vs C_TEMPLATE 路由（已确认，tiling.cpp `InitParams` L259-269）

```cpp
if (s2Size != 0 && sparseBlockSize <= 4) perfMode_ = V_TEMPLATE_MODE;  // 细粒度稀疏，有 MergeKv
else                                      perfMode_ = C_TEMPLATE_MODE;  // 粗粒度/PA
if (isDenseMode)                          perfMode_ = C_TEMPLATE_MODE;  // step3b 强制
```

### MergeKv 对块大小的硬依赖（已确认，vector_mla.h）

MergeKv 写死了三处小块假设，所以 `return_packed_kv` 只在 `sparseBlockSize<=4` 安全：
1. **UB ping-pong 缓冲固定 32 token**（L839/899 `32*512`，L843/905 `32*64`）
2. **flush 阈值写死 32**（L979 `mte2Size-mte3Size + 2*sparseBlockSize > 32`），要求 `2*sparseBlockSize<=32`
3. **输出 slot 固定 512×576**（隐含总 merge 长度 `sparseBlockCount*sparseBlockSize<=512`）

---

## 改动逐文件（比原草案少：无 packed_value、cube_mla.h 零改）

| 文件 | 改动 | 难度 |
|---|---|---|
| `def.cpp` | 新增 2 REQUIRED 输出 packed_key/packed_key_rope（同 query dtype）+ 1 OPTIONAL 输出 actual_packed_len(int32) + 1 OPTIONAL attr return_packed_kv(bool) | 低 |
| `proto.cpp` | InferShape：BSND→ packed_key `[B,S1,N2,S2,512]`、packed_key_rope `[...,64]`、actual_packed_len `[B,S1,N2]`；TND 换前缀。**永远给真实 shape，不要 `[0]`** | 低 |
| `tiling.h` | `SFATilingInfo` 加 `bool returnPackedKv`；baseParams 加 `uint32_t returnPackedKv` | 低 |
| `tiling.cpp` | GenerateInfo 读 attr；InitParams 强制 V_TEMPLATE + guard；FillTilingBaseParamsMla 写入 | 低 |
| `sparse_flash_attention.cpp` | kernel `__global__` 入口加 3 个 GM 指针 + SFA_OP_IMPL 透传 | 低 |
| `kernel_mla.h` | 加 3 个成员指针；SetGlobalBuffer 指向用户输出；按 (b,s1,n2) 段 + 段内 s2 偏移算 copy-out offset；维护 actual_packed_len | **中** |
| `vector_mla.h` | MergeKv 写完 workspace 后，从同一 UB 额外 copy-out 到 packedKeyGm_/packedKeyRopeGm_；尾部清零保留；段末写 actual_packed_len | **中** |
| `torch_adpt.h` | 新增 2 output + 1 output(len) + 1 attr；dense + return_packed_kv 报错 | 低 |
| `torch_binding.cpp` / `torch_binding_meta.cpp` | 照搬 step 2 LSE 三输出模式 | 低 |
| 测试（新建） | 仿 dense 测例，PyTorch gather 做参考，marker `step4_packed_kv` | 低-中 |
| `cube_mla.h` | **零改动** ✅ | — |

### copy-out offset（Strategy B 核心）

ring 索引 `loop`(=gloop) 是 **per-s2-内层循环**自增（kernel_mla.h L823），不是 per-(b,s1,n2)。所以 copy-out 到用户输出的 offset 不是简单段基址，而是：

```
段基址(b,s1,n2)  +  s2LoopIdx * s2BaseSize * 维度
```

每个 (b,s1,n2) 段跑 s2LoopTimes 次 MergeKv，分别落到段内不同 s2 偏移。**只改 vector 写侧，不碰 cube 读**（cube 仍读 workspace ring）。

---

## 架构要点（新增会话必须知道）

### 两层 build artifact

| 层 | 安装物 | 重建命令 | 何时需要 |
|---|---|---|---|
| CANN 算子包 | `_cann_ops_custom/vendors/` | `rm -rf csrc/build csrc/output vllm_ascend/_cann_ops_custom/vendors && bash csrc/build_aclnn.sh $(pwd) <soc>` | 改 def.cpp / tiling.{h,cpp} / proto.cpp / op_kernel/* |
| Python 扩展 | `_C_ascend.so` | `pip install -e . --no-build-isolation --no-deps --force-reinstall` | 改 torch_binding.cpp / torch_binding_meta.cpp / *_torch_adpt.h |

**`--no-build-isolation` 是关键**：不加 pip 会开临时 venv 重装 torch，国内/ARM 卡死或极慢。

### 测试框架约定

- 新测例必加 `enable_custom_op()`（top of module，import torch_npu / vllm_ascend 之后）
- `sparse_indices range` 必须 `[0, max_blocks-1]`（kernel 只用 -1 当终止符）
- NPU `at::empty` 不归零 —— padding 区靠 kernel `InitOutputSingleCore` 清零
- `EXEC_NPU_CMD` 不能传 nullptr / 空 tensor —— 会 segfault

### def.cpp 改输出 → kernel 入口必须同步

step 2 最大的坑。**一旦 def.cpp 新增输出**，kernel 入口（`sparse_flash_attention.cpp` 的 `__global__` 函数）必须同步加 GM 指针，否则 CANN binary gen 报 template mismatch。不存在"先只改 def 不改 kernel"。

---

## 工作流建议：拆 3 个子步骤

| 子步骤 | 内容 | 可独立验证 |
|---|---|---|
| **4a** | def + proto + tiling plumbing + kernel 入口签名 + torch 绑定 | 跑 sparse 回归（return_packed_kv=false），0 退化 |
| **4b** | MergeKv copy-out（packed_key + packed_key_rope）+ offset + actual_packed_len | return_packed_kv=true BSND sparse case 对照 PyTorch gather |
| **4c** | 测试完善 + PA 路径验证 | 完整覆盖 |

每步一个 commit（用户在远端 NPU server 跑 build/test；每次改动 commit + push，不等验证）。

### 每步后跑

```bash
# 回归
cd tests/op_test/sparse_flash_attention/framework && bash test_run.sh single
# 该步新测例
pytest test_sparse_flash_attention_packed_kv.py -m step4_packed_kv -s -v
```

---

## 已知的坑（来自 step 2/3）

1. **def 改输出 → kernel 签名必须同步**，否则 binary gen mismatch
2. **InferShape 永远给真实 shape**，不要 `[0]` —— EXEC_NPU_CMD 不接受 null 输出
3. **`at::empty` 在 NPU 上是脏值**，kernel 侧 `InitOutputSingleCore` 要清零尾部
4. **两层 artifact 独立重装**，改什么装什么
5. **sparse_indices range** 必须 `[0, max_blocks-1]`
6. **OPS_LOG_I 默认不输出**，调试用 `OPS_LOG_E`
7. **Workaround 在正确层做** —— 一步改动触发 3+ 处 defensive 就停下换思路
8. **测试脚本必加 `enable_custom_op()`**

---

## 待落地时确认的细节

1. **"同一 at::Tensor 同时当 key 和 value 两个输入"**：aclnn 对只读 aliased 输入通常 OK，但需下游首次集成实测确认。退路：物理输出一份 packed_value（多 512 写），不影响整体设计。
2. **packed_key 内容布局**：每个 (b,s1,n2) 段内按 sparse_indices 顺序、token-major `[S2,512]` / `[S2,64]`，与下游 dense 的 KV 输入 layout `[.., S2, N2, dim]` 对齐（注意 N2 维位置，BSND）。

---

## 参考

- ops-transformer arch22 MergeKv：`/Users/hongyizhao/projects/MY-SFA/ops-transformer/attention/sparse_flash_attention/op_kernel/arch22/`
- sfa_modify_plan.md：本仓库根目录（step 4 节同步定稿）
- 关键源码索引见 sfa_modify_plan.md L158-167
- memory：`~/.claude/projects/-Users-hongyizhao-projects-MY-SFA-vllm-ascend/memory/MEMORY.md`
