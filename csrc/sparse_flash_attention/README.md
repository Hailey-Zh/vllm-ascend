# Sparse Flash Attention (SFA) 算子使用文档

`torch.ops._C_ascend.npu_sparse_flash_attention` —— 面向 **DeepSeek MLA** 的稀疏 Flash Attention 算子（Ascend NPU）。

它在标准 Flash Attention 的基础上多做一件事：**只在你指定的那部分 KV token 上算注意力**，而不是全部。这样长序列 decode 时能省掉大量无用的访存和计算。

---

## 构建与接入

这个算子有**两层独立产物**，改不同文件要重建不同层：

| 你改了什么 | 要重建哪层 |
|---|---|
| `op_kernel/*`、`op_host/*`（def/proto/tiling/kernel） | **层1**：CANN 算子包 |
| `torch_binding*`、`*_torch_adpt.h` | **层2**：Python 扩展 `_C_ascend.so` |
| `def` 接口（影响两层签名） | **两层都要** |

用一键脚本重建（在仓库内任意目录跑都行，脚本会自动定位仓库根）：

```bash
# 全清重建两层（改了 def 接口时用）
bash tests/op_test/sparse_flash_attention/diag/diag_clean_rebuild.sh ascend910b

# 只重建算子包（改了 kernel/tiling）
bash tests/op_test/sparse_flash_attention/diag/diag_clean_rebuild.sh ascend910b layer1

# 只重建 Python 扩展（改了 torch_adpt / binding）
bash tests/op_test/sparse_flash_attention/diag/diag_clean_rebuild.sh "" layer2
```

参数：第 1 个是 SoC（`ascend910b` / `ascend910_93`，层1 必填），第 2 个是模式（`all`默认 / `layer1` / `layer2`）。

> ⚠️ **重建算子包后，必须在你自己的 shell 里 source 一次环境变量**，否则运行算子会报 `EZ9999 ... binary bin not found`：
> ```bash
> source vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash
> ```
> 脚本内部的 `source` 只对脚本自身生效、退出后不保留，所以它会在结尾打印这条命令提醒你手动再跑一次。脚本本身会自动校验算子产物是否真的进了 `vendors/`、并验证算子是否注册成功。

接入 Python 侧只要：`import vllm_ascend` 注册算子，再 `enable_custom_op()` 真正加载，之后即可调用 `torch.ops._C_ascend.npu_sparse_flash_attention(...)`。

## 一分钟上手

```python
import torch, torch_npu
import vllm_ascend                       # 注册 torch.ops._C_ascend
from vllm_ascend.utils import enable_custom_op
enable_custom_op()

B, S1, S2, N1, N2, D, ROPE = 1, 1, 4096, 16, 1, 512, 64
dev, dt = "npu", torch.float16
scale = 1.0 / (576 ** 0.5)             # 注意是 576 = 512 + 64，不是 512

query      = torch.randn(B, S1, N1, D,    dtype=dt, device=dev) * 0.1
key        = torch.randn(B, S2, N2, D,    dtype=dt, device=dev) * 0.1
value      = key                         # MLA 里 value 就是 key 的 latent（见“注意事项”）
query_rope = torch.randn(B, S1, N1, ROPE, dtype=dt, device=dev) * 0.1
key_rope   = torch.randn(B, S2, N2, ROPE, dtype=dt, device=dev) * 0.1

# 每个 query 选 2048 个 KV token（这里简单地选前 2048 个 block）
K = 2048
sparse_indices = torch.arange(K, dtype=torch.int32, device=dev).view(1,1,1,K).expand(B,S1,N2,K).contiguous()

out, lse_max, lse_sum, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
    query=query, key=key, value=value,
    sparse_indices=sparse_indices,       # None 则是 dense（全选）
    scale_value=scale,
    sparse_block_size=1,                 # 每个 index 代表 1 个 token
    actual_seq_lengths_query=torch.tensor([S1], dtype=torch.int32, device=dev),
    actual_seq_lengths_kv=torch.tensor([S2], dtype=torch.int32, device=dev),
    query_rope=query_rope, key_rope=key_rope,
    layout_query="BSND", layout_kv="BSND",
    sparse_mode=0,
)
# out: [B, S1, N1, D]，和 query 同形
```

---

## 这个算子算的是什么

MLA 把 KV 压缩成一个低秩 latent `c_KV`（512 维）+ 一段旋转位置编码 `k_rope`（64 维）。注意力分数用完整的 576 维：

```
score = (Q_nope · K_nope  +  Q_rope · K_rope) * scale        # 576 = 512 + 64
P     = softmax(score)
out   = P · Value                                            # Value 就是 512 维的 c_KV
```

所以这个算子是 **MLA 专用**：head 维度固定 **512(NoPE) + 64(RoPE)**，`scale` 必须用 `1/sqrt(576)`。

“稀疏”体现在 `softmax` 只覆盖 `sparse_indices` 选中的那批 KV token，而不是全部历史。

---

## 完整签名

```python
npu_sparse_flash_attention(
    query, key, value,
    sparse_indices, scale_value, sparse_block_size,   # 前 6 个是位置参数
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    return_softmax_lse=False,
    return_packed_kv=False,
) -> (attn_out, softmax_max, softmax_sum, packed_key, packed_key_rope)
```

### 参数说明

| 参数 | 是否必填 | 含义 |
|---|---|---|
| `query` | ✔ | 查询。NoPE 部分，形状见下方 layout 表，head 维 = 512 |
| `key` | ✔ | KV 的 NoPE 部分（latent `c_KV`），head 维 = 512 |
| `value` | ✔ | 值。MLA 里**应与 `key` 相同**（同一个 latent），见“注意事项” |
| `sparse_indices` | 可空 | 选哪些 KV block。`None` = dense（全选）。非空时形状见下 |
| `scale_value` | ✔ | 注意力缩放，**`1/sqrt(576)`** |
| `sparse_block_size` | ✔ | 1 个 block = 多少个**连续** KV token。dense 时传 1 即可 |
| `block_table` | PA 必填 | Paged KV 的页表，`[B, 每batch最大页数]`，`int32` |
| `actual_seq_lengths_query` | 建议填 | 每个 batch 的有效 query 长度，`int32`（语义见下） |
| `actual_seq_lengths_kv` | 建议填 | 每个 batch 的有效 KV 长度，`int32` |
| `query_rope` / `key_rope` | ✔(MLA) | RoPE 部分，head 维 = 64 |
| `layout_query` / `layout_kv` | ✔ | 排布格式：`BSND` / `TND` / `PA_BSND`（见下） |
| `sparse_mode` | 默认 3 | `0` = 看全部有效 KV；`3` = causal（每个 query 只看自己及之前） |
| `return_softmax_lse` | 默认 False | 是否返回 `softmax_max`/`softmax_sum`（flash-decode 跨核合并用） |
| `return_packed_kv` | 默认 False | 是否额外输出“gather 后的连续 KV”，仅支持 sparse 且 `block_size<=4` |

> `query`/`key`/`value` 支持 `float16` 和 `bfloat16`。

---

## 三种 layout

| layout | query 形状 | key/value 形状 | 说明 |
|---|---|---|---|
| `BSND` | `[B, S1, N1, D]` | `[B, S2, N2, D]` | 最常规，定长 batch |
| `TND` | `[T1, N1, D]` | `[T2, N2, D]` | 变长打包：所有 batch 的 token 拼成一条，长度由 `actual_seq` 划分 |
| `PA_BSND`（仅 KV） | — | `[block_num, block_size, N2, D]` | Paged KV cache，配合 `block_table` 用 |

- **N1 / N2**：`N1` 是 query 头数，`N2` 是 KV 头数，GQA 下 `N1` 是 `N2` 的整数倍（一组 query 头共享一个 KV 头）。MLA 通常 `N2=1`。
- `query_rope`/`key_rope` 的形状与 query/key 一致，只是最后一维换成 64。

### `actual_seq_lengths_*` 的两种语义（容易踩坑）

- **BSND / PA_BSND**：传**每个 batch 的长度**，例如 2 个 batch 各 128 → `[128, 128]`。
- **TND**：传**累积前缀和**（cumulative），例如 2 个 batch 各 4 → `[4, 8]`。
- 注意：query 用 TND 而 KV 用 PA_BSND 时，**query 端用 cumulative，KV 端用 per-batch**。

---

## 稀疏是怎么选的

`sparse_indices` 的形状：

- `BSND`：`[B, S1, N2, K]`
- `TND`：`[T1, N2, K]`

它的含义：**对每个 (batch, query, kv_head)，给一串 block 编号，表示这个 query 要 attend 哪些 KV block。**

- 第 `idx` 个 block 覆盖的 KV token 是 `[idx * sparse_block_size, idx * sparse_block_size + sparse_block_size)`。
- 例：`sparse_block_size=8`，`idx=4` → 选中 token `[32, 40)`。
- `K` 维里没用满的位置填 **`-1`** 作为结束哨兵，kernel 读到 `-1` 就停。
- 选中的 token 会被 `threshold` 截断（由 `sparse_mode` + `actual_seq` 决定），超出有效长度的部分自动丢弃。

**dense（`sparse_indices=None`）** = attend `[0, actual_seq_kv)` 全部 KV，等价于普通 Flash Attention，但走的是更快的连续读路径。

---

## 返回值

```python
attn_out, softmax_max, softmax_sum, packed_key, packed_key_rope = op(...)
```

| 返回 | 形状 | 说明 |
|---|---|---|
| `attn_out` | 同 `query` | 注意力输出，**主结果** |
| `softmax_max` | `[B,N2,S1,N1/N2]`(BSND) / `[N2,T1,N1/N2]`(TND) | 仅 `return_softmax_lse=True` 时有效 |
| `softmax_sum` | 同上 | flash-decode 跨核合并 LSE 用 |
| `packed_key` / `packed_key_rope` | 见下 | 仅 `return_packed_kv=True` 时有效，否则为 `None` |

不需要的输出可以用 `_` 接收。`return_packed_kv` 会额外输出“按 `sparse_indices` gather 好的连续 KV”，方便下游复用（如二段式 KV 复用 pipeline），形状 `[..., N2, K*block_size, head_dim]`；**只支持 sparse 且 `sparse_block_size<=4`**。

---

## 注意事项（重要）

1. **MLA 专用，维度写死**：head 维必须是 512(NoPE) + 64(RoPE)，`scale_value` 用 `1/sqrt(576)`。不是通用 attention 算子。

2. **`value` 应当等于 `key`**：MLA absorb 形式下 value 就是 latent `c_KV`，与 `key` 的 NoPE 部分是同一个张量。
   - 实际使用直接传 `value=key` 即可。
   - ⚠️ 如果你传**独立**的 `value`（`value != key`）：`sparse_block_size <= 4` 这条路径（内部走 merge-KV 优化）**会忽略你的独立 value、复用 key**，结果不对；`sparse_block_size > 4` 与 dense 路径则会正确读独立 value。换句话说，**只在 `value==key` 的 MLA 场景使用**。

3. **`sparse_block_size` 影响内部走哪条 kernel**：`<=4` 走“先 gather 成连续再算”（V_TEMPLATE），`>4` 与 dense 走“边算边直读”（C_TEMPLATE）。两者数学等价（在 `value==key` 前提下），按你的稀疏块大小自然选择即可。

4. **`actual_seq` 的 per-batch / cumulative 区别**见上面 layout 小节，TND 用累积和，最易错。

5. **PA_BSND 必须配 `block_table`**，且 `block_table` 是 `int32`。

---

## 常见场景示例

### 1) Decode（单 query，稀疏选 top-k KV）
```python
out, *_ = op(query=q, key=kv, value=kv, sparse_indices=topk_idx, scale_value=scale,
             sparse_block_size=1, actual_seq_lengths_query=[1]*B, actual_seq_lengths_kv=seq_kv,
             query_rope=q_rope, key_rope=kv_rope, layout_query="BSND", layout_kv="BSND",
             sparse_mode=0)
```

### 2) Prefill（多 query，causal）
```python
out, *_ = op(..., sparse_mode=3)   # 每个 query 只 attend 自己及之前的 KV
```

### 3) Paged KV
```python
out, *_ = op(query=q, key=key_cache, value=value_cache, sparse_indices=idx, scale_value=scale,
             sparse_block_size=1, block_table=block_table,
             actual_seq_lengths_kv=seq_kv, query_rope=q_rope, key_rope=kr_cache,
             layout_query="BSND", layout_kv="PA_BSND", sparse_mode=3)
```

### 4) Dense（全选，等价普通 FA 但走快路）
```python
out, *_ = op(query=q, key=kv, value=kv, sparse_indices=None, scale_value=scale,
             sparse_block_size=1, actual_seq_lengths_kv=seq_kv,
             query_rope=q_rope, key_rope=kv_rope, layout_query="BSND", layout_kv="BSND",
             sparse_mode=0)
```

---

## 测试参考

- 正确性回归：`tests/op_test/sparse_flash_attention/correctness/`
  - `test_dense.py`（dense 4 种 layout）、`test_sparse_subset.py`（真稀疏子集）、`test_lse.py`、`test_packed_kv*.py`、`test_guards.py`
- 框架集成（对 CPU golden 严格对比）：`tests/op_test/sparse_flash_attention/framework/`，`bash test_run.sh single`
- 性能：`benchmark/bench_sfa.py`
