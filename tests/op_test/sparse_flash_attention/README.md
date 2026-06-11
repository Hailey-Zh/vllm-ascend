# sparse_flash_attention 算子测试

`torch.ops._C_ascend.npu_sparse_flash_attention` 的测试集，分为两套独立体系：

| 体系 | 目录 | 验证方式 | 触发 |
|---|---|---|---|
| **correctness** | `correctness/` | 直调算子 + PyTorch 参考实现逐元素比对 | `pytest -m <marker>` |
| **framework** | `framework/` | paramset/Excel → CPU golden → NPU 精度比对 | `bash test_run.sh <模式>` |

- `correctness/`：每个功能点一个独立测试文件，参考值在文件内用 PyTorch 现写，按 pytest marker 选择性运行。改造新增功能（LSE / dense / packed_kv）的精度验证主要在这里。
- `framework/`：原 ops-transformer arch22 移植的回归框架，覆盖大 shape / 多 batch / bf16 的 attn_out 回归。

```
sparse_flash_attention/
├── README.md                  # 本文件
├── pytest.ini                 # 全局 pytest marker 定义（rootdir 锚点）
├── correctness/               # 独立算子级精度测试（详见下方矩阵）
│   ├── test_lse.py
│   ├── test_dense.py
│   ├── test_dense_extended.py
│   ├── test_packed_kv.py
│   ├── test_packed_kv_extended.py
│   └── test_guards.py
├── framework/                 # paramset/Excel → CPU golden → NPU 比对
│   ├── test_sparse_flash_attention_single.py   # paramset 直跑
│   ├── test_sparse_flash_attention_batch.py    # .pt 回放
│   ├── sparse_flash_attention_golden.py        # CPU golden 实现
│   ├── sparse_flash_attention_paramset.py      # 单用例入参配置
│   ├── generate_tensor_data.py                 # 输入张量生成
│   ├── result_compare_method.py                # 精度比对
│   ├── check_valid_param.py                    # 参数约束校验
│   ├── utils.py                                # 参数解析 / 执行入口
│   ├── test_run.sh                             # 框架统一入口脚本
│   ├── run_baseline.sh                         # 5 个 baseline 用例一键回归
│   ├── RUN_ON_NPU.md                           # NPU 侧环境与回归详细说明
│   └── batch/ , excel/                         # Excel 批量用例生成 / 回放
└── diag/                      # 调试用 shell 脚本（aclnn nullptr / 重编译等）
```

## 前置环境

在 NPU 机器（Atlas A2 / A3，ascend910b 或 ascend910_93）上：

```bash
# 1) torch / torch_npu / vllm_ascend 可正常 import
python -c "import torch_npu, vllm_ascend; print('ok')"

# 2) 编译安装 vllm-ascend（含 csrc 自定义算子），并 source 算子环境
source vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash
```

> ⚠️ `def.cpp` 改了输出/属性后，必须重编 **CANN op_plugin 二进制**（不只是 `_C_ascend.so`），
> 否则 `return_softmax_lse=True` / `return_packed_kv=True` 会报
> `AclNN_Parameter_Error / Check executor != nullptr failed`。详见 `framework/RUN_ON_NPU.md`。

---

## correctness/ —— 算子级精度测试

每个文件带一个 pytest marker，可独立运行。在 `correctness/` 目录下执行：

```bash
cd correctness
pytest test_lse.py                 -m lse          -s -v   # LSE（softmax_max/sum）
pytest test_dense.py               -m step3c_dense -s -v   # 稠密 == 全选稀疏
pytest test_dense_extended.py      -m dense_ext    -s -v   # 稠密 TND / dense+LSE
pytest test_packed_kv.py           -m step4_packed_kv -s -v # packed_kv 基础
pytest test_packed_kv_extended.py  -m packed_ext   -s -v   # packed_kv 扩展
pytest test_guards.py              -m guards       -s -v   # 入参守卫 / 边界

# 全部 correctness 一次跑完
pytest . -s -v
```

`-s` 显示精度对比 print，`-v` 显示用例名，`-k <name>` 单跑某个用例。

### 覆盖矩阵

| 功能 | BSND/BSND | BSND/PA_BSND | TND/TND | TND/PA_BSND | 文件 |
|---|---|---|---|---|---|
| 基础 attn_out | framework | framework | framework | framework | （framework 体系） |
| **LSE** mode0/3 | ✅ | ✅ mode3 | ✅ mode0 | ✅ mode3 | `test_lse.py` |
| LSE 变长 actual_seq | ✅ | | | | `test_lse.py` |
| LSE bf16 | ✅ | | | | `test_lse.py` |
| **Dense**（==全选稀疏） | ✅ | ✅ | ✅ | ✅ | `test_dense.py` / `test_dense_extended.py` |
| Dense + LSE | ✅ | | | | `test_dense_extended.py` |
| **Packed KV** | ✅ | ✅ | ✅ | ✅ | `test_packed_kv.py` / `test_packed_kv_extended.py` |
| Packed KV mode3 截断 | ✅ | | | | `test_packed_kv_extended.py` |
| Packed KV block_size 2/4 | ✅ | | | | `test_packed_kv_extended.py` / `test_guards.py` |
| Packed KV + LSE | ✅ | | | | `test_packed_kv_extended.py` |
| **Guards**：dense+packed 报错 | ✅ | | | | `test_guards.py` |
| Guards：block_size>4 报错 | ✅ | | | | `test_guards.py` |

> 说明：MLA 恒为 `N2=1`、`D=512`、`rope_head_dim=64`，故矩阵不再展开这些维度。
> QK 打分用完整 576 维（NoPE 512 + RoPE 64），value 仅用 NoPE 512——参考实现与
> `framework/sparse_flash_attention_golden.py` 一致。

---

## framework/ —— golden 回归框架

paramset 直跑（默认 `sparse_flash_attention_paramset.py` 的 6 组用例，含 LSE 回归）：

```bash
cd framework
bash test_run.sh single                  # 跑 paramset
bash run_baseline.sh                      # 5 个 baseline 用例一键回归
```

其它模式（Excel 批量生成 .pt、回放执行等）见 `framework/RUN_ON_NPU.md` 与 `test_run.sh` 内注释。

### 当前支持范围（算子约束）

- `layout_query` ∈ {BSND, TND}，`layout_kv` ∈ {BSND, TND, PA_BSND}；非 PA 要求 query/kv layout 相同
- `q_type` / `kv_type` ∈ {float16, bfloat16}
- `N2 = 1`，`g = N1/N2` ∈ {1,2,4,8,16,32,64,128}
- `D = 512`，`rope_head_dim = 64`，`attention_mode = 2`
- `sparse_mode` ∈ {0, 3}
- `sparse_block_size`：普通计算 ∈ [1,128] 且 2 的幂；**packed_kv 仅支持 ≤ 4**（MergeKv 硬约束）
- `K ≤ ceil(S2 / sparse_block_size)`
- `block_size` 仅 PA_BSND 生效，需正整数且 16 对齐；`block_num` 需覆盖实际 KV 长度
- `actual_seq_q` / `actual_seq_kv` 若传入，长度须等于 `B`

### 结果文件

- `result.xlsx`：每个用例的入参、状态与 `fulfill_percent`
- `pt_files/*.pt`：batch 流程生成的中间用例

---

## 关键概念速查

- **sparse_indices**：lightning_indexer 选出的 KV **block 号**列表，`-1` 为终止符。
  一个 index 展开 `sparse_block_size` 个连续 token（block_size=4 → index 5 选中 token 20~23）。
- **packed_key / packed_key_rope**：把选中 KV 压实成连续 buffer 输出，供 decode 跨 step 复用；
  仅稀疏模式 + `sparse_block_size ≤ 4` 产出（MergeKv 的 UB ping-pong / flush 阈值写死）。
- **softmax_max / softmax_sum**（LSE）：`return_softmax_lse=True` 时输出，fp32；
  BSND→`[B,N2,S1,g]`，TND→`[N2,T1,g]`。
