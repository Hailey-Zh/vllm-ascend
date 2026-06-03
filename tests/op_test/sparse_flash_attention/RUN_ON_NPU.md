# vllm-ascend SparseFlashAttention 回归测试（NPU 侧使用说明）

> 配合 `sfa_modify_plan.md` 的步骤 0~4，每步改完都跑一次。
> 框架来自 ops-transformer arch22 pytest，已改造为调 `torch.ops._C_ascend.npu_sparse_flash_attention`。

## 前置环境

在 NPU 机器（Atlas A2 / A3，ascend910b 或 ascend910_93）上：

```bash
# 1) torch_npu 已就绪
python -c "import torch_npu; print(torch_npu.__version__)"

# 2) 编译并安装 vllm-ascend（含 csrc 自定义算子）
cd /path/to/vllm-ascend
git checkout feature/sfa-extend
pip install -e .          # 或按项目正式构建流程
bash csrc/build_aclnn.sh  # 构建自定义 aclnn 算子（包含 sparse_flash_attention）

# 3) 验证算子已注册
python -c "import torch, vllm_ascend; print(hasattr(torch.ops._C_ascend, 'npu_sparse_flash_attention'))"
# 期望输出 True
```

## 跑 baseline

```bash
cd tests/op_test/sparse_flash_attention
bash run_baseline.sh
```

会跑 `sparse_flash_attention_paramset.py::ENABLED_PARAMS` 里的 5 个用例：

| 用例 | layout_query | layout_kv | 用途 |
|---|---|---|---|
| `bsnd_basic` | BSND | BSND | 最简形状 |
| `bsnd_multi_batch` | BSND | BSND | 多 batch 变长 |
| `pa_bsnd` | BSND | PA_BSND | PageAttention |
| `tnd_basic` | TND | TND | 变长 TND |
| `tnd_pa_multi_batch` | TND | PA_BSND | **生产路径** |

每个用例 = CPU golden（`sparse_flash_attention_golden.py` 内置参考实现）+ NPU 直调 + 精度对比。

退出码：
- `0` = 全部 Pass
- 非 0 = 有用例失败，看 pytest 输出定位

## 改造步骤与回归映射

按 `sfa_modify_plan.md`：

| 改造步骤 | 完成后要回归 | 额外开启 |
|---|---|---|
| 步骤 1 (TND+PA_BSND 联调) | `bash run_baseline.sh` 全 Pass | — |
| 步骤 2 (LSE 输出) | baseline + 把 paramset 里 `return_softmax_lse` 改回 `True` 跑一遍 | utils.py 已支持 LSE 对比 |
| 步骤 3 (sparse_indices 可选) | baseline + 新增稠密 FA 用例（无 sparse_indices） | 需在 paramset 加 dense 组 |
| 步骤 4 (packed_kv 输出) | baseline + 新增 packed_kv 输出比对 | 需在 utils.py 加 packed_kv 对比逻辑 |

## 目录说明

```
tests/op_test/sparse_flash_attention/
├── run_baseline.sh                 # 一键回归脚本（本次新增）
├── test_run.sh                     # ops-transformer 原始多模式入口（single / batch_save / batch_exec）
├── sparse_flash_attention_paramset.py   # 5 个 baseline 用例（已统一关 LSE）
├── sparse_flash_attention_golden.py     # CPU 参考实现
├── batch/sparse_flash_attention_process.py  # NPU 算子调用（已切到 torch.ops._C_ascend）
├── utils.py                        # 测试驱动 & 精度对比
├── result_compare_method.py        # 精度比较算法
├── generate_tensor_data.py         # 输入张量生成
├── check_valid_param.py            # 参数校验
├── test_sparse_flash_attention_single.py  # pytest 入口（single 模式）
├── test_sparse_flash_attention_batch.py   # pytest 入口（batch 模式）
├── pytest.ini
├── README.md                       # ops-transformer 原始说明
├── RUN_ON_NPU.md                   # 本文件
├── excel/                          # 批量生成的 xlsx（gitignored）
└── golden/                         # 保存的 NPU 输出（gitignored，预留）
```

## 故障排查

- `torch.ops._C_ascend 未注册` → `csrc/build_aclnn.sh` 没跑或 `vllm-ascend` 没装好
- `npu_sparse_flash_attention 未注册` → 检查 `csrc/torch_binding.cpp:664` 那段是否被编进来
- pytest 报参数缺失 → 看 `check_valid_param.py` 的校验规则；当前 N1 限制 1/2/4/8/16/32/64/128
- LSE 用例直接挂 → 应该不会发生，已统一关 LSE；如发生检查 `paramset.py` 是否被本地改回去
