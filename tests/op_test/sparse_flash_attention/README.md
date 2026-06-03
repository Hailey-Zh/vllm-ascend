# sparse_flash_attention pytest 测试框架



## 功能说明

基于pytest测试框架，实现 sparse_flash_attention 算子的功能验证：

-   **CPU侧**：复现算子功能用以生成golden数据
-   **NPU侧**：通过torch_npu进行算子直调获取实际数据
-   **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

支持三条主流程：

- `single`：基于 `sparse_flash_attention_paramset.py` 的固定参数直接构造输入并拉起 NPU 单算子执行。
- `batch_save`：从 Excel 读取参数，生成包含 CPU golden 的 `.pt` 用例文件。
- `gen_excel_from_paramset`：从 paramset 生成 Excel 文件。
- `batch_exec`：从已有 `.pt` 文件批量回放执行 NPU 算子并对比精度。



## 当前实现范围

### 参数说明

以下参数约束已经下沉到框架校验和本文档中，便于统一维护：

- `layout_query`: 支持 `BSND`、`TND`
- `layout_kv`: 支持 `BSND`、`TND`、`PA_BSND`
- 非 PA 场景要求 `layout_query == layout_kv`
- `q_type`: 仅支持 `torch.float16`、`torch.bfloat16`
- `kv_type`: 仅支持 `torch.float16`、`torch.bfloat16`
- `N2`: 仅支持 `1`
- `g = N1 / N2`: 仅支持 `1/2/4/8/16/32/64/128`
- `D`: 当前仅支持 `512`
- `rope_head_dim`: 当前仅支持 `64`
- `attention_mode`: 当前仅支持 `2`
- `sparse_mode`: 仅支持 `0` 和 `3`
- `sparse_block_size`: 当前仅支持 `1`，需位于 `[1, 128]` 且为 2 的幂
- `K`: 不能超过 `ceil(S2 / sparse_block_size)`
- `block_size`: 仅在 `PA_BSND` 生效，且要求为正整数且 16 对齐
- `block_num`: 需覆盖实际 KV 长度
- `actual_seq_q` / `actual_seq_kv`: 如果传入，长度必须等于 `B`

更完整的算子定义和输入约束，请同步参考：

- `attention/sparse_flash_attention/README.md`

### 环境配置

#### 前置要求

1、 确认torch_npu为最新版本
2、 参考[Attention融合算子Experimental使用说明](https://gitcode.com/cann/ops-transformer/blob/master/attention/Attention融合算子Experimental使用说明.md)激活CANN包和自定义算子包

#### custom包调用

支持custom包调用



## 文件结构

```text
pytest/
├── README.md
├── pytest.ini			# 创建测试标记
├── test_run.sh			# 执行脚本
├── check_valid_param.py			# 参数约束拦截
├── sparse_flash_attention_golden.py		# tensor转换/cpu侧算子golden实现
├── sparse_flash_attention_paramset.py		# 单用例入参配置
├── result_compare_method.py		# 输出精度对比
├── utils.py			# 参数解析/cpu npu执行入口
├── test_sparse_flash_attention_single.py	# 单用例运行主程序
├── test_sparse_flash_attention_batch.py	# 从 pt 文件批量执行 NPU 测试
└── batch/
    ├── sparse_flash_attention_process.py	# npu接口
    ├── test_sparse_flash_attention_pt_save.py		# 从 Excel 批量生成 pt 文件
    ├── gen_excel_from_paramset.py	# 从 paramset 生成 Excel 文件
    └── excel/
        ├── example.xlsx		# 示例 Excel 用例文件
        └── .gitkeep			# 目录占位符
```



## 使用方法

在 `attention/sparse_flash_attention/tests/pytest` 目录下执行：

### 命令格式

```bash
bash test_run.sh <模式> [-E excel_path] [-S sheet] [-P path] [-O output_path]
```

### 参数选项

| 选项 | 说明 | 适用模式 |
| --- | --- | --- |
| `-E excel_path` | 指定 Excel 文件路径，默认 `./excel/example.xlsx` | batch_save |
| `-S sheet` | 指定 Excel Sheet 页名，默认 `Sheet1` | batch_save/gen_excel_from_paramset |
| `-P path` | 指定路径（不同模式含义不同，详见下表） | single/batch_save/batch_exec/gen_excel_from_paramset |

| 模式 | `-P` 参数含义 | 默认值 |
| --- | --- | --- |
| single | paramset 文件名 | `sparse_flash_attention_paramset` |
| batch_save | pt 文件保存路径 | `./pt_files/` |
| batch_exec | pt 文件执行路径（目录或单个文件） | `./pt_files/` |
| gen_excel_from_paramset | paramset 文件名 | `sparse_flash_attention_paramset` |

**gen_excel_from_paramset 模式额外参数：**

| 选项 | 说明 | 默认值 |
| --- | --- | --- |
| `-E excel_output` | 输出 Excel 文件路径 | `./excel/example.xlsx` |
| `-S sheet` | Excel Sheet 页名 | `Sheet1` |

### single

手动配置 `sparse_flash_attention_paramset.py` 的参数，或使用 `-P` 指定其他 paramset 文件。

```bash
bash test_run.sh single                              # 使用默认 paramset
bash test_run.sh single -P my_paramset                # 使用指定的 paramset 文件
```

### batch_save

从 Excel 读取参数，生成包含 CPU golden 的 `.pt` 用例文件。

```bash
bash test_run.sh batch_save                           # 使用默认 Excel 和 Sheet
bash test_run.sh batch_save -E ./test.xlsx            # 指定 Excel 文件
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1  # 指定 Excel 和 Sheet
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1 -P ./output_pt/  # 指定全部参数
bash test_run.sh batch_save -S Sheet1 -E ./test.xlsx  # 参数顺序可任意
```

### gen_excel_from_paramset

从 paramset 生成 Excel 文件。

```bash
bash test_run.sh gen_excel_from_paramset                           # 使用默认 paramset
bash test_run.sh gen_excel_from_paramset -P my_paramset            # 指定 paramset 文件
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx  # 指定输出路径
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx -S decode  # 指定 Sheet 名
```

### batch_exec

从 `.pt` 文件批量回放执行 NPU 算子并对比精度。

```bash
bash test_run.sh batch_exec                           # 执行默认目录下所有 pt 文件
bash test_run.sh batch_exec -P ./pt_files/test.pt     # 执行单个 pt 文件
bash test_run.sh batch_exec -P ./custom_pt_dir/       # 执行指定目录下所有 pt 文件
```

下面给一个可直接参考的 Excel 用例，列名需与 batch 框架读取字段保持一致：

| Testcase_Prefix | Testcase_Number | layout_query | layout_kv | q_type | kv_type | B | T | T2 | S1 | S2 | N1 | N2 | D | K | scale_value | sparse_block_size | rope_head_dim | sparse_mode | attention_mode | return_softmax_lse | block_size | block_num | actual_seq_q | actual_seq_kv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bsnd_basic | 1 | BSND | BSND | torch.float16 | torch.float16 | 1 |  |  | 5 | 262144 | 8 | 1 | 512 | 16 | 0.04419 | 1 | 64 | 0 | 2 | False |  |  | [4] | [4] |
| tnd_basic | 1 | TND | TND | torch.float16 | torch.float16 | 2 | 8 | 3072 | 4 | 3072 | 8 | 1 | 512 | 32 | 0.04419 | 1 | 64 | 0 | 2 | False |  |  | [4,8] | [1111,3000] |



## 结果文件

- `result.xlsx`：记录每个用例的关键信息、执行状态与 `fulfill_percent`
- `./pt_files/*.pt`：batch 流程生成的中间测试用例