# 成员 B 缺陷复现记录

> 当前状态：修复前记录。
>
> 本文内容来自本地实际执行结果。正式写入缺陷报告模板前，应由成员 B 再次复现、截图，并用自己的语言补充影响分析。

## 1. 执行环境

```text
Python       3.12.10
PyTorch      2.11.0+cpu
Transformers 4.50.3
Tokenizers   0.21.4
pytest       9.1.1
设备         CPU
```

复现命令：

```powershell
cd "C:\Users\Administrator\OneDrive\Desktop\课程任务\软件质量测试\MMSep_test"
& ".\.venv\Scripts\python.exe" -m pytest tests\test_mm_separators.py -v --tb=short
```

本轮执行结果：

```text
9 collected
6 passed
3 failed
```

## 2. B-DEF-01：无图像记录触发未初始化变量异常

### 缺陷位置

`src/mm_separators.py` 第 64 行附近。

### 前置条件

- batch 中记录不包含图像；
- `image_token_posi` 使用 `-1` 表示无图像；
- 为函数提供合法的 features、position ids 和 attention mask。

### 复现用例

```text
test_record_without_image_keeps_original_data
```

单独复现：

```powershell
python -m pytest tests\test_mm_separators.py::test_record_without_image_keeps_original_data -v
```

### 预期结果

无图像记录不需要执行视觉 token 筛选，原始 features、position ids 和 attention mask 应保持不变并正常返回。

### 实际结果

```text
UnboundLocalError: cannot access local variable 'attention_mask_list'
where it is not associated with a value
```

### 初步原因

无图像分支在函数前半段执行：

```python
attention_mask_list.append(attention_mask[i])
```

但 `attention_mask_list` 在后面的 mask 重建阶段才被创建，因此执行该分支时变量尚不存在。

### 当前状态

已修复。无图像分支不再提前访问尚未创建的 `attention_mask_list`，而是在统一重建阶段保留原 mask 并重新生成有效位置编号。对应测试已由失败转为通过。

## 3. B-DEF-02：训练模式下 Query 变量没有赋值

### 缺陷位置

`src/mm_separators.py` 第 72-76 行附近。

### 前置条件

- 模型处于训练状态，即 `self.training=True`；
- 当前记录包含视觉 token；
- 其他输入满足函数要求。

### 复现用例

```text
test_training_mode_can_rank_visual_tokens
```

单独复现：

```powershell
python -m pytest tests\test_mm_separators.py::test_training_mode_can_rank_visual_tokens -v
```

### 预期结果

函数应明确支持训练状态，或者在不支持训练状态时抛出说明清楚的受控异常，不应因局部变量未赋值而崩溃。

当前测试按“训练状态同样可以完成筛选”的预期编写。

### 实际结果

```text
UnboundLocalError: cannot access local variable 'text_query_states'
where it is not associated with a value
```

### 初步原因

源码在训练分支中只有 `pass`：

```python
if self.training:
    pass
else:
    text_query_states = ...
```

随后无论是否处于训练状态，都会使用 `text_query_states` 计算注意力，因此训练状态下变量没有定义。

### 当前状态

已复现，尚未修复。修复前需要确认设计上是否允许训练模式调用该功能。

## 4. B-DEF-03：不同视觉长度的 batch 无法堆叠 attention mask

### 缺陷位置

`src/mm_separators.py` 第 119 行附近。

### 前置条件

- batch size 大于 1；
- 不同记录拥有不同数量的视觉 token；
- 筛选后各记录的新序列长度不同。

### 复现用例

```text
test_batch_with_different_visual_lengths_is_padded_consistently
```

单独复现：

```powershell
python -m pytest tests\test_mm_separators.py::test_batch_with_different_visual_lengths_is_padded_consistently -v
```

### 预期结果

features、position ids 和 attention mask 应统一补齐到 batch 内最大新长度，最终三个结果的序列维度保持一致。

### 实际结果

```text
RuntimeError: stack expects each tensor to be equal size,
but got [5] at entry 0 and [7] at entry 1
```

### 初步原因

源码已经将每条记录的 feature 补到 `max_len`，但重建后的 `new_attention_mask` 没有执行相同补齐，便直接调用：

```python
torch.stack(attention_mask_list, dim=0)
```

当两条 mask 长度分别为 5 和 7 时无法堆叠。

### 当前状态

已复现，尚未修复。

## 5. 本轮通过的行为

以下 6 项未发现异常：

1. 根据注意力保留排名最高的视觉 token；
2. 选择多个视觉 token 后恢复原序列顺序；
3. 小规模视觉区域允许筛选为 0 个 token；
4. 输入 position ids 和 attention mask 为 `None` 时，输出相应恢复为 `None`；
5. 左侧 padding 被明确拒绝；
6. 非 `flash_attention_2` 注意力类型被明确拒绝。

## 6. 修复验证计划

下一阶段按以下顺序处理：

1. 保留当前失败结果或截图作为修复前证据；
2. 分别修改三个问题对应的最小代码区域；
3. 先单独运行对应失败用例；
4. 再运行 `test_mm_separators.py`；
5. 最后运行全部测试，确认缓存压缩功能没有回归；
6. 把修复后结果、提交编号和截图补充到正式缺陷报告。
