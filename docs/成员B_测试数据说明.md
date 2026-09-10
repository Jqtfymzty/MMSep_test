# 成员 B 固定测试数据说明

> 本文记录 `tests/conftest.py` 中成员 B fixture 的设计依据。正式提交前需结合本人实际测试设计进行核对和改写。

## 1. 本阶段目的

本阶段只建立可复用、可追踪的小尺寸测试数据，不编写具体测试步骤和断言。这样可以把“准备输入数据”和“设计测试用例”分开，减少重复代码，并保证两次运行得到相同结果。

成员 B 的 fixture 统一使用以下前缀：

- `cache_compress_`：缓存压缩相关数据；
- `mmsep_`：视觉 token 筛选相关数据。

成员 A 可以使用其他前缀，避免双方在 `conftest.py` 中出现同名 fixture。

## 2. 缓存对象工厂

`cache_compress_factory` 返回一个创建 `MMSepCache` 的工厂函数。默认参数如下：

| 参数 | 默认值 | 选择原因 |
|---|---:|---|
| `layer_num` | 2 | 能观察按层属性，同时保持运行量较小 |
| `init_cache_size` | 2 | cache 开头有明确的固定保留区域 |
| `sep_cache_size` | 3 | 容量较小，便于人工追踪 separator 区域 |
| `local_size` | 2 | 最近窗口短，压缩前后容易核对 |
| `cache_size` | 8 | 满足 `init + sep + local < cache_size` |
| `separator_token_ids` | `[2, 3]` | 使用容易识别的小整数作为分隔符 |
| `PADDING_ID` | 0 | 与正常 token 和分隔符区分 |
| `image_token_length` | 4 | 代替实际模型中的大规模视觉 token |
| `image_start_pos` | `[2, 2]` | 两条 batch 记录使用相同起点作为基础数据 |
| `mmsep_layer` | 1 | 两层设置中可区分启用前和启用后的层 |
| `pe_dim` | 4 | 与小尺寸 head dimension 对齐 |

工厂函数接受关键字参数覆盖默认值。因此后续需要研究某个配置时，不必重复填写其他无关参数。

## 3. KV tensor 工厂

`cache_compress_kv_factory` 默认产生：

```text
Key shape   = [2, 2, 8, 4]
Value shape = [2, 2, 8, 4]
```

四个维度依次表示：

```text
[batch_size, num_heads, sequence_length, head_dim]
```

Key 使用从 0 开始连续递增的数值，Value 等于 Key 加 10000。这样设计有两个用途：

1. 每个序列位置的内容可以直接追踪，不依赖随机种子；
2. Key 和 Value 不会混淆，但二者的位置关系完全一致。

工厂允许覆盖 batch、head、序列长度和 head dimension，以便后续由成员 B 自行构造不同规模的输入。

## 4. token id 数据

`cache_compress_token_ids` 提供两条长度为 8 的序列：

```text
记录 0：[9, 2, 8, 3, 7, 2, 6, 5]
记录 1：[9, 8, 2, 7, 6, 5, 4, 1]
```

在默认 `separator_token_ids=[2, 3]` 下，两条记录中的分隔符数量不同。这组数据适合用于人工观察 batch 对齐逻辑，但本文不预先规定具体测试步骤或期望值。

## 5. 多模态 feature 数据

`mmsep_features` 的形状为：

```text
[batch_size=2, sequence_length=10, hidden_size=8]
```

数据由连续整数除以 100 得到，不使用随机生成。每个位置的特征数值不同，发生切片、重排或补零后，可以根据数值追踪它来自原序列的哪个位置。

## 6. position id 和 attention mask

`mmsep_position_ids` 为两条 `0..9` 的位置编号，与 feature 的序列长度一致。

`mmsep_attention_mask` 包含：

- 一条长度全部有效的记录；
- 一条末尾有两个右侧 padding 位置的记录。

这与 `graph_rank_separators()` 只允许右侧 padding 的当前实现一致，也能让成员 B 后续自行观察完整序列与右侧 padding 序列之间的差异。

## 7. 使用约束

1. fixture 只负责提供数据，不在 `conftest.py` 中写断言。
2. 需要不同输入时优先调用工厂参数，不直接修改公共默认数据。
3. 测试中如需原地修改 tensor，应先使用 `clone()`，避免影响同一测试中的其他计算。
4. 不依赖 CUDA、网络、API Key 或模型权重。
5. 后续新增 fixture 时继续使用成员 B 的命名前缀。

## 8. 本人复核记录

在进入测试用例实现阶段前，由成员 B 补充：

- [ ] 已确认每个 tensor 维度的含义；
- [ ] 已手工定位 token id 中的分隔符位置；
- [ ] 已确认默认 cache 参数之间的容量关系；
- [ ] 已确认 Value 加 10000 的追踪方式容易识别；
- [ ] 已根据个人测试设计修改或补充 fixture；
- [ ] 已与成员 A 确认 fixture 命名前缀没有冲突。

## 9. Transformers 兼容性记录

首次使用系统环境中的 Transformers 4.57.6 创建 `MMSepCache` 时，基类 `Cache` 要求提供新的 layer 参数，导致 `super().__init__()` 直接抛出 `ValueError`。这说明该版本与当前 `src/cache.py` 的继承方式不兼容。

随后使用隔离目录中的 Transformers 4.50.3 和 Tokenizers 0.21.4 复核，`MMSepCache` 能正常创建，所有固定 fixture 的形状也符合本说明。因此项目依赖基线改为这两个版本。该记录属于环境兼容性结论，不计作成员 B 的功能缺陷。
