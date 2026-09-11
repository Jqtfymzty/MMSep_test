# pytest 基础学习笔记（结合 MMSep_test）

> 这份笔记用于理解当前项目的测试代码。建议边看 `tests/test_cache_compress.py` 边阅读，并用自己的语言补充实际运行体会。

## 1. pytest 做了什么

pytest 主要负责三件事：

1. 自动发现测试文件和测试函数；
2. 准备 fixture，并把它们传给测试函数；
3. 执行断言，汇总通过、失败和异常信息。

本项目从仓库根目录执行：

```powershell
python -m pytest tests -v
```

使用 `python -m pytest` 的好处是明确使用当前 Python 环境中的 pytest，并且项目根目录会进入模块搜索路径，所以 `from src.cache import MMSepCache` 可以正常工作。

不要直接执行：

```powershell
python tests\test_cache_compress.py
```

直接执行时，Python 会把 `tests` 当作脚本入口目录，不会按照 pytest 的方式发现 fixture，也可能找不到同级的 `src`。

## 2. pytest 如何发现测试

pytest 默认识别：

- 文件名以 `test_` 开头；
- 函数名以 `test_` 开头；
- 测试类名以 `Test` 开头，并且通常不写自定义构造函数。

本项目中的文件名：

```text
tests/test_cache_compress.py
```

其中的函数如：

```python
def test_no_separator_returns_empty_cache(...):
    ...
```

都符合自动发现规则。

可以只查看 pytest 能发现哪些测试，而不实际执行：

```powershell
python -m pytest tests --collect-only -q
```

## 3. 一个测试的基本结构

测试通常可以拆成 Arrange、Act、Assert 三部分。

### 3.1 Arrange：准备数据

```python
cache = cache_compress_factory()
key, value = cache_compress_kv_factory()
token_ids = torch.full((2, 8), 9, dtype=torch.long)
```

这里创建被测对象和输入数据。

### 3.2 Act：调用被测功能

```python
(sep_key, sep_value), sep_ids, min_count, max_count = (
    cache.compress_past_win_2_seps((key, value), token_ids)
)
```

这里应只突出本用例真正要观察的操作。如果测试函数中混入很多不相关调用，失败时就难以判断是哪一步造成的。

### 3.3 Assert：判断结果

```python
assert min_count.item() == 0
assert sep_ids.shape == (2, 0)
```

断言表达预期。断言失败时，pytest 会显示实际值和预期值之间的差异。

## 4. fixture 是什么

fixture 是可复用的测试准备逻辑。它使用 `@pytest.fixture` 标记：

```python
@pytest.fixture
def cache_compress_token_ids() -> torch.Tensor:
    return torch.tensor(...)
```

测试函数只需把 fixture 名写成参数：

```python
def test_something(cache_compress_token_ids):
    ...
```

pytest 会根据参数名找到同名 fixture，执行 fixture，再把返回值传进测试函数。这种方式叫依赖注入。

fixture 的主要价值是：

- 减少重复准备代码；
- 统一常用数据；
- 每次测试重新取得数据，降低测试相互影响；
- 修改基础数据时有明确入口。

## 5. 为什么 fixture 里还要返回工厂函数

`cache_compress_factory` 不是直接返回一个固定 cache，而是返回内部函数 `_factory`：

```python
cache = cache_compress_factory()
```

需要改变设置时可以写：

```python
cache = cache_compress_factory(
    SEP_ACCUMULATION=False,
    USE_MAX_SEP_CACHE=True,
)
```

这样既保留默认值，又允许每个测试只覆盖自己关心的参数。KV fixture 也采用相同设计，可以改变 batch 或序列长度。

## 6. 常用断言

### 6.1 普通 `assert`

适合整数、列表、形状和布尔条件：

```python
assert cache.sep_exrange[0] == 4
assert result_ids.tolist() == [[2, 3, 7, 6]]
assert result_key.shape == (1, 2, 6, 4)
```

测试 tensor 形状时可以直接比较 `torch.Size` 和元组。

### 6.2 `torch.testing.assert_close()`

适合比较 tensor：

```python
torch.testing.assert_close(actual, expected)
```

浮点计算可能存在很小的舍入误差，因此通常不建议直接写：

```python
assert actual == expected
```

而且 tensor 的 `==` 会返回一个布尔 tensor，不能直接代表整体是否相等。`assert_close()` 会检查形状、数值、dtype 等信息，并在失败时给出更清楚的差异。

### 6.3 `pytest.raises()`

如果预期代码应该抛出异常，可以写：

```python
with pytest.raises(AssertionError, match="at least one"):
    cache.compress_past_win_2_seps(...)
```

它同时检查：

1. 是否真的抛出了指定异常；
2. 异常消息是否包含指定文本。

如果没有抛异常，或者异常类型不对，该测试都会失败。

## 7. 为什么测试数据使用连续数字

KV fixture 使用 `torch.arange()`，Value 再整体加 10000。例如某个输出位置值来自原序列位置 5，可以直接与：

```python
key[:, :, 5, :]
```

比较。这比随机 tensor 更容易人工核对。

测试数据不一定要模拟真实模型的巨大规模。单元测试更关注逻辑是否正确，小尺寸数据通常更快、更稳定，也更容易找出错误位置。

## 8. 如何读取失败信息

出现失败时先看以下三部分：

1. 测试函数名称：说明哪个行为没有满足预期；
2. traceback 最底部：通常是最直接的异常位置；
3. pytest 给出的 actual/expected 差异：判断是数据准备错误、断言错误还是源码错误。

建议失败后只运行当前用例：

```powershell
python -m pytest tests\test_cache_compress.py::test函数名 -vv
```

确认原因后再运行整个文件，最后运行全部测试。

## 9. 常用运行参数

| 参数 | 作用 |
|---|---|
| `-v` | 显示每条测试的名称和结果 |
| `-vv` | 输出更详细的信息 |
| `-q` | 简化输出 |
| `-x` | 第一条失败后立即停止 |
| `-s` | 不捕获 `print()` 输出，适合临时调试 |
| `-k 关键词` | 只运行名称包含关键词的测试 |
| `--collect-only` | 只收集测试，不执行 |
| `--tb=short` | 使用较短的 traceback |

例如只运行名称中含 `accumulation` 的测试：

```powershell
python -m pytest tests -v -k accumulation
```

## 10. 测试应当相互独立

一个测试不应依赖另一个测试先执行。pytest 可能改变执行顺序，也可能以后使用并行插件。

当前每个测试都会重新调用 `cache_compress_factory()` 创建对象，因此某条测试对 `sep_exrange` 或 `cache_size` 的修改不会泄漏到其他测试。

如果直接复用全局 cache，就可能出现“单独运行通过、全量运行失败”的顺序依赖问题。

## 11. 当前缓存测试在验证什么

前 6 条测试关注 `compress_past_win_2_seps()`：

- 是否识别正确的 token id；
- 是否选中相同位置的 Key/Value；
- 多个位置是否保持顺序；
- batch 数量不同时如何截断或补齐；
- 最少分隔符保护是否生效。

后 5 条测试关注层级压缩：

- initial、separator、local 的拼接顺序；
- `past_tok_ids` 是否包含 initial 部分时的差别；
- 是否累积旧 separator；
- separator 超过容量时是否扩容；
- 固定容量模式是否保留较新的 separator。

最后 6 条测试关注 `update()` 主流程：

- 首次预填充能否写入 cache；
- 解码 token 能否连续追加；
- 新长度达到容量时是否触发压缩；
- 预填充标志是否绕过压缩；
- 普通 token 是否可以返回筛选后的视觉 KV；
- 文本分隔符是否优先返回完整 KV。

测试不仅检查最终长度，还检查实际保留位置。只检查 shape 可能漏掉“长度正确但内容顺序错误”的问题。

## 12. 建议自己动手的小练习

为了确认已经理解，可以自行完成下面的小改动，再观察失败信息：

1. 把某条用例的预期位置故意改错，运行后阅读 tensor 差异，再恢复。
2. 用 `-k accumulation` 只运行两条累积相关测试。
3. 在某条测试中临时使用 `print(result_ids)`，分别比较带 `-s` 和不带 `-s` 的输出。
4. 修改 KV 工厂的 `seq_len`，根据 initial/local 大小重新计算应保留的位置。
5. 用自己的语言给一条测试标注 Arrange、Act、Assert 三部分。

这些练习产生的临时改动不要提交；确认理解后恢复即可。
