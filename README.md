# MMSep 自动化测试实践

本仓库以 MMSep 的缓存压缩和多模态分隔符代码为被测对象，完成软件测试与质量保证课程的实践任务。

## 项目结构

```text
MMSep_test/
├── src/                  # 被测源码
│   ├── auto_eval.py
│   ├── cache.py
│   └── mm_separators.py
├── tests/                # pytest 自动化测试
├── docs/                 # 作业要求、模板、分工和分析文档
├── check_env.py          # 环境一致性检查
├── requirements.txt      # Python 依赖
└── run_test.bat          # Windows 一键测试入口
```

## 环境要求

- Windows 10/11
- Python 3.12
- CPU 环境即可运行单元测试
- Git

当前依赖基线记录在 `requirements.txt` 中。测试设计应使用小尺寸固定 tensor、mock 和本地样例，不依赖 GPU、大模型权重、API Key 或网络请求。

## 安装

在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可在当前终端临时执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

不要把 API Key 写入源码、README 或 Git 历史。真实运行 `auto_eval.py` 时，应由本地环境变量提供 `ARK_API_KEY`；单元测试应使用 mock，不调用真实服务。

## 环境检查

激活虚拟环境后执行：

```powershell
python check_env.py
python -m pip check
```

两名成员应比较 Python、PyTorch、Transformers 和 pytest 等关键版本，避免因环境差异产生不一致结果。

## 运行测试

Windows 下可双击或在终端运行：

```powershell
.\run_test.bat
```

也可以直接使用 pytest：

```powershell
python -m pytest tests -v
```

运行某个测试文件时使用：

```powershell
python -m pytest tests\文件名.py -v
```

`run_test.bat` 会优先使用项目内 `.venv` 的 Python；不存在 `.venv` 时才使用系统 `python`。

## 协作约定

- 成员 A 负责 `auto_eval.py` 和 `cache.py` 基础功能测试。
- 李晨希负责 `cache.py` 压缩功能和 `mm_separators.py` 测试。
- 共用 fixture 应采用清晰的命名前缀，避免覆盖对方内容。
- 每个提交只包含一个完整、可说明的工作单元。
- 提交前先查看 `git diff`，确认没有密钥、缓存、模型权重或无关文件。
- 测试用例、缺陷报告和测试报告中的编号应保持一致。

## 当前阶段

李晨希负责部分现有 27 条自动化用例，2026 年 9 月 13 日在锁定环境下复测全部通过；4 项缺陷均已完成修复前复现和修复后回归。视觉筛选测试使用轻量替身，结论限于局部单元测试，不包含真实模型、GPU 或端到端推理。小组最终报告仍需合并成员 A 的执行结果、贡献比例和汇报材料。
