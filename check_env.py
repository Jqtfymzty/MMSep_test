# -*- coding: utf-8 -*-
r"""
环境一致性校验脚本（成员 A / 李晨希各运行一次，比对输出）

用法：
    .venv\Scripts\activate
    python check_env.py

用途：
    1. 确认两人 Python 版本、关键依赖版本完全一致
    2. 确认被测源码文件齐全
    3. 确认无依赖冲突
    4. 把输出贴到群里比对，不一致的部分先对齐再开发

运行前请确保已激活项目虚拟环境（命令行前缀出现 (.venv)）。
"""

import sys
import platform
import pathlib
from importlib import metadata as md

# 需要版本完全一致的关键包
KEY_PACKAGES = [
    "torch",
    "transformers",
    "tokenizers",
    "pytest",
    "numpy",
    "tqdm",
    "volcengine-python-sdk",
]

# 被测源码文件（相对 src/ 目录）
SOURCE_FILES = [
    "cache.py",
    "mm_separators.py",
    "auto_eval.py",
]


def show_python():
    print("=== Python ===")
    print("version    :", sys.version.split()[0])
    print("executable :", sys.executable)
    print("platform   :", platform.platform())
    print("in venv    :", "YES" if sys.prefix != sys.base_prefix else "NO (警告：未激活虚拟环境)")
    print()


def show_packages():
    print("=== 关键依赖 ===")
    for pkg in KEY_PACKAGES:
        try:
            print("{:<14}: {}".format(pkg, md.version(pkg)))
        except md.PackageNotFoundError:
            print("{:<14}: NOT INSTALLED  <-- 需要安装".format(pkg))
    print()

    print("=== torch 设备 ===")
    try:
        import torch

        print("torch cuda available:", torch.cuda.is_available())
        print("（本项目要求 CPU 可跑，输出 False 属正常）")
    except ImportError:
        print("torch 未安装，跳过设备检查")
    print()


def show_sources():
    print("=== 被测源码 ===")
    src_dir = pathlib.Path(__file__).parent / "src"
    for name in SOURCE_FILES:
        path = src_dir / name
        if path.exists():
            size = path.stat().st_size
            print("{:<20}: found ({} bytes)".format(name, size))
        else:
            print("{:<20}: MISSING  <-- 检查路径".format(name))
    print()


def show_transformers_api():
    print("=== transformers API 兼容性 ===")
    try:
        from transformers.cache_utils import Cache  # noqa: F401

        print("from transformers.cache_utils import Cache : OK")
    except ImportError as exc:
        print("from transformers.cache_utils import Cache : FAILED")
        print("   原因:", exc)
        print("   处理: pip install \"transformers>=4.36\"")
    print()


def show_ark_api():
    print("=== Ark SDK 兼容性 ===")
    try:
        from volcenginesdkarkruntime import Ark  # noqa: F401

        print("from volcenginesdkarkruntime import Ark : OK")
    except ImportError as exc:
        print("from volcenginesdkarkruntime import Ark : FAILED")
        print("   原因:", exc)
        print('   处理: pip install "volcengine-python-sdk[ark]"')
    print()


def show_conflicts():
    print("=== 依赖冲突检查 ===")
    print("请在命令行单独执行： pip check")
    print("期望输出： No broken requirements found.")
    print()


def main():
    print()
    print("########################################")
    print("# MMSep_test 环境校验")
    print("########################################")
    print()
    show_python()
    show_packages()
    show_transformers_api()
    show_ark_api()
    show_sources()
    show_conflicts()
    print("把以上全部输出复制到群里，与另一名成员逐行比对。")
    print()


if __name__ == "__main__":
    main()
