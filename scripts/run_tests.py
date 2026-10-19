#!/usr/bin/env python
"""最小测试运行入口：用项目虚拟环境启动 pytest。

每次运行都会在 logs/（已被 Git 忽略）下分配一个独立的临时目录作为
--basetemp，避免不同对话/进程共用同一临时目录而互相踩踏。本入口只负责
启动一次 pytest 并原样返回其退出码，不做自动重试，也不删除旧临时目录。
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 冲突/环境错误的退出码，避免与 pytest 的正常退出码(0/1/2/…)混淆
CONFLICT_EXIT = 2
NO_VENV_EXIT = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LOGS_DIR = PROJECT_ROOT / "logs"


def has_basetemp_conflict(args: list[str]) -> bool:
    """用户显式传入 --basetemp 时返回 True。

    --basetemp 由本入口自行管理，既不覆盖也不复用已有目录，
    因此对同时出现的两种写法都明确拒绝。
    """
    return "--basetemp" in args or any(a.startswith("--basetemp=") for a in args)


def alloc_tmp_dir(base_dir: Path) -> Path:
    """分配一个新的独立临时目录。只会新建，不删除、不复用已有目录。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tmp_dir = base_dir / f"pytest_tmp_{stamp}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def build_pytest_command(venv_python: Path, tmp_dir: Path, passthrough: list[str]) -> list[str]:
    """构造子进程命令行。返回列表而非 shell 字符串，不使用 shell=True。"""
    return [
        str(venv_python),
        "-m",
        "pytest",
        "--basetemp",
        str(tmp_dir),
        "-p",
        "no:cacheprovider",
        *passthrough,
    ]


def main(argv: list[str] | None = None) -> int:
    # 不用 argparse：需要把所有剩余参数原样传给 pytest（含以“-”开头的参数）
    passthrough = list(sys.argv[1:] if argv is None else argv)

    if has_basetemp_conflict(passthrough):
        print("参数冲突：本入口自行管理 --basetemp，拒绝用户显式传入，pytest 未启动。", file=sys.stderr)
        return CONFLICT_EXIT

    if not VENV_PYTHON.exists():
        print(f"未找到项目虚拟环境解释器：{VENV_PYTHON}", file=sys.stderr)
        return NO_VENV_EXIT

    LOGS_DIR.mkdir(exist_ok=True)
    tmp_dir = alloc_tmp_dir(LOGS_DIR)
    print(f"本轮临时目录：{tmp_dir}")

    cmd = build_pytest_command(VENV_PYTHON, tmp_dir, passthrough)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())