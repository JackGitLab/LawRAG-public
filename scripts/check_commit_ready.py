#!/usr/bin/env python
"""提交前状态检查：只在 index 存在已暂存变更时才允许执行 git commit。

只读检查 Git，不执行 git add/commit/reset/restore，不读取文件内容，
不输出密钥。退出码约定：

- READY（存在已暂存变更）：0
- NOT_READY（没有已暂存变更）：0，属于正常检查结果
- 非 Git 顶层目录或 Git 命令失败：2，明确失败

「没有已暂存变更」是正常状态，用 NOT_READY 表达，不制造新错误；
git commit 是否有内容以 index 为准，不检查工作区未暂存变更。
"""

from __future__ import annotations

import os
import subprocess
import sys


def run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def main() -> int:
    cwd = os.getcwd()

    root_result = run_git(["rev-parse", "--show-toplevel"])
    if root_result.returncode != 0:
        print("COMMIT_CHECK_ERROR", file=sys.stderr)
        print(
            f"无法确认当前目录属于 Git 仓库：{root_result.stderr.strip()}",
            file=sys.stderr,
        )
        return 2
    top_level = os.path.abspath(root_result.stdout.strip())
    if os.path.normcase(top_level) != os.path.normcase(os.path.abspath(cwd)):
        print("COMMIT_CHECK_ERROR", file=sys.stderr)
        print(f"当前目录不是 Git 顶层目录（顶层：{top_level}）", file=sys.stderr)
        return 2

    staged_result = run_git(["diff", "--cached", "--name-only"])
    if staged_result.returncode != 0:
        print("COMMIT_CHECK_ERROR", file=sys.stderr)
        print(f"读取暂存区失败：{staged_result.stderr.strip()}", file=sys.stderr)
        return 2

    staged = [line for line in staged_result.stdout.splitlines() if line.strip()]
    if not staged:
        print("NOT_READY")
        print("没有已暂存变更；不要执行 git commit。")
        return 0

    print("READY")
    print("已暂存文件：")
    for name in staged:
        print(name)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
