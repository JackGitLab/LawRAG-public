#!/usr/bin/env python
"""只检索明确目录的直接普通日志文件，不递归或跟随链接。"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path
from typing import Protocol


NO_MATCH = 1
INPUT_ERROR = 2


class EntryLike(Protocol):
    name: str

    def is_symlink(self) -> bool: ...

    def is_junction(self) -> bool: ...

    def is_file(self) -> bool: ...


def is_direct_regular_file(entry: EntryLike, file_glob: str) -> bool:
    """链接和目录联接一律跳过，再判断是否为匹配的普通文件。"""
    if entry.is_symlink() or entry.is_junction():
        return False
    return entry.is_file() and fnmatch.fnmatch(entry.name, file_glob)


def search_directories(
    pattern: re.Pattern[str], directories: list[Path], file_glob: str = "*.log"
) -> tuple[list[str], list[str]]:
    matches: list[str] = []
    errors: list[str] = []
    for directory in directories:
        if not directory.exists():
            errors.append(f"输入目录不存在：{directory}")
            continue
        if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
            errors.append(f"输入不是可检索的真实目录：{directory}")
            continue
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            errors.append(f"无法列举目录 {directory}: {exc}")
            continue
        for entry in entries:
            try:
                if not is_direct_regular_file(entry, file_glob):
                    continue
                text = entry.read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"无法读取文件 {entry}: {exc}")
                continue
            except UnicodeError as exc:
                errors.append(f"无法按 UTF-8 解码文件 {entry}: {exc}")
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{entry}:{line_number}:{line}")
    return matches, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", help="Python 正则表达式")
    parser.add_argument("directories", nargs="+", type=Path, help="明确的非递归目录")
    parser.add_argument("--glob", default="*.log", dest="file_glob")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pattern = re.compile(args.pattern)
    except re.error as exc:
        print(f"无效正则表达式：{exc}", file=sys.stderr)
        return INPUT_ERROR
    matches, errors = search_directories(pattern, args.directories, args.file_glob)
    for match in matches:
        print(match)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return INPUT_ERROR
    if not matches:
        print("检索成功，但没有匹配。")
        return NO_MATCH
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
