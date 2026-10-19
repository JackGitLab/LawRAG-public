#!/usr/bin/env python
"""只读核对本地错误档案；使用显式快照比较，不依赖 Git 跟踪。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ERRORS_PATH = Path("docs/ERRORS.md")
HEADING_RE = re.compile(r"^## (E-\d{3})\b.*$", re.MULTILINE)
INPUT_ERROR = 2


class ArchiveError(RuntimeError):
    """归档内容或 Git 基线无法可靠核对。"""


def split_entries(text: str) -> tuple[str, dict[str, str]]:
    """按二级 E-xxx 标题分块，并拒绝重复编号。"""
    matches = list(HEADING_RE.finditer(text))
    preamble = text[: matches[0].start()] if matches else text
    entries: dict[str, str] = {}
    for index, match in enumerate(matches):
        error_id = match.group(1)
        if error_id in entries:
            raise ArchiveError(f"重复错误编号：{error_id}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries[error_id] = text[match.start() : end].rstrip()
    return preamble, entries


def read_disk_archive(repo: Path) -> str:
    path = repo / ERRORS_PATH
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArchiveError(f"无法读取当前归档 {path}: {exc}") from exc


def read_baseline_archive(baseline: Path) -> str:
    try:
        return baseline.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArchiveError(f"无法读取本地基线 {baseline}：{exc}") from exc


def changed_entries(repo: Path, baseline: Path) -> set[str]:
    """比较任务开始前的本地快照与当前磁盘文件。"""
    if baseline.resolve() == (repo / ERRORS_PATH).resolve():
        raise ArchiveError("基线不能是当前错误档案本身，请使用修改前的独立快照")
    baseline_preamble, baseline_entries = split_entries(read_baseline_archive(baseline))
    disk_preamble, disk_entries = split_entries(read_disk_archive(repo))
    if baseline_preamble != disk_preamble:
        raise ArchiveError("检测到无法归属到 E-xxx 条目的归档前言变化")
    all_ids = set(baseline_entries) | set(disk_entries)
    return {
        error_id
        for error_id in all_ids
        if baseline_entries.get(error_id) != disk_entries.get(error_id)
    }


def require_known(requested: list[str], known: set[str]) -> None:
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ArchiveError(f"未知错误编号：{', '.join(unknown)}")


def show_entries(repo: Path, requested: list[str]) -> str:
    _, entries = split_entries(read_disk_archive(repo))
    require_known(requested, set(entries))
    return "\n\n".join(entries[error_id] for error_id in requested)


def check_expected_changes(repo: Path, allowed: list[str], baseline: Path) -> set[str]:
    baseline_text = read_baseline_archive(baseline)
    disk_text = read_disk_archive(repo)
    _, baseline_entries = split_entries(baseline_text)
    _, disk_entries = split_entries(disk_text)
    require_known(allowed, set(baseline_entries) | set(disk_entries))
    # 再调用统一比较函数，确保前言变化等无法归属情况也明确失败。
    actual = changed_entries(repo, baseline)
    unexpected = sorted(actual - set(allowed))
    if unexpected:
        raise ArchiveError(f"检测到允许列表之外的变化：{', '.join(unexpected)}")
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument("--baseline", type=Path, help="修改前的本地错误档案快照；比较时必需")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", nargs="+", metavar="E-XXX")
    group.add_argument("--expect-changed", nargs="+", metavar="E-XXX")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.show:
            print(show_entries(args.repo, args.show))
        else:
            if args.baseline is None:
                raise ArchiveError("--expect-changed 必须提供 --baseline 本地快照")
            actual = check_expected_changes(args.repo, args.expect_changed, args.baseline)
            rendered = ", ".join(sorted(actual)) if actual else "无"
            print(f"归档变化范围通过；实际变化编号：{rendered}")
    except ArchiveError as exc:
        print(f"归档检查失败：{exc}", file=sys.stderr)
        return INPUT_ERROR
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
