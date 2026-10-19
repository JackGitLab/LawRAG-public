"""check_commit_ready.py 的机制测试：只读、区分 READY/NOT_READY、非仓库明确失败。

全部使用 pytest 临时目录内的临时 Git 仓库，不在真实 LawRAG 仓库创建探针提交。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_commit_ready.py"


def run_ready(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    return tmp_path


def has_any_commit(repo: Path) -> bool:
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    return log.returncode == 0


def test_no_staged_changes_reports_not_ready(git_repo: Path) -> None:
    result = run_ready(git_repo)

    assert result.returncode == 0  # 无暂存变更是正常结果
    assert "NOT_READY" in result.stdout
    assert "没有已暂存变更" in result.stdout
    assert not has_any_commit(git_repo)  # 未产生提交


def test_staged_changes_reports_ready_and_lists_files(git_repo: Path) -> None:
    (git_repo / "a.txt").write_text("a", encoding="utf-8")
    (git_repo / "b.txt").write_text("b", encoding="utf-8")
    subprocess.run(
        ["git", "add", "a.txt", "b.txt"],
        cwd=git_repo, check=True, capture_output=True,
    )

    result = run_ready(git_repo)

    assert result.returncode == 0
    assert "READY" in result.stdout
    assert "已暂存文件" in result.stdout
    assert "a.txt" in result.stdout
    assert "b.txt" in result.stdout
    assert not has_any_commit(git_repo)  # 只检查，不自动提交


def test_non_git_directory_fails_with_distinct_exit_code(tmp_path: Path) -> None:
    result = run_ready(tmp_path)

    assert result.returncode != 0  # 与 NOT_READY(0) 明确区分
    assert "COMMIT_CHECK_ERROR" in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == []  # 不修改目录
