"""归档范围核对与非递归日志检索助手的定向测试。"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


archive = _load_script("check_error_archive")
log_search = _load_script("search_log_files")


def _entry(error_id: str, value: str) -> str:
    return f"## {error_id} 示例\n\n- 值：{value}\n"


def _repo_with_archive(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "ERRORS.md").write_text(
        "# 错误档案\n\n" + _entry("E-001", "a") + "\n" + _entry("E-002", "b"),
        encoding="utf-8",
    )
    (repo / "before.md").write_bytes((repo / "docs" / "ERRORS.md").read_bytes())
    return repo


def test_show_uses_current_disk_entry(tmp_path) -> None:
    repo = _repo_with_archive(tmp_path)
    path = repo / "docs" / "ERRORS.md"
    path.write_text(path.read_text(encoding="utf-8").replace("值：a", "值：disk"), encoding="utf-8")
    shown = archive.show_entries(repo, ["E-001"])
    assert "值：disk" in shown


def test_multiple_disk_changes_are_detected_without_git(tmp_path) -> None:
    repo = _repo_with_archive(tmp_path)
    path = repo / "docs" / "ERRORS.md"
    path.write_text(path.read_text(encoding="utf-8").replace("值：a", "值：first"), encoding="utf-8")
    path.write_text(path.read_text(encoding="utf-8").replace("值：b", "值：second"), encoding="utf-8")
    assert archive.check_expected_changes(repo, ["E-001", "E-002"], repo / "before.md") == {"E-001", "E-002"}


def test_allowed_ids_are_a_superset_not_a_required_set(tmp_path) -> None:
    repo = _repo_with_archive(tmp_path)
    path = repo / "docs" / "ERRORS.md"
    path.write_text(path.read_text(encoding="utf-8").replace("值：a", "值：changed"), encoding="utf-8")
    assert archive.check_expected_changes(repo, ["E-001", "E-002"], repo / "before.md") == {"E-001"}


def test_added_and_deleted_entries_are_detected(tmp_path) -> None:
    repo = _repo_with_archive(tmp_path)
    path = repo / "docs" / "ERRORS.md"
    path.write_text("# 错误档案\n\n" + _entry("E-002", "b") + "\n" + _entry("E-003", "c"), encoding="utf-8")
    assert archive.check_expected_changes(repo, ["E-001", "E-003"], repo / "before.md") == {"E-001", "E-003"}


def test_duplicate_unknown_unowned_and_unexpected_changes_fail(tmp_path) -> None:
    with pytest.raises(archive.ArchiveError, match="重复错误编号"):
        archive.split_entries(_entry("E-001", "a") + _entry("E-001", "b"))

    repo = _repo_with_archive(tmp_path)
    with pytest.raises(archive.ArchiveError, match="未知错误编号"):
        archive.show_entries(repo, ["E-999"])

    path = repo / "docs" / "ERRORS.md"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("# 错误档案", "# 已改前言"), encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="无法归属"):
        archive.changed_entries(repo, repo / "before.md")

    path.write_text(original.replace("值：a", "值：unexpected"), encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="允许列表之外"):
        archive.check_expected_changes(repo, ["E-002"], repo / "before.md")


def test_missing_or_same_baseline_is_not_treated_as_no_changes(tmp_path) -> None:
    repo = _repo_with_archive(tmp_path)
    with pytest.raises(archive.ArchiveError, match="无法读取本地基线"):
        archive.changed_entries(repo, repo / "missing.md")
    with pytest.raises(archive.ArchiveError, match="不能是当前错误档案"):
        archive.changed_entries(repo, repo / "docs" / "ERRORS.md")


def test_cli_requires_snapshot_and_supports_untracked_archive(tmp_path, capsys) -> None:
    repo = _repo_with_archive(tmp_path)
    args = ["--repo", str(repo), "--expect-changed", "E-001"]
    assert archive.main(args) == archive.INPUT_ERROR
    assert "--baseline" in capsys.readouterr().err
    path = repo / "docs" / "ERRORS.md"
    path.write_text(path.read_text(encoding="utf-8").replace("值：a", "值：new"), encoding="utf-8")
    assert archive.main(args + ["--baseline", str(repo / "before.md")]) == 0
    assert "E-001" in capsys.readouterr().out
    assert not (repo / ".git").exists()


def test_missing_local_archive_is_reported(tmp_path, capsys) -> None:
    assert archive.main(["--repo", str(tmp_path), "--show", "E-001"]) == archive.INPUT_ERROR
    assert "无法读取当前归档" in capsys.readouterr().err


def test_log_search_is_direct_and_distinguishes_no_match(tmp_path) -> None:
    (tmp_path / "direct.log").write_text("needle\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.log").write_text("needle\n", encoding="utf-8")
    matches, errors = log_search.search_directories(re.compile("needle"), [tmp_path])
    assert errors == []
    assert len(matches) == 1
    assert "direct.log:1:needle" in matches[0]
    assert log_search.main(["missing", str(tmp_path)]) == log_search.NO_MATCH


def test_log_search_reports_missing_and_simulated_read_failure(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "missing"
    assert log_search.main(["x", str(missing)]) == log_search.INPUT_ERROR

    log_file = tmp_path / "blocked.log"
    log_file.write_text("x", encoding="utf-8")
    original = Path.read_text

    def fail_selected(self, *args, **kwargs):
        if self == log_file:
            raise PermissionError("simulated")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_selected)
    matches, errors = log_search.search_directories(re.compile("x"), [tmp_path])
    assert matches == []
    assert len(errors) == 1 and "simulated" in errors[0]


class _FakeEntry:
    name = "linked.log"

    def __init__(self, *, symlink=False, junction=False, file=True) -> None:
        self._symlink = symlink
        self._junction = junction
        self._file = file

    def is_symlink(self) -> bool:
        return self._symlink

    def is_junction(self) -> bool:
        return self._junction

    def is_file(self) -> bool:
        return self._file


@pytest.mark.parametrize("entry", [_FakeEntry(symlink=True), _FakeEntry(junction=True)])
def test_log_search_rejects_links_and_junctions(entry) -> None:
    assert not log_search.is_direct_regular_file(entry, "*.log")
