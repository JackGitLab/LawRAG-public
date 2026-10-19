"""scripts/run_tests.py 测试入口的行为测试（R05）。

只校验入口自身的纯函数与合作行为（参数冲突、独立目录、退出码传播），
不依赖真实 pytest 业务结果，因此运行快且隔离。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("run_tests", _SCRIPTS / "run_tests.py")
assert _spec is not None and _spec.loader is not None
run_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_tests)


class _FakeProc:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


@pytest.mark.parametrize(
    "args",
    [["--basetemp", "x"], ["--basetemp=x"], ["-q", "--basetemp", "x"], ["--basetemp="]],
)
def test_basetemp_conflict_is_detected(args) -> None:
    assert run_tests.has_basetemp_conflict(args)


@pytest.mark.parametrize("args", [[], ["-q"], ["tests/test_tools.py"], ["-x", "-v", "tests"]])
def test_no_basetemp_conflict(args) -> None:
    assert not run_tests.has_basetemp_conflict(args)


def test_alloc_tmp_dir_yields_distinct_new_dirs(tmp_path) -> None:
    first = run_tests.alloc_tmp_dir(tmp_path)
    second = run_tests.alloc_tmp_dir(tmp_path)
    assert first != second
    assert first.name.startswith("pytest_tmp_")
    assert second.name.startswith("pytest_tmp_")
    assert first.exists() and second.exists()


def test_main_rejects_basetemp_without_launching_pytest(monkeypatch) -> None:
    launched = []

    def fake_run(*args, **kwargs):  # noqa: ARG001
        launched.append(args)
        return _FakeProc(0)

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    assert run_tests.main(["--basetemp", "x"]) == run_tests.CONFLICT_EXIT
    assert launched == []


def test_main_propagates_exit_code_and_runs_once(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(run_tests, "LOGS_DIR", tmp_path / "logs")
    # CI 等全新检出自检没有 .venv，需替身化解释器路径，否则入口会按设计
    # 返回 NO_VENV_EXIT 而不是传播 pytest 退出码（E-074）。
    monkeypatch.setattr(run_tests, "VENV_PYTHON", Path(sys.executable))

    def fake_run(cmd, **kwargs):  # noqa: ARG001
        calls.append(cmd)
        return _FakeProc(7)

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    assert run_tests.main(["-q", "tests/test_tools.py"]) == 7
    assert len(calls) == 1
    assert "--basetemp" in calls[0]
    assert "-p" in calls[0]
    assert "no:cacheprovider" in calls[0]


def test_build_command_uses_venv_python_and_forwarded_args() -> None:
    passthrough = ["-q", "tests/test_tools.py"]
    cmd = run_tests.build_pytest_command(
        run_tests.VENV_PYTHON, run_tests.LOGS_DIR / "tmp", passthrough
    )
    assert cmd[0] == str(run_tests.VENV_PYTHON)
    assert cmd == [str(run_tests.VENV_PYTHON), "-m", "pytest", "--basetemp",
                   str(run_tests.LOGS_DIR / "tmp"), "-p", "no:cacheprovider", *passthrough]