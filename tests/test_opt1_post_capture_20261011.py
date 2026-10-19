"""OPT1 优化后真实对照复测入口的离线验收（PRE-01～PRE-07）。

不调用真实模型、不启动真实服务：以离线桩穿过包装器的**实际 collect 入口**（不是只测参数解析），
核对实际读取的场景文件、manifest 身份、固定计划与首错停止行为；
测量统计公式、原始时间线/usage/attempts 与报告重建均沿用已验收实现。
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "outputs/live_opt1_20261011/run_capture_opt1.py"
CASES = ROOT / "tests/performance_cases_opt1.json"
OLD_CASES = ROOT / "tests/performance_cases.json"
PRICES = ROOT / "outputs/live_opt1_20261011/pricing.json"

SPEC = importlib.util.spec_from_file_location("opt1_post_capture", WRAPPER)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


@pytest.fixture(autouse=True)
def isolated_business_dirs(tmp_path, monkeypatch):
    """离线采集使用临时源文件，不依赖开发机的索引、数据库或语料目录。"""
    for name in ("CHROMA_SOURCE_DIR", "REAL_DATA_DIR", "REAL_KB_DIR"):
        directory = tmp_path / "source" / name
        directory.mkdir(parents=True)
        (directory / "fixture.txt").write_text(name, encoding="utf-8")
        monkeypatch.setattr(wrapper.perf, name, directory)


def _metrics(status):
    return {
        "status": status,
        "duration_ms": 300.0,
        "first_text_ms": None if status != "ok" else 100.0,
        "done_ms": None if status != "ok" else 200.0,
        "observed_first_text_ms": 100.0,
        "observed_done_ms": 200.0,
        "failure_ms": 50.0 if status != "ok" else None,
        "done_count": 1,
        "text_event_count": 1,
        "notes": [] if status == "ok" else ["连接异常：HTTPError: 401"],
    }


def _stub_request(calls, fail=False):
    counter = {"n": 0}

    def request(base_url, session_id, question, use_web_search, expected_path):
        counter["n"] += 1
        calls.append({"session": session_id, "web": bool(use_web_search), "question": question})
        record = {
            "session_id": session_id,
            "question": question,
            "use_web_search": bool(use_web_search),
            "expected_path": expected_path,
            "http_status": 401 if fail else 200,
            "wall_start": f"2026-10-11T10:00:{counter['n']:02d}.000+00:00",
            "wall_end": f"2026-10-11T10:00:{counter['n']:02d}.500+00:00",
            "metrics": _metrics("failed" if fail else "ok"),
            "event_types": {},
            "server_session_id": None if fail else (session_id or f"s{counter['n']}"),
        }
        if not fail:
            # 原始 SSE 时间线：报告重建与指标重算沿用已验收实现（PRE-06）
            record.update({
                "t_start": 0.0,
                "t_end": 0.3,
                "events": [
                    {"order": 1, "type": "token", "t": 0.1, "text_nonempty": True},
                    {"order": 2, "type": "done", "t": 0.2, "text_nonempty": False},
                ],
                "failure": None,
            })
        return record

    return request


def _prepare_run(tmp_path, monkeypatch, request_stub):
    monkeypatch.setattr(wrapper.perf, "PERF_ROOT", tmp_path)
    monkeypatch.setattr(wrapper.perf, "start_server", lambda run_dir, port: (object(), 0.1))
    monkeypatch.setattr(
        wrapper.perf,
        "stop_server",
        lambda process, port, run_dir: {
            "terminated": True,
            "port_released": True,
            "cleanup_complete": True,  # R1-05 整体判定字段（包装器退出判定依据）
            "method": "stub",
        },
    )
    monkeypatch.setattr(wrapper.perf, "run_one_request", request_stub)


def test_pre01_pre02_pre03_entry_uses_opt1_cases_and_records_identity(tmp_path, monkeypatch):
    """PRE-01/02/03：实际 collect 读取优化后场景；manifest 记录场景与包装器身份；固定计划不变。"""
    calls = []
    _prepare_run(tmp_path, monkeypatch, _stub_request(calls))
    monkeypatch.setattr(sys, "argv", ["run_capture_opt1.py", "--run-id", "opt1-post-stub"])

    assert wrapper.main() == 0

    run_dir = tmp_path / "opt1-post-stub"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    # PRE-01：实际执行的场景来自优化后对照文件，不是旧默认场景
    opt1_cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert manifest["cases_file_sha256"] == hashlib.sha256(CASES.read_bytes()).hexdigest()
    assert manifest["cases_sha256"] == wrapper.perf.cases_sha256(opt1_cases)
    assert manifest["cases_file_sha256"] != hashlib.sha256(OLD_CASES.read_bytes()).hexdigest()
    assert [s["id"] for s in manifest["scenarios"]] == [s["id"] for s in opt1_cases["scenarios"]]

    # PRE-02：manifest 记录实际场景路径、包装器与价格依据（含未核实项）的版本身份
    code_hashes = manifest["code_hashes"]
    assert "tests/performance_cases_opt1.json" in code_hashes
    assert "outputs/live_opt1_20261011/run_capture_opt1.py" in code_hashes
    assert "outputs/live_opt1_20261011/pricing.json" in code_hashes
    assert "outputs/live_baseline_20261011/run_capture.py" in code_hashes  # 复用来源
    assert code_hashes["tests/performance_cases_opt1.json"] == manifest["cases_file_sha256"]
    prices = json.loads(PRICES.read_text(encoding="utf-8"))
    assert manifest["pricing"] == prices
    assert any(entry.get("price_status") == "unknown" for entry in manifest["pricing"])  # 未核实项保留
    assert manifest["warmup_passes"] == 1 and manifest["formal_reps"] == 3
    assert manifest["models"] and manifest["retrieval_strategy"] == "hybrid_rerank"
    assert manifest["isolation"]["snapshot_before"]
    assert manifest["isolation"]["real_dirs_unchanged"] is True
    assert manifest["cleanup"]["terminated"] is True and manifest["cleanup"]["port_released"] is True

    # PRE-03：固定计划 24 业务请求（预热 6 + 正式 18）、16 任务、4 次联网场景；会话规则不变
    requests = wrapper.perf._read_jsonl(run_dir / "requests.jsonl")
    assert len(requests) == 24
    assert sum(1 for r in requests if r["phase"] == "warmup") == 6
    assert sum(1 for r in requests if r["phase"] == "formal") == 18
    assert sum(1 for c in calls if c["web"]) == 4
    assert len(calls) == 24  # 首条预热即冒烟，成功路径不重复冒烟（PRE-04 成功侧）
    tasks = wrapper.perf.evaluate_tasks([], manifest)
    assert len(tasks) == 16  # 预热 4 + 正式 12
    assert sum(1 for t in tasks if t["phase"] == "warmup") == 4
    assert sum(1 for t in tasks if t["phase"] == "formal") == 12
    # 会话规则：每场景每重复首轮不带 session，后续轮复用
    assert sum(1 for c in calls if c["session"] is None) == 16
    assert all(calls[i]["session"] is None for i, r in enumerate(requests) if r["turn_index"] == 1)
    assert all(
        calls[i]["session"] == requests[i - 1]["server_session_id"]
        for i, r in enumerate(requests)
        if r["turn_index"] == 2
    )

    # PRE-06：报告重建与统计沿用已验收实现（同一采集器生成）
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "计划样本" in report
    assert "原始 SSE 时间线" in report or "可重算" in report


def test_pre04_first_failure_saved_then_stops_without_extra_requests(tmp_path, monkeypatch):
    """PRE-04：首错落盘即停止，不产生额外请求；失败样本先落盘，不重试整轮。"""
    calls = []
    _prepare_run(tmp_path, monkeypatch, _stub_request(calls, fail=True))
    monkeypatch.setattr(sys, "argv", ["run_capture_opt1.py", "--run-id", "opt1-post-fail"])

    assert wrapper.main() == 1

    run_dir = tmp_path / "opt1-post-fail"
    assert len(calls) == 1
    rows = wrapper.perf._read_jsonl(run_dir / "requests.jsonl")
    assert len(rows) == 1  # 失败样本已落盘
    assert rows[0]["metrics"]["status"] == "failed"
    stop = json.loads((run_dir / "stop.json").read_text(encoding="utf-8"))
    assert stop["sent"] == 1 and stop["reason"] == "request_failed_or_session_missing"


def test_pre04_missing_session_stops_before_next_turn(tmp_path, monkeypatch):
    """PRE-04：缺 session 同样在下一轮之前停止。"""
    calls = []

    def request(base_url, session_id, question, use_web_search, expected_path):
        calls.append({"session": session_id})
        return {
            "http_status": 200,
            "metrics": _metrics("ok"),
            "expected_path": expected_path,
            "server_session_id": None,
            "session_id": None,
            "wall_start": "2026-10-11T10:00:00.000+00:00",
            "wall_end": "2026-10-11T10:00:05.000+00:00",
            "event_types": {},
        }

    _prepare_run(tmp_path, monkeypatch, request)
    monkeypatch.setattr(sys, "argv", ["run_capture_opt1.py", "--run-id", "opt1-post-nosession"])
    assert wrapper.main() == 1
    assert len(calls) == 1


def test_pre05_old_cases_and_old_evidence_untouched(tmp_path, monkeypatch):
    """PRE-05：旧冻结场景不变，新采集不覆盖既有结果；真实历史由交付哈希核对。"""
    assert hashlib.sha256(OLD_CASES.read_bytes()).hexdigest() == (
        "4e8f5e1cab7380483014a9150e8df99b1498516ee12d10cd05d0b78e53bfaf02"
    )
    # 旧包装器与旧价格快照仍在原位（本入口只读复用，不覆盖）
    assert (ROOT / "outputs/live_baseline_20261011/run_capture.py").exists()
    assert (ROOT / "outputs/live_baseline_20261011/pricing.json").exists()
    old_manifest = tmp_path / "perf-20261011-live1" / "manifest.json"
    old_manifest.parent.mkdir()
    old_manifest.write_text('{"run_id": "perf-20261011-live1"}', encoding="utf-8")
    protected = [
        OLD_CASES,
        ROOT / "outputs/live_baseline_20261011/run_capture.py",
        ROOT / "outputs/live_baseline_20261011/pricing.json",
        old_manifest,
    ]
    before = {path: path.read_bytes() for path in protected}
    _prepare_run(tmp_path, monkeypatch, _stub_request([]))
    monkeypatch.setattr(sys, "argv", ["run_capture_opt1.py", "--run-id", "opt1-preserve-old"])

    assert wrapper.main() == 0
    assert {path: path.read_bytes() for path in protected} == before


def test_pre07_wrapper_is_thin_delegate():
    """PRE-06/07：包装器只做入口编排，统计/重建逻辑来自已验收采集器同一模块实例。"""
    assert wrapper.perf is wrapper.live1.perf  # 同一采集器模块实例
    assert wrapper.guarded_samples is wrapper.live1.guarded_samples  # 已验收首错停止逻辑
    assert wrapper.perf.build_report_md.__module__ == "accepted_perf"
    assert wrapper.perf.derive_metrics.__module__ == "accepted_perf"
    assert wrapper.perf.run_one_request.__module__ == "accepted_perf"


# ---------------------------------------------------------------------------
# E-102 R2-02：taskkill 两条路径的真实 bytes stderr 穿过 cmd_collect 落盘
# ---------------------------------------------------------------------------


class _SerialProc:
    """虚拟进程对象：terminate 拒绝→走父进程 taskkill 回退路径（不触碰真实 PID）。"""

    def __init__(self):
        self.pid = 424242
        self.returncode = None

    def terminate(self):
        raise RuntimeError("terminate denied")

    def wait(self, timeout=None):
        self.returncode = -1
        return self.returncode


def test_r202_taskkill_bytes_stderr_paths_land_in_manifest_and_wrapper_fails(
    tmp_path, monkeypatch
):
    """R2-02：父回退＋后代两条 taskkill 路径的真实 bytes stderr 经 cmd_collect 落盘为文本。

    taskkill 由无害本地 Python 子进程替代（真实 subprocess 管道类型：非空 bytes stderr、
    返回码 5）；不执行真实 taskkill、无业务请求、不触碰无关 PID。
    """
    import subprocess as sp

    real_run = sp.run
    taskkill_calls = []

    def routed_run(args, capture_output=False, timeout=None, **kwargs):
        if args and args[0] == "taskkill":
            taskkill_calls.append(list(args))
            # 保留实际 subprocess 管道类型：不带 text=True → stderr 为 bytes
            return real_run(
                [sys.executable, "-c", "import sys; sys.stderr.write('taskkill-boom'); sys.exit(5)"],
                capture_output=True,
                timeout=30,
            )
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(wrapper.perf, "PERF_ROOT", tmp_path)
    monkeypatch.setattr(wrapper.perf, "start_server", lambda run_dir, port: (_SerialProc(), 0.1))
    monkeypatch.setattr(wrapper.perf.subprocess, "run", routed_run)
    monkeypatch.setattr(wrapper.perf, "_descendant_pids", lambda _pid: ([112233], None))
    monkeypatch.setattr(wrapper.perf, "_pid_alive", lambda _pid: (True, "still_active"))
    monkeypatch.setattr(sys, "argv", ["run_capture_opt1.py", "--run-id", "e102r2-serial"])
    calls = []
    monkeypatch.setattr(wrapper.perf, "run_one_request", _stub_request(calls))

    rc = wrapper.main()

    assert rc == 1  # 当前包装器仍返回失败（cleanup_complete=false）
    run_dir = tmp_path / "e102r2-serial"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))  # 可解析
    cleanup = manifest["cleanup"]

    # 两条 taskkill 路径都被实际调用（父 /T /F 与后代 /F），参数正确
    assert taskkill_calls == [
        ["taskkill", "/PID", "424242", "/T", "/F"],
        ["taskkill", "/PID", "112233", "/F"],
    ]
    assert [c["args"] for c in cleanup["kill_commands"]] == taskkill_calls
    for command in cleanup["kill_commands"]:
        assert command["returncode"] == 5
        assert isinstance(command["stderr"], str) and "taskkill-boom" in command["stderr"]

    # 父进程回退事实与后代记账：失败不冒充成功
    assert cleanup["method"] == "taskkill"
    assert cleanup["terminated"] is False  # 非零返回不得变成成功
    assert "taskkill_exit_5" in cleanup["error"]
    entry = cleanup["descendant_results"][0]
    assert isinstance(entry["command_stderr"], str) and "taskkill-boom" in entry["command_stderr"]
    assert entry["outcome"] == "still_alive" and cleanup["descendants_killed"] == []
    assert cleanup["cleanup_complete"] is False

    # 已采集记录与首次清理证据保留
    requests = wrapper.perf._read_jsonl(run_dir / "requests.jsonl")
    assert len(requests) == 24
    assert cleanup["port_initial_check"]  # 首次检查保留
