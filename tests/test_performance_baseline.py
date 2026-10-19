"""性能基线测量实现（tests/run_performance_baseline.py）的行为测试（R1 返工版）。

全部离线：真实模型、搜索、索引与用户数据目录均不触碰；
计时公式用可控时钟（事件时间由用例给出）验证，不靠 sleep 制造时长。
覆盖 R1-01～R1-07 回归（对应 outputs/STAGE3_STEP1_ACCEPTANCE_20261010.md F-01～F-07）。
"""

import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

import app.db as db_module
import app.graph as graph_module
from app import ask, observability

SCRIPT_PATH = Path(__file__).with_name("run_performance_baseline.py")
CASES_PATH = Path(__file__).with_name("performance_cases.json")

SPEC = spec_from_file_location("performance_baseline_runner", SCRIPT_PATH)
assert SPEC and SPEC.loader
perf = module_from_spec(SPEC)
SPEC.loader.exec_module(perf)


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def make_expected(**overrides) -> dict:
    expected = {
        "nodes": ["prepare", "retrieve", "route", "generate"],
        "llm_calls": ["generate"],
        "tool_executions": [],
        "tool_inits": [],
        "retrieval_steps": ["vector", "bm25", "rerank"],
        "answer_mode": "generated",
    }
    expected.update(overrides)
    return expected


def make_raw(first_text=0.1, done=0.4, failure=None, extra_events=()) -> dict:
    events = [
        {"order": 1, "type": "session", "t": 0.01, "text_nonempty": False},
        {"order": 2, "type": "token", "t": first_text, "text_nonempty": True},
        {"order": 3, "type": "done", "t": done, "text_nonempty": False},
    ]
    events.extend(extra_events)
    return {"t_start": 0.0, "t_end": round(done + 0.1, 6), "events": events, "failure": failure}


def make_metrics(**overrides) -> dict:
    metrics = {
        "status": "ok",
        "duration_ms": 500.0,
        "first_text_ms": 100.0,
        "done_ms": 400.0,
        "observed_first_text_ms": 100.0,
        "observed_done_ms": 400.0,
        "failure_ms": None,
        "done_count": 1,
        "text_event_count": 1,
        "notes": [],
    }
    metrics.update(overrides)
    return metrics


def make_request(**overrides) -> dict:
    record = {
        "request_id": "S-1-formal1-t1",
        "scenario_id": "S-1",
        "scenario_name": "场景一",
        "phase": "formal",
        "rep": 1,
        "turn_index": 1,
        "session_id": "sess-1",
        "question": "测试问题",
        "use_web_search": False,
        "expected_path": make_expected(),
        "http_status": 200,
        "wall_start": "2026-10-10T10:00:00.000+00:00",
        "wall_end": "2026-10-10T10:00:05.000+00:00",
        "metrics": make_metrics(),
    }
    record.update(overrides)
    return record


def make_obs(**overrides) -> dict:
    record = {
        "seq": 1,
        "ts_utc": "2026-10-10T10:00:01.000+00:00",
        "kind": "node",
        "name": "generate",
        "duration_ms": 100.0,
        "ok": True,
    }
    record.update(overrides)
    return record


def make_manifest(**overrides) -> dict:
    manifest = {
        "run_id": "test-run",
        "git_head": "abc123",
        "git_dirty": False,
        "code_hashes": {"app/graph.py": "hash"},
        "cases_sha256": "caseshash",
        "cases_frozen_date": "2026-10-10",
        "scenarios": [{"id": "S-1", "name": "场景一", "turns": 1}],
        "warmup_passes": 1,
        "formal_reps": 3,
        "models": {"chat": "deepseek-chat", "router": "deepseek-flash",
                   "embedding": "BAAI/bge-m3", "rerank": "BAAI/bge-reranker-v2-m3"},
        "retrieval_strategy": "hybrid_rerank",
        "langfuse": "disabled",
        "started_utc": "2026-10-10T09:59:00.000+00:00",
        "finished_utc": "2026-10-10T10:10:00.000+00:00",
        "server_start_seconds": 3.2,
        "pricing": copy.deepcopy(perf.PRICING),
    }
    manifest.update(overrides)
    return manifest


# ---------------------------------------------------------------------------
# 场景冻结
# ---------------------------------------------------------------------------


def test_frozen_cases_validate_clean():
    cases = perf.load_cases(CASES_PATH)
    assert perf.validate_cases(cases) == []
    assert len(cases["scenarios"]) == 4
    assert sum(len(scenario["turns"]) for scenario in cases["scenarios"]) == 6
    assert perf.cases_sha256(cases) == perf.cases_sha256(json.loads(json.dumps(cases)))


def test_validate_cases_reports_missing_fields():
    problems = perf.validate_cases({"format": "bad", "scenarios": [{"id": "S", "turns": [{}]}]})
    assert any("format" in problem for problem in problems)
    assert any("expected_path" in problem for problem in problems)


# ---------------------------------------------------------------------------
# R1-01 失败请求不进成功统计（六种事件顺序 + 正常对照）
# ---------------------------------------------------------------------------


def test_r101_token_error_done_is_failed_observed_kept():
    events = [
        {"t": 0.1, "event": {"type": "token", "text": "a"}},
        {"t": 0.2, "event": {"type": "error", "status": 502}},
        {"t": 0.3, "event": {"type": "done"}},
    ]
    metrics = perf.compute_request_metrics(events, 0.0, 0.4)
    assert metrics["status"] == "failed"
    assert metrics["first_text_ms"] is None and metrics["done_ms"] is None  # 不进成功统计
    assert metrics["observed_first_text_ms"] == 100.0 and metrics["observed_done_ms"] == 300.0
    assert metrics["failure_ms"] == 200.0


def test_r101_token_done_then_connection_failure_is_failed():
    events = [
        {"t": 0.1, "event": {"type": "token", "text": "a"}},
        {"t": 0.2, "event": {"type": "done"}},
    ]
    metrics = perf.compute_request_metrics(events, 0.0, 0.3, {"t": 0.25, "error": "ConnectionResetError"})
    assert metrics["status"] == "failed"
    assert metrics["done_ms"] is None and metrics["observed_done_ms"] == 200.0
    assert any("连接异常" in note for note in metrics["notes"])


def test_r101_duplicate_done_is_failed():
    events = [
        {"t": 0.1, "event": {"type": "token", "text": "a"}},
        {"t": 0.2, "event": {"type": "done"}},
        {"t": 0.3, "event": {"type": "done"}},
    ]
    metrics = perf.compute_request_metrics(events, 0.0, 0.4)
    assert metrics["status"] == "failed"
    assert metrics["first_text_ms"] is None and metrics["done_ms"] is None
    assert any("done 事件数 2 > 1" in note for note in metrics["notes"])


def test_r101_missing_done_is_failed():
    metrics = perf.compute_request_metrics(
        [{"t": 0.1, "event": {"type": "token", "text": "a"}}], 0.0, 0.2
    )
    assert metrics["status"] == "failed"
    assert metrics["done_ms"] is None and metrics["observed_first_text_ms"] == 100.0


def test_r101_done_before_token_is_failed():
    events = [
        {"t": 0.1, "event": {"type": "done"}},
        {"t": 0.2, "event": {"type": "token", "text": "late"}},
    ]
    metrics = perf.compute_request_metrics(events, 0.0, 0.3)
    assert metrics["status"] == "failed"  # 异常事件顺序必须影响有效性，不能只写备注
    assert metrics["first_text_ms"] is None and metrics["done_ms"] is None
    assert any("done 之后仍有非空文本" in note for note in metrics["notes"])


def test_r101_normal_token_done_is_ok():
    events = [
        {"t": 1.0, "event": {"type": "session", "session_id": "s1"}},
        {"t": 2.5, "event": {"type": "token", "text": "第一段"}},
        {"t": 4.0, "event": {"type": "done", "citations": []}},
    ]
    metrics = perf.compute_request_metrics(events, 1.0, 4.5)
    assert metrics["status"] == "ok"
    assert metrics["first_text_ms"] == 1500.0 and metrics["done_ms"] == 3000.0  # session 不算首字
    assert metrics["failure_ms"] is None


def test_r101_failed_sample_reduces_success_n_keeps_planned_denominator():
    """注入失败样本：成功统计 n 减少、失败/缺失计数增加，计划分母不变。"""
    raw_ok = make_raw()
    requests = [
        make_request(request_id=f"S-1-formal{rep}-t1", rep=rep,
                     wall_start=f"2026-10-10T10:0{rep}:00.000+00:00",
                     wall_end=f"2026-10-10T10:0{rep}:05.000+00:00",
                     events=raw_ok["events"], t_start=raw_ok["t_start"], t_end=raw_ok["t_end"],
                     failure=None)
        for rep in (1, 2, 3)
    ]
    manifest = make_manifest()
    report = perf.build_report_md(copy.deepcopy(requests), [], manifest)
    assert "| S-1 T1 | 首段可见回答耗时(ms) | 3 | 3 | 0 |" in report

    failed_events = [
        {"order": 1, "type": "token", "t": 0.1, "text_nonempty": True},
        {"order": 2, "type": "error", "t": 0.2, "text_nonempty": False},
        {"order": 3, "type": "done", "t": 0.3, "text_nonempty": False},
    ]
    injected = copy.deepcopy(requests)
    injected[0]["events"] = failed_events
    injected[0]["t_end"] = 0.4
    report2 = perf.build_report_md(injected, [], manifest)
    # 从原始时间线重算：该样本变失败；成功 n 3→2、失败 0→1、计划仍为 3
    assert "| S-1 T1 | 首段可见回答耗时(ms) | 2 | 3 | 1 |" in report2
    assert "计划样本：预热 1 + 正式 3 = 4" in report2
    assert "成功 2，失败 1" in report2  # 成功统计不含失败样本
    assert "成功率 2/4" in report2  # 分母（计划 4）不随失败减少
    assert "S-1-formal1-t1：收到错误事件" in report2  # 原因可追溯


def test_r101_multi_turn_task_requires_all_turns_success_and_path():
    manifest = make_manifest(scenarios=[{"id": "M", "name": "两轮", "turns": 2}])
    ok_raw = make_raw()
    empty_expected = make_expected(nodes=[], llm_calls=[], tool_executions=[],
                                   tool_inits=[], retrieval_steps=[])
    requests = [
        make_request(request_id="M-formal1-t1", scenario_id="M", rep=1, turn_index=1,
                     expected_path=empty_expected,
                     wall_start="2026-10-10T10:00:00.000+00:00", wall_end="2026-10-10T10:00:05.000+00:00",
                     events=ok_raw["events"], t_start=0.0, t_end=ok_raw["t_end"], failure=None),
        make_request(request_id="M-formal1-t2", scenario_id="M", rep=1, turn_index=2,
                     expected_path=empty_expected,
                     wall_start="2026-10-10T10:00:05.000+00:00", wall_end="2026-10-10T10:00:09.000+00:00",
                     events=[{"order": 1, "type": "token", "t": 0.1, "text_nonempty": True}],
                     t_start=0.0, t_end=0.2, failure=None),  # 缺 done → 失败
        make_request(request_id="M-formal2-t1", scenario_id="M", rep=2, turn_index=1,
                     expected_path=empty_expected,
                     wall_start="2026-10-10T10:01:00.000+00:00", wall_end="2026-10-10T10:01:05.000+00:00",
                     events=ok_raw["events"], t_start=0.0, t_end=ok_raw["t_end"], failure=None),
        make_request(request_id="M-formal2-t2", scenario_id="M", rep=2, turn_index=2,
                     expected_path=empty_expected,
                     wall_start="2026-10-10T10:01:05.000+00:00", wall_end="2026-10-10T10:01:09.000+00:00",
                     events=ok_raw["events"], t_start=0.0, t_end=ok_raw["t_end"], failure=None),
    ]
    annotated, _ = perf.annotate_requests(requests, [])
    tasks = perf.evaluate_tasks(annotated, manifest)
    by_id = {t["task_id"]: t for t in tasks}
    assert by_id["M-formal1"]["complete_success"] is False  # 一轮失败 → 非完整成功任务
    assert by_id["M-formal1"]["problems"]
    assert by_id["M-formal2"]["complete_success"] is True  # 两轮都成功才算完整
    report = perf.build_report_md(copy.deepcopy(requests), [], manifest)
    # 部分成功轮次不相加为完整场景耗时：M 场景耗时只含 formal2 一个完整任务
    assert "未计入不完整任务：['M-formal1']" in report


# ---------------------------------------------------------------------------
# R1-02 重排尝试次数（真实 rerank_candidates + 离线网络桩）
# ---------------------------------------------------------------------------


class FakeRerankResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return b'{"results": []}'


@pytest.mark.parametrize(
    "failures,expected_attempts,expected_ok",
    [(0, 1, True), (1, 2, True), (2, 2, False)],
)
def test_r102_rerank_attempts_recorded_via_real_call_path(
    monkeypatch, tmp_path: Path, failures, expected_attempts, expected_ok
):
    obs_path = tmp_path / "observations.jsonl"
    effects = [ConnectionError("INJECTED OFFLINE RETRY")] * failures + (
        [FakeRerankResponse()] if failures < 2 else []
    )
    monkeypatch.setenv("LAWRAG_PERF_OBS_FILE", str(obs_path))
    monkeypatch.setattr(ask, "require_env", lambda name: "OFFLINE_DUMMY")
    calls = []

    def fake_urlopen(*args, **kwargs):
        calls.append(1)
        effect = effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    monkeypatch.setattr(ask, "urlopen", fake_urlopen)
    monkeypatch.setattr(ask.time, "sleep", lambda *_: None)
    try:
        ask.rerank_candidates("offline", [], 1)
        returned_ok = True
    except Exception:  # noqa: BLE001
        returned_ok = False
    assert returned_ok is expected_ok
    assert len(calls) == expected_attempts  # 实际网络尝试次数
    record = perf.load_observations(obs_path)[0]
    assert record["kind"] == "retrieval" and record["name"] == "rerank"
    assert record["attempts"] == expected_attempts
    assert record["ok"] is expected_ok


def test_r102_missing_attempts_marked_unknown_not_zero():
    old_records = [
        make_obs(seq=1, kind="retrieval", name="rerank", duration_ms=10.0,
                 ts_utc="2026-10-10T10:00:01.000+00:00", attempts=None),
        make_obs(seq=2, kind="retrieval", name="rerank", duration_ms=12.0,
                 ts_utc="2026-10-10T10:00:02.000+00:00"),
    ]
    bucket = perf.aggregate_observations(old_records)["retrieval"]["rerank"]
    assert bucket["attempts"] == 0  # 有字段的合计（无）
    assert bucket["attempts_unknown"] == 2  # 缺字段单列 UNKNOWN，不当 0 次
    # 旧数据（如 r2）观测归属于正式请求：报告必须标 UNKNOWN 而非 0
    request = make_request(
        events=make_raw()["events"], t_start=0.0, t_end=0.5, failure=None,
        expected_path=make_expected(retrieval_steps=["rerank"], nodes=[], llm_calls=[]),
    )
    report = perf.build_report_md([request], old_records, make_manifest())
    assert "UNKNOWN（2 条记录无 attempts 字段）" in report
    assert "无重试" not in report


# ---------------------------------------------------------------------------
# R1-03 只用冻结价格 + report 不回写原始文件
# ---------------------------------------------------------------------------


def _dataset_with_llm():
    requests = [
        make_request(request_id="S-1-formal1-t1", rep=1,
                     wall_start="2026-10-10T10:00:00.000+00:00", wall_end="2026-10-10T10:00:05.000+00:00"),
        make_request(request_id="S-1-formal2-t1", rep=2,
                     wall_start="2026-10-10T10:01:00.000+00:00", wall_end="2026-10-10T10:01:05.000+00:00"),
    ]
    observations = [
        make_obs(seq=1, kind="llm", name="generate", model="deepseek-chat", duration_ms=100.0,
                 usage={"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 200}),
    ]
    return requests, observations


def test_r103_uses_frozen_pricing_not_current_constant():
    requests, observations = _dataset_with_llm()
    manifest = make_manifest()
    report = perf.build_report_md(requests, observations, manifest)
    altered_global = copy.deepcopy(perf.PRICING)
    altered_global[0]["output_per_m"] = 999.0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(perf, "PRICING", altered_global)
        global_report = perf.build_report_md(requests, observations, manifest)
    assert global_report == report  # 当前代码默认价格不影响历史重建


def test_r103_frozen_pricing_change_changes_report():
    requests, observations = _dataset_with_llm()
    manifest = make_manifest()
    report = perf.build_report_md(requests, observations, manifest)
    altered_manifest = make_manifest()
    altered_manifest["pricing"][0]["output_per_m"] *= 100
    altered_report = perf.build_report_md(requests, observations, altered_manifest)
    assert altered_report != report  # 冻结副本价格变动 → 费用按预期变化


def test_r103_missing_frozen_pricing_marks_unknown():
    requests, observations = _dataset_with_llm()
    report = perf.build_report_md(requests, observations, make_manifest(pricing=None))
    assert "缺少冻结价格" in report
    assert "全部费用未知" in report
    # 不得回退到当前默认价格算出金额
    assert "0.001804" not in report and "CNY 0.00" not in report


def test_r103_report_cmd_writes_new_dir_and_never_rewrites_run_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perf, "PERF_ROOT", tmp_path)
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    requests, observations = _dataset_with_llm()
    raw_lines = [json.dumps(r, ensure_ascii=False, default=str) for r in requests]
    (run_dir / "requests.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(make_manifest(), ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "observations.jsonl").write_text("", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    output_dir = tmp_path / "derived-1"
    rc = perf.main(["report", "--run-id", "run-a", "--output-dir", str(output_dir)])
    assert rc == 0
    after = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    assert after == before  # 原始 requests.jsonl / manifest / observations 未被回写
    assert (output_dir / "report.md").exists()
    assert (output_dir / "requests_derived.jsonl").exists()
    # 派生输出可再次同输入重建且字节一致
    output_dir2 = tmp_path / "derived-2"
    assert perf.main(["report", "--run-id", "run-a", "--output-dir", str(output_dir2)]) == 0
    assert (output_dir / "report.md").read_bytes() == (output_dir2 / "report.md").read_bytes()


# ---------------------------------------------------------------------------
# R1-04 usage 字段级未知、费用范围与任务口径
# ---------------------------------------------------------------------------


def _llm_record(usage, **overrides):
    record = make_obs(seq=1, kind="llm", name="generate", model="deepseek-chat",
                      duration_ms=1.0, usage=usage)
    record.update(overrides)
    return record


def test_r104_all_usage_none_is_unknown_not_zero():
    cost = perf.compute_cost(
        perf.aggregate_observations([_llm_record({"input_tokens": None, "output_tokens": None,
                                                  "cache_read_tokens": None})]),
        perf.PRICING,
    )
    assert cost["known_total"] == 0.0
    assert cost["items"][0]["calls_costed"] == 0
    assert cost["usage_field_unknowns"] == {"input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 1}


def test_r104_single_field_missing_stays_unknown():
    for field in ("input_tokens", "output_tokens", "cache_read_tokens"):
        usage = {"input_tokens": 5, "output_tokens": 5, "cache_read_tokens": 0}
        usage[field] = None
        cost = perf.compute_cost(perf.aggregate_observations([_llm_record(usage)]), perf.PRICING)
        assert cost["items"][0]["calls_costed"] == 0, field
        assert cost["usage_field_unknowns"][field] == 1, field
        assert cost["known_total"] == 0.0
    # 缓存拆分缺失：不得当零命中出“完整费用”，只给标注上界
    usage = {"input_tokens": 5, "output_tokens": 5, "cache_read_tokens": None}
    cost = perf.compute_cost(perf.aggregate_observations([_llm_record(usage)]), perf.PRICING)
    assert cost["items"][0]["estimated_cost"] is None
    assert cost["items"][0]["assumed_upper_bound"] is not None
    assert "上界假设" in cost["items"][0]["assumed_upper_bound_note"]


def test_r104_missing_price_marks_unknown():
    cost = perf.compute_cost(
        perf.aggregate_observations([_llm_record({"input_tokens": 5, "output_tokens": 5,
                                                  "cache_read_tokens": 0})]),
        [{"model": "m"}],
    )
    assert cost["known_total"] == 0.0
    assert any("单价" in entry for entry in cost["unknowns"])


def test_r104_task_unit_and_per_success_task_cost():
    raw = make_raw()
    manifest = make_manifest(scenarios=[{"id": "M", "name": "两轮", "turns": 2}])
    empty_expected = make_expected(nodes=[], llm_calls=[], tool_executions=[],
                                   tool_inits=[], retrieval_steps=[])
    requests = [
        make_request(request_id="M-formal1-t1", scenario_id="M", rep=1, turn_index=1,
                     expected_path=empty_expected,
                     wall_start="2026-10-10T10:00:00.000+00:00", wall_end="2026-10-10T10:00:05.000+00:00",
                     events=raw["events"], t_start=0.0, t_end=raw["t_end"], failure=None),
        make_request(request_id="M-formal1-t2", scenario_id="M", rep=1, turn_index=2,
                     expected_path=empty_expected,
                     wall_start="2026-10-10T10:00:05.000+00:00", wall_end="2026-10-10T10:00:09.000+00:00",
                     events=raw["events"], t_start=0.0, t_end=raw["t_end"], failure=None),
    ]
    annotated, _ = perf.annotate_requests(requests, [])
    tasks = perf.evaluate_tasks(annotated, manifest)
    # 两轮场景一次执行 = 一个任务（R2 起 evaluate_tasks 另含冻结计划中的未执行任务，
    # 接口适配：只统计实际执行的任务，断言语义不变）
    executed_tasks = [t for t in tasks if t["executed"]]
    assert len(executed_tasks) == 1
    split = {"warmup": [], "formal": [
        make_obs(seq=1, ts_utc="2026-10-10T10:00:01.000+00:00", kind="llm", name="generate",
                 model="deepseek-chat", duration_ms=1.0,
                 usage={"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 200}),
    ], "unassigned": []}
    costs = perf.compute_scoped_costs(split, perf.PRICING, tasks)
    assert costs["formal_complete_success_tasks"] == 1
    # (800/1M*1 + 200/1M*0.02 + 500/1M*2) = 0.001804；每成功任务 = 该值 / 1
    assert costs["per_success_task_formal"] == pytest.approx(0.001804, abs=1e-9)


def test_r104_all_failed_tasks_cost_na_and_failed_attempts_kept_in_numerator():
    manifest = make_manifest()
    failed_raw_events = [
        {"order": 1, "type": "token", "t": 0.1, "text_nonempty": True},
        {"order": 2, "type": "error", "t": 0.2, "text_nonempty": False},
    ]
    requests = [
        make_request(request_id="S-1-formal1-t1", rep=1,
                     wall_start="2026-10-10T10:00:00.000+00:00", wall_end="2026-10-10T10:00:05.000+00:00",
                     events=failed_raw_events, t_start=0.0, t_end=0.3, failure=None),
    ]
    annotated, _ = perf.annotate_requests(requests, [])
    tasks = perf.evaluate_tasks(annotated, manifest)
    assert not tasks[0]["complete_success"]
    split = {"warmup": [], "formal": [
        make_obs(seq=1, ts_utc="2026-10-10T10:00:01.000+00:00", kind="llm", name="generate",
                 model="deepseek-chat", duration_ms=1.0,
                 usage={"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 200}),
    ], "unassigned": []}
    costs = perf.compute_scoped_costs(split, perf.PRICING, tasks)
    assert costs["per_success_task_formal"] == "N/A"  # 全失败分母为 0
    assert costs["formal"]["known_total"] == pytest.approx(0.001804, abs=1e-9)  # 失败尝试已发生费用保留在分子


def test_r104_scoped_costs_warmup_formal_whole_run():
    warm = make_obs(seq=1, ts_utc="2026-10-10T10:00:01.000+00:00", kind="llm", name="generate",
                    model="deepseek-chat", duration_ms=1.0,
                    usage={"input_tokens": 100, "output_tokens": 100, "cache_read_tokens": 0})
    formal = make_obs(seq=2, ts_utc="2026-10-10T10:01:01.000+00:00", kind="llm", name="generate",
                      model="deepseek-chat", duration_ms=1.0,
                      usage={"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 200})
    split = {"warmup": [warm], "formal": [formal], "unassigned": []}
    costs = perf.compute_scoped_costs(split, perf.PRICING, [])
    assert costs["warmup"]["known_total"] == pytest.approx(0.0003, abs=1e-9)
    assert costs["formal"]["known_total"] == pytest.approx(0.001804, abs=1e-9)
    assert costs["whole_run"]["known_total"] == pytest.approx(0.002104, abs=1e-9)
    assert costs["per_success_task_formal"] == "N/A"  # 无任务记录 → 无成功任务


# ---------------------------------------------------------------------------
# R1-05 预热与正式彻底分离
# ---------------------------------------------------------------------------


def _phase_dataset():
    raw = make_raw()
    requests = []
    for phase, reps, base in (("warmup", 1, 0), ("formal", 3, 10)):
        for rep in range(1, reps + 1):
            requests.append(
                make_request(
                    request_id=f"S-1-{phase}{rep}-t1", phase=phase, rep=rep,
                    wall_start=f"2026-10-10T10:{base + rep:02d}:00.000+00:00",
                    wall_end=f"2026-10-10T10:{base + rep:02d}:05.000+00:00",
                    events=raw["events"], t_start=0.0, t_end=raw["t_end"], failure=None,
                    metrics=make_metrics(first_text_ms=100.0 + rep, done_ms=400.0 + rep,
                                         observed_first_text_ms=100.0 + rep, observed_done_ms=400.0 + rep),
                )
            )
    observations = []
    seq = 0
    for phase, reps, base in (("warmup", 1, 0), ("formal", 3, 10)):
        for rep in range(1, reps + 1):
            for kind, name, dur in (("node", "generate", 1000000.0 if phase == "warmup" else 100.0),
                                    ("llm", "generate", 50.0)):
                seq += 1
                obs = make_obs(seq=seq, kind=kind, name=name, duration_ms=dur,
                               ts_utc=f"2026-10-10T10:{base + rep:02d}:01.000+00:00")
                if kind == "llm":
                    obs["model"] = "deepseek-chat"
                    obs["usage"] = {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0}
                observations.append(obs)
    return requests, observations


def test_r105_warmup_giant_values_do_not_change_formal_stats_or_bottleneck():
    requests, observations = _phase_dataset()
    manifest = make_manifest()
    report = perf.build_report_md(requests, observations, manifest)
    # 预热 generate 节点耗时 1000000ms：正式统计/瓶颈不含它
    assert "| generate | 3 | 0 | 300.0 |" in report  # 5.1 正式节点表（3 次 × 100ms）
    bottleneck_lines = [ln for ln in report.splitlines() if ln.startswith("| generate | 100.0%")]
    assert bottleneck_lines  # 瓶颈占比只按正式样本
    sec_51 = report.split("### 5.1")[1].split("### 5.2")[0]
    sec_52 = report.split("### 5.2")[1].split("### 5.3")[0]
    assert "1000000" not in sec_51  # 正式表无预热数值
    assert "1000000" in sec_52  # 预热单列可见
    assert "### 5.2 预热样本" in report
    # 调用审计能对账：正式 3 + 预热 1 = 整次 4
    assert "LLM 4 次（正式 3 + 预热 1" in report
    assert "含预热" in report


def test_r105_formal_call_counts_and_no_p95():
    requests, observations = _phase_dataset()
    report = perf.build_report_md(requests, observations, make_manifest())
    formal_rows = [ln for ln in report.splitlines() if ln.startswith("| generate | 3 |")]
    assert formal_rows  # 正式 generate 3 次（不含预热）
    assert "不发布 P95（样本不足）" in report


# ---------------------------------------------------------------------------
# R1-06 原始 SSE 时间线：采集→落盘→离线报告（本地 HTTP/SSE 桩）
# ---------------------------------------------------------------------------


class _SSEHandler(BaseHTTPRequestHandler):
    scripts: dict = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        script = self.scripts.get(payload.get("question"), [
            {"type": "session", "session_id": "stub-sess", "text": ""},
            {"type": "token", "text": "回答第一段"},
            {"type": "done", "text": ""},
        ])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in script:
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

    def log_message(self, *args):  # 静默
        pass


def _start_stub_server(scripts):
    _SSEHandler.scripts = scripts
    server = HTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_r106_e2e_stub_collect_save_offline_report(tmp_path: Path):
    server, base_url = _start_stub_server({
        "正常问题": [
            {"type": "session", "session_id": "stub-sess", "text": ""},
            {"type": "token", "text": "回答第一段"},
            {"type": "token", "text": ""},
            {"type": "token", "text": "回答第二段"},
            {"type": "done", "text": ""},
        ],
        "异常问题": [
            {"type": "token", "text": "半截"},
            {"type": "error", "status": 502, "text": ""},
            {"type": "done", "text": ""},
        ],
    })
    try:
        selected = [{"id": "S-1", "name": "桩场景", "turns": [
            {"question": "正常问题", "use_web_search": False, "expected_path": make_expected()},
            {"question": "异常问题", "use_web_search": False, "expected_path": make_expected()},
        ]}]
        requests_path = tmp_path / "requests.jsonl"
        count = perf.run_samples(base_url, selected, 1, 1, requests_path, log=lambda *_: None)
    finally:
        server.shutdown()
    assert count == 4  # 预热 2 + 正式 2
    records = perf._read_jsonl(requests_path)
    ok_records = [r for r in records if r["question"] == "正常问题"]
    bad_records = [r for r in records if r["question"] == "异常问题"]
    # 原始证据：顺序、类型、相对单调时间、是否非空可见文本、HTTP 状态、失败信息
    for record in records:
        assert record["t_start"] == 0.0 and record["t_end"] > 0
        assert [e["order"] for e in record["events"]] == list(range(1, len(record["events"]) + 1))
        assert all("t" in e and "type" in e and "text_nonempty" in e for e in record["events"])
        assert record["http_status"] == 200
    assert ok_records[0]["metrics"]["status"] == "ok"
    assert ok_records[0]["metrics"]["first_text_ms"] is not None
    assert bad_records[0]["metrics"]["status"] == "failed"
    # 报告从原始时间线重算：篡改派生 metrics 不改变结果
    manifest = make_manifest()
    baseline = perf.build_report_md(copy.deepcopy(records), [], manifest)
    tampered = copy.deepcopy(records)
    for record in tampered:
        record["metrics"] = make_metrics(first_text_ms=999999.0, done_ms=999999.0)
    tampered_report = perf.build_report_md(tampered, [], manifest)
    assert tampered_report == baseline  # 重算来源是原始时间线，不是保存的 metrics
    # 删除原始时间线 → 标 NOT_VERIFIED 且不倒造
    no_raw = copy.deepcopy(records)
    for record in no_raw:
        for key in ("events", "t_start", "t_end", "failure"):
            record.pop(key, None)
    report_no_raw = perf.build_report_md(no_raw, [], manifest)
    assert "NOT_VERIFIED" in report_no_raw
    assert report_no_raw != baseline


def test_r106_metrics_from_raw_recomputes_protocol_anomalies():
    record = {
        "t_start": 0.0,
        "t_end": 0.5,
        "events": [
            {"order": 1, "type": "done", "t": 0.1, "text_nonempty": False},
            {"order": 2, "type": "token", "t": 0.2, "text_nonempty": True},
        ],
        "failure": None,
    }
    metrics = perf.metrics_from_raw(record)
    assert metrics["status"] == "failed"
    assert metrics["first_text_ms"] is None and metrics["done_ms"] is None
    assert metrics["observed_first_text_ms"] == 200.0 and metrics["observed_done_ms"] == 100.0
    assert perf.metrics_from_raw({"metrics": {}}) is None  # 缺原始时间线 → None，不倒造


# ---------------------------------------------------------------------------
# R1-07 启动失败清理本次进程
# ---------------------------------------------------------------------------


class FakeProcess:
    def __init__(self, poll_result=None, terminate_raises=False):
        self.pid = 4242
        self.returncode = None
        self._poll_result = poll_result
        self.terminate_raises = terminate_raises
        self.terminated = 0

    def poll(self):
        return self._poll_result

    def terminate(self):
        self.terminated += 1
        if self.terminate_raises:
            raise RuntimeError("terminate denied")

    def wait(self, timeout=None):
        self.returncode = self._poll_result if self._poll_result is not None else -15


def test_r107_timeout_branch_cleans_up_own_process(tmp_path: Path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(perf.subprocess, "Popen", lambda *a, **k: process)
    ticks = iter([0.0, 0.0, 999.0])
    monkeypatch.setattr(perf.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(perf.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError) as excinfo:
        perf.start_server(tmp_path, 12345)
    assert "未在限时内就绪" in str(excinfo.value)
    assert process.terminated == 1  # 超时分支清理了本次进程
    assert excinfo.value.cleanup["terminated"] is True  # 清理证据如实记录


def test_r107_early_exit_branch_cleans_up(tmp_path: Path, monkeypatch):
    process = FakeProcess(poll_result=3)
    monkeypatch.setattr(perf.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(perf.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError) as excinfo:
        perf.start_server(tmp_path, 12346)
    assert "提前退出" in str(excinfo.value)
    assert process.terminated == 1
    assert excinfo.value.cleanup["exit_code"] == 3  # 提前退出正确记录


def test_r107_health_check_exception_waits_until_timeout(tmp_path: Path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(perf.subprocess, "Popen", lambda *a, **k: process)
    ticks = iter([0.0, 0.0, 0.5, 1.0, 999.0])
    monkeypatch.setattr(perf.time, "perf_counter", lambda: next(ticks))
    sleeps = []
    monkeypatch.setattr(perf.time, "sleep", lambda s: sleeps.append(s))

    def broken_urlopen(*a, **k):
        raise ValueError("healthz probe boom")

    monkeypatch.setattr(perf.urllib.request, "urlopen", broken_urlopen)
    with pytest.raises(RuntimeError) as excinfo:
        perf.start_server(tmp_path, 12347)
    assert "未在限时内就绪" in str(excinfo.value)  # 检查异常不提前放弃
    assert sleeps and process.terminated == 1


def test_r107_normal_start_no_cleanup_then_single_stop(tmp_path: Path, monkeypatch):
    process = FakeProcess()

    class FakeHealthResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(perf.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(perf.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(perf.urllib.request, "urlopen", lambda *a, **k: FakeHealthResponse())
    started_process, _ = perf.start_server(tmp_path, 12348)
    assert started_process is process
    assert process.terminated == 0  # 正常启动不预清理
    info = perf.stop_server(process, 12348, tmp_path)
    assert process.terminated == 1  # 收尾只清一次
    assert info["terminated"] is True and info["method"] == "terminate"


def test_r107_forced_kill_branch_reports_failure_truthfully(tmp_path: Path, monkeypatch):
    process = FakeProcess(terminate_raises=True)

    class DummyRun:
        returncode = 0

    monkeypatch.setattr(perf.subprocess, "run", lambda *a, **k: DummyRun())
    info = perf.stop_server(process, 12349, tmp_path)
    assert info["method"] == "taskkill"  # 强制分支
    assert info["terminated"] is True


# ---------------------------------------------------------------------------
# E-102 采集服务清理：真实进程路径（不只依赖进程桩）
# ---------------------------------------------------------------------------

_LISTENER_SRC = (
    "import socket,sys,time\n"
    "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
    "s.bind(('127.0.0.1',int(sys.argv[1])));s.listen()\n"
    "print('listening',flush=True)\n"
    "time.sleep(120)\n"
)

_PARENT_SRC = (
    "import subprocess,sys\n"
    "child=subprocess.Popen([sys.executable,sys.argv[2],sys.argv[1]])\n"
    "print('ready '+str(child.pid),flush=True)\n"
    "child.wait()\n"
)


def _spawn_listener_tree(tmp_path: Path, port: int):
    """真实父子进程树：子进程持有监听套接字（模拟 venv python shim → 真实服务进程）。"""
    (tmp_path / "child_listener.py").write_text(_LISTENER_SRC, encoding="utf-8")
    (tmp_path / "parent_tree.py").write_text(_PARENT_SRC, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(tmp_path / "parent_tree.py"), str(port),
         str(tmp_path / "child_listener.py")],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    line = process.stdout.readline().strip()
    child_pid = int(line.split()[1])
    deadline = time.time() + 15
    while time.time() < deadline and not perf._port_accepts_connections(port, timeout=0.5):
        time.sleep(0.1)
    return process, child_pid


def test_e102_real_tree_cleanup_releases_port_with_timestamped_evidence(tmp_path: Path):
    """正常结束路径：真实子进程持有监听；清理需终止本次树并确认端口释放（带时间戳证据）。"""
    port = perf._free_port()
    process, child_pid = _spawn_listener_tree(tmp_path, port)
    try:
        assert perf._port_accepts_connections(port, timeout=1.0)  # 子进程真实持有监听
        info = perf.stop_server(process, port, tmp_path)
        assert info["terminated"] is True and info["method"] == "terminate"
        assert child_pid in (info["descendants"] or []), info  # 后代按父子链确认归属
        assert info["descendants_alive_after"][str(child_pid)] is False  # 本次树的后代已终止
        assert info["port_initial_check"]["state"] in {"listening", "released", "unknown"}  # 首次检查如实保留
        assert info["port_released"] is True  # 有限等待内经明确依据确认释放
        assert info["port_released_source"] in {"tcp_table", "connect_refused"}
        assert info["cleanup_complete"] is True  # 整体判定：父+后代+端口均确认
        assert isinstance(info["port_release_wait_ms"], (int, float))
        assert info["checked_at_utc"]  # 带时间戳证据
        assert not perf._port_accepts_connections(port, timeout=1.0)  # 实际端口状态复核
    finally:
        if process.poll() is None:
            process.kill()


def test_e102_forced_branch_kills_real_tree_and_reports(tmp_path: Path):
    """已知异常分支（terminate 拒绝）：真实进程树走 taskkill 强制分支，结果如实记录。"""

    class NoTerminate:
        def __init__(self, real):
            self._real = real

        @property
        def pid(self):
            return self._real.pid

        @property
        def returncode(self):
            return self._real.returncode

        def terminate(self):
            raise RuntimeError("terminate denied")

        def wait(self, timeout=None):
            return self._real.wait(timeout=timeout)

    port = perf._free_port()
    process, child_pid = _spawn_listener_tree(tmp_path, port)
    try:
        info = perf.stop_server(NoTerminate(process), port, tmp_path)
        assert info["method"] == "taskkill" and info["terminated"] is True
        assert info["port_released"] is True
        assert info["cleanup_complete"] is True, info
        assert info["descendants_alive_after"][str(child_pid)] is False
        assert not perf._port_accepts_connections(port, timeout=1.0)
    finally:
        if process.poll() is None:
            process.kill()


def test_e102_unrelated_listener_not_killed_and_failure_reported(tmp_path: Path, monkeypatch):
    """端口未释放时如实失败；不按端口误杀不属于本次的进程。"""
    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.5)
    port = perf._free_port()
    (tmp_path / "child_listener.py").write_text(_LISTENER_SRC, encoding="utf-8")
    unrelated = subprocess.Popen(
        [sys.executable, str(tmp_path / "child_listener.py"), str(port)],
        cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not perf._port_accepts_connections(port, timeout=0.5):
            time.sleep(0.1)
        process = FakeProcess()
        process.pid = 999999999  # 与无关服务无父子关系
        info = perf.stop_server(process, port, tmp_path)
        assert info["port_released"] is False  # 如实报告未释放
        assert info["port_error"]
        assert info["port_state"] in {"listening", "unknown"}  # 不把未知/仍监听写成释放
        assert info["cleanup_complete"] is False  # 整体判定不成立（R1-05）
        assert info["port_check_samples"]  # 有限等待的真实状态采样
        assert unrelated.poll() is None  # 无关服务未被误杀
    finally:
        unrelated.kill()


# ---------------------------------------------------------------------------
# R1-02 辅助：perf_span 保留 extra 引用（回归守护）
# ---------------------------------------------------------------------------


def test_perf_span_preserves_extra_dict_reference(monkeypatch, tmp_path: Path):
    obs_path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("LAWRAG_PERF_OBS_FILE", str(obs_path))
    extra = {}
    with observability.perf_span("retrieval", "rerank", extra=extra):
        extra["attempts"] = 2
    record = perf.load_observations(obs_path)[0]
    assert record["attempts"] == 2


# ---------------------------------------------------------------------------
# 原有回归（口径按 R1 更新后保留意图）
# ---------------------------------------------------------------------------


def test_summarize_keeps_missing_out_of_stats():
    summary = perf.summarize([100.0, None, 300.0])
    assert summary == {"n": 2, "missing": 1, "min": 100.0, "median": 200.0, "max": 300.0}
    assert perf.summarize([None, None])["min"] is None


def test_deterministic_clarify_marked_not_model_first_token():
    request = make_request(
        request_id="C-formal1-t1",
        expected_path=make_expected(nodes=["route", "clarify"], llm_calls=["router"],
                                    retrieval_steps=[], answer_mode="deterministic_clarify"),
    )
    observations = [
        make_obs(seq=1, kind="node", name="route"),
        make_obs(seq=2, kind="llm", name="router"),
        make_obs(seq=3, kind="node", name="clarify"),
    ]
    annotated, unassigned = perf.annotate_requests([request], observations)
    assert unassigned == 0
    assert annotated[0]["first_text_kind"] == "deterministic"
    assert annotated[0]["path_match"] is True
    assert annotated[0]["actual_path"]["llm_calls"] == ["router"]


def test_path_mismatch_and_unassigned_observation():
    request = make_request()
    observations = [
        make_obs(seq=1, kind="node", name="generate"),
        make_obs(seq=2, ts_utc="2026-10-10T09:00:00.000+00:00", kind="node", name="prepare"),
    ]
    annotated, unassigned = perf.annotate_requests([request], observations)
    assert unassigned == 1
    assert annotated[0]["path_match"] is False


def test_adjacent_windows_do_not_steal_next_request_records():
    first = make_request(request_id="S-1-formal1-t1",
                         wall_start="2026-10-10T10:00:00.000+00:00",
                         wall_end="2026-10-10T10:00:05.000+00:00")
    second = make_request(request_id="S-1-formal1-t2", turn_index=2,
                          wall_start="2026-10-10T10:00:05.200+00:00",
                          wall_end="2026-10-10T10:00:10.000+00:00")
    observations = [
        make_obs(seq=1, ts_utc="2026-10-10T10:00:05.300+00:00", kind="node", name="prepare"),
        make_obs(seq=2, ts_utc="2026-10-10T10:00:09.000+00:00", kind="node", name="generate"),
    ]
    correlation = perf.correlate_observations(observations, [first, second])
    assert correlation["by_request"]["S-1-formal1-t1"] == []
    assert [record["seq"] for record in correlation["by_request"]["S-1-formal1-t2"]] == [1, 2]


def test_dedup_observations_by_seq(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    lines = [
        json.dumps(make_obs(seq=1, kind="node", name="generate")),
        json.dumps(make_obs(seq=1, kind="node", name="generate")),
        json.dumps(make_obs(seq=2, kind="llm", name="generate")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert [record["seq"] for record in perf.load_observations(path)] == [1, 2]


def test_layered_aggregation_no_double_count():
    observations = [
        make_obs(seq=1, kind="node", name="generate", duration_ms=100.0),
        make_obs(seq=2, kind="llm", name="generate", duration_ms=80.0,
                 usage={"input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0}),
        make_obs(seq=3, kind="retrieval", name="vector", duration_ms=15.0),
        make_obs(seq=4, kind="tool", name="calculate_labor_compensation", duration_ms=5.0),
    ]
    agg = perf.aggregate_observations(observations)
    assert agg["node_layer_ms"]["generate"]["total_ms"] == 100.0
    assert agg["llm_by_purpose"]["generate"]["total_ms"] == 80.0
    assert agg["retrieval"]["vector"]["total_ms"] == 15.0
    assert agg["tools"]["calculate_labor_compensation"]["total_ms"] == 5.0
    assert "llm" not in agg["node_layer_ms"]


def test_session_id_from_events_prefers_session_event():
    events = [
        {"t": 1.0, "event": {"type": "session", "session_id": "server-sess"}},
        {"t": 1.1, "event": {"type": "token", "text": "答"}},
    ]
    assert perf.session_id_from_events(events) == "server-sess"
    assert perf.session_id_from_events([{"t": 1.0, "event": {"type": "token", "text": "x"}}]) is None


def test_run_one_request_first_turn_omits_session_id(monkeypatch):
    calls = []

    def fake_sse_request(base_url, session_id, question, use_web_search):
        calls.append(session_id)
        return {
            "events": [
                {"t": 1.0, "event": {"type": "session", "session_id": "server-sess"}},
                {"t": 1.2, "event": {"type": "token", "text": "答"}},
                {"t": 1.4, "event": {"type": "done"}},
            ],
            "failure": None,
            "t_start": 1.0,
            "t_end": 1.5,
            "wall_start": "2026-10-10T10:00:00.000+00:00",
            "wall_end": "2026-10-10T10:00:01.000+00:00",
            "http_status": 200,
        }

    monkeypatch.setattr(perf, "sse_request", fake_sse_request)
    first = perf.run_one_request("http://x", None, "问", False, make_expected())
    second = perf.run_one_request("http://x", first["server_session_id"], "追问", False, make_expected())
    assert calls == [None, "server-sess"]
    assert first["session_id"] == "server-sess"
    assert first["metrics"]["status"] == "ok"
    # 原始时间线落盘（R1-06）
    assert first["t_start"] == 0.0 and first["t_end"] == pytest.approx(0.5)
    assert [e["order"] for e in first["events"]] == [1, 2, 3]
    assert first["events"][1]["text_nonempty"] is True


class RouterStub:
    def bind_tools(self, tools):
        return self

    def invoke(self, prompt, config=None):
        from types import SimpleNamespace

        return SimpleNamespace(tool_calls=[])


def test_perf_observation_does_not_change_stream(monkeypatch, tmp_path: Path):
    """观测开启后，实际图的回答与流式事件与关闭时完全一致（P3-01）。"""
    from types import SimpleNamespace

    from llama_index.core.schema import NodeWithScore, TextNode

    nodes = [
        NodeWithScore(
            node=TextNode(text="法条 1", metadata={"source": "劳动合同法", "article": "第一条"}),
            score=0.9,
        )
    ]

    class FakeModel:
        def invoke(self, prompt, config=None):
            return SimpleNamespace(content="测试回答")

    graph_module.get_graph.cache_clear()
    for getter in (db_module.get_connection, db_module.get_checkpointer_connection):
        if getter.cache_info().currsize:
            getter().close()
        getter.cache_clear()
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path / "t.sqlite3")
    monkeypatch.setattr(graph_module, "get_checkpointer_connection",
                        db_module.get_checkpointer_connection)
    monkeypatch.setattr(graph_module, "retrieve", lambda query, top_k=3, session_id=None: nodes)
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())
    monkeypatch.setattr(graph_module, "get_router_model", lambda: RouterStub())

    def snapshot(session_id: str):
        answer, _ = graph_module.run_turn(session_id, "劳动合同到期有补偿吗？")
        events = list(graph_module.stream_turn(session_id, "那补偿怎么算？"))
        return answer, events

    try:
        monkeypatch.delenv("LAWRAG_PERF_OBS_FILE", raising=False)
        answer_off, events_off = snapshot("sess-off")
        obs_path = tmp_path / "observations.jsonl"
        monkeypatch.setenv("LAWRAG_PERF_OBS_FILE", str(obs_path))
        answer_on, events_on = snapshot("sess-on")
    finally:
        graph_module.get_graph.cache_clear()
        for getter in (db_module.get_connection, db_module.get_checkpointer_connection):
            if getter.cache_info().currsize:
                getter().close()
            getter.cache_clear()

    assert answer_on == answer_off
    assert [event["type"] for event in events_on] == [event["type"] for event in events_off]
    assert "".join(event.get("text", "") for event in events_on) == "".join(
        event.get("text", "") for event in events_off
    )
    records = perf.load_observations(obs_path)
    assert any(record["kind"] == "node" for record in records)
    assert any(record["kind"] == "llm" for record in records)


def test_traced_invoke_writes_single_llm_record(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace

    obs_path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("LAWRAG_PERF_OBS_FILE", str(obs_path))

    class UsageModel:
        model_name = "deepseek-chat"

        def invoke(self, prompt, config=None):
            return SimpleNamespace(
                content="回答",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                response_metadata={},
            )

    observability.traced_invoke(UsageModel(), "提示", session_id="s1", tags=["generate"])
    records = perf.load_observations(obs_path)
    assert len(records) == 1
    assert records[0]["kind"] == "llm" and records[0]["model"] == "deepseek-chat"
    assert records[0]["usage"]["input_tokens"] == 10


def test_traced_invoke_missing_usage_recorded_unknown(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace

    obs_path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("LAWRAG_PERF_OBS_FILE", str(obs_path))

    class PlainModel:
        def invoke(self, prompt, config=None):
            return SimpleNamespace(content="回答")

    observability.traced_invoke(PlainModel(), "提示", session_id="s1", tags=["rewrite"])
    records = perf.load_observations(obs_path)
    assert records[0]["usage"] is None
    assert records[0]["usage_source"] == "missing"


def test_perf_disabled_writes_nothing(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("LAWRAG_PERF_OBS_FILE", raising=False)
    assert observability.perf_enabled() is False
    observability.perf_write({"kind": "node", "name": "x"})
    with observability.perf_span("node", "y"):
        pass
    assert not (tmp_path / "observations.jsonl").exists()


# ---------------------------------------------------------------------------
# R2 返工回归（R1 独立复验五项阻塞 R1-F01～F05，验收夹具同 outputs/STAGE3_STEP1_R1_ACCEPTANCE_20261010.md）
# ---------------------------------------------------------------------------

FULL_USAGE = {"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 0}


def _good_llm(seq=1, ok=True, usage=FULL_USAGE, **overrides):
    record = make_obs(seq=seq, kind="llm", name="generate", model="deepseek-chat",
                      duration_ms=1.0, ok=ok, usage=usage)
    record.update(overrides)
    return record


def test_r201_known_call_keeps_cost_beside_same_model_unknown_calls():
    good = _good_llm()
    one = perf.compute_cost(perf.aggregate_observations([good]), perf.PRICING)
    assert one["known_total"] == 0.002 and one["items"][0]["calls_costed"] == 1  # 夹具：完整调用 CNY 0.002
    cases = {
        "all_missing": None,
        "input_missing": {"input_tokens": None, "output_tokens": 50, "cache_read_tokens": 0},
        "output_missing": {"input_tokens": 100, "output_tokens": None, "cache_read_tokens": 0},
        "cache_missing": {"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": None},
        "multi_missing": {"input_tokens": None, "output_tokens": None, "cache_read_tokens": None},
    }
    for tag, usage in cases.items():
        bad = _good_llm(seq=2, ok=False, usage=usage)
        cost = perf.compute_cost(perf.aggregate_observations([good, bad]), perf.PRICING)
        assert cost["known_total"] == 0.002, tag  # 同模型未知调用不抹掉已知费用
        assert cost["items"][0]["calls_costed"] == 1, tag
        assert cost["items"][0]["calls_unknown"] == 1, tag  # 缺多个字段只计一次未知调用
        assert cost["items"][0]["estimated_cost"] == 0.002, tag


def test_r201_failed_call_usable_usage_counted_and_scoped_numerator():
    failed = _good_llm(seq=2, ok=False)  # 失败但 usage 可核实 → 照常计费
    cost = perf.compute_cost(perf.aggregate_observations([failed]), perf.PRICING)
    assert cost["known_total"] == 0.002 and cost["items"][0]["calls_costed"] == 1
    # MIX-SCOPED 夹具：一个完整成功任务 + 一个失败未知调用
    scoped = perf.compute_scoped_costs(
        {"formal": [_good_llm(seq=1), _good_llm(seq=2, ok=False, usage=None)], "warmup": [], "unassigned": []},
        perf.PRICING,
        [{"phase": "formal", "complete_success": True}, {"phase": "formal", "complete_success": False}],
    )
    assert scoped["formal"]["known_total"] == 0.002  # 正式已知费用保留 0.002
    assert scoped["per_success_task_formal"] == 0.002  # 每成功任务保留 0.002


def test_r201_upper_bound_only_for_eligible_unknown_calls():
    good = _good_llm(seq=1)
    cache_missing = _good_llm(seq=2, usage={"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": None})
    cost = perf.compute_cost(perf.aggregate_observations([good, cache_missing]), perf.PRICING)
    assert cost["known_total"] == 0.002  # 已知合计不含上界
    item = cost["items"][0]
    # 上界只含缓存缺失的未知调用（100/1M*1 + 50/1M*2），不含已计费的完整调用
    assert item["assumed_upper_bound"] == pytest.approx(100 / 1_000_000 * 1 + 50 / 1_000_000 * 2, abs=1e-12)
    assert "上界假设" in item["assumed_upper_bound_note"]
    assert item["calls_costed"] == 1 and item["calls_unknown"] == 1


def test_r201_all_unknown_and_zero_success_tasks_keep_na():
    cost = perf.compute_cost(perf.aggregate_observations([_good_llm(usage=None)]), perf.PRICING)
    assert cost["known_total"] == 0.0 and cost["items"][0]["calls_costed"] == 0
    assert cost["items"][0]["calls_unknown"] == 1
    assert any("费用未知" in entry for entry in cost["unknowns"])  # 未知说明保留，不补零
    scoped = perf.compute_scoped_costs(
        {"formal": [_good_llm(usage=None)], "warmup": [], "unassigned": []},
        perf.PRICING, [{"phase": "formal", "complete_success": False}],
    )
    assert scoped["per_success_task_formal"] == "N/A"


def _single_turn_dataset():
    raw = make_raw()
    requests, observations = [], []
    minute = 0
    for phase, reps in (("warmup", 1), ("formal", 3)):
        for rep in range(1, reps + 1):
            minute += 1
            rid = f"S-1-{phase}{rep}-t1"
            observations.append(make_obs(
                seq=minute, kind="llm", name="generate", model="deepseek-chat", duration_ms=1.0,
                ts_utc=f"2026-10-10T11:{minute:02d}:01.000+00:00", usage=dict(FULL_USAGE), rid=rid))
    # 机制性修正（E-095 复发 ≥2）：预期路径由该请求自己的构造观测经 build_actual_path 派生，
    # 不手写 expected_path，杜绝“预期 vs 构造观测”不一致。
    for phase, reps in (("warmup", 1), ("formal", 3)):
        for rep in range(1, reps + 1):
            minute = {"warmup": rep, "formal": rep + 1}[phase]
            rid = f"S-1-{phase}{rep}-t1"
            expected = perf.build_actual_path([o for o in observations if o.get("rid") == rid])
            requests.append(make_request(
                request_id=rid, phase=phase, rep=rep, expected_path=expected,
                wall_start=f"2026-10-10T11:{minute:02d}:00.000+00:00",
                wall_end=f"2026-10-10T11:{minute:02d}:05.000+00:00",
                events=raw["events"], t_start=0.0, t_end=raw["t_end"], failure=None))
    return requests, observations


def test_r202_planned_denominators_from_frozen_manifest():
    requests, observations = _single_turn_dataset()
    manifest = make_manifest()
    drop_id = "S-1-formal3-t1"
    kept = [r for r in requests if r["request_id"] != drop_id]
    kept_obs = [o for o in observations if o.get("rid") != drop_id]
    annotated, _ = perf.annotate_requests(kept, kept_obs)
    tasks = perf.evaluate_tasks(annotated, manifest)
    formal = [t for t in tasks if t["phase"] == "formal"]
    assert len(formal) == 3  # 计划正式任务仍为 3，不因缺记录缩小
    assert sum(t["complete_success"] for t in formal) == 2
    missing = [t for t in formal if t["task_id"] == "S-1-formal3"][0]
    assert missing["status"] == "not_executed" and missing["executed"] is False
    report = perf.build_report_md(kept, kept_obs, manifest)
    assert "| S-1 T1 | 首段可见回答耗时(ms) | 2 | 3 | 0 | 1 |" in report  # 计划 3、成功 2、未执行 1
    assert "| S-1 T1 | 请求完成耗时(ms) | 2 | 3 | 0 | 1 |" in report
    assert "| S-1 | 场景耗时(ms) | 2 | 3 | 0 | 1 |" in report
    assert "计划正式任务 3" in report
    assert "正式完整成功任务数：2（分母；计划正式任务 3）" in report
    costs = perf.compute_scoped_costs(perf.split_observations_by_phase(kept_obs, annotated), perf.PRICING, tasks)
    assert costs["formal_tasks"] == 3 and costs["formal_complete_success_tasks"] == 2
    assert costs["per_success_task_formal"] == pytest.approx(0.004 / 2, abs=1e-9)  # 分母用实际成功 2，不用计划 3


def test_r202_zero_records_shows_frozen_plan_without_crash():
    report = perf.build_report_md([], [], make_manifest())
    assert "计划样本：预热 1 + 正式 3 = 4" in report
    assert "实际请求 0，成功 0，失败 0" in report
    assert "| S-1 T1 | 首段可见回答耗时(ms) | 0 | 3 | 0 | 3 | N/A | N/A | N/A |" in report
    assert "| S-1 | 场景耗时(ms) | 0 | 3 | 0 | 3 | N/A | N/A | N/A |" in report
    assert "正式完整成功任务数：0（分母；计划正式任务 3）" in report
    assert "正式每成功任务已知费用：**N/A**" in report
    tasks = perf.evaluate_tasks([], make_manifest())
    assert len([t for t in tasks if t["phase"] == "formal"]) == 3
    assert all(t["status"] == "not_executed" for t in tasks)


def _two_turn_requests(ids_and_turns, minute_base=0):
    raw = make_raw()
    empty_expected = make_expected(nodes=[], llm_calls=[], tool_executions=[],
                                   tool_inits=[], retrieval_steps=[])
    out = []
    for offset, (rid, rep, turn) in enumerate(ids_and_turns):
        minute = minute_base + offset
        out.append(make_request(
            request_id=rid, scenario_id="M", phase="formal", rep=rep, turn_index=turn,
            expected_path=empty_expected,
            wall_start=f"2026-10-10T12:{minute:02d}:00.000+00:00",
            wall_end=f"2026-10-10T12:{minute:02d}:05.000+00:00",
            events=raw["events"], t_start=0.0, t_end=raw["t_end"], failure=None))
    return out


def test_r202_whole_task_unexecuted_keeps_plan_row():
    manifest = make_manifest(scenarios=[{"id": "M", "name": "两轮", "turns": 2}],
                             warmup_passes=0, formal_reps=2)
    requests = _two_turn_requests([("M-formal1-t1", 1, 1), ("M-formal1-t2", 1, 2)])
    annotated, _ = perf.annotate_requests(requests, [])
    tasks = {t["task_id"]: t for t in perf.evaluate_tasks(annotated, manifest)}
    assert tasks["M-formal1"]["complete_success"] is True
    assert tasks["M-formal2"]["status"] == "not_executed"  # 整个重复未执行仍占计划行
    report = perf.build_report_md(requests, [], manifest)
    assert "| M | 场景耗时(ms) | 1 | 2 | 0 | 1 |" in report


def test_r202_two_turn_scenario_with_single_turn_recorded():
    manifest = make_manifest(scenarios=[{"id": "M", "name": "两轮", "turns": 2}],
                             warmup_passes=0, formal_reps=2)
    requests = _two_turn_requests(
        [("M-formal1-t1", 1, 1), ("M-formal1-t2", 1, 2), ("M-formal2-t1", 2, 1)])
    annotated, _ = perf.annotate_requests(requests, [])
    tasks = {t["task_id"]: t for t in perf.evaluate_tasks(annotated, manifest)}
    assert tasks["M-formal1"]["complete_success"] is True
    assert tasks["M-formal2"]["status"] == "executed_incomplete"
    assert any("缺少轮次 T2" in problem for problem in tasks["M-formal2"]["problems"])
    report = perf.build_report_md(requests, [], manifest)
    assert "| M T2 | 首段可见回答耗时(ms) | 1 | 2 | 0 | 1 |" in report  # T2 计划仍为 2


def test_r203_turn_identity_and_uniqueness_matrix():
    manifest = make_manifest(scenarios=[{"id": "M", "name": "两轮", "turns": 2}],
                             warmup_passes=0, formal_reps=5)
    requests = _two_turn_requests([
        ("M-formal1-t1", 1, 1), ("M-formal1-t2", 1, 2),          # 合法 T1/T2
        ("M-formal2-t1", 2, 1), ("M-formal2-dup", 2, 1),          # T1/T1
        ("M-formal3-t1", 3, 1),                                    # 缺 T2
        ("M-formal4-t1", 4, 1), ("M-formal4-t2", 4, 2), ("M-formal4-dup", 4, 2),  # 同轮重复
        ("M-formal5-t1", 5, 1), ("M-formal5-t3", 5, 3),          # 越界轮次（且缺 T2）
    ])
    annotated, _ = perf.annotate_requests(requests, [])
    tasks = {t["task_id"]: t for t in perf.evaluate_tasks(annotated, manifest)}
    assert tasks["M-formal1"]["complete_success"] is True and tasks["M-formal1"]["status"] == "complete_success"
    for task_id in ("M-formal2", "M-formal3", "M-formal4", "M-formal5"):
        assert tasks[task_id]["complete_success"] is False, task_id
        assert tasks[task_id]["problems"], task_id
        assert tasks[task_id]["records"], task_id  # 保留实际记录
    assert any("重复" in problem for problem in tasks["M-formal2"]["problems"])
    assert any("缺少轮次 T2" in problem for problem in tasks["M-formal3"]["problems"])
    assert any("重复" in problem for problem in tasks["M-formal4"]["problems"])
    assert any("越界" in problem for problem in tasks["M-formal5"]["problems"])


def test_r203_invalid_task_excluded_from_duration_and_denominator():
    manifest = make_manifest(scenarios=[{"id": "M", "name": "两轮", "turns": 2}],
                             warmup_passes=0, formal_reps=2)
    requests = _two_turn_requests([
        ("M-formal1-t1", 1, 1), ("M-formal1-t2", 1, 2),
        ("M-formal2-t1", 2, 1), ("M-formal2-dup", 2, 1),
    ])
    observations = [make_obs(seq=1, kind="llm", name="generate", model="deepseek-chat",
                             duration_ms=1.0, ts_utc="2026-10-10T12:03:01.000+00:00",
                             usage=dict(FULL_USAGE))]  # 归属 formal2（无效任务）的可核实费用
    annotated, _ = perf.annotate_requests(requests, observations)
    tasks = perf.evaluate_tasks(annotated, manifest)
    assert [t["complete_success"] for t in tasks if t["phase"] == "formal"] == [True, False]
    report = perf.build_report_md(requests, observations, manifest)
    assert "| M | 场景耗时(ms)（未计入不完整任务：['M-formal2']） | 1 | 2 | 1 | 0 |" in report
    costs = perf.compute_scoped_costs(perf.split_observations_by_phase(observations, annotated),
                                      perf.PRICING, tasks)
    assert costs["formal_complete_success_tasks"] == 1  # 无效任务不进每成功任务费用分母
    assert costs["formal"]["known_total"] == 0.002  # 但可核实费用仍计分子
    assert costs["per_success_task_formal"] == pytest.approx(0.002, abs=1e-9)


def test_r204_same_timestamp_order_decides_validity():
    tied_ok = perf.derive_metrics(
        [{"t": 0.2, "type": "token", "text_nonempty": True},
         {"t": 0.2, "type": "done", "text_nonempty": False}], 0.0, 0.3)
    assert tied_ok["status"] == "ok"  # 同刻 token → done：正常
    tied_bad = perf.derive_metrics(
        [{"t": 0.2, "type": "done", "text_nonempty": False},
         {"t": 0.2, "type": "token", "text_nonempty": True}], 0.0, 0.3)
    assert tied_bad["status"] == "failed"  # 同刻 done → token：失败
    assert tied_bad["done_ms"] is None and tied_bad["first_text_ms"] is None
    assert tied_bad["observed_first_text_ms"] == 200.0 and tied_bad["observed_done_ms"] == 200.0
    assert any("done 之后仍有非空文本" in note for note in tied_bad["notes"])
    mixed = perf.derive_metrics(
        [{"t": 0.1, "type": "token", "text_nonempty": True},
         {"t": 0.2, "type": "done", "text_nonempty": False},
         {"t": 0.2, "type": "token", "text_nonempty": True}], 0.0, 0.3)
    assert mixed["status"] == "failed" and mixed["done_ms"] is None
    assert mixed["observed_done_ms"] == 200.0


def test_r204_same_timestamp_counterexample_via_raw_timeline_report():
    events = [
        {"order": 1, "type": "token", "t": 0.1, "text_nonempty": True},
        {"order": 2, "type": "done", "t": 0.2, "text_nonempty": False},
        {"order": 3, "type": "token", "t": 0.2, "text_nonempty": True},  # 同刻、done 之后
    ]
    request = make_request(request_id="S-1-formal1-t1", rep=1,
                           events=events, t_start=0.0, t_end=0.3, failure=None)
    annotated, _ = perf.annotate_requests([request], [])  # 原始时间线落盘 → 离线重算入口
    metrics = annotated[0]["metrics"]
    assert metrics["status"] == "failed"
    assert metrics["first_text_ms"] is None and metrics["done_ms"] is None
    assert metrics["observed_first_text_ms"] == 100.0 and metrics["observed_done_ms"] == 200.0
    report = perf.build_report_md([request], [], make_manifest())
    assert "done 之后仍有非空文本事件" in report


def test_r205_missing_or_none_single_rate_gives_traceable_unknown():
    for field in ("input_cache_hit_per_m", "input_cache_miss_per_m", "output_per_m"):
        for mode in ("absent", "none"):
            pricing = copy.deepcopy(perf.PRICING)
            if mode == "absent":
                pricing[0].pop(field)
            else:
                pricing[0][field] = None
            cost = perf.compute_cost(perf.aggregate_observations([_good_llm()]), pricing)
            assert cost["known_total"] == 0.0, (field, mode)  # 不进 float*None、不静默记零
            assert cost["items"][0]["estimated_cost"] is None, (field, mode)
            assert any(field in entry for entry in cost["unknowns"]), (field, mode)  # 未知原因含字段名
            assert any("费率" in entry for entry in cost["unknowns"]), (field, mode)


def test_r205_explicit_zero_rate_is_valid():
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 100}
    for field in ("input_cache_hit_per_m", "input_cache_miss_per_m", "output_per_m"):
        pricing = copy.deepcopy(perf.PRICING)
        pricing[0][field] = 0.0  # 显式数值 0 是有效费率
        cost = perf.compute_cost(perf.aggregate_observations([_good_llm(usage=usage)]), pricing)
        assert cost["items"][0]["calls_costed"] == 1, field
        assert cost["items"][0]["estimated_cost"] is not None, field


def _extra_pricing():
    return {
        "service": "测试", "model": "m-test", "applies_to_models": ["m-test"],
        "currency": "CNY", "unit": "元/百万tokens",
        "input_cache_miss_per_m": 1.0, "input_cache_hit_per_m": 0.5, "output_per_m": 2.0,
        "source_url": "https://example.test/pricing", "checked_on": "2026-10-10",
        "price_status": "assumed",
    }


def test_r205_broken_model_price_does_not_affect_other_model():
    pricing = copy.deepcopy(perf.PRICING)
    pricing[0]["output_per_m"] = None  # deepseek 费率不完整
    pricing.append(_extra_pricing())
    cost = perf.compute_cost(
        perf.aggregate_observations([_good_llm(seq=1), _good_llm(seq=2, model="m-test")]), pricing)
    items = {item["model"]: item for item in cost["items"]}
    assert items["deepseek-chat"]["estimated_cost"] is None
    assert items["m-test"]["estimated_cost"] == 0.002  # 其他模型可计费部分不受影响
    assert cost["known_total"] == 0.002


def test_r205_report_and_cli_generate_with_missing_rate(tmp_path: Path, monkeypatch):
    pricing = copy.deepcopy(perf.PRICING)
    pricing[0]["output_per_m"] = None
    pricing.append(_extra_pricing())
    manifest = make_manifest(pricing=pricing)
    requests = [make_request()]
    observations = [
        _good_llm(seq=1),                                          # deepseek：缺费率 → 未知
        _good_llm(seq=2, model="m-test"),                          # 可计费 → 0.002
        _good_llm(seq=3, model="m-test", usage=None),              # 同模型未知调用
    ]
    report = perf.build_report_md(requests, observations, manifest)
    assert "缺少计价所需费率" in report and "output_per_m" in report  # 未知原因可追溯
    assert "CNY 0.002" in report  # 已知费用不丢失
    assert "已知部分，另 1 次调用费用未知" in report  # 不把部分已知合计描述为完整费用
    assert report == perf.build_report_md(requests, observations, manifest)  # 重复重建字节一致

    monkeypatch.setattr(perf, "PERF_ROOT", tmp_path)
    run_dir = tmp_path / "run-missing-rate"
    run_dir.mkdir()
    (run_dir / "requests.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in requests), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (run_dir / "observations.jsonl").write_text(
        "".join(json.dumps(o, ensure_ascii=False, default=str) + "\n" for o in observations), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    out1, out2 = tmp_path / "derived-a", tmp_path / "derived-b"
    assert perf.main(["report", "--run-id", "run-missing-rate", "--output-dir", str(out1)]) == 0
    assert perf.main(["report", "--run-id", "run-missing-rate", "--output-dir", str(out2)]) == 0
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before  # 原文件不回写
    assert (out1 / "report.md").read_bytes() == (out2 / "report.md").read_bytes()


# ---------------------------------------------------------------------------
# R3 返工回归（R2 独立验收 R2-F01：计划外记录不得污染冻结计划统计）
# ---------------------------------------------------------------------------


def _slot_fixture(specs):
    """specs: [(rid, rep, turn, minute[, raw])] → (requests, observations)。

    每条请求带原始时间线与一条 generate llm 观测（FULL_USAGE=0.002/条）；
    预期路径逐请求由自己的观测派生（E-095 机制性修正）。
    """
    requests, observations = [], []
    for index, spec in enumerate(specs, start=1):
        rid, rep, turn, minute = spec[:4]
        raw = spec[4] if len(spec) > 4 else make_raw()
        obs = make_obs(seq=index, kind="llm", name="generate", model="deepseek-chat",
                       duration_ms=1.0, ts_utc=f"2026-10-10T13:{minute:02d}:01.000+00:00",
                       usage=dict(FULL_USAGE), rid=rid)
        observations.append(obs)
        requests.append(make_request(
            request_id=rid, scenario_id="S", rep=rep, turn_index=turn,
            expected_path=perf.build_actual_path([obs]),
            wall_start=f"2026-10-10T13:{minute:02d}:00.000+00:00",
            wall_end=f"2026-10-10T13:{minute:02d}:05.000+00:00",
            events=raw["events"], t_start=0.0, t_end=raw["t_end"], failure=None))
    return requests, observations


def _slow_raw():
    return make_raw(first_text=0.9, done=1.0)


def _slot_manifest(**overrides):
    kwargs = {"scenarios": [{"id": "S", "name": "单轮", "turns": 1}],
              "warmup_passes": 0, "formal_reps": 3}
    kwargs.update(overrides)
    return make_manifest(**kwargs)


def _report_cli(run_dir: Path, manifest, requests, observations, out: Path) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "requests.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in requests), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (run_dir / "observations.jsonl").write_text(
        "".join(json.dumps(o, ensure_ascii=False, default=str) + "\n" for o in observations), encoding="utf-8")
    assert perf.main(["report", "--run-id", run_dir.name, "--output-dir", str(out)]) == 0
    return (out / "report.md").read_text(encoding="utf-8")


def test_r301_case_a_normal_control():
    """用例 A：三次计划内全部成功——计划 3/3、无未执行、无额外记录、0.006、每成功任务 0.002。"""
    requests, observations = _slot_fixture([
        ("S-formal1-t1", 1, 1, 1), ("S-formal2-t1", 2, 1, 2), ("S-formal3-t1", 3, 1, 3)])
    manifest = _slot_manifest()
    annotated, _ = perf.annotate_requests(requests, observations)
    counts = perf.classify_request_slots(annotated, manifest)["counts"]
    assert counts["planned_slots"] == 3 and counts["success_slots"] == 3
    assert counts["empty_slots"] == 0 and counts["anomaly_records"] == 0
    tasks = perf.evaluate_tasks(annotated, manifest)
    assert len(tasks) == 3 and all(t["planned"] and t["complete_success"] for t in tasks)
    costs = perf.compute_scoped_costs(perf.split_observations_by_phase(observations, annotated),
                                      manifest["pricing"], tasks)
    assert costs["formal_tasks"] == 3 and costs["formal_complete_success_tasks"] == 3
    assert costs["formal"]["known_total"] == 0.006
    assert costs["per_success_task_formal"] == 0.002
    report = perf.build_report_md(requests, observations, manifest)
    assert "计划样本：预热 0 + 正式 3 = 3" in report
    assert "计划槽位结果：成功 3，失败 0，重复异常 0，未执行 0，成功率 3/3" in report
    assert "实际请求 3，成功 3，失败 0" in report
    assert "计划外/重复异常记录：无。" in report
    assert "正式任务：计划 3，实际执行 3，完整成功 3，未执行 0，已执行未完成或失败 0" in report
    assert "| S T1 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 100.0 | 100.0 | 100.0 |" in report
    assert "| S | 场景耗时(ms) | 3 | 3 | 0 | 0 | 500.0 | 500.0 | 500.0 |" in report
    assert "正式完整成功任务数：3（分母；计划正式任务 3）" in report
    assert "正式每成功任务已知费用：**0.002**" in report


def test_r301_case_b_extra_record_keeps_plan_columns():
    """用例 B：追加 S-formal4-t1——计划列仍 3，计划内成功 3/3，额外 1 单列，费用 0.008，每成功任务 0.008÷3。"""
    requests, observations = _slot_fixture([
        ("S-formal1-t1", 1, 1, 1), ("S-formal2-t1", 2, 1, 2), ("S-formal3-t1", 3, 1, 3),
        ("S-formal4-t1", 4, 1, 4, _slow_raw())])
    manifest = _slot_manifest()
    annotated, _ = perf.annotate_requests(requests, observations)
    counts = perf.classify_request_slots(annotated, manifest)["counts"]
    assert counts["planned_slots"] == 3 and counts["success_slots"] == 3
    assert counts["empty_slots"] == 0
    assert counts["out_of_plan_records"] == 1 and counts["anomaly_records"] == 1
    tasks = perf.evaluate_tasks(annotated, manifest)
    assert len([t for t in tasks if t["planned"]]) == 3
    assert sum(t["complete_success"] for t in tasks) == 3  # 完整成功任务仍为 3
    unplanned = [t for t in tasks if not t["planned"]]
    assert len(unplanned) == 1 and unplanned[0]["task_id"] == "S-formal4"
    costs = perf.compute_scoped_costs(perf.split_observations_by_phase(observations, annotated),
                                      manifest["pricing"], tasks)
    assert costs["formal_tasks"] == 3 and costs["formal_unplanned_tasks"] == 1
    assert costs["formal"]["known_total"] == 0.008  # 分子保留计划外请求已发生费用
    assert costs["per_success_task_formal"] == 0.00266667  # 0.008 ÷ 3（计划内完整成功任务）
    report = perf.build_report_md(requests, observations, manifest)
    assert "计划槽位结果：成功 3，失败 0，重复异常 0，未执行 0，成功率 3/3" in report
    assert "实际请求 4，成功 4，失败 0" in report
    assert "S-formal4-t1" in report and "S/formal/4/T1" in report and "计划外记录" in report
    assert "正式任务：计划 3，实际执行 3，完整成功 3，未执行 0，已执行未完成或失败 0；计划外任务 1 个" in report
    # 计划内逐轮与场景耗时不受第四次影响（慢样本 900/1000/1100ms 未混入）
    assert "| S T1 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 100.0 | 100.0 | 100.0 |" in report
    assert "| S T1 | 请求完成耗时(ms) | 3 | 3 | 0 | 0 | 400.0 | 400.0 | 400.0 |" in report
    assert "| S | 场景耗时(ms) | 3 | 3 | 0 | 0 | 500.0 | 500.0 | 500.0 |" in report
    assert "正式完整成功任务数：3（分母；计划正式任务 3）" in report
    assert "正式每成功任务已知费用：**0.00266667**" in report


def test_r301_case_b_report_cli_validated(tmp_path: Path, monkeypatch):
    """用例 B 经落盘后的 report CLI 验证：计划列冻结、字节一致、原文件不回写。"""
    requests, observations = _slot_fixture([
        ("S-formal1-t1", 1, 1, 1), ("S-formal2-t1", 2, 1, 2), ("S-formal3-t1", 3, 1, 3),
        ("S-formal4-t1", 4, 1, 4, _slow_raw())])
    manifest = _slot_manifest()
    monkeypatch.setattr(perf, "PERF_ROOT", tmp_path)
    run_dir = tmp_path / "case-b-run"
    before = None
    out1, out2 = tmp_path / "case-b-out1", tmp_path / "case-b-out2"
    report1 = _report_cli(run_dir, manifest, requests, observations, out1)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    report2 = _report_cli(run_dir, manifest, requests, observations, out2)
    assert "正式任务：计划 3" in report1 and "计划正式任务 3" in report1
    assert "| S | 场景耗时(ms) | 3 | 3 |" in report1
    assert "成功率 3/3" in report1 and "计划外 1" in report1
    assert report1 == report2  # 同输入重复重建字节一致
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before  # 原文件不回写


def test_r301_case_c_missing_slot_plus_extra_record():
    """用例 C：删 formal3、留 formal1/2、追加 formal4——计划仍 3、成功 2/3、未执行 1、计划外 1。"""
    requests, observations = _slot_fixture([
        ("S-formal1-t1", 1, 1, 1), ("S-formal2-t1", 2, 1, 2),
        ("S-formal4-t1", 4, 1, 4, _slow_raw())])
    manifest = _slot_manifest()
    annotated, _ = perf.annotate_requests(requests, observations)
    counts = perf.classify_request_slots(annotated, manifest)["counts"]
    assert counts["planned_slots"] == 3 and counts["success_slots"] == 2
    assert counts["empty_slots"] == 1  # 实际条数 3 == 计划 3，也不显示“全部执行”
    assert counts["out_of_plan_records"] == 1
    tasks = perf.evaluate_tasks(annotated, manifest)
    assert [t["status"] for t in tasks if t["planned"] and t["rep"] == 3] == ["not_executed"]
    assert sum(t["complete_success"] for t in tasks) == 2
    report = perf.build_report_md(requests, observations, manifest)
    assert "计划槽位结果：成功 2，失败 0，重复异常 0，未执行 1，成功率 2/3" in report
    assert "实际请求 3，成功 3，失败 0" in report
    assert "S-formal4-t1" in report and "计划外记录" in report
    assert "正式任务：计划 3，实际执行 2，完整成功 2，未执行 1，已执行未完成或失败 0；计划外任务 1 个" in report
    assert "| S T1 | 首段可见回答耗时(ms) | 2 | 3 | 0 | 1 | 100.0 | 100.0 | 100.0 |" in report
    assert "| S | 场景耗时(ms) | 2 | 3 | 0 | 1 | 500.0 | 500.0 | 500.0 |" in report
    assert "正式完整成功任务数：2（分母；计划正式任务 3）" in report


def test_r301_case_c_report_cli_validated(tmp_path: Path, monkeypatch):
    """用例 C 经落盘后的 report CLI 验证：未执行槽位不被计划外记录填补。"""
    requests, observations = _slot_fixture([
        ("S-formal1-t1", 1, 1, 1), ("S-formal2-t1", 2, 1, 2),
        ("S-formal4-t1", 4, 1, 4, _slow_raw())])
    manifest = _slot_manifest()
    monkeypatch.setattr(perf, "PERF_ROOT", tmp_path)
    run_dir = tmp_path / "case-c-run"
    out1, out2 = tmp_path / "case-c-out1", tmp_path / "case-c-out2"
    report1 = _report_cli(run_dir, manifest, requests, observations, out1)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    report2 = _report_cli(run_dir, manifest, requests, observations, out2)
    assert "计划槽位结果：成功 2，失败 0，重复异常 0，未执行 1，成功率 2/3" in report1
    assert "正式任务：计划 3" in report1 and "计划正式任务 3" in report1
    assert report1 == report2
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def test_r301_case_d_missing_slot_plus_duplicate_record():
    """用例 D：删 formal3、重复 formal1——重复异常不计成功、不填补 formal3，费用保留。"""
    requests, observations = _slot_fixture([
        ("S-formal1-t1", 1, 1, 1), ("S-formal1-t1-dup", 1, 1, 2), ("S-formal2-t1", 2, 1, 3)])
    manifest = _slot_manifest()
    annotated, _ = perf.annotate_requests(requests, observations)
    slot_info = perf.classify_request_slots(annotated, manifest)
    counts = slot_info["counts"]
    assert counts["planned_slots"] == 3 and counts["success_slots"] == 1
    assert counts["duplicate_slots"] == 1 and counts["empty_slots"] == 1  # formal3 仍未执行
    assert counts["duplicate_records"] == 2 and counts["out_of_plan_records"] == 0
    status = slot_info["slot_status"]
    assert status[("S", "formal", 1, 1)] == "duplicate"  # 不任选一条算成功
    assert status[("S", "formal", 2, 1)] == "success"
    assert status[("S", "formal", 3, 1)] == "not_executed"
    tasks = perf.evaluate_tasks(annotated, manifest)
    assert [t["complete_success"] for t in tasks] == [False, True, False]  # formal1 重复无效、formal2 成功
    costs = perf.compute_scoped_costs(perf.split_observations_by_phase(observations, annotated),
                                      manifest["pricing"], tasks)
    assert costs["formal"]["known_total"] == 0.006  # 已发生可核实费用保留（3 条 × 0.002）
    assert costs["formal_complete_success_tasks"] == 1
    assert costs["per_success_task_formal"] == 0.006
    report = perf.build_report_md(requests, observations, manifest)
    assert "计划槽位结果：成功 1，失败 0，重复异常 1，未执行 1，成功率 1/3" in report
    assert "S-formal1-t1-dup" in report and "重复记录" in report
    assert "| S T1 | 首段可见回答耗时(ms) | 1 | 3 | 1 | 1 | 100.0 | 100.0 | 100.0 |" in report
    assert "| S | 场景耗时(ms)（未计入不完整任务：['S-formal1']） | 1 | 3 | 1 | 1 | 500.0 | 500.0 | 500.0 |" in report
    assert "正式完整成功任务数：1（分母；计划正式任务 3）" in report


def test_r305_case_e_zero_plan_no_division_and_extra_kept():
    """用例 E：计划为 0 时不除零，额外记录保留。"""
    manifest = _slot_manifest(warmup_passes=0, formal_reps=0)
    requests, observations = _slot_fixture([("S-formal1-t1", 1, 1, 1)])
    annotated, _ = perf.annotate_requests(requests, observations)
    counts = perf.classify_request_slots(annotated, manifest)["counts"]
    assert counts["planned_slots"] == 0 and counts["out_of_plan_records"] == 1
    report = perf.build_report_md(requests, observations, manifest)
    assert "计划样本：预热 0 + 正式 0 = 0" in report
    assert "成功率 N/A（计划槽位 0）" in report
    assert "S-formal1-t1" in report and "计划外记录" in report
    costs = perf.compute_scoped_costs(perf.split_observations_by_phase(observations, annotated),
                                      manifest["pricing"], perf.evaluate_tasks(annotated, manifest))
    assert costs["formal_tasks"] == 0 and costs["per_success_task_formal"] == "N/A"


def test_r305_case_e_warmup_formal_slots_do_not_fill_each_other():
    """用例 E：预热与正式身份匹配规则一致、互不填补。"""
    manifest = _slot_manifest(warmup_passes=1, formal_reps=1)
    requests, observations = _slot_fixture([
        ("S-formal1-t1", 1, 1, 1), ("S-formal2-t1", 2, 1, 2)])  # warmup 槽位无记录
    annotated, _ = perf.annotate_requests(requests, observations)
    slot_info = perf.classify_request_slots(annotated, manifest)
    counts = slot_info["counts"]
    assert counts["planned_slots"] == 2 and counts["success_slots"] == 1
    assert counts["empty_slots"] == 1  # warmup1-t1 仍为未执行
    assert counts["out_of_plan_records"] == 1  # formal rep2 不填补 warmup 槽位
    assert slot_info["slot_status"][("S", "warmup", 1, 1)] == "not_executed"
    report = perf.build_report_md(requests, observations, manifest)
    assert "计划槽位结果：成功 1，失败 0，重复异常 0，未执行 1，成功率 1/2" in report


# ---------------------------------------------------------------------------
# 第三阶段第二步（OPT-07）：冻结场景对照材料（新版本 + 哈希 + 逐项差异）
# ---------------------------------------------------------------------------

OPT1_CASES_PATH = Path(__file__).with_name("performance_cases_opt1.json")


def test_opt7_opt1_cases_keep_plan_and_only_allow_path_differences():
    """新版本场景文件：问题/顺序/轮数/联网开关/计划一致，差异仅限节点顺序与 clarify 检索省略。"""
    import hashlib

    old = perf.load_cases(CASES_PATH)
    new = perf.load_cases(OPT1_CASES_PATH)
    assert perf.validate_cases(new) == []
    # 记录的旧哈希与实际冻结文件一致（不覆盖旧文件）
    assert new["derived_from"]["file_sha256"] == hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    assert new["derived_from"]["content_sha256"] == perf.cases_sha256(old)

    old_scenarios = {s["id"]: s for s in old["scenarios"]}
    new_scenarios = {s["id"]: s for s in new["scenarios"]}
    assert [s["id"] for s in new["scenarios"]] == [s["id"] for s in old["scenarios"]]

    documented = {
        (d["scenario"], d["turn"], d["field"]): (d["old"], d["new"])
        for d in new["path_differences"]
    }
    actual_diffs = {}
    for sid, old_scenario in old_scenarios.items():
        new_scenario = new_scenarios[sid]
        assert len(new_scenario["turns"]) == len(old_scenario["turns"])  # 轮数一致
        for turn_index, (old_turn, new_turn) in enumerate(
            zip(old_scenario["turns"], new_scenario["turns"]), start=1
        ):
            assert new_turn["question"] == old_turn["question"]  # 问题与顺序一致
            assert new_turn["use_web_search"] == old_turn["use_web_search"]  # 联网开关一致
            old_path, new_path = old_turn["expected_path"], new_turn["expected_path"]
            assert new_path["answer_mode"] == old_path["answer_mode"]
            for key in ("llm_calls", "tool_executions", "tool_inits"):
                assert new_path[key] == old_path[key], (sid, turn_index, key)  # 不改调用计划
            for key in ("nodes", "retrieval_steps"):
                if new_path[key] != old_path[key]:
                    actual_diffs[(sid, turn_index, key)] = (old_path[key], new_path[key])
    assert actual_diffs == documented  # 实际逐项差异与文件内记录一致
    for (sid, turn, key), (old_value, new_value) in actual_diffs.items():
        if key == "nodes":
            if (sid, turn) == ("PERF-CALC-01", 1):
                expected = sorted(old_value)
                expected.remove("retrieve")
                assert sorted(new_value) == expected
            else:
                assert sorted(new_value) == sorted(old_value)  # 仅顺序重排，不放宽路径检查
        else:
            assert (sid, turn, key) == ("PERF-CALC-01", 1, "retrieval_steps")
            assert old_value == ["vector", "bm25", "rerank"] and new_value == []


def test_opt7_opt1_cases_replay_projection_counts():
    """四场景原计划离线重放的检索次数投影（OPT-07）：槽位不变，仅 CALC T1 跳过检索。"""
    old = perf.load_cases(CASES_PATH)
    new = perf.load_cases(OPT1_CASES_PATH)
    warmup_passes, formal_reps = 1, 3
    total_turns = sum(len(s["turns"]) for s in new["scenarios"])
    clarify_turns = sum(
        1
        for s in new["scenarios"]
        for t in s["turns"]
        if t["expected_path"]["answer_mode"] == "deterministic_clarify"
    )
    assert (warmup_passes + formal_reps) * total_turns == 24  # 业务请求槽位不变
    assert (warmup_passes + formal_reps) * clarify_turns == 4  # 4 个 CALC T1 跳过检索
    assert (warmup_passes + formal_reps) * total_turns - (
        warmup_passes + formal_reps
    ) * clarify_turns == 20  # 检索/向量/重排逻辑次数 24 → 20
    assert formal_reps * (total_turns - clarify_turns) == 15  # 正式检索次数 18 → 15
    # 固定模型桩下 LLM 逻辑调用数保持原计划
    for old_s, new_s in zip(old["scenarios"], new["scenarios"]):
        for old_t, new_t in zip(old_s["turns"], new_s["turns"]):
            assert old_t["expected_path"]["llm_calls"] == new_t["expected_path"]["llm_calls"]


# ---------------------------------------------------------------------------
# E-102 R1：清理状态真实性（端口/进程诊断/终止记账/整体判定）
# ---------------------------------------------------------------------------


class _CmdResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_r101_saturated_backlog_listener_is_not_reported_released(tmp_path: Path, monkeypatch):
    """R1-01 真实反例：接收队列满时连接超时，但监听器 SO_ACCEPTCONN 仍为 True。"""
    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.5)
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    clients = []
    try:
        for _ in range(4):  # 填满接收队列，使后续连接超时
            try:
                clients.append(socket.create_connection(("127.0.0.1", port), timeout=0.3))
            except Exception:  # noqa: BLE001
                break
        assert server.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1  # 独立判据：仍监听
        state, source, reason = perf._listen_state(port, probe_timeout=0.3)
        assert state == "listening"  # 不得判释放（连接超时但监听仍在）
        process = FakeProcess()
        process.pid = 999999999
        info = perf.stop_server(process, port, tmp_path)
        assert info["port_released"] is False  # 真实仍监听→不能报成功
        assert info["port_state"] in {"listening", "unknown"}
        assert info["cleanup_complete"] is False
    finally:
        for client in clients:
            client.close()
        server.close()
    # 收尾后独立判据：监听已关闭
    assert server.fileno() == -1


def test_r101_timeout_permission_generic_unknown_and_positive_controls(monkeypatch):
    """R1-01：TimeoutError/PermissionError/一般异常＝未知；拒绝连接/连接成功为正向对照。"""
    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(perf, "_tcp_listen_table", lambda: None)

    def probe(_port, timeout=1.0):
        return ("not_connected", "TimeoutError: timed out")

    monkeypatch.setattr(perf, "_port_probe", probe)
    assert perf._listen_state(1)[0] == "unknown"

    for reason in ("PermissionError: access denied", "OSError: boom"):
        monkeypatch.setattr(perf, "_port_probe", lambda _p, timeout=1.0, r=reason: ("not_connected", r))
        assert perf._listen_state(1)[0] == "unknown", reason

    # 正向对照：明确拒绝连接＝确认释放；连接成功＝确认仍可连接
    monkeypatch.setattr(
        perf, "_port_probe",
        lambda _p, timeout=1.0: ("not_connected", "ConnectionRefusedError: refused"),
    )
    assert perf._listen_state(1) == ("released", "connect_refused", "ConnectionRefusedError: refused")
    monkeypatch.setattr(perf, "_port_probe", lambda _p, timeout=1.0: ("connected", None))
    assert perf._listen_state(1)[0] == "listening"

    # 未知状态不得在 stop_server 中被写成释放成功
    monkeypatch.setattr(perf, "_port_probe", lambda _p, timeout=1.0: ("not_connected", "TimeoutError: timed out"))
    process = FakeProcess()
    process.pid = 999999999
    info = perf.stop_server(process, 41234, Path("."))
    assert info["port_released"] is False
    assert info["port_state"] == "unknown"
    assert "TimeoutError" in info["port_error"]


def test_r101_released_and_listening_positive_controls(tmp_path: Path):
    """R1-01 正向对照（真实套接字）：正常监听判 listening；真正关闭判 released（明确依据）。"""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]
    try:
        state, source, _ = perf._listen_state(port, probe_timeout=1.0)
        assert state == "listening" and source in {"tcp_table", "connect_connected"}
    finally:
        server.close()
    state, source, _ = perf._listen_state(port, probe_timeout=1.0)
    assert state == "released" and source in {"tcp_table", "connect_refused"}


def test_r102_bounded_wait_persistent_unknown_and_listening_never_succeed(tmp_path: Path, monkeypatch):
    """R1-02：持续诊断错误/持续监听到截止时间仍未确认释放→不成功或未知；首次与后续样本分开带时间戳。"""
    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.4)
    for fixed in (("unknown", "connect_unclassified", "TimeoutError: x"),
                  ("listening", "tcp_table", "owner_pid=1")):
        monkeypatch.setattr(perf, "_listen_state", lambda _p, probe_timeout=1.0, f=fixed: f)
        monkeypatch.setattr(perf, "_descendant_pids", lambda _pid: ([], None))
        process = FakeProcess()
        process.pid = 999999999
        started = time.monotonic()
        info = perf.stop_server(process, 41235, tmp_path)
        elapsed = time.monotonic() - started
        assert elapsed < 5  # 有上限，不会无限等待
        assert info["port_released"] is False
        assert info["port_state"] == fixed[0]
        assert info["cleanup_complete"] is False
        assert info["port_initial_check"]["state"] == fixed[0]  # 首次结果单独保留
        assert len(info["port_check_samples"]) >= 2  # 首次＋后续样本
        assert all("t_ms" in s for s in info["port_check_samples"])  # 带时间戳


def test_r103_process_table_failures_are_unknown_not_empty(monkeypatch):
    """进程查询非零/空白/异常解析/命令异常＝未知，不能当作空进程表。"""
    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **k: _CmdResult(returncode=1, stderr="Access is denied"),
    )
    table, reason = perf._process_table()
    assert table is None and "cim_exit_1" in reason

    monkeypatch.setattr(perf.subprocess, "run", lambda *a, **k: _CmdResult(returncode=0, stdout=""))
    table, reason = perf._process_table()
    assert table is None and reason == "cim_blank_output"

    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **k: _CmdResult(returncode=0, stdout="not json"),
    )
    table, reason = perf._process_table()
    assert table is None and "untrusted_parse" in reason

    def boom(*a, **k):
        raise FileNotFoundError("powershell missing")

    monkeypatch.setattr(perf.subprocess, "run", boom)
    table, reason = perf._process_table()
    assert table is None and "cim_exception" in reason

    # 后代语义：未知 ≠ 无后代
    descendants, reason = perf._descendant_pids(1)
    assert descendants is None and reason  # 诊断失败→未知，不是 []

    # 正向对照：真实可信进程表非空
    monkeypatch.undo()
    table, reason = perf._process_table()
    assert table and reason is None, reason

    # 可信表且无后代（不存在的父 PID）
    descendants, reason = perf._descendant_pids(999999999)
    assert descendants == [] and reason is None  # 已确认无后代


def test_r103_pid_alive_query_failures_unknown_with_real_controls(monkeypatch):
    """R1-03：OpenProcess/GetExitCodeProcess 查询失败＝未知；真实存活/退出为对照。"""
    import ctypes as ctypes_mod

    class FakeKernel:
        def __init__(self, handle=0, getexit_ok=True, code=259, err=5):
            self._handle = handle
            self._getexit_ok = getexit_ok
            self._code = code
            self._err = err

        def OpenProcess(self, *_):
            monkeypatch.setattr(ctypes_mod, "get_last_error", lambda: self._err)
            return self._handle

        def GetExitCodeProcess(self, _h, out):
            if not self._getexit_ok:
                monkeypatch.setattr(ctypes_mod, "get_last_error", lambda: self._err)
                return 0
            out._obj.value = self._code
            return 1

        def CloseHandle(self, _h):
            return 1

    monkeypatch.setattr(ctypes_mod, "WinDLL", lambda *a, **k: FakeKernel(handle=0, err=5))
    assert perf._pid_alive(4) == (None, "openprocess_error_5")  # 权限拒绝＝未知

    monkeypatch.setattr(ctypes_mod, "WinDLL", lambda *a, **k: FakeKernel(handle=0, err=87))
    assert perf._pid_alive(999999999) == (False, "process_not_found")  # OS 明确应答无此进程

    monkeypatch.setattr(
        ctypes_mod, "WinDLL",
        lambda *a, **k: FakeKernel(handle=1, getexit_ok=False, err=6),
    )
    assert perf._pid_alive(4)[0] is None  # GetExitCodeProcess 失败＝未知

    monkeypatch.undo()
    assert perf._pid_alive(os.getpid()) == (True, "still_active")  # 真实存活对照
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    alive, reason = perf._pid_alive(child.pid)  # 真实退出对照
    assert alive is False and reason in {"process_not_found", f"exit_code_{child.returncode}"}


def _patch_descendant_cleanup(monkeypatch, desc_pid, alive_seq, cmd_results):
    """注入后代清单、存活查询序列与 taskkill 命令桩，返回命令调用记录。"""
    calls = []
    monkeypatch.setattr(perf, "_descendant_pids", lambda _pid: ([desc_pid], None))
    alive_iter = iter(alive_seq)
    monkeypatch.setattr(perf, "_pid_alive", lambda _pid: next(alive_iter))
    results = iter(cmd_results)

    def fake_run(args, capture_output=False, timeout=None):
        if args[0] == "taskkill":
            calls.append(list(args))
            item = next(results)
            if isinstance(item, Exception):
                raise item
            return item
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(perf.subprocess, "run", fake_run)
    return calls


def test_r104_taskkill_failure_and_false_success_not_recorded_as_killed(monkeypatch):
    """R1-04：非零返回＋stderr＋仍存活、抛错、返回0仍存活均不得列入已确认终止。"""
    # 非零返回 + stderr + 仍存活
    calls = _patch_descendant_cleanup(
        monkeypatch, 112233, [(True, "still_active"), (True, "still_active")],
        [_CmdResult(returncode=5, stderr="Access denied")],
    )
    info = perf._terminate_own_process(FakeProcess())
    assert calls == [["taskkill", "/PID", "112233", "/F"]]  # 命令分支实际被调用且参数正确
    entry = info["descendant_results"][0]
    assert entry["command_ok"] is False and "Access denied" in entry["command_stderr"]
    assert entry["outcome"] == "still_alive"
    assert info["descendants_killed"] == []

    # 命令抛错
    _patch_descendant_cleanup(
        monkeypatch, 112233, [(True, "still_active"), (None, "openprocess_error_5")],
        [RuntimeError("taskkill crashed")],
    )
    info = perf._terminate_own_process(FakeProcess())
    entry = info["descendant_results"][0]
    assert entry["command_ok"] is False and "RuntimeError" in entry["command_stderr"]
    assert entry["outcome"] == "unconfirmed_after"  # 终止后查询失败＝无法确认
    assert info["descendants_killed"] == []

    # 返回 0 但仍存活
    _patch_descendant_cleanup(
        monkeypatch, 112233, [(True, "still_active"), (True, "still_active")],
        [_CmdResult(returncode=0)],
    )
    info = perf._terminate_own_process(FakeProcess())
    entry = info["descendant_results"][0]
    assert entry["command_ok"] is True and entry["outcome"] == "still_alive"  # 返回0≠确认
    assert info["descendants_killed"] == []


def test_r104_confirmed_killed_exited_on_own_recorded_separately(monkeypatch):
    """R1-04：成功后确认退出＝confirmed_killed；清理前已自行退出＝exited_on_own（未发起命令）。"""
    calls = _patch_descendant_cleanup(
        monkeypatch, 112233, [(True, "still_active"), (False, "process_not_found")],
        [_CmdResult(returncode=0)],
    )
    info = perf._terminate_own_process(FakeProcess())
    entry = info["descendant_results"][0]
    assert calls and entry["outcome"] == "confirmed_killed"
    assert info["descendants_killed"] == [112233]

    calls = _patch_descendant_cleanup(
        monkeypatch, 112233, [(False, "process_not_found"), (False, "process_not_found")],
        [],
    )
    info = perf._terminate_own_process(FakeProcess())
    entry = info["descendant_results"][0]
    assert calls == [] and entry["kill_attempted"] is False  # 已自行退出，不发起命令
    assert entry["outcome"] == "exited_on_own"
    assert info["descendants_killed"] == []


def test_r105_cleanup_complete_requires_all_confirmed(monkeypatch, tmp_path: Path):
    """R1-05：整体判定贯通——父进程退出≠整棵树清理完成；未知/仍存活即不成功。"""
    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.3)
    released_port = perf._free_port()  # 真正无监听的端口

    # 后代仍存活 → 不完整
    _patch_descendant_cleanup(
        monkeypatch, 112233, [(True, "still_active"), (True, "still_active")],
        [_CmdResult(returncode=5, stderr="Access denied")],
    )
    info = perf.stop_server(FakeProcess(), released_port, tmp_path)
    assert info["terminated"] is True  # 父进程事实为真
    assert info["cleanup_complete"] is False  # 但整体判定不成立
    assert "descendants_unconfirmed_or_alive" in info["cleanup_incomplete_reasons"]

    # 后代诊断未知 → 不完整
    monkeypatch.setattr(perf, "_descendant_pids", lambda _pid: (None, "wmic_exit_1: Access is denied"))
    info = perf.stop_server(FakeProcess(), released_port, tmp_path)
    assert info["cleanup_complete"] is False

    # 全部确认（无后代＋端口确认释放）→ 完整
    monkeypatch.setattr(perf, "_descendant_pids", lambda _pid: ([], None))
    info = perf.stop_server(FakeProcess(), released_port, tmp_path)
    assert info["port_released"] is True
    assert info["cleanup_complete"] is True
    assert info["cleanup_incomplete_reasons"] == []


# ---------------------------------------------------------------------------
# E-102 R2：可信进程表 / 可落盘错误证据 / 剩余等待预算
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('rows', [
    [], None, {}, [None],
    [{"ProcessId": True, "ParentProcessId": 0}],
    [{"ProcessId": 1, "ParentProcessId": -1}],
    [{"ProcessId": "1", "ParentProcessId": 0}],
    [{"ProcessId": 1, "ParentProcessId": 0}, {"ProcessId": 1, "ParentProcessId": 2}],
])
def test_cim_invalid_process_table_remains_unknown(monkeypatch, rows):
    monkeypatch.setattr(perf.subprocess, 'run', lambda *a, **k: _CmdResult(stdout=json.dumps(rows)))
    table, reason = perf._process_table()
    assert table is None and 'untrusted_parse' in reason


def test_cim_query_does_not_depend_on_wmic(monkeypatch):
    def run(command, **kwargs):
        assert command[:4] == ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command']
        assert 'Get-CimInstance -ClassName Win32_Process' in command[4]
        assert '-ErrorAction Stop' in command[4]
        assert kwargs['timeout'] == 30
        return _CmdResult(stdout='[{"ProcessId":2,"ParentProcessId":1}]')

    monkeypatch.setattr(perf.subprocess, 'run', run)
    assert perf._process_table() == ({2: 1}, None)


def test_cim_timeout_remains_unknown(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired('powershell.exe', 30)

    monkeypatch.setattr(perf.subprocess, 'run', run)
    table, reason = perf._process_table()
    assert table is None and 'TimeoutExpired' in reason


def test_r201_missing_or_nonnumeric_parent_is_unknown_not_zero(monkeypatch):
    """R2-01：空白/非数字父字段＝不可信父关系，不能静默变 0；合法 0/正数父 PID 正常解析。"""
    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **k: _CmdResult(
            returncode=0,
            stdout='[{"ProcessId":112233,"ParentProcessId":null},{"ProcessId":123456,"ParentProcessId":0}]',
        ),
    )
    table, reason = perf._process_table()
    assert table is None and "untrusted_parse" in reason  # 空白父字段→未知，不是 {112233: 0}

    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **k: _CmdResult(
            returncode=0,
            stdout='[{"ProcessId":123,"ParentProcessId":"abc"},{"ProcessId":222,"ParentProcessId":0}]',
        ),
    )
    table, reason = perf._process_table()
    assert table is None and "untrusted_parse" in reason  # 非数字父字段→未知

    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **k: _CmdResult(
            returncode=0,
            stdout='[{"ProcessId":123456,"ParentProcessId":0},{"ProcessId":222222,"ParentProcessId":123456}]',
        ),
    )
    table, reason = perf._process_table()
    assert reason is None
    assert table == {123456: 0, 222222: 123456}  # 合法 0 与合法正数父 PID

    # 有效行与无效行混合的表：整体不可信，不得跳过问题行后宣布完整
    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **k: _CmdResult(
            returncode=0,
            stdout='[{"ProcessId":1,"ParentProcessId":0},{"ProcessId":2}]',
        ),
    )
    table, reason = perf._process_table()
    assert table is None and "untrusted_parse" in reason


def test_r201_unknown_parent_relations_block_cleanup_complete(tmp_path: Path, monkeypatch):
    """R2-01：unknown 贯穿 _descendant_pids → stop_server，cleanup_complete 不得为 true。"""
    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **k: _CmdResult(
            returncode=0,
            stdout='[{"ProcessId":112233,"ParentProcessId":null},{"ProcessId":123456,"ParentProcessId":0}]',
        ),
    )
    released_port = perf._free_port()
    process = FakeProcess()
    process.pid = 999999999
    info = perf.stop_server(process, released_port, tmp_path)
    assert info["descendants"] is None  # 未知，不是 []
    assert "untrusted_parse" in info["descendants_diagnostic"]
    assert info["port_released"] is True  # 端口确有明确依据
    assert info["cleanup_complete"] is False  # 但父关系未知→整体不成立
    assert "descendants_unconfirmed_or_alive" in info["cleanup_incomplete_reasons"]


def test_r203_probe_timeout_never_exceeds_remaining_budget(tmp_path: Path, monkeypatch):
    """R2-03：可控时钟验证传入 timeout/sleep 参数——剩余10ms/不足100ms/恰好截止边界。"""

    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

    clock = FakeClock()
    sleeps = []
    probes = []

    def fake_sleep(seconds):
        sleeps.append(round(seconds, 6))
        clock.now += seconds

    def fake_listen(port, probe_timeout=1.0):
        probes.append(round(probe_timeout, 6))
        return ("unknown", "connect_unclassified", "TimeoutError: x")

    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(perf, "PORT_RELEASE_POLL_SECONDS", 0.02)
    monkeypatch.setattr(perf.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(perf.time, "sleep", fake_sleep)
    monkeypatch.setattr(perf, "_listen_state", fake_listen)
    monkeypatch.setattr(perf, "_descendant_pids", lambda _pid: ([], None))
    process = FakeProcess()
    process.pid = 999999999
    info = perf.stop_server(process, 41236, tmp_path)

    # 传入 timeout：首次 0.05；随后 0.03、0.01（最后一次探测恰好耗尽剩余预算=10ms）
    assert probes == [0.05, 0.03, 0.01]
    assert all(t <= 0.05 for t in probes)  # 均不超过预算
    assert probes[-1] == 0.01  # 剩余 10ms 边界：不再抬高到 100ms
    # 传入 sleep：均不超过安排时的剩余预算
    assert sleeps == [0.02, 0.02, 0.01]
    assert clock.now == 0.05  # 恰好截止：不再探测、不再 sleep
    assert info["port_released"] is False  # 截止时仍 unknown→不判成功
    assert info["port_state"] == "unknown"
    assert info["cleanup_complete"] is False
    assert len(info["port_check_samples"]) == len(probes)  # 首次＋后续样本齐全带时间戳


def test_r203_no_probe_or_sleep_past_deadline(tmp_path: Path, monkeypatch):
    """R2-03：首次探测已耗尽预算（已超时）→ 不再发起探测或 sleep。"""

    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

    clock = FakeClock()
    sleeps = []
    probes = []

    def fake_listen(port, probe_timeout=1.0):
        probes.append(round(probe_timeout, 6))
        clock.now += 0.1  # 探测本身耗尽预算之外的时间
        return ("unknown", "connect_unclassified", "TimeoutError: x")

    monkeypatch.setattr(perf, "PORT_RELEASE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(perf.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(perf.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(perf, "_listen_state", fake_listen)
    monkeypatch.setattr(perf, "_descendant_pids", lambda _pid: ([], None))
    process = FakeProcess()
    process.pid = 999999999
    info = perf.stop_server(process, 41237, tmp_path)
    assert probes == [0.05]  # 仅首次探测
    assert sleeps == []  # 截止后不 sleep
    assert info["port_released"] is False
    assert info["cleanup_complete"] is False
