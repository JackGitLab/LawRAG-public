"""评测实现（tests/run_quality_evaluation.py）的返工行为测试（R1）。

覆盖 outputs/STAGE1_ACCEPTANCE_20261009.md F-01～F-08 的反例，并保留原评测测试意图。
全部离线：真实模型、搜索、索引与用户数据目录均不触碰；
“实际图 + 离线桩”用例使用真实 LangGraph 编排 + 离线模型/检索桩 + tmp 数据库。
"""

import hashlib
import json
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("run_quality_evaluation.py")
CASES_PATH = Path(__file__).with_name("evaluation_cases.json")

SPEC = spec_from_file_location("quality_evaluation_runner", SCRIPT_PATH)
assert SPEC and SPEC.loader
evaluation = module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def cases_sha(cases: dict) -> str:
    """与 run_quality_evaluation.rescore_convert 相同的题集 hash 口径。"""
    payload = json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def make_turn(**overrides) -> dict:
    turn = {
        "question": "测试问题",
        "use_web_search": False,
        "expected_articles": [],
        "required_points": ["要点A"],
        "forbidden_claims": ["禁止结论A"],
        "expected_tools": [],
    }
    turn.update(overrides)
    return turn


def make_scenario(**overrides) -> dict:
    scenario = {
        "id": "S-1",
        "category": "library_qa",
        "subcategory": "multi_article",
        "execution_scope": "graph",
        "annotation_status": "executor_annotated",
        "turns": [make_turn()],
        "completion_criteria": ["完成条件A"],
        "evidence": [{"note": "note"}],
        "difference_from_regression": "d",
    }
    scenario.update(overrides)
    return scenario


def make_record(scenario_id="S-1", turn_index=0, attempt=1, **overrides) -> dict:
    record = {
        "record_type": "turn_result",
        "run_id": "mini",
        "scenario_id": scenario_id,
        "turn_index": turn_index,
        "attempt": attempt,
        "execution_scope": "graph",
        "status": "ok",
        "question": "测试问题",
        "citations": [],
        "answer": "回答",
        "planned_tool_calls": [],
        "tool_text": "",
        "web_sources": [],
    }
    record.update(overrides)
    record["result_hash"] = evaluation.compute_result_hash(record)
    return record


def citation(source: str, article: str, text: str = "条文内容") -> dict:
    return {"source": source, "article": article, "score": 0.9, "text": text}


def make_turn_review(turn: dict, record: dict, is_route_only: bool = False, **overrides) -> dict:
    expects_refusal = evaluation.frozen_expects_refusal(turn)
    entry = {
        "scenario_id": record["scenario_id"],
        "turn_index": record["turn_index"],
        "attempt_used": record["attempt"],
        "result_hash": record["result_hash"],
        "response_type": "answered",
        "required_points": [
            {"point": point, "result": "PASS", "reason": ""} for point in turn.get("required_points") or []
        ],
        "forbidden_claims": [
            {"claim": claim, "result": "PASS", "reason": ""} for claim in turn.get("forbidden_claims") or []
        ],
        "conclusions": [],
        "conclusions_extraction_complete": True,
        "insufficiency_handling": {
            "applicable": expects_refusal,
            "result": "PASS" if expects_refusal else "N/A",
            "reason": "",
        },
        "wrong_refusal": {
            "applicable": (not expects_refusal) and not is_route_only,
            "result": "PASS" if (not expects_refusal and not is_route_only) else "N/A",
            "reason": "",
        },
        "evidence_location": "results.jsonl :: fixture",
        "reviewer": "tester",
        "reviewer_role": "executor_preliminary",
    }
    entry.update(overrides)
    return entry


def make_scenario_review(scenario: dict, group: dict, **overrides) -> dict:
    entry = {
        "scenario_id": scenario["id"],
        "attempt_used": group.get("attempt"),
        "result_hashes": {
            str(index): (record.get("result_hash") or evaluation.compute_result_hash(record))
            for index, record in (group.get("turns") or {}).items()
        },
        "criteria_results": [
            {"criterion": criterion, "result": "PASS", "reason": ""} for criterion in scenario["completion_criteria"]
        ],
        "reconciles_turn_failures": "",
        "evidence_location": "results.jsonl :: fixture",
        "reviewer": "tester",
        "reviewer_role": "executor_preliminary",
    }
    entry.update(overrides)
    return entry


def write_run_fixture(run_dir: Path, cases: dict, results: list, reviews: dict | None, manifest: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True)
    frozen_sha = hashlib.sha256(cases_payload.encode("utf-8")).hexdigest()
    # 按字节写入，避免 Windows 换行翻译导致文件 hash ≠ 内容 hash（与 write_frozen_cases_if_absent 一致）
    (run_dir / "cases_frozen.json").write_bytes(cases_payload.encode("utf-8"))
    (run_dir / "manifest.json").write_text(
        json.dumps(
            manifest
            or {
                "run_id": run_dir.name,
                "git_head": "test",
                "code_sha256": {"tests/evaluation_cases.json": frozen_sha},
                "models": {"chat": "stub"},
                "retrieval_strategy": "hybrid_rerank",
                "attempt_selection_policy": "最新完整 attempt",
                "real_dirs_unchanged": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if reviews is not None:
        # 复核必须绑定冻结身份（R2-02）；调用方可预先覆盖该键构造失配反例
        reviews.setdefault("cases_sha256", frozen_sha)
        (run_dir / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False), encoding="utf-8")


def snapshot_run_dir_files(run_dir: Path) -> dict:
    """运行目录普通文件 hash 清单（用于核对失败路径不静默改写证据文件）。"""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# 原测试意图保留：题集校验与无答案泄漏
# ---------------------------------------------------------------------------


def test_validate_rejects_duplicate_ids_and_bad_structure() -> None:
    bad_cases = {
        "version": "0.0.1",
        "scenarios": [
            make_scenario(
                id="DUP-1",
                evidence=[{"source": "中华人民共和国劳动合同法", "article": "第十条", "quote": "不是原文"}],
            ),
            {
                "id": "DUP-1",
                "category": "unknown",
                "subcategory": "x",
                "execution_scope": "graph",
                "annotation_status": "executor_annotated",
                "turns": [],
                "completion_criteria": [],
                "evidence": [],
                "difference_from_regression": "",
            },
        ],
    }
    kb = {("中华人民共和国劳动合同法", "第十条"): "第十条 建立劳动关系，应当订立书面劳动合同。"}
    errors = evaluation.validate_case_set(bad_cases, regression_questions=[], kb_articles=kb)
    assert any("重复的场景 ID" in error for error in errors)
    assert any("quote 不是 kb/ 原文子串" in error for error in errors)
    assert any("category 非法" in error for error in errors)


def test_validate_rejects_duplicate_questions_against_regression() -> None:
    cases = {
        "version": "0.0.1",
        "scenarios": [
            make_scenario(
                id="DUP-Q",
                category="boundary",
                subcategory="out_of_scope",
                turns=[make_turn(question="建立劳动关系后，最迟应在什么时候订立书面劳动合同？")],
            )
        ],
    }
    regression = [{"id": 1, "question": "建立劳动关系后，最迟应在什么时候订立书面劳动合同？"}]
    errors = evaluation.validate_case_set(cases, regression, kb_articles={})
    assert any("明显重复" in error for error in errors)


def test_expected_fields_never_enter_model_input(tmp_path, monkeypatch) -> None:
    captured_graph_calls = []
    captured_route_calls = []

    def fake_invoke_graph(session_id, question, use_web_search):
        captured_graph_calls.append((session_id, question, use_web_search))
        return {"search_query": question, "citations": [], "answer": "回答", "planned_tool_calls": []}

    def fake_plan_route(question, use_web_search):
        captured_route_calls.append((question, use_web_search))
        return []

    case = make_scenario(id="QA-LAB-01", turns=[make_turn(question="库内问题")])
    route_case = make_scenario(
        id="TOOL-WEB-01",
        category="tool_routing",
        subcategory="web_route",
        execution_scope="route_only",
        turns=[
            make_turn(
                question="联网问题",
                use_web_search=True,
                required_points=[],
                forbidden_claims=[],
                expected_tools=[{"name": "verify_statute_status", "args": {"query": {"match": "question"}}}],
            )
        ],
    )
    cases = {"version": "mini", "scenarios": [case, route_case]}
    monkeypatch.setattr(evaluation, "REAL_DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(evaluation, "CHROMA_SOURCE_DIR", tmp_path / "chroma")
    monkeypatch.setattr(evaluation, "REAL_KB_DIR", tmp_path / "kb")

    evaluation.run_collection(
        run_dir=tmp_path / "run",
        scenarios=[case, route_case],
        run_id="test-run",
        cases=cases,
        with_isolation=False,
        invoke_graph=fake_invoke_graph,
        plan_route=fake_plan_route,
    )

    assert len(captured_graph_calls) == 1
    assert captured_graph_calls[0][1:] == ("库内问题", False)
    assert captured_graph_calls[0][0].startswith("eval-test-run-QA-LAB-01-a1-")
    assert captured_route_calls == [("联网问题", True)]
    for record in evaluation.read_existing_results(tmp_path / "run"):
        serialized = json.dumps(record, ensure_ascii=False)
        for leaked in ("expected_articles", "required_points", "forbidden_claims", "expected_tools"):
            assert leaked not in serialized


# ---------------------------------------------------------------------------
# 原测试意图保留：工具与检索评分单测
# ---------------------------------------------------------------------------


def test_tool_name_correct_but_param_wrong_fails() -> None:
    turn = make_turn(
        expected_tools=[
            {"name": "calculate_labor_compensation", "args": {"monthly_wage": 8000, "years": 4, "kind": "N"}}
        ]
    )
    record = make_record(
        planned_tool_calls=[
            {"name": "calculate_labor_compensation", "args": {"monthly_wage": 8000, "years": 5, "kind": "N"}}
        ]
    )
    assert evaluation.score_tool_selection(turn, record)["result"] == "PASS"
    params = evaluation.score_tool_params(turn, record)
    assert params["result"] == "FAIL"
    assert "years" in params["reason"]


def test_unexpected_tool_call_fails_selection() -> None:
    turn = make_turn(expected_tools=[])
    record = make_record(
        planned_tool_calls=[{"name": "calculate_labor_compensation", "args": {"monthly_wage": 1, "years": 1}}]
    )
    selection = evaluation.score_tool_selection(turn, record)
    assert selection["result"] == "FAIL"
    assert "预期不调用" in selection["reason"]


def test_missing_call_and_fabricated_param_fail() -> None:
    turn = make_turn(
        expected_tools=[
            {"name": "calculate_labor_compensation", "args": {"monthly_wage": 8000, "years": 4, "kind": "N"}}
        ]
    )
    assert evaluation.score_tool_params(turn, make_record(planned_tool_calls=[]))["result"] == "FAIL"
    fabricated = make_record(
        planned_tool_calls=[
            {
                "name": "calculate_labor_compensation",
                "args": {"monthly_wage": 8000, "years": 4, "kind": "N", "social_average_wage": 9999},
            }
        ]
    )
    params = evaluation.score_tool_params(turn, fabricated)
    assert params["result"] == "FAIL"
    assert "编造/多余参数" in params["reason"]


def test_param_normalization_and_default_kind() -> None:
    turn = make_turn(
        expected_tools=[
            {"name": "calculate_labor_compensation", "args": {"monthly_wage": 8000, "years": 4, "kind": "N"}}
        ]
    )
    record = make_record(
        planned_tool_calls=[
            {"name": "calculate_labor_compensation", "args": {"monthly_wage": "8000 元", "years": "4 年"}}
        ]
    )
    assert evaluation.score_tool_params(turn, record)["result"] == "PASS"
    kind_wrong = make_record(
        planned_tool_calls=[
            {"name": "calculate_labor_compensation", "args": {"monthly_wage": 8000, "years": 4, "kind": "2N"}}
        ]
    )
    assert evaluation.score_tool_params(turn, kind_wrong)["result"] == "FAIL"


def test_retrieval_hit_reports_hit_and_missing_articles() -> None:
    turn = make_turn(
        expected_articles=["中华人民共和国劳动合同法 第十条", "中华人民共和国劳动合同法 第八十二条"]
    )
    record = make_record(
        citations=[
            citation("中华人民共和国劳动合同法", "第十条"),
            citation("中华人民共和国消费者权益保护法", "第二十五条"),
            citation("中华人民共和国劳动合同法", "第十九条"),
        ]
    )
    result = evaluation.score_retrieval_turn(turn, record)
    assert result["result"] == "PASS"
    assert result["hit"] == ["中华人民共和国劳动合同法 第十条"]
    assert result["missing"] == ["中华人民共和国劳动合同法 第八十二条"]


# ---------------------------------------------------------------------------
# R1-01：attempt 会话隔离与整组 attempt 选择
# ---------------------------------------------------------------------------


def test_r1_01_real_graph_two_passes_message_counts(tmp_path, monkeypatch) -> None:
    """实际 LangGraph + 离线桩：两轮场景采集两次，消息数必须为 2、4、2、4。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LAWRAG_DB_PATH", str(tmp_path / "db.sqlite3"))
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(key, "")

    import app.db
    import app.graph

    monkeypatch.setattr(app.db, "DATABASE_PATH", tmp_path / "db.sqlite3")
    app.graph.get_graph.cache_clear()
    app.db.get_connection.cache_clear()
    app.db.get_checkpointer_connection.cache_clear()

    rewrite_prompts = []

    monkeypatch.setattr(app.graph, "get_chat_model", lambda: object())
    monkeypatch.setattr(
        app.graph, "get_router_model", lambda: types.SimpleNamespace(bind_tools=lambda tools: object())
    )

    def fake_traced_invoke(model, prompt, session_id=None, tags=None):
        if tags and "rewrite" in tags:
            rewrite_prompts.append(prompt)
            return types.SimpleNamespace(content="改写后的检索问题")
        return types.SimpleNamespace(content="离线桩回答", tool_calls=[])

    monkeypatch.setattr(app.graph, "traced_invoke", fake_traced_invoke)
    monkeypatch.setattr(app.graph, "retrieve", lambda question, top_k=3, session_id=None: [])

    message_counts = []
    session_ids = []
    for pass_index in range(2):
        session_id = evaluation.new_session_id("sess", "MT-REF-01", pass_index + 1)
        session_ids.append(session_id)
        for question in ("第一轮问题", "那第二轮问题呢？"):
            evaluation._invoke_graph(session_id, question, False)
            state = app.graph.get_graph().get_state(app.graph.graph_config(session_id))
            message_counts.append(len(state.values.get("messages", [])))

    assert message_counts == [2, 4, 2, 4]
    # 改写只发生在每个会话的第二轮；第二次首轮不得因旧历史触发改写
    assert len(rewrite_prompts) == 2
    assert session_ids[0] != session_ids[1]

    app.graph.get_graph.cache_clear()
    app.db.get_connection.cache_clear()
    app.db.get_checkpointer_connection.cache_clear()


def test_r1_01_second_pass_first_turn_no_history_rewrite(tmp_path, monkeypatch) -> None:
    """第二次 attempt 的首轮 search_query 应等于原问题（无历史 → 不改写）。"""
    seen = []

    def fake_invoke_graph(session_id, question, use_web_search):
        seen.append(session_id)
        return {"search_query": question, "citations": [], "answer": "回答", "planned_tool_calls": []}

    monkeypatch.setattr(evaluation, "REAL_DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(evaluation, "CHROMA_SOURCE_DIR", tmp_path / "chroma")
    monkeypatch.setattr(evaluation, "REAL_KB_DIR", tmp_path / "kb")
    scenario = make_scenario(id="MT-REF-01", turns=[make_turn(question="Q1"), make_turn(question="Q2")])
    cases = {"version": "mini", "scenarios": [scenario]}
    run_dir = tmp_path / "run"
    for _ in range(2):
        evaluation.run_collection(
            run_dir=run_dir,
            scenarios=[scenario],
            run_id="sess",
            cases=cases,
            with_isolation=False,
            invoke_graph=fake_invoke_graph,
        )
    records = evaluation.read_existing_results(run_dir)
    attempts = sorted({record["attempt"] for record in records})
    assert attempts == [1, 2]
    first_turns = [record for record in records if record["turn_index"] == 0]
    assert first_turns[0]["session_id"] != first_turns[1]["session_id"]
    # 两轮共享会话：同一 attempt 的两条记录 session_id 相同
    attempt_one = [record for record in records if record["attempt"] == 1]
    assert attempt_one[0]["session_id"] == attempt_one[1]["session_id"]


def test_r1_01_interrupted_before_first_record_still_uses_new_session(tmp_path, monkeypatch) -> None:
    """中断发生在首条结果落盘前，下一次运行也不得复用旧会话。"""
    sessions = []
    calls = {"count": 0}

    def crashing_invoke(session_id, question, use_web_search):
        sessions.append(session_id)
        calls["count"] += 1
        raise KeyboardInterrupt("模拟中断")

    def ok_invoke(session_id, question, use_web_search):
        sessions.append(session_id)
        return {"search_query": question, "citations": [], "answer": "回答", "planned_tool_calls": []}

    monkeypatch.setattr(evaluation, "REAL_DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(evaluation, "CHROMA_SOURCE_DIR", tmp_path / "chroma")
    monkeypatch.setattr(evaluation, "REAL_KB_DIR", tmp_path / "kb")
    scenario = make_scenario(id="QA-LAB-01", turns=[make_turn(question="Q1")])
    cases = {"version": "mini", "scenarios": [scenario]}
    run_dir = tmp_path / "run"

    with pytest.raises(KeyboardInterrupt):
        evaluation.run_collection(
            run_dir=run_dir,
            scenarios=[scenario],
            run_id="sess",
            cases=cases,
            with_isolation=False,
            invoke_graph=crashing_invoke,
        )
    assert evaluation.read_existing_results(run_dir) == []  # 首条结果未落盘

    evaluation.run_collection(
        run_dir=run_dir,
        scenarios=[scenario],
        run_id="sess",
        cases=cases,
        with_isolation=False,
        invoke_graph=ok_invoke,
    )
    assert sessions[0] != sessions[1]  # attempt 编号可能相同，会话身份必须不同


def test_r1_01_attempt_group_selection_never_merges_attempts() -> None:
    """attempt 1 完整、attempt 2 只有一轮时，不得拼接旧轮次补出完成 PASS。"""
    scenario = make_scenario(id="MT-REF-01", turns=[make_turn(question="Q1"), make_turn(question="Q2")])
    cases = {"version": "mini", "scenarios": [scenario]}
    records = [
        make_record("MT-REF-01", 0, attempt=1, question="Q1", answer="旧答案一"),
        make_record("MT-REF-01", 1, attempt=1, question="Q2", answer="旧答案二"),
        make_record("MT-REF-01", 0, attempt=2, question="Q1", answer="新答案一"),
    ]
    selected = evaluation.select_attempt_groups(records, cases["scenarios"])
    group = selected["MT-REF-01"]
    assert group["attempt"] == 2
    assert set(group["turns"]) == {0}
    assert group["missing_turns"] == [1]

    result = evaluation.aggregate(cases, selected, reviews=None)
    completion = result["semantic"]["completion_graph"]
    assert completion["applicable"] == 1
    assert completion["not_verified"] == 1  # 不完整 attempt 如实显示，不回退 attempt 1
    assert completion["pass"] == 0


# ---------------------------------------------------------------------------
# R1-02：评分绑定结果身份 + 评分项完整性
# ---------------------------------------------------------------------------


def _binding_cases():
    scenario = make_scenario(
        id="S-1",
        turns=[make_turn(required_points=["要点A", "要点B"], forbidden_claims=["禁止结论A"])],
        completion_criteria=["完成条件A"],
    )
    return {"version": "mini", "scenarios": [scenario]}, scenario


def test_r1_02_stale_review_not_reused_for_new_attempt(tmp_path) -> None:
    """新答案 attempt=2 配旧 attempt=1 评分：旧 PASS 不得沿用；不完整 attempt 不得拼接旧轮次。"""
    scenario = make_scenario(id="S-1", turns=[make_turn(question="Q1"), make_turn(question="Q2")])
    cases = {"version": "mini", "scenarios": [scenario]}
    old_0 = make_record("S-1", 0, attempt=1, question="Q1", answer="旧答案一")
    old_1 = make_record("S-1", 1, attempt=1, question="Q2", answer="旧答案二")
    new_0 = make_record("S-1", 0, attempt=2, question="Q1", answer="新答案一")
    selected = evaluation.select_attempt_groups([old_0, old_1, new_0], cases["scenarios"])
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [
            make_turn_review(scenario["turns"][0], old_0),
            make_turn_review(scenario["turns"][1], old_1),
        ],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: old_0, 1: old_1}})],
    }
    write_run_fixture(tmp_path, cases, [old_0, old_1, new_0], reviews)
    result = evaluation.run_report(tmp_path)["result"]

    assert result["semantic"]["required_points"]["pending_review"] == 1  # turn0 旧评分绑定失配
    assert result["semantic"]["required_points"]["not_verified"] == 1  # attempt 2 缺 turn1
    assert result["semantic"]["required_points"]["pass"] == 0
    # attempt 2 缺 turn 1：完成如实显示不完整（未采集），不得回退 attempt 1 拼出 PASS
    assert result["semantic"]["completion_graph"]["not_verified"] == 1
    assert result["semantic"]["completion_graph"]["pass"] == 0
    assert result["invalid_reviews"]


def test_r1_02_changed_answer_same_attempt_invalidates_review(tmp_path) -> None:
    """attempt 不变但答案/证据变化（result_hash 变化），旧评分失效。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1, answer="原答案")
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [make_turn_review(scenario["turns"][0], record)],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }
    changed = dict(record)
    changed["answer"] = "改过的答案"
    changed["result_hash"] = evaluation.compute_result_hash(changed)
    write_run_fixture(tmp_path, cases, [changed], reviews)
    result = evaluation.run_report(tmp_path)["result"]

    assert result["semantic"]["required_points"]["pending_review"] == 1
    assert result["semantic"]["required_points"]["pass"] == 0


def test_r1_02_deleted_review_item_cannot_raise_score(tmp_path) -> None:
    """删除原本 FAIL 的评分项 → 评分无效进入待复核，不得提高成绩或声称全部已审核。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    entry = make_turn_review(scenario["turns"][0], record)
    entry["required_points"][0]["result"] = "FAIL"  # 要点A 原本 FAIL
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    baseline = evaluation.run_report(tmp_path)["result"]
    assert baseline["semantic"]["required_points"]["fail"] == 1

    # 删除 FAIL 项后重跑报告
    entry["required_points"] = [entry["required_points"][1]]  # 只剩要点B
    (tmp_path / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False), encoding="utf-8")
    result = evaluation.run_report(tmp_path)["result"]
    assert result["semantic"]["required_points"]["pending_review"] == 1
    assert result["semantic"]["required_points"]["pass"] == 0
    assert result["coverage"]["semantic_items_reviewed"] == 0


def test_r1_02_report_never_backfills_review_hash(tmp_path) -> None:
    """报告生成不得自动替旧评分补新 hash（不得伪装为已审核）。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1, answer="原答案")
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [make_turn_review(scenario["turns"][0], record)],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }
    changed = dict(record)
    changed["answer"] = "改过的答案"
    changed["result_hash"] = evaluation.compute_result_hash(changed)
    write_run_fixture(tmp_path, cases, [changed], reviews)
    before = (tmp_path / "reviews.json").read_text(encoding="utf-8")
    evaluation.run_report(tmp_path)
    after = (tmp_path / "reviews.json").read_text(encoding="utf-8")
    assert before == after
    saved = json.loads(after)
    assert saved["turn_reviews"][0]["result_hash"] == record["result_hash"]  # 仍是旧 hash，未被修补


def test_r1_02_na_without_basis_is_invalid(tmp_path) -> None:
    """无依据的 N/A（缺 na_basis）不能按完整有效评分处理。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    entry = make_turn_review(scenario["turns"][0], record)
    entry["required_points"][0] = {"point": "要点A", "result": "N/A", "reason": ""}
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    result = evaluation.run_report(tmp_path)["result"]
    assert result["semantic"]["required_points"]["pending_review"] == 1
    assert any("无依据的 N/A" in problem for item in result["invalid_reviews"] for problem in item["problems"])


# ---------------------------------------------------------------------------
# R1-03：冻结版本与重算来源可追溯
# ---------------------------------------------------------------------------


def test_r1_03_report_uses_frozen_cases_not_workspace(tmp_path, monkeypatch) -> None:
    """修改工作区题集后，旧运行报告不得静默改变成绩且仍展示旧 hash。"""
    cases, scenario = _binding_cases()
    record = make_record(
        "S-1",
        0,
        attempt=1,
        citations=[citation("中华人民共和国劳动合同法", "第十条")],
    )
    scenario["turns"][0]["expected_articles"] = ["中华人民共和国劳动合同法 第十条"]
    write_run_fixture(tmp_path, cases, [record], None)
    first = evaluation.run_report(tmp_path)
    first_report = (tmp_path / "report.md").read_text(encoding="utf-8")
    first_hash = first["cases_sha256"]

    # 工作区题集被改动（目标法条变化）
    tampered = json.loads(json.dumps(cases))
    tampered["scenarios"][0]["turns"][0]["expected_articles"] = ["中华人民共和国劳动合同法 第九十九条"]
    monkeypatch.setattr(evaluation, "load_cases", lambda *args, **kwargs: tampered)

    second = evaluation.run_report(tmp_path)
    second_report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert second["cases_sha256"] == first_hash
    assert first_hash in second_report  # 报告展示的是冻结快照 hash
    assert "35/37" not in second_report  # 冻结口径下成绩不变（夹具中不出现真实运行数字）
    assert second["result"]["auto"]["retrieval"] == first["result"]["auto"]["retrieval"]


def test_r1_03_rescore_conversion_records_provenance(tmp_path, monkeypatch) -> None:
    """修正规则重算：转换记录来源核验、原始/新评分 SHA 与变更原因。"""
    cases, scenario = _binding_cases()
    source_dir = tmp_path / "eval-src"
    record = make_record("S-1", 0, attempt=1, answer="原始答案")
    source_manifest = {
        "run_id": "eval-src",
        "git_head": "old",
        "code_sha256": {
            "tests/evaluation_cases.json": cases_sha(cases),
            "tests/run_quality_evaluation.py": "old-scorer-sha",
        },
    }
    write_run_fixture(source_dir, cases, [record], None, manifest=source_manifest)
    (source_dir / "cases_frozen.json").unlink()  # 模拟旧产物缺少冻结快照

    monkeypatch.setattr(evaluation, "load_cases", lambda *args, **kwargs: cases)
    target_dir = tmp_path / "eval-r1"
    conversion = evaluation.rescore_convert(source_dir, target_dir)

    assert conversion["report_mode"] == "rescore_with_repaired_rules"
    assert conversion["source_run"] == str(source_dir)
    assert conversion["source_results_sha256"] == evaluation.sha256_file(source_dir / "results.jsonl")
    assert conversion["source_scorer_sha256"] == "old-scorer-sha"
    assert conversion["new_scorer_sha256"] == evaluation.sha256_file(Path(evaluation.__file__))
    assert "不等同于原版成绩的无变化重放" in conversion["note"]
    converted = evaluation.read_existing_results(target_dir)
    assert converted[0]["answer"] == "原始答案"
    assert converted[0]["result_hash"] == record["result_hash"]
    assert (target_dir / "cases_frozen.json").exists()
    assert (target_dir / "conversion.json").exists()


def test_r1_03_rescore_refuses_on_cases_mismatch(tmp_path, monkeypatch) -> None:
    """源运行题集与工作区题集不一致时明确拒绝，不静默换 gold。"""
    cases, _ = _binding_cases()
    source_dir = tmp_path / "eval-src"
    record = make_record("S-1", 0, attempt=1)
    source_manifest = {
        "run_id": "eval-src",
        "code_sha256": {"tests/evaluation_cases.json": "different-hash"},
    }
    write_run_fixture(source_dir, cases, [record], None, manifest=source_manifest)
    (source_dir / "cases_frozen.json").unlink()
    monkeypatch.setattr(evaluation, "load_cases", lambda *args, **kwargs: cases)

    with pytest.raises(RuntimeError, match="拒绝转换"):
        evaluation.rescore_convert(source_dir, tmp_path / "eval-r1")


def test_r1_03_new_attempt_preserves_old_provenance(tmp_path, monkeypatch) -> None:
    """新增 attempt 后旧 attempt 的版本信息仍可查（manifest 只追加）。"""
    monkeypatch.setattr(evaluation, "REAL_DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(evaluation, "CHROMA_SOURCE_DIR", tmp_path / "chroma")
    monkeypatch.setattr(evaluation, "REAL_KB_DIR", tmp_path / "kb")

    def fake_invoke(session_id, question, use_web_search):
        return {"search_query": question, "citations": [], "answer": "回答", "planned_tool_calls": []}

    scenario = make_scenario(id="QA-LAB-01", turns=[make_turn(question="Q1")])
    cases = {"version": "mini", "scenarios": [scenario]}
    run_dir = tmp_path / "run"
    evaluation.run_collection(
        run_dir=run_dir, scenarios=[scenario], run_id="p", cases=cases,
        with_isolation=False, invoke_graph=fake_invoke,
    )
    first_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    first_pass = first_manifest["collection_passes"][0]

    evaluation.run_collection(
        run_dir=run_dir, scenarios=[scenario], run_id="p", cases=cases,
        with_isolation=False, invoke_graph=fake_invoke,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["collection_passes"]) == 2
    assert manifest["collection_passes"][0] == first_pass  # 旧 provenance 未被覆盖
    records = evaluation.read_existing_results(run_dir)
    assert {record["attempt"] for record in records} == {1, 2}
    assert all(record["cases_sha256_at_collection"] for record in records)


# ---------------------------------------------------------------------------
# R1-04：待标注复核隔离
# ---------------------------------------------------------------------------


def test_r1_04_pending_annotation_excluded_from_trusted_scores() -> None:
    """needs_legal_review 场景评分单列待标注复核，不进入可信 PASS/FAIL。"""
    trusted = make_scenario(
        id="S-T",
        turns=[make_turn(expected_articles=["中华人民共和国劳动合同法 第十条"])],
    )
    pending = make_scenario(
        id="S-P",
        annotation_status="needs_legal_review",
        turns=[make_turn(expected_articles=["中华人民共和国劳动合同法 第十条"])],
    )
    cases = {"version": "mini", "scenarios": [trusted, pending]}
    records = [
        make_record("S-T", 0, citations=[citation("中华人民共和国劳动合同法", "第十条")]),
        make_record("S-P", 0, citations=[citation("中华人民共和国劳动合同法", "第十条")]),
    ]
    selected = evaluation.select_attempt_groups(records, cases["scenarios"])
    result = evaluation.aggregate(cases, selected, reviews=None)

    retrieval = result["auto"]["retrieval"]
    assert retrieval["applicable"] == 2
    assert retrieval["pending_annotation"] == 1
    assert retrieval["pass"] == 1  # 可信题的确定性判定计入 PASS；待审标注题不进可信 PASS/FAIL
    assert all(result["identity_checks"].values())


def test_r1_04_no_hardcoded_diagnostic_targets() -> None:
    """32/33、5/5 只是诊断性复算，不得硬编码成目标成绩。"""
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "32/33" not in source
    assert "5/5" not in source


def test_r1_04_report_shows_pending_annotation_scope(tmp_path) -> None:
    cases, scenario = _binding_cases()
    cases["scenarios"][0]["annotation_status"] = "needs_legal_review"
    record = make_record("S-1", 0, attempt=1)
    write_run_fixture(tmp_path, cases, [record], None)
    evaluation.run_report(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "待标注复核" in report
    assert "待法律复核标注题：1 个（S-1）" in report
    assert "可信子集覆盖" in report


# ---------------------------------------------------------------------------
# R1-05：冻结分母、范围与多轮状态
# ---------------------------------------------------------------------------


def _real_cases():
    return evaluation.load_cases(CASES_PATH)


def test_r1_05_frozen_denominator_not_shrunk_by_collection() -> None:
    """全集仅采集一条 graph 记录时，工具选择适用总数仍为 46，未采集 45。"""
    cases = _real_cases()
    record = make_record("QA-LAB-01", 0, attempt=1, question="我在公司干了快两年了一直没签劳动合同，现在还在上班，公司要不要给我双倍工资？从哪天开始算？")
    selected = evaluation.select_attempt_groups([record], cases["scenarios"])
    result = evaluation.aggregate(cases, selected, reviews=None)

    tool_selection = result["auto"]["tool_selection"]
    assert tool_selection["applicable"] == 46
    assert tool_selection["not_verified"] == 45
    assert result["auto"]["retrieval"]["applicable"] == 37
    assert all(result["identity_checks"].values())


def test_r1_05_empty_results_show_not_verified_within_frozen_scope() -> None:
    cases = _real_cases()
    selected = evaluation.select_attempt_groups([], cases["scenarios"])
    result = evaluation.aggregate(cases, selected, reviews=None)

    assert result["auto"]["tool_selection"]["not_verified"] == 46
    assert result["auto"]["retrieval"]["not_verified"] == 37
    assert result["semantic"]["completion_graph"]["not_verified"] == 38
    assert result["semantic"]["completion_graph"]["applicable"] == 38
    assert result["semantic"]["completion_route_only"]["applicable"] == 2
    assert all(result["identity_checks"].values())


def test_r1_05_multi_turn_error_invalidates_completion(tmp_path) -> None:
    """MT-REF-01 第二轮报错时场景进入错误状态，旧完成 PASS 失效。"""
    cases = _real_cases()
    mt = next(scenario for scenario in cases["scenarios"] if scenario["id"] == "MT-REF-01")
    records = [
        make_record("MT-REF-01", 0, attempt=1, question=mt["turns"][0]["question"], answer="回答一"),
        make_record(
            "MT-REF-01", 1, attempt=1, question=mt["turns"][1]["question"],
            status="error", error_type="ExternalServiceError", error_message="生成服务调用失败",
        ),
    ]
    selected = evaluation.select_attempt_groups(records, cases["scenarios"])
    result = evaluation.aggregate(cases, selected, reviews=None)
    completion = result["semantic"]["completion_graph"]
    assert completion["error"] >= 1
    assert completion["pass"] == 0


def test_r1_05_route_only_completion_separate_from_graph(tmp_path) -> None:
    """route_only 完成指标独立：graph 38 题与 route_only 2 题分开，不出现 33/40 混合分母。"""
    cases = _real_cases()
    selected = evaluation.select_attempt_groups([], cases["scenarios"])
    result = evaluation.aggregate(cases, selected, reviews=None)
    assert result["semantic"]["completion_graph"]["applicable"] == 38
    assert result["semantic"]["completion_route_only"]["applicable"] == 2
    report_source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "33/40" not in report_source


# ---------------------------------------------------------------------------
# R1-06：拒答两类指标（三类反例）
# ---------------------------------------------------------------------------


def test_r1_06_three_refusal_counterexamples(tmp_path) -> None:
    """错误拒答 / 恰当拒答 / 拒答后编造分别计入对应指标。"""
    answerable = make_scenario(
        id="S-ANS",
        turns=[make_turn(required_points=["要点A"], forbidden_claims=["禁止结论A"])],
    )
    out_of_scope = make_scenario(
        id="S-OOS",
        category="boundary",
        subcategory="out_of_scope",
        turns=[make_turn(required_points=["明确说明现有资料无法回答"], forbidden_claims=["不得编造金额"])],
    )
    cases = {"version": "mini", "scenarios": [answerable, out_of_scope]}
    r_ans = make_record("S-ANS", 0, answer="现有资料无法回答。", status="ok")
    r_oos = make_record("S-OOS", 0, answer="现有资料无法回答，但你大概能拿到十万元。", status="ok")
    selected = evaluation.select_attempt_groups([r_ans, r_oos], cases["scenarios"])

    entry_ans = make_turn_review(answerable["turns"][0], r_ans)
    entry_ans["wrong_refusal"]["result"] = "FAIL"  # 反例1：可答题被错误拒答
    entry_oos = make_turn_review(out_of_scope["turns"][0], r_oos)
    entry_oos["insufficiency_handling"]["result"] = "FAIL"  # 反例3：拒答语后仍编造
    entry_oos["forbidden_claims"][0]["result"] = "FAIL"
    entry_oos["required_points"][0]["result"] = "PASS"

    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry_ans, entry_oos],
        "scenario_reviews": [
            make_scenario_review(answerable, selected["S-ANS"]),
            make_scenario_review(out_of_scope, selected["S-OOS"]),
        ],
    }
    write_run_fixture(tmp_path, cases, [r_ans, r_oos], reviews)
    result = evaluation.run_report(tmp_path)["result"]

    # 反例1：错误拒答计入 wrong_refusal FAIL
    assert result["semantic"]["wrong_refusal"]["fail"] == 1
    # 反例3：拒答后编造 → 拒答处理 FAIL + 禁止结论 FAIL（不因含拒答语而 PASS）
    assert result["semantic"]["insufficiency_handling"]["fail"] == 1
    assert result["semantic"]["forbidden_claims"]["fail"] == 1
    # 反例2：恰当拒答 → 拒答处理 PASS
    assert result["semantic"]["insufficiency_handling"]["pass"] == 0  # 本夹具中库外题判 FAIL（编造）
    assert result["semantic"]["wrong_refusal"]["not_applicable"] == 1  # 库外题不进错误拒答分母


def test_r1_06_proper_refusal_counts_pass(tmp_path) -> None:
    cases = {
        "version": "mini",
        "scenarios": [
            make_scenario(
                id="S-OOS",
                category="boundary",
                subcategory="out_of_scope",
                turns=[make_turn(required_points=["明确说明现有资料无法回答"], forbidden_claims=["不得编造金额"])],
            )
        ],
    }
    record = make_record("S-OOS", 0, answer="现有资料无法回答")
    selected = evaluation.select_attempt_groups([record], cases["scenarios"])
    entry = make_turn_review(cases["scenarios"][0]["turns"][0], record)
    entry["response_type"] = "refused"
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry],
        "scenario_reviews": [make_scenario_review(cases["scenarios"][0], selected["S-OOS"])],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    result = evaluation.run_report(tmp_path)["result"]
    assert result["semantic"]["insufficiency_handling"]["pass"] == 1
    assert result["semantic"]["insufficiency_handling"]["fail"] == 0


def test_r1_06_applicability_from_frozen_definition_not_id_special_case() -> None:
    turn_refusal = make_turn(required_points=["说明缺少月工资无法算出具体金额"])
    turn_normal = make_turn(required_points=["要点A"])
    assert evaluation.frozen_expects_refusal(turn_refusal) is True
    assert evaluation.frozen_expects_refusal(turn_normal) is False
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "MT-SWT-02" not in source  # 不再依赖场景 ID 特判


# ---------------------------------------------------------------------------
# R1-07：引用审核覆盖率
# ---------------------------------------------------------------------------


def test_r1_07_pending_conclusion_lowers_review_coverage(tmp_path) -> None:
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    entry = make_turn_review(scenario["turns"][0], record)
    entry["conclusions"] = [
        {"text": "结论一", "needs_evidence": True, "citation": "第十条", "supported": True, "reason": ""},
        {"text": "结论二", "needs_evidence": True, "citation": None, "supported": None, "reason": "待审"},
    ]
    entry["conclusions_extraction_complete"] = True
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    citation = evaluation.run_report(tmp_path)["result"]["citation"]

    assert citation["extracted_needing_evidence"] == 2
    assert citation["reviewed"] == 1
    assert citation["supported"] == 1
    assert citation["pending"] == 1
    assert citation["support_rate"] == 1.0
    assert citation["review_coverage_rate"] == 0.5  # 待审结论使审核覆盖率下降


def test_r1_07_empty_conclusions_without_confirmation_not_counted_reviewed(tmp_path) -> None:
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    entry = make_turn_review(scenario["turns"][0], record)
    entry["conclusions"] = []
    entry["conclusions_extraction_complete"] = None  # 未完成提取，不是“确认无此类结论”
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    evaluation.run_report(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    result = evaluation.run_report(tmp_path)["result"]

    assert result["citation"]["extraction_unconfirmed_answers"] == 1
    assert "结论提取完整性尚未确认的回答数：1" in report
    assert "已提取结论审核覆盖率" in report  # 与语义评分项覆盖率分开展示
    assert "语义评分项复核覆盖率" in report


def test_r1_07_confirmed_empty_conclusions_counts_as_none(tmp_path) -> None:
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    entry = make_turn_review(scenario["turns"][0], record)
    entry["conclusions"] = []
    entry["conclusions_extraction_complete"] = True
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    citation = evaluation.run_report(tmp_path)["result"]["citation"]
    assert citation["extraction_unconfirmed_answers"] == 0
    assert citation["extracted_needing_evidence"] == 0
    assert citation["support_rate"] is None  # 分母为零显示 N/A（None → 报告 N/A 语义）


# ---------------------------------------------------------------------------
# R1-08：完成判定与冻结条件逐项对应
# ---------------------------------------------------------------------------


def test_r1_08_completion_pass_requires_all_criteria_pass(tmp_path) -> None:
    cases, scenario = _binding_cases()
    scenario["completion_criteria"] = ["完成条件A", "完成条件B"]
    record = make_record("S-1", 0, attempt=1)
    selected = evaluation.select_attempt_groups([record], cases["scenarios"])
    entry = make_scenario_review(
        scenario,
        selected["S-1"],
        criteria_results=[
            {"criterion": "完成条件A", "result": "PASS", "reason": ""},
            {"criterion": "完成条件B", "result": "FAIL", "reason": "未满足"},
        ],
    )
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [make_turn_review(scenario["turns"][0], record)],
        "scenario_reviews": [entry],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    result = evaluation.run_report(tmp_path)["result"]
    assert result["semantic"]["completion_graph"]["fail"] == 1
    assert result["semantic"]["completion_graph"]["pass"] == 0


def test_r1_08_completion_pass_with_turn_fail_requires_reconciliation(tmp_path) -> None:
    """必要条件（轮次评分）有 FAIL 时，完成 PASS 必须显式解释，否则评分无效。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    selected = evaluation.select_attempt_groups([record], cases["scenarios"])
    turn_entry = make_turn_review(scenario["turns"][0], record)
    turn_entry["required_points"][0]["result"] = "FAIL"
    completion_entry = make_scenario_review(scenario, selected["S-1"])  # criteria 全 PASS、无解释
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [turn_entry],
        "scenario_reviews": [completion_entry],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    result = evaluation.run_report(tmp_path)["result"]
    assert result["semantic"]["completion_graph"]["pending_review"] == 1
    assert any(
        "reconciles_turn_failures" in problem
        for item in result["invalid_reviews"]
        for problem in item["problems"]
    )

    # 提供显式解释后，完成判定仍须逐项对应冻结条件；本例解释后可判 PASS
    completion_entry["reconciles_turn_failures"] = "要点A 失败不影响完成条件A 的判定（附证据）"
    (tmp_path / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False), encoding="utf-8")
    result2 = evaluation.run_report(tmp_path)["result"]
    assert result2["semantic"]["completion_graph"]["pass"] == 1


def test_r1_08_criteria_must_match_frozen_definition(tmp_path) -> None:
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    selected = evaluation.select_attempt_groups([record], cases["scenarios"])
    entry = make_scenario_review(
        scenario,
        selected["S-1"],
        criteria_results=[{"criterion": "被替换的完成条件", "result": "PASS", "reason": ""}],
    )
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [make_turn_review(scenario["turns"][0], record)],
        "scenario_reviews": [entry],
    }
    write_run_fixture(tmp_path, cases, [record], reviews)
    result = evaluation.run_report(tmp_path)["result"]
    assert result["semantic"]["completion_graph"]["pending_review"] == 1
    assert any("completion_criteria" in problem for item in result["invalid_reviews"] for problem in item["problems"])


# ---------------------------------------------------------------------------
# 原测试意图保留：隔离、离线汇总与模板
# ---------------------------------------------------------------------------


def make_fake_real_dirs(tmp_path) -> dict:
    chroma = tmp_path / "real_chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"index-bytes")
    data = tmp_path / "real_data"
    data.mkdir()
    (data / "lawrag.sqlite3").write_bytes(b"db-bytes")
    kb = tmp_path / "real_kb"
    kb.mkdir()
    (kb / "law.txt").write_text("kb", encoding="utf-8")
    return {"chroma": chroma, "data": data, "kb": kb}


def test_setup_isolation_points_db_and_index_into_run_dir(tmp_path, monkeypatch) -> None:
    fake = make_fake_real_dirs(tmp_path)
    monkeypatch.setattr(evaluation, "CHROMA_SOURCE_DIR", fake["chroma"])
    monkeypatch.setattr(evaluation, "REAL_DATA_DIR", fake["data"])
    monkeypatch.setattr(evaluation, "REAL_KB_DIR", fake["kb"])
    for key in ("LAWRAG_DB_PATH", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(key, "sentinel-original")

    import app.ask
    import app.config
    import app.db

    monkeypatch.setattr(app.db, "DATABASE_PATH", app.db.DATABASE_PATH)
    monkeypatch.setattr(app.ask, "CHROMA_DIR", app.ask.CHROMA_DIR)
    monkeypatch.setattr(app.config, "CHROMA_DIR", app.config.CHROMA_DIR)

    run_dir = tmp_path / "run"
    isolation = evaluation.setup_isolation(run_dir)

    assert app.db.DATABASE_PATH == run_dir / "lawrag.sqlite3"
    assert app.ask.CHROMA_DIR == run_dir / "chroma_copy"
    assert (Path(isolation["chroma_copy"]) / "chroma.sqlite3").read_bytes() == b"index-bytes"
    assert isolation["index_copy_consistent"] is True
    assert (fake["data"] / "lawrag.sqlite3").read_bytes() == b"db-bytes"
    evaluation.teardown_isolation()


def test_inconsistent_index_copy_aborts_collection(tmp_path, monkeypatch) -> None:
    fake = make_fake_real_dirs(tmp_path)
    monkeypatch.setattr(evaluation, "CHROMA_SOURCE_DIR", fake["chroma"])
    monkeypatch.setattr(evaluation, "REAL_DATA_DIR", fake["data"])
    monkeypatch.setattr(evaluation, "REAL_KB_DIR", fake["kb"])
    for key in ("LAWRAG_DB_PATH", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(key, "sentinel-original")

    snapshots = iter(
        [
            {"exists": True, "entries": {"chroma.sqlite3": {"v": 1}}},
            {"exists": True, "entries": {"chroma.sqlite3": {"v": 2}}},
        ]
    )
    monkeypatch.setattr(evaluation, "snapshot_dir", lambda path: next(snapshots))

    with pytest.raises(RuntimeError, match="并发入库"):
        evaluation.setup_isolation(tmp_path / "run")
    assert not (tmp_path / "run" / "chroma_copy").exists()


def test_offline_report_never_calls_model(tmp_path, monkeypatch) -> None:
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)

    def forbid(*args, **kwargs):  # pragma: no cover - 触发即失败
        raise AssertionError("离线汇总不得调用模型")

    monkeypatch.setattr(evaluation, "_invoke_graph", forbid)
    monkeypatch.setattr(evaluation, "_plan_route", forbid)
    monkeypatch.setattr(evaluation, "load_kb_articles", forbid)

    write_run_fixture(tmp_path, cases, [record], None)
    result = evaluation.run_report(tmp_path)
    assert result["report_path"].exists()
    assert result["reviews_created"] is True


def test_review_template_marks_pending_not_keyword_judged(tmp_path) -> None:
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1, answer="现有资料无法回答，还有编造的十万元")
    write_run_fixture(tmp_path, cases, [record], None)
    evaluation.run_report(tmp_path)
    template = json.loads((tmp_path / "reviews.json").read_text(encoding="utf-8"))
    entry = template["turn_reviews"][0]
    # 模板不做关键词判定：拒答与禁止结论都保持 PENDING
    assert entry["forbidden_claims"][0]["result"] == "PENDING"
    assert entry["wrong_refusal"]["result"] in {"PENDING", "N/A"}
    assert entry["result_hash"] == record["result_hash"]
    assert entry["conclusions_extraction_complete"] is None


# ---------------------------------------------------------------------------
# R2-01：结果身份三方校验（保存 hash — 实际内容 hash — 复核绑定 hash）
# ---------------------------------------------------------------------------


def _identity_reviews(scenario: dict, record: dict) -> dict:
    return {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [make_turn_review(scenario["turns"][0], record)],
        "scenario_reviews": [make_scenario_review(scenario, {"attempt": 1, "turns": {0: record}})],
    }


def _tamper_ok_record(field: str, record: dict) -> None:
    mutations = {
        "answer": {"answer": "完全不同的答案：所有劳动关系均不受任何法律约束。"},
        "citations": {"citations": [citation("中华人民共和国劳动合同法", "第八十七条")]},
        "planned_tool_calls": {
            "planned_tool_calls": [
                {"name": "calculate_labor_compensation", "args": {"monthly_wage": 1, "years": 1}}
            ]
        },
        "tool_text": {"tool_text": "[被替换的工具输出]"},
        "web_sources": {"web_sources": [{"title": "被替换的外部来源", "url": "https://example.invalid/x"}]},
        "search_query": {"search_query": "被替换的检索词"},
        "status": {"status": "error"},
    }
    record.update(mutations[field])


R2_CONTENT_TAMPER_FIELDS = [
    "answer",
    "citations",
    "planned_tool_calls",
    "tool_text",
    "web_sources",
    "search_query",
    "status",
]


@pytest.mark.parametrize("field", R2_CONTENT_TAMPER_FIELDS)
def test_r2_01_content_change_with_stale_hash_never_keeps_pass(tmp_path, field) -> None:
    """只修改内容、保留旧 result_hash 与旧 reviews：轮次与场景旧 PASS 均不得继续有效。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    reviews = _identity_reviews(scenario, record)
    tampered = dict(record)
    _tamper_ok_record(field, tampered)
    assert tampered["result_hash"] == record["result_hash"]  # 反例前提：旧 hash 保留

    write_run_fixture(tmp_path, cases, [tampered], reviews)
    result = evaluation.run_report(tmp_path)["result"]

    problems = [problem for item in result["invalid_reviews"] for problem in item["problems"]]
    assert any("失配" in problem for problem in problems), problems
    assert result["semantic"]["required_points"]["pass"] == 0
    assert result["semantic"]["completion_graph"]["pass"] == 0


@pytest.mark.parametrize("field", ["error_type", "error_message"])
def test_r2_01_error_field_change_with_stale_hash_detected(tmp_path, field) -> None:
    """只修改记录实际使用的错误字段、保留旧 hash 与旧 reviews：必须触发身份问题。"""
    cases, scenario = _binding_cases()
    record = make_record(
        "S-1", 0, attempt=1,
        status="error", error_type="ExternalServiceError", error_message="生成服务调用失败",
    )
    reviews = _identity_reviews(scenario, record)
    tampered = dict(record)
    tampered[field] = "被替换的错误信息"
    assert tampered["result_hash"] == record["result_hash"]

    write_run_fixture(tmp_path, cases, [tampered], reviews)
    result = evaluation.run_report(tmp_path)["result"]

    problems = [problem for item in result["invalid_reviews"] for problem in item["problems"]]
    assert any("失配" in problem for problem in problems), problems
    assert result["semantic"]["required_points"]["pass"] == 0
    assert result["semantic"]["completion_graph"]["pass"] == 0


def test_r2_01_untouched_record_passes_three_way_identity(tmp_path) -> None:
    """未修改记录：保存 hash = 内容 hash = 复核绑定 hash，身份校验通过。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    reviews = _identity_reviews(scenario, record)
    write_run_fixture(tmp_path, cases, [record], reviews)
    result = evaluation.run_report(tmp_path)["result"]

    assert evaluation.result_identity_problems(record) == []
    assert result["invalid_reviews"] == []
    assert result["semantic"]["required_points"]["pass"] == 1
    assert result["semantic"]["completion_graph"]["pass"] == 1


def test_r2_01_missing_saved_hash_requires_explicit_conversion(tmp_path) -> None:
    """缺少保存 hash 的历史记录未经显式转换不得当作可信身份复用评分。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    reviews = _identity_reviews(scenario, record)
    unhashed = dict(record)
    unhashed.pop("result_hash")
    write_run_fixture(tmp_path, cases, [unhashed], reviews)
    result = evaluation.run_report(tmp_path)["result"]

    problems = [problem for item in result["invalid_reviews"] for problem in item["problems"]]
    assert any("缺少保存的结果 hash" in problem for problem in problems), problems
    assert result["semantic"]["required_points"]["pending_review"] == 1


def test_r2_01_real_r1_records_pass_identity_check() -> None:
    """未修改的 R1 记录与复核应通过身份校验（只读 R1 目录，不写任何文件）。"""
    r1_dir = Path(__file__).resolve().parents[1] / "logs" / "evaluation" / "eval-20261009-r1"
    if not (r1_dir / "results.jsonl").exists():
        pytest.skip("R1 运行目录不存在（离线环境）")
    records = [
        json.loads(line)
        for line in (r1_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reviews = json.loads((r1_dir / "reviews.json").read_text(encoding="utf-8"))
    cases = json.loads((r1_dir / "cases_frozen.json").read_text(encoding="utf-8"))
    scenario_by_id = {scenario["id"]: scenario for scenario in cases["scenarios"]}
    selected = evaluation.select_attempt_groups(records, cases["scenarios"])

    for record in records:
        assert evaluation.result_identity_problems(record) == []
    for entry in reviews["turn_reviews"]:
        scenario = scenario_by_id[entry["scenario_id"]]
        group = selected[entry["scenario_id"]]
        record = (group.get("turns") or {}).get(entry["turn_index"])
        problems = evaluation.validate_turn_review(
            entry,
            scenario["turns"][entry["turn_index"]],
            record,
            is_route_only=scenario.get("execution_scope") == "route_only",
        )
        assert problems == [], (entry["scenario_id"], entry["turn_index"], problems)
    for entry in reviews["scenario_reviews"]:
        scenario = scenario_by_id[entry["scenario_id"]]
        problems = evaluation.validate_scenario_review(entry, scenario, selected[entry["scenario_id"]])
        assert problems == [], (entry["scenario_id"], problems)


# ---------------------------------------------------------------------------
# R2-02：评分前核验冻结题集身份
# ---------------------------------------------------------------------------


def test_r2_02_tampered_frozen_expected_articles_refused(tmp_path) -> None:
    """只改冻结 expected_articles：拦截，不能产生提高后的正常成绩，也不覆盖既有报告。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    write_run_fixture(tmp_path, cases, [record], _identity_reviews(scenario, record))
    evaluation.run_report(tmp_path)
    baseline_report = (tmp_path / "report.md").read_bytes()

    frozen = json.loads((tmp_path / "cases_frozen.json").read_text(encoding="utf-8"))
    frozen["scenarios"][0]["turns"][0]["expected_articles"] = ["中华人民共和国劳动合同法 第九十九条"]
    (tmp_path / "cases_frozen.json").write_bytes(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    )
    before = snapshot_run_dir_files(tmp_path)
    with pytest.raises(RuntimeError, match="冻结"):
        evaluation.run_report(tmp_path)
    assert snapshot_run_dir_files(tmp_path) == before  # 失败路径不静默改写任何证据文件
    assert (tmp_path / "report.md").read_bytes() == baseline_report  # 不覆盖此前的有效报告


def test_r2_02_tampered_frozen_annotation_status_refused(tmp_path) -> None:
    """只改冻结 annotation_status：拦截，不能解除待审限制。"""
    cases, scenario = _binding_cases()
    cases["scenarios"][0]["annotation_status"] = "needs_legal_review"
    cases["scenarios"][0]["annotation_note"] = "待独立法律复核"
    record = make_record("S-1", 0, attempt=1)
    write_run_fixture(tmp_path, cases, [record], _identity_reviews(scenario, record))
    evaluation.run_report(tmp_path)

    frozen = json.loads((tmp_path / "cases_frozen.json").read_text(encoding="utf-8"))
    frozen["scenarios"][0]["annotation_status"] = "executor_annotated"
    (tmp_path / "cases_frozen.json").write_bytes(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    )
    before = snapshot_run_dir_files(tmp_path)
    with pytest.raises(RuntimeError, match="冻结"):
        evaluation.run_report(tmp_path)
    assert snapshot_run_dir_files(tmp_path) == before


def test_r2_02_tampered_reviews_binding_invalidates_all_reviews(tmp_path) -> None:
    """只改 reviews.cases_sha256：旧语义评分全部失效并明确报告原因。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    reviews = _identity_reviews(scenario, record)
    reviews["turn_reviews"][0]["conclusions"] = [
        {"text": "结论一", "needs_evidence": True, "citation": "第十条", "supported": True, "reason": ""}
    ]
    reviews["turn_reviews"][0]["conclusions_extraction_complete"] = True
    write_run_fixture(tmp_path, cases, [record], reviews)
    baseline = evaluation.run_report(tmp_path)["result"]
    assert baseline["citation"]["reviewed"] == 1

    reviews["cases_sha256"] = "0" * 64
    (tmp_path / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False), encoding="utf-8")
    result = evaluation.run_report(tmp_path)["result"]

    assert result["semantic"]["required_points"]["pass"] == 0
    assert result["semantic"]["required_points"]["pending_review"] == 1
    assert result["semantic"]["completion_graph"]["pass"] == 0
    assert result["semantic"]["completion_graph"]["pending_review"] == 1
    assert result["citation"]["reviewed"] == 0  # 失配复核的结论不得计入可信已审
    problems = [problem for item in result["invalid_reviews"] for problem in item["problems"]]
    assert any("冻结身份" in problem for problem in problems), problems
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "冻结身份" in report


def test_r2_02_tampered_manifest_frozen_identity_refused(tmp_path) -> None:
    """只改 manifest 中对应冻结身份：明确失配并拒绝出报告。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    write_run_fixture(tmp_path, cases, [record], _identity_reviews(scenario, record))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["code_sha256"]["tests/evaluation_cases.json"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    before = snapshot_run_dir_files(tmp_path)
    with pytest.raises(RuntimeError, match="失配"):
        evaluation.run_report(tmp_path)
    assert snapshot_run_dir_files(tmp_path) == before


def test_r2_02_tampered_conversion_frozen_identity_refused(tmp_path) -> None:
    """只改 conversion.frozen_cases_sha256：明确失配并拒绝出报告。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    write_run_fixture(tmp_path, cases, [record], _identity_reviews(scenario, record))
    (tmp_path / "conversion.json").write_text(
        json.dumps({"frozen_cases_sha256": cases_sha(cases), "report_mode": "rescore_with_repaired_rules"}),
        encoding="utf-8",
    )
    evaluation.run_report(tmp_path)

    conversion = json.loads((tmp_path / "conversion.json").read_text(encoding="utf-8"))
    conversion["frozen_cases_sha256"] = "0" * 64
    (tmp_path / "conversion.json").write_text(json.dumps(conversion, ensure_ascii=False), encoding="utf-8")
    before = snapshot_run_dir_files(tmp_path)
    with pytest.raises(RuntimeError, match="失配"):
        evaluation.run_report(tmp_path)
    assert snapshot_run_dir_files(tmp_path) == before


def test_r2_02_consistent_identities_generate_report(tmp_path, monkeypatch) -> None:
    """全部原始身份一致 + 工作区题集被改：仍正常离线重算（只读冻结快照）。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    write_run_fixture(tmp_path, cases, [record], _identity_reviews(scenario, record))
    (tmp_path / "conversion.json").write_text(
        json.dumps({"frozen_cases_sha256": cases_sha(cases), "report_mode": "rescore_with_repaired_rules"}),
        encoding="utf-8",
    )
    tampered_workspace = json.loads(json.dumps(cases))
    tampered_workspace["scenarios"][0]["turns"][0]["required_points"] = ["被替换的工作区要点"]
    monkeypatch.setattr(evaluation, "load_cases", lambda *args, **kwargs: tampered_workspace)

    result = evaluation.run_report(tmp_path)
    assert result["report_path"].exists()
    assert result["result"]["invalid_reviews"] == []
    assert result["result"]["semantic"]["required_points"]["pass"] == 1


# ---------------------------------------------------------------------------
# R2-03：待法律复核题的引用结论全部分流
# ---------------------------------------------------------------------------


def _citation_fixture(supported_trusted=True):
    trusted = make_scenario(id="S-T", turns=[make_turn()])
    pending = make_scenario(
        id="S-P",
        annotation_status="needs_legal_review",
        annotation_note="待独立法律复核",
        turns=[make_turn()],
    )
    cases = {"version": "mini", "scenarios": [trusted, pending]}
    return cases, trusted, pending


@pytest.mark.parametrize("supported", [True, False, None])
def test_r2_03_pending_annotation_conclusions_never_enter_trusted_counts(tmp_path, supported) -> None:
    """待标注题结论无论初评 supported=True/False/None，均不进可信已审/支持/不支持。"""
    cases, trusted, pending = _citation_fixture()
    record_t = make_record("S-T", 0, attempt=1)
    record_p = make_record("S-P", 0, attempt=1)
    entry_t = make_turn_review(trusted["turns"][0], record_t)
    entry_t["conclusions"] = [
        {"text": "可信结论", "needs_evidence": True, "citation": "第十条", "supported": True, "reason": ""}
    ]
    entry_t["conclusions_extraction_complete"] = True
    entry_p = make_turn_review(pending["turns"][0], record_p)
    entry_p["conclusions"] = [
        {"text": "待标注结论", "needs_evidence": True, "citation": "第十条", "supported": supported,
         "reason": "执行者初评（诊断用，不进可信成绩）"}
    ]
    entry_p["conclusions_extraction_complete"] = True
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry_t, entry_p],
        "scenario_reviews": [
            make_scenario_review(trusted, {"attempt": 1, "turns": {0: record_t}}),
            make_scenario_review(pending, {"attempt": 1, "turns": {0: record_p}}),
        ],
    }
    write_run_fixture(tmp_path, cases, [record_t, record_p], reviews)
    result = evaluation.run_report(tmp_path)["result"]
    citation = result["citation"]

    assert citation["extracted_needing_evidence"] == 2
    assert citation["extracted_trusted_annotation"] == 1
    assert citation["pending_annotation_conclusions"] == 1
    assert citation["reviewed"] == 1
    assert citation["supported"] == 1
    assert citation["unsupported"] == 0
    assert citation["pending"] == 0
    # 待标注结论的执行者初评仅保留为诊断信息
    diag = citation["pending_annotation_executor_preliminary"]
    assert diag["supported_true"] + diag["supported_false"] + diag["supported_none"] == 1
    assert citation["extracted_identity_ok"] is True
    assert citation["reviewed_identity_ok"] is True


def test_r2_03_citation_accounting_matches_r1_shape(tmp_path) -> None:
    """R1 数据复验口径（夹具数字，不硬编码进实现）：50 = 46（非待标注）+ 4（待标注）。

    可核算关系：已提取 = 可信已审 + 可信标注下待答案复核 + 待标注复核；可信已审 = 支持 + 不支持。
    """
    cases, trusted, pending = _citation_fixture()
    record_t = make_record("S-T", 0, attempt=1)
    record_p = make_record("S-P", 0, attempt=1)
    entry_t = make_turn_review(trusted["turns"][0], record_t)
    entry_t["conclusions"] = [
        {"text": f"可信结论{i}", "needs_evidence": True, "citation": "第十条",
         "supported": (False if i == 0 else None if i == 1 else True), "reason": ""}
        for i in range(46)
    ]
    entry_t["conclusions_extraction_complete"] = True
    entry_p = make_turn_review(pending["turns"][0], record_p)
    entry_p["conclusions"] = [
        {"text": f"待标注结论{i}", "needs_evidence": True, "citation": "第十条", "supported": True, "reason": ""}
        for i in range(4)
    ]
    entry_p["conclusions_extraction_complete"] = True
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry_t, entry_p],
        "scenario_reviews": [
            make_scenario_review(trusted, {"attempt": 1, "turns": {0: record_t}}),
            make_scenario_review(pending, {"attempt": 1, "turns": {0: record_p}}),
        ],
    }
    write_run_fixture(tmp_path, cases, [record_t, record_p], reviews)
    result = evaluation.run_report(tmp_path)["result"]
    citation = result["citation"]

    assert citation["extracted_needing_evidence"] == 50
    assert citation["extracted_trusted_annotation"] == 46
    assert citation["pending_annotation_conclusions"] == 4
    assert citation["reviewed"] == 45 == citation["supported"] + citation["unsupported"]
    assert citation["supported"] == 44
    assert citation["unsupported"] == 1
    assert citation["pending"] == 1
    assert citation["extracted_needing_evidence"] == (
        citation["reviewed"] + citation["pending"] + citation["pending_annotation_conclusions"]
    )
    assert citation["extracted_identity_ok"] is True
    assert citation["reviewed_identity_ok"] is True
    # 覆盖率分母不排除待标注结论：45/50，而非 45/46 或 100%
    assert citation["review_coverage_rate"] == round(45 / 50, 4)


def test_r2_03_report_explains_three_pending_reasons(tmp_path) -> None:
    """报告区分三类未完成原因：标注待审、结论支持待审、结论提取未确认（按回答数）。"""
    cases, trusted, pending = _citation_fixture()
    record_t = make_record("S-T", 0, attempt=1)
    record_p = make_record("S-P", 0, attempt=1)
    entry_t = make_turn_review(trusted["turns"][0], record_t)
    entry_t["conclusions"] = [
        {"text": "待审结论", "needs_evidence": True, "citation": None, "supported": None, "reason": ""},
    ]
    entry_t["conclusions_extraction_complete"] = None  # 提取未确认（按回答数计）
    entry_p = make_turn_review(pending["turns"][0], record_p)
    entry_p["conclusions"] = [
        {"text": "待标注结论", "needs_evidence": True, "citation": "第十条", "supported": True, "reason": ""}
    ]
    entry_p["conclusions_extraction_complete"] = True
    reviews = {
        "format": evaluation.REVIEW_FORMAT,
        "turn_reviews": [entry_t, entry_p],
        "scenario_reviews": [
            make_scenario_review(trusted, {"attempt": 1, "turns": {0: record_t}}),
            make_scenario_review(pending, {"attempt": 1, "turns": {0: record_p}}),
        ],
    }
    write_run_fixture(tmp_path, cases, [record_t, record_p], reviews)
    evaluation.run_report(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "可信已审结论数" in report
    assert "待标注复核结论数" in report
    assert "可信标注下待答案复核结论数" in report
    assert "可核算关系" in report
    assert "结论提取完整性尚未确认的回答数：1" in report
    assert "结论提取未确认按回答数另外报告，不与结论数相加" in report


def test_r2_03_executor_preliminary_disclaimer_in_report(tmp_path) -> None:
    """A-02：报告明确语义评分为 executor_preliminary，待答案复核 0 不代表独立法律复核完成。"""
    cases, scenario = _binding_cases()
    record = make_record("S-1", 0, attempt=1)
    write_run_fixture(tmp_path, cases, [record], _identity_reviews(scenario, record))
    evaluation.run_report(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "executor_preliminary" in report
    assert "不代表独立法律复核完成" in report
