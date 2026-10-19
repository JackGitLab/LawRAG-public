"""tests/evaluation_cases.json 的离线数据校验测试。

只读题集、kb/ 原文与旧回归题，不访问网络、不触碰真实 data/ 或 chroma_db/。
"""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CASES_PATH = Path(__file__).with_name("evaluation_cases.json")
REGRESSION_PATH = Path(__file__).with_name("regression_questions.json")
SCRIPT_PATH = Path(__file__).with_name("run_quality_evaluation.py")

SPEC = spec_from_file_location("quality_evaluation_cases", SCRIPT_PATH)
assert SPEC and SPEC.loader
evaluation = module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)

EXPECTED_SCENARIO_IDS = {
    *(f"QA-LAB-{index:02d}" for index in range(1, 9)),
    *(f"QA-CON-{index:02d}" for index in range(1, 9)),
    *(f"BND-OOS-{index:02d}" for index in range(1, 5)),
    *(f"BND-INS-{index:02d}" for index in range(1, 5)),
    *(f"TOOL-CALC-{index:02d}" for index in range(1, 5)),
    "TOOL-MISS-01",
    "TOOL-MISS-02",
    "TOOL-WEB-01",
    "TOOL-WEB-02",
    "MT-REF-01",
    "MT-REF-02",
    "MT-ELL-01",
    "MT-ELL-02",
    "MT-COR-01",
    "MT-COR-02",
    "MT-SWT-01",
    "MT-SWT-02",
}


@pytest.fixture(scope="module")
def cases() -> dict:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def regression_questions() -> list:
    return json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kb_articles() -> dict:
    return evaluation.load_kb_articles()


def test_case_set_passes_offline_validation(cases, regression_questions, kb_articles) -> None:
    errors = evaluation.validate_case_set(cases, regression_questions, kb_articles)
    assert errors == []


def test_scenario_ids_and_totals(cases) -> None:
    scenarios = cases["scenarios"]
    assert len(scenarios) == 40
    assert {scenario["id"] for scenario in scenarios} == EXPECTED_SCENARIO_IDS
    category_counts = {}
    for scenario in scenarios:
        category_counts[scenario["category"]] = category_counts.get(scenario["category"], 0) + 1
    assert category_counts == {
        "library_qa": 16,
        "boundary": 8,
        "tool_routing": 8,
        "multi_turn": 8,
    }


def test_library_qa_split_eight_per_law(cases) -> None:
    law_counts = {}
    for scenario in cases["scenarios"]:
        if scenario["category"] != "library_qa":
            continue
        laws = {
            article.rsplit(" ", 1)[0]
            for turn in scenario["turns"]
            for article in turn["expected_articles"]
        }
        assert len(laws) == 1, scenario["id"]
        law = laws.pop()
        law_counts[law] = law_counts.get(law, 0) + 1
    assert sorted(law_counts.values()) == [8, 8]


def test_multi_turn_scenarios_have_two_turns_others_one(cases) -> None:
    for scenario in cases["scenarios"]:
        expected = 2 if scenario["category"] == "multi_turn" else 1
        assert len(scenario["turns"]) == expected, scenario["id"]
    two_turn = [
        scenario["id"]
        for scenario in cases["scenarios"]
        if scenario["category"] == "multi_turn"
    ]
    assert len(two_turn) == 8


def test_execution_scope_and_web_flags(cases) -> None:
    route_only_ids = set()
    for scenario in cases["scenarios"]:
        is_route_only = scenario["execution_scope"] == "route_only"
        assert is_route_only == (scenario["subcategory"] == "web_route"), scenario["id"]
        if is_route_only:
            route_only_ids.add(scenario["id"])
        for turn in scenario["turns"]:
            assert turn["use_web_search"] is is_route_only, scenario["id"]
    assert route_only_ids == {"TOOL-WEB-01", "TOOL-WEB-02"}


def test_evidence_quotes_are_kb_original_text(cases, kb_articles) -> None:
    for scenario in cases["scenarios"]:
        for entry in scenario["evidence"]:
            if entry.get("note"):
                assert entry["note"].strip()
                continue
            key = (entry["source"], entry["article"])
            assert key in kb_articles, f"{scenario['id']}：{key}"
            assert entry["quote"] in kb_articles[key], f"{scenario['id']}：quote 非原文"


def test_expected_articles_reference_real_kb_labels(cases, kb_articles) -> None:
    for scenario in cases["scenarios"]:
        for turn in scenario["turns"]:
            for label in turn["expected_articles"]:
                law_name, article = label.rsplit(" ", 1)
                assert (law_name, article) in kb_articles, f"{scenario['id']}：{label}"


def test_no_obvious_duplication_with_old_regression(cases, regression_questions) -> None:
    old_questions = [item["question"] for item in regression_questions]
    for scenario in cases["scenarios"]:
        for turn in scenario["turns"]:
            for old_question in old_questions:
                assert not evaluation._is_obvious_duplicate(
                    turn["question"], old_question
                ), f"{scenario['id']} 与旧题重复：{old_question}"


def test_missing_param_scenarios_declare_missing_params(cases) -> None:
    for scenario in cases["scenarios"]:
        if scenario["subcategory"] != "calc_missing_params":
            continue
        for turn in scenario["turns"]:
            assert set(turn["missing_params"]) <= {"years", "monthly_wage"}
            assert turn["expected_tools"] == []


def test_needs_legal_review_scenarios_have_annotation_note(cases) -> None:
    flagged = []
    for scenario in cases["scenarios"]:
        if scenario["annotation_status"] == "needs_legal_review":
            assert scenario["annotation_note"].strip(), scenario["id"]
            flagged.append(scenario["id"])
    assert flagged, "应存在标注待复核场景或明确说明全部可直接支持"


def test_route_only_scenarios_carry_route_only_notes(cases) -> None:
    for scenario in cases["scenarios"]:
        if scenario["execution_scope"] != "route_only":
            continue
        expected_names = [entry["name"] for entry in scenario["turns"][0]["expected_tools"]]
        assert expected_names in (["verify_statute_status"], ["web_search"])
        assert scenario["turns"][0]["expected_articles"] == []
