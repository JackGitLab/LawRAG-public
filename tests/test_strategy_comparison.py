from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from app.ask import ExternalServiceError


SCRIPT_PATH = Path(__file__).with_name("run_strategy_comparison.py")
SPEC = spec_from_file_location("strategy_comparison", SCRIPT_PATH)
assert SPEC and SPEC.loader
comparison = module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_comparison_includes_all_seven_strategies() -> None:
    assert comparison.STRATEGIES == (
        "vector",
        "vector_rerank",
        "hyde",
        "multi_query",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    )


def test_retrieve_with_retry_recovers_from_one_external_failure(monkeypatch) -> None:
    calls = 0
    delays = []

    def retrieve(question, strategy, top_k):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ExternalServiceError("临时失败")
        return ["result"]

    monkeypatch.setattr(comparison, "retrieve_with_strategy", retrieve)
    monkeypatch.setattr(comparison.time, "sleep", delays.append)

    assert comparison.retrieve_with_retry("问题", "vector", 3) == ["result"]
    assert calls == 2
    assert delays == [comparison.RETRY_DELAY_SECONDS]


def test_retrieve_with_retry_raises_after_second_failure(monkeypatch) -> None:
    calls = 0

    def retrieve(question, strategy, top_k):
        nonlocal calls
        calls += 1
        raise ExternalServiceError("持续失败")

    monkeypatch.setattr(comparison, "retrieve_with_strategy", retrieve)
    monkeypatch.setattr(comparison.time, "sleep", lambda seconds: None)

    with pytest.raises(ExternalServiceError, match="持续失败"):
        comparison.retrieve_with_retry("问题", "vector", 3)

    assert calls == 2


def test_group_result_counts_only_requested_question_type() -> None:
    cases = [
        {"id": 1, "type": "colloquial"},
        {"id": 2, "type": "terminology"},
        {"id": 3, "type": "terminology"},
    ]
    outcomes = {
        1: {"bm25": False},
        2: {"bm25": True},
        3: {"bm25": False},
    }

    assert comparison.group_result(cases, outcomes, "bm25", "terminology") == {
        "count": 2,
        "hits": 1,
        "rate": 50.0,
    }
