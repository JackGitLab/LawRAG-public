import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from llama_index.core.schema import NodeWithScore, TextNode

import app.main as main_module
import app.db as db_module
import app.graph as graph_module
from app.ask import ExternalServiceError, IndexNotReadyError


def set_ready(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "initialize_resources", lambda: object())


def parse_sse_events(text: str) -> list[dict]:
    return [json.loads(block.removeprefix("data: ")) for block in text.strip().split("\n\n")]


@pytest.fixture(autouse=True)
def isolated_graph(monkeypatch, tmp_path: Path):
    # 捕获原始缓存函数引用：部分测试会把 graph_module.get_graph 替换为替身类，
    # teardown 时原属性已不在，但此处持有的引用仍然有效。
    graph_getter = graph_module.get_graph
    graph_getter.cache_clear()
    if db_module.get_checkpointer_connection.cache_info().currsize:
        db_module.get_checkpointer_connection().close()
    db_module.get_checkpointer_connection.cache_clear()
    if db_module.get_connection.cache_info().currsize:
        db_module.get_connection().close()
    db_module.get_connection.cache_clear()
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path / "chat.sqlite3")
    monkeypatch.setattr(
        graph_module, "get_checkpointer_connection", db_module.get_checkpointer_connection
    )
    monkeypatch.setattr(main_module, "run_turn", graph_module.run_turn)
    yield
    graph_getter.cache_clear()
    if db_module.get_checkpointer_connection.cache_info().currsize:
        db_module.get_checkpointer_connection().close()
    db_module.get_checkpointer_connection.cache_clear()
    if db_module.get_connection.cache_info().currsize:
        db_module.get_connection().close()
    db_module.get_connection.cache_clear()


def test_healthz_responds_while_index_is_not_ready(monkeypatch) -> None:
    """容器健康检查：索引未就绪时也返回 200，且不影响既有业务端点行为。"""

    def missing_index():
        raise IndexNotReadyError("未找到 chroma_db，请先运行入库命令")

    monkeypatch.setattr(main_module, "initialize_resources", missing_index)

    with TestClient(main_module.app) as client:
        health = client.get("/healthz")
        ask = client.post("/api/ask", json={"question": "测试问题"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ask.status_code == 503  # 业务端点保持原行为


def test_minimal_serves_demo_page_with_disclaimer(monkeypatch) -> None:
    set_ready(monkeypatch)

    with TestClient(main_module.app) as client:
        response = client.get("/minimal")

    assert response.status_code == 200
    assert "LawRAG 法律问答演示" in response.text
    assert "不构成法律意见，不得用于实际法律决策" in response.text
    assert "检索中…" in response.text
    assert "引用来源" in response.text


def test_ask_returns_answer_and_citations(monkeypatch) -> None:
    set_ready(monkeypatch)
    node = TextNode(
        text="第四十七条 经济补偿按工作年限计算。",
        metadata={"source": "中华人民共和国劳动合同法", "article": "第四十七条"},
    )
    monkeypatch.setattr(
        main_module,
        "ask",
        lambda question: ("应按工作年限计算。（第四十七条）", [NodeWithScore(node=node, score=0.91)]),
    )

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": "  如何计算经济补偿？  "})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "应按工作年限计算。（第四十七条）",
        "citations": [
            {
                "source": "中华人民共和国劳动合同法",
                "article": "第四十七条",
                "score": 0.91,
            }
        ],
    }


def test_static_mount_does_not_shadow_api(monkeypatch) -> None:
    set_ready(monkeypatch)
    node = TextNode(
        text="第四十七条 经济补偿按工作年限计算。",
        metadata={"source": "中华人民共和国劳动合同法", "article": "第四十七条"},
    )
    monkeypatch.setattr(
        main_module,
        "ask",
        lambda question: ("应按工作年限计算。（第四十七条）", [NodeWithScore(node=node, score=0.91)]),
    )

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": "如何计算经济补偿？"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "应按工作年限计算。（第四十七条）",
        "citations": [
            {
                "source": "中华人民共和国劳动合同法",
                "article": "第四十七条",
                "score": 0.91,
            }
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   \t"},
        {},
        {"question": None},
    ],
)
def test_ask_rejects_invalid_question(monkeypatch, payload) -> None:
    set_ready(monkeypatch)

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_ask_explains_whitespace_only_question(monkeypatch) -> None:
    set_ready(monkeypatch)

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 422
    assert "问题不能为空或仅包含空白字符" in response.text


@pytest.mark.parametrize(
    ("question", "status"),
    [
        ("x" * 2000, 200),
        ("x" * 2001, 422),
        ("  " + "x" * 2000 + "  ", 200),
        ("  " + "x" * 2001 + "  ", 422),
    ],
)
def test_question_length_boundary(monkeypatch, question, status) -> None:
    set_ready(monkeypatch)
    captured = {}

    def fake_ask(q):
        captured["q"] = q
        return "回答", []

    monkeypatch.setattr(main_module, "ask", fake_ask)

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": question})

    assert response.status_code == status
    if status == 200:
        # 合法问题传入 endpoint 的是 strip 后的内容
        assert captured["q"] == question.strip()


def test_overlong_question_error_message(monkeypatch) -> None:
    set_ready(monkeypatch)

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": "x" * 2001})

    assert response.status_code == 422
    assert "问题长度不能超过 2000 个字符" in response.text


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/chat", "/api/chat/stream"])
def test_overlong_question_rejected_before_business_logic(monkeypatch, endpoint) -> None:
    set_ready(monkeypatch)
    calls = []

    def fake_ask(q):
        calls.append("ask")
        return "回答", []

    monkeypatch.setattr(main_module, "ask", fake_ask)
    monkeypatch.setattr(
        main_module,
        "create_session",
        lambda session_id, title: calls.append("create_session"),
    )
    monkeypatch.setattr(
        main_module,
        "run_turn",
        lambda *args, **kwargs: calls.append("run_turn") or ("回答", []),
    )
    monkeypatch.setattr(
        main_module,
        "stream_turn",
        lambda *args, **kwargs: calls.append("stream_turn") or iter([]),
    )

    with TestClient(main_module.app) as client:
        response = client.post(endpoint, json={"question": "x" * 2001})

    assert response.status_code == 422
    assert "问题长度不能超过 2000 个字符" in response.text
    assert calls == []  # 未进入任何业务逻辑，未创建空会话


def test_ask_reports_missing_index(monkeypatch) -> None:
    def missing_index():
        raise IndexNotReadyError("未找到 chroma_db，请先运行入库命令")

    monkeypatch.setattr(main_module, "initialize_resources", missing_index)

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": "测试问题"})

    assert response.status_code == 503
    assert response.json() == {"detail": "未找到 chroma_db，请先运行入库命令"}


def test_ask_logs_external_failure_without_leaking_details(monkeypatch) -> None:
    set_ready(monkeypatch)
    logged = {}

    def fail_ask(question):
        raise ExternalServiceError("上游响应包含内部细节")

    def fake_log(scene, message):
        logged.update(scene=scene, message=message)
        return Path("logs/test.log")

    monkeypatch.setattr(main_module, "ask", fail_ask)
    monkeypatch.setattr(main_module, "log_current_exception", fake_log)

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": "测试问题"})

    assert response.status_code == 502
    assert response.json() == {"detail": "外部服务调用失败，请稍后重试"}
    assert "内部细节" not in response.text
    assert logged == {"scene": "api-ask", "message": "HTTP 问答外部服务调用失败"}


def test_missing_environment_variable_fails_during_startup(monkeypatch) -> None:
    def fake_require_env(name):
        if name == "DEEPSEEK_API_KEY":
            raise RuntimeError("环境变量缺失：DEEPSEEK_API_KEY")
        return "configured"

    monkeypatch.setattr(main_module, "require_env", fake_require_env)

    with pytest.raises(RuntimeError, match="环境变量缺失：DEEPSEEK_API_KEY"):
        with TestClient(main_module.app):
            pass


def test_resources_are_initialized_once_for_multiple_requests(monkeypatch) -> None:
    initialized = 0

    def initialize():
        nonlocal initialized
        initialized += 1
        return object()

    monkeypatch.setattr(main_module, "initialize_resources", initialize)
    monkeypatch.setattr(main_module, "ask", lambda question: ("测试回答", []))

    with TestClient(main_module.app) as client:
        assert client.post("/api/ask", json={"question": "问题一"}).status_code == 200
        assert client.post("/api/ask", json={"question": "问题二"}).status_code == 200

    assert initialized == 1


def test_initialize_resources_loads_vector_before_bm25(monkeypatch) -> None:
    events = []
    index = object()

    monkeypatch.setattr(main_module, "require_env", lambda name: "configured")
    monkeypatch.setattr(main_module, "get_embedding", lambda: object())
    monkeypatch.setattr(main_module, "get_llm_client", lambda: object())

    def load_vector():
        events.append("vector")
        return index

    def load_bm25():
        events.append("bm25")
        return object()

    monkeypatch.setattr(main_module, "load_index", load_vector)
    monkeypatch.setattr(main_module, "get_bm25_index", load_bm25)

    assert main_module.initialize_resources() is index
    assert events == ["vector", "bm25"]


def test_bm25_index_is_preheated_once_per_lifespan(monkeypatch) -> None:
    bm25_calls = 0

    monkeypatch.setattr(main_module, "require_env", lambda name: "configured")
    monkeypatch.setattr(main_module, "get_embedding", lambda: object())
    monkeypatch.setattr(main_module, "get_llm_client", lambda: object())
    monkeypatch.setattr(main_module, "load_index", lambda: object())

    def load_bm25():
        nonlocal bm25_calls
        bm25_calls += 1
        return object()

    monkeypatch.setattr(main_module, "get_bm25_index", load_bm25)
    monkeypatch.setattr(main_module, "ask", lambda question: ("测试回答", []))

    with TestClient(main_module.app) as client:
        assert client.post("/api/ask", json={"question": "问题一"}).status_code == 200
        assert client.post("/api/ask", json={"question": "问题二"}).status_code == 200

    assert bm25_calls == 1


def test_bm25_missing_index_uses_existing_503_path(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "require_env", lambda name: "configured")
    monkeypatch.setattr(main_module, "get_embedding", lambda: object())
    monkeypatch.setattr(main_module, "get_llm_client", lambda: object())
    monkeypatch.setattr(main_module, "load_index", lambda: object())

    def missing_bm25_index():
        raise IndexNotReadyError("LawRAG 索引为空，请先运行入库命令")

    monkeypatch.setattr(main_module, "get_bm25_index", missing_bm25_index)

    with TestClient(main_module.app) as client:
        response = client.post("/api/ask", json={"question": "测试问题"})

    assert response.status_code == 503
    assert response.json() == {"detail": "LawRAG 索引为空，请先运行入库命令"}


def test_chat_success(monkeypatch) -> None:
    set_ready(monkeypatch)
    node = TextNode(text="测试法条", metadata={"source": "劳动合同法", "article": "第一条"})
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)
    captured = {}

    def fake_run_turn(session_id, question, use_web_search):
        captured.update(
            session_id=session_id,
            question=question,
            use_web_search=use_web_search,
        )
        return "测试回答", [NodeWithScore(node=node, score=0.9)]

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/chat", json={"question": "测试问题", "use_web_search": True}
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "测试回答"
    assert response.json()["citations"][0]["article"] == "第一条"
    assert captured["question"] == "测试问题"
    assert captured["use_web_search"] is True


def test_chat_maps_index_and_external_errors(monkeypatch) -> None:
    set_ready(monkeypatch)
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)
    monkeypatch.setattr(
        main_module,
        "run_turn",
        lambda session_id, question, use_web_search: (_ for _ in ()).throw(IndexNotReadyError("索引未就绪")),
    )
    with TestClient(main_module.app) as client:
        assert client.post("/api/chat", json={"question": "问题"}).status_code == 503

    monkeypatch.setattr(
        main_module,
        "run_turn",
        lambda session_id, question, use_web_search: (_ for _ in ()).throw(ExternalServiceError("内部错误")),
    )
    monkeypatch.setattr(main_module, "log_current_exception", lambda scene, message: Path("logs/test.log"))
    with TestClient(main_module.app) as client:
        response = client.post("/api/chat", json={"question": "问题"})
    assert response.status_code == 502
    assert "内部错误" not in response.text


def test_chat_first_generation_failure_uses_actual_graph_boundary(
    monkeypatch, isolated_graph
) -> None:
    set_ready(monkeypatch)
    node = TextNode(text="测试法条", metadata={"source": "劳动合同法", "article": "第一条"})

    class FailingModel:
        def invoke(self, prompt):
            raise OSError("upstream-secret-detail")

    monkeypatch.setattr(graph_module, "retrieve", lambda question, top_k=3, session_id=None: [NodeWithScore(node=node, score=0.9)])
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FailingModel())
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)
    logged = {}

    def fake_log(scene, message):
        logged.update(scene=scene, message=message, cause=sys.exc_info()[1].__cause__)
        return Path("logs/test.log")

    monkeypatch.setattr(main_module, "log_current_exception", fake_log)

    with TestClient(main_module.app) as client:
        response = client.post("/api/chat", json={"question": "普通问题"})

    assert response.status_code == 502
    assert response.json() == {"detail": "外部服务调用失败，请稍后重试"}
    assert "upstream-secret-detail" not in response.text
    assert isinstance(logged["cause"], OSError)


def test_chat_follow_up_rewrite_failure_uses_actual_graph_boundary(
    monkeypatch, isolated_graph
) -> None:
    set_ready(monkeypatch)
    node = TextNode(text="测试法条", metadata={"source": "劳动合同法", "article": "第一条"})

    class FailingRewriteModel:
        def invoke(self, prompt):
            if "改写成可独立检索" in prompt:
                raise OSError("rewrite-upstream-detail")
            return SimpleNamespace(content="正常回答")

    monkeypatch.setattr(graph_module, "retrieve", lambda question, top_k=3, session_id=None: [NodeWithScore(node=node, score=0.9)])
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FailingRewriteModel())
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)
    monkeypatch.setattr(main_module, "get_session", lambda session_id: {"id": session_id})
    monkeypatch.setattr(main_module, "touch_session", lambda session_id: None)
    logged = {}

    def fake_log(scene, message):
        logged.update(scene=scene, message=message, cause=sys.exc_info()[1].__cause__)
        return Path("logs/test.log")

    monkeypatch.setattr(main_module, "log_current_exception", fake_log)

    with TestClient(main_module.app) as client:
        first = client.post("/api/chat", json={"question": "第一轮问题"})
        session_id = first.json()["session_id"]
        second = client.post(
            "/api/chat",
            json={"question": "第二轮追问", "session_id": session_id},
        )

    assert first.status_code == 200
    assert first.json()["answer"] == "正常回答"
    assert second.status_code == 502
    assert second.json() == {"detail": "外部服务调用失败，请稍后重试"}
    assert "rewrite-upstream-detail" not in second.text
    assert isinstance(logged["cause"], OSError)


def test_sessions_empty_and_unknown_session(monkeypatch) -> None:
    set_ready(monkeypatch)
    monkeypatch.setattr(main_module, "list_sessions", lambda: [])
    monkeypatch.setattr(main_module, "get_session", lambda session_id: None)
    with TestClient(main_module.app) as client:
        assert client.get("/api/sessions").json() == []
        assert client.get("/api/sessions/missing").status_code == 404


def test_session_detail_returns_messages(monkeypatch) -> None:
    set_ready(monkeypatch)
    session = {
        "id": "s1",
        "title": "t",
        "created_at": "2026-09-16T00:00:00",
        "updated_at": "2026-09-16T00:00:00",
    }
    monkeypatch.setattr(
        main_module,
        "get_session",
        lambda session_id: session if session_id == "s1" else None,
    )

    class FakeGraph:
        def get_state(self, config):
            assert config == {"configurable": {"thread_id": "s1"}}
            return SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="问"),
                        AIMessage(
                            content="答",
                            additional_kwargs={
                                "citations": [
                                    {"source": "劳动合同法", "article": "第一条", "score": 0.9}
                                ]
                            },
                        ),
                        SystemMessage(content="内部"),
                    ]
                }
            )

    monkeypatch.setattr(main_module, "get_graph", lambda: FakeGraph())
    with TestClient(main_module.app) as client:
        response = client.get("/api/sessions/s1")
        missing = client.get("/api/sessions/missing")

    assert response.status_code == 200
    assert response.json()["messages"] == [
        {"role": "user", "content": "问"},
        {
            "role": "assistant",
            "content": "答",
            "citations": [{"source": "劳动合同法", "article": "第一条", "score": 0.9}],
        },
    ]
    assert missing.status_code == 404


def test_chat_stream_emits_sse_events(monkeypatch) -> None:
    set_ready(monkeypatch)
    node = TextNode(text="测试法条", metadata={"source": "劳动合同法", "article": "第一条"})
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)
    captured = {}

    def fake_stream_turn(session_id, question, use_web_search):
        captured.update(
            session_id=session_id,
            question=question,
            use_web_search=use_web_search,
        )
        return iter(
            [
                {"type": "token", "text": "测试"},
                {"type": "token", "text": "回答"},
                {"type": "done", "citations": [NodeWithScore(node=node, score=0.9)]},
            ]
        )

    monkeypatch.setattr(main_module, "stream_turn", fake_stream_turn)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"question": "测试问题", "use_web_search": True},
        )

    events = parse_sse_events(response.text)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert events[0]["type"] == "session"
    assert "".join(event["text"] for event in events if event["type"] == "token") == "测试回答"
    assert sum(event["type"] == "done" for event in events) == 1
    assert not any(event["type"] == "error" for event in events)
    assert events[-1] == {
        "type": "done",
        "citations": [{"source": "劳动合同法", "article": "第一条", "score": 0.9}],
        "web_sources": [],
    }
    assert captured["question"] == "测试问题"
    assert captured["use_web_search"] is True


def test_chat_stream_emits_one_error_before_first_token(monkeypatch) -> None:
    set_ready(monkeypatch)
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)

    class FailingGraph:
        def stream(self, input, config, stream_mode):
            raise ExternalServiceError("upstream-detail")
            yield

    monkeypatch.setattr(graph_module, "get_graph", lambda: FailingGraph())
    monkeypatch.setattr(main_module, "stream_turn", graph_module.stream_turn)
    monkeypatch.setattr(main_module, "log_current_exception", lambda scene, message: Path("logs/test.log"))

    with TestClient(main_module.app) as client:
        response = client.post("/api/chat/stream", json={"question": "问题"})

    events = parse_sse_events(response.text)
    assert [event["type"] for event in events] == ["session", "error"]
    assert sum(event["type"] == "error" for event in events) == 1
    assert not any(event["type"] == "done" for event in events)
    assert events[-1] == {
        "type": "error",
        "status": 502,
        "detail": "外部服务调用失败，请稍后重试",
    }


def test_chat_stream_preserves_tokens_and_emits_one_error_after_partial_output(
    monkeypatch,
) -> None:
    set_ready(monkeypatch)
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)

    class PartialFailureGraph:
        def stream(self, input, config, stream_mode):
            yield AIMessageChunk(content="已生成") , {"langgraph_node": "generate"}
            raise ExternalServiceError("upstream-detail")

    monkeypatch.setattr(graph_module, "get_graph", lambda: PartialFailureGraph())
    monkeypatch.setattr(main_module, "stream_turn", graph_module.stream_turn)
    monkeypatch.setattr(main_module, "log_current_exception", lambda scene, message: Path("logs/test.log"))

    with TestClient(main_module.app) as client:
        response = client.post("/api/chat/stream", json={"question": "问题"})

    events = parse_sse_events(response.text)
    assert [event["type"] for event in events] == ["session", "token", "error"]
    assert "".join(event["text"] for event in events if event["type"] == "token") == "已生成"
    assert sum(event["type"] == "error" for event in events) == 1
    assert not any(event["type"] == "done" for event in events)


def test_citation_payload_accepts_checkpoint_dicts() -> None:
    metadata = {"source": "中华人民共和国劳动合同法", "article": "第四十七条"}
    text = "第四十七条 经济补偿按工作年限计算。"
    live = NodeWithScore(node=TextNode(text=text, metadata=metadata), score=0.81)
    replayed = NodeWithScore.model_construct(
        node={"metadata": metadata, "text": text},
        score=0.81,
    )
    expected = [{"source": "中华人民共和国劳动合同法", "article": "第四十七条", "score": 0.81}]

    assert main_module._citation_payload([live]) == expected
    assert main_module._citation_payload([replayed]) == expected


def test_chat_stream_unknown_session_returns_404(monkeypatch) -> None:
    set_ready(monkeypatch)
    monkeypatch.setattr(main_module, "get_session", lambda session_id: None)
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"question": "问题", "session_id": "missing"},
        )
    assert response.status_code == 404


def test_chat_stream_error_event_on_external_failure(monkeypatch) -> None:
    set_ready(monkeypatch)
    monkeypatch.setattr(main_module, "create_session", lambda session_id, title: None)

    def failed_stream(session_id, question, use_web_search):
        raise ExternalServiceError("内部错误")
        yield

    monkeypatch.setattr(main_module, "stream_turn", failed_stream)
    monkeypatch.setattr(main_module, "log_current_exception", lambda scene, message: Path("logs/test.log"))
    with TestClient(main_module.app) as client:
        response = client.post("/api/chat/stream", json={"question": "问题"})

    assert response.status_code == 200
    assert '"type": "error"' in response.text
    assert '"status": 502' in response.text


def test_delete_session_endpoint(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = []
    monkeypatch.setattr(
        main_module,
        "get_session",
        lambda session_id: {"id": session_id} if session_id == "s1" else None,
    )
    monkeypatch.setattr(main_module, "delete_session_state", lambda session_id: calls.append(("state", session_id)))
    monkeypatch.setattr(main_module, "delete_session", lambda session_id: calls.append(("row", session_id)))

    with TestClient(main_module.app) as client:
        response = client.delete("/api/sessions/s1")
        missing = client.delete("/api/sessions/missing")

    assert response.status_code == 200
    assert response.json() == {"deleted": "s1"}
    assert calls == [("state", "s1"), ("row", "s1")]
    assert missing.status_code == 404


def test_web_source_payload_tolerates_missing_and_invalid_items() -> None:
    assert main_module._web_source_payload(
        [{"title": "标题", "url": "https://example.com"}, "invalid", {"site": "站点"}]
    ) == [
        {"title": "标题", "url": "https://example.com", "site": "", "date": ""},
        {"title": "", "url": "", "site": "站点", "date": ""},
    ]
    assert main_module._web_source_payload([]) == []


# ---------------------------------------------------------------------------
# P11：清理处理失败后新建的空会话
# ---------------------------------------------------------------------------


def _cleanup_factory(monkeypatch, state_error=None, row_error=None):
    calls = []
    monkeypatch.setattr(
        main_module, "create_session",
        lambda session_id, title: calls.append(("create", session_id)),
    )
    monkeypatch.setattr(main_module, "touch_session", lambda session_id: None)

    def fake_state(session_id):
        calls.append(("state", session_id))
        if state_error:
            raise state_error

    def fake_row(session_id):
        calls.append(("row", session_id))
        if row_error:
            raise row_error

    monkeypatch.setattr(main_module, "delete_session_state", fake_state)
    monkeypatch.setattr(main_module, "delete_session", fake_row)
    return calls


def test_p11_t01_sync_new_session_cleanup_on_index_not_ready(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "run_turn",
        lambda sid, q, w: (_ for _ in ()).throw(IndexNotReadyError("索引未就绪")),
    )
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat", json={"question": "问题"})

    assert resp.status_code == 503
    assert resp.json() == {"detail": "索引未就绪"}
    ops = [op for op, _ in calls]
    assert "create" in ops
    assert ops[-2:] == ["state", "row"]  # 清理顺序 state → row


def test_p11_t02_sync_new_session_cleanup_on_external_error(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)
    logged = {}

    def fail_run_turn(sid, q, w):
        raise ExternalServiceError("上游内部细节")
        yield

    monkeypatch.setattr(main_module, "run_turn", fail_run_turn)
    monkeypatch.setattr(
        main_module, "log_current_exception",
        lambda scene, message: logged.update(scene=scene, message=message) or Path("logs/test.log"),
    )
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat", json={"question": "问题"})

    assert resp.status_code == 502
    assert resp.json() == {"detail": "外部服务调用失败，请稍后重试"}
    assert "内部细节" not in resp.text
    assert logged == {"scene": "api-chat", "message": "HTTP 多轮问答外部服务调用失败"}
    assert [op for op, _ in calls][-2:] == ["state", "row"]  # 原始 502 不被清理覆盖


def test_p11_t03_sync_new_session_unexpected_exception_cleans_then_rethrows(
    monkeypatch,
) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)

    def boom(sid, q, w):
        raise RuntimeError("unexpected")
        yield

    monkeypatch.setattr(main_module, "run_turn", boom)
    with TestClient(main_module.app, raise_server_exceptions=False) as client:
        resp = client.post("/api/chat", json={"question": "问题"})

    assert resp.status_code == 500  # 普通异常不转 502/503
    assert [op for op, _ in calls][-2:] == ["state", "row"]


def test_p11_t04_sync_existing_session_failure_preserved(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)
    monkeypatch.setattr(main_module, "get_session", lambda sid: {"id": sid})
    monkeypatch.setattr(main_module, "touch_session", lambda sid: None)
    monkeypatch.setattr(
        main_module,
        "run_turn",
        lambda sid, q, w: (_ for _ in ()).throw(ExternalServiceError("内部")),
    )
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat", json={"question": "问题", "session_id": "s1"})

    assert resp.status_code == 502
    # 已有会话不触发自动清理
    assert [op for op, _ in calls if op in ("state", "row")] == []


def test_p11_t05_stream_new_session_error_before_first_token(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)

    def fail_stream(sid, q, w):
        raise ExternalServiceError("内部错误")
        yield

    monkeypatch.setattr(main_module, "stream_turn", fail_stream)
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat/stream", json={"question": "问题"})

    events = parse_sse_events(resp.text)
    assert [event["type"] for event in events] == ["session", "error"]
    assert sum(event["type"] == "error" for event in events) == 1
    assert not any(event["type"] == "done" for event in events)
    assert [op for op, _ in calls][-2:] == ["state", "row"]


def test_p11_t06_stream_new_session_error_after_partial_output(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)

    def partial_stream(sid, q, w):
        yield {"type": "token", "text": "部分"}
        raise ExternalServiceError("内部错误")

    monkeypatch.setattr(main_module, "stream_turn", partial_stream)
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat/stream", json={"question": "问题"})

    events = parse_sse_events(resp.text)
    assert [event["type"] for event in events] == ["session", "token", "error"]
    assert sum(event["type"] == "error" for event in events) == 1
    assert not any(event["type"] == "done" for event in events)
    assert [op for op, _ in calls][-2:] == ["state", "row"]


def test_p11_t07_stream_new_session_explicit_error_event(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "stream_turn",
        lambda sid, q, w: iter([{"type": "error", "status": 502}]),
    )
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat/stream", json={"question": "问题"})

    events = parse_sse_events(resp.text)
    assert events[0]["type"] == "session"
    assert events[-1] == {
        "type": "error",
        "status": 502,
        "detail": "外部服务调用失败，请稍后重试",
    }
    assert not any(event["type"] == "done" for event in events)
    assert [op for op, _ in calls][-2:] == ["state", "row"]


def test_p11_t08_stream_new_session_missing_done_is_cleaned(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "stream_turn",
        lambda sid, q, w: iter([{"type": "token", "text": "x"}]),
    )
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat/stream", json={"question": "问题"})

    events = parse_sse_events(resp.text)
    assert not any(event["type"] == "done" for event in events)
    assert not any(event["type"] == "error" for event in events)
    assert [op for op, _ in calls][-2:] == ["state", "row"]


def test_p11_t09_sync_and_stream_success_preserve_new_session(monkeypatch) -> None:
    for kind in ("sync", "stream"):
        set_ready(monkeypatch)
        calls = _cleanup_factory(monkeypatch)
        if kind == "sync":
            monkeypatch.setattr(
                main_module, "run_turn", lambda sid, q, w: ("回答", [])
            )
            with TestClient(main_module.app) as client:
                resp = client.post("/api/chat", json={"question": "问题"})
        else:
            node = TextNode(text="t", metadata={"source": "s", "article": "a"})
            monkeypatch.setattr(
                main_module,
                "stream_turn",
                lambda sid, q, w: iter(
                    [{"type": "done", "citations": [NodeWithScore(node=node, score=0.9)]}]
                ),
            )
            with TestClient(main_module.app) as client:
                resp = client.post("/api/chat/stream", json={"question": "问题"})

        assert resp.status_code == 200
        # 成功路径不触发自动清理
        assert [op for op, _ in calls if op in ("state", "row")] == []


def test_p11_t10_stream_existing_session_failure_preserved(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)
    monkeypatch.setattr(main_module, "get_session", lambda sid: {"id": sid})
    monkeypatch.setattr(main_module, "touch_session", lambda sid: None)

    def fail_stream(sid, q, w):
        raise ExternalServiceError("内部错误")
        yield

    monkeypatch.setattr(main_module, "stream_turn", fail_stream)
    with TestClient(main_module.app) as client:
        resp = client.post(
            "/api/chat/stream", json={"question": "问题", "session_id": "s1"}
        )

    events = parse_sse_events(resp.text)
    assert events[-1]["type"] == "error"
    assert [op for op, _ in calls if op in ("state", "row")] == []


def test_p11_t11_generator_close_does_not_trigger_cleanup(monkeypatch) -> None:
    set_ready(monkeypatch)
    calls = _cleanup_factory(monkeypatch)

    def endless_stream(sid, q, w):
        while True:
            yield {"type": "token", "text": "t"}

    monkeypatch.setattr(main_module, "stream_turn", endless_stream)
    gen = main_module.chat_event_stream("new-sid", "q", False, is_new_session=True)
    next(gen)  # session
    next(gen)  # token
    gen.close()  # GeneratorExit，不按失败清理
    gen = None
    assert [op for op, _ in calls if op in ("state", "row")] == []


def test_p11_t12_cleanup_one_side_failure_does_not_block_other(monkeypatch) -> None:
    set_ready(monkeypatch)

    # 方向 1：state 清理失败仍删 sessions 行
    calls = _cleanup_factory(monkeypatch, state_error=OSError("state-fail-on-cleanup"))
    logged = []

    def boom(sid, q, w):
        raise ExternalServiceError("内部")
        yield

    monkeypatch.setattr(main_module, "run_turn", boom)
    monkeypatch.setattr(
        main_module, "log_current_exception",
        lambda scene, message: logged.append((scene, message)) or Path("logs/test.log"),
    )
    with TestClient(main_module.app) as client:
        resp = client.post("/api/chat", json={"question": "问题"})

    assert resp.status_code == 502  # 原始业务失败保持
    assert resp.json() == {"detail": "外部服务调用失败，请稍后重试"}
    ops = [op for op, _ in calls]
    assert "state" in ops
    assert "row" in ops  # state 失败后仍尝试删 row
    assert ("cleanup-failed-session", "清理失败会话 Checkpointer 状态") in logged

    # 方向 2：row 清理失败时 state 清理结果保留，原始失败不被覆盖
    calls2 = _cleanup_factory(monkeypatch, state_error=None, row_error=OSError("row-fail"))
    logged2 = []
    monkeypatch.setattr(main_module, "run_turn", boom)
    monkeypatch.setattr(
        main_module, "log_current_exception",
        lambda scene, message: logged2.append((scene, message)) or Path("logs/test.log"),
    )
    with TestClient(main_module.app) as client:
        resp2 = client.post("/api/chat", json={"question": "问题"})

    assert resp2.status_code == 502
    assert resp2.json() == {"detail": "外部服务调用失败，请稍后重试"}
    ops2 = [op for op, _ in calls2]
    assert "state" in ops2 and "row" in ops2
    assert ("cleanup-failed-session", "清理失败会话 sessions 行") in logged2


def test_database_is_isolated_to_tmp_path(monkeypatch, tmp_path: Path) -> None:
    """P11-R2 隔离机制回归：所有 test_main 测试默认使用 tmp_path 数据库。"""
    expected_db = tmp_path / "chat.sqlite3"
    assert db_module.DATABASE_PATH == expected_db
    real_db = Path(__file__).resolve().parent.parent / "data" / "lawrag.sqlite3"
    assert db_module.DATABASE_PATH != real_db

    data_dir = Path(__file__).resolve().parent.parent / "data"
    # data/ 被 Git 忽略，CI 全新检出没有该目录；缺失时视为空集合，
    # 若本轮操作新建了 data/ 仍会被下方差集捕获（E-074）。
    names_before = {p.name for p in data_dir.iterdir()} if data_dir.exists() else set()

    graph = graph_module.get_graph()
    checkpointer_conn = graph.checkpointer.conn
    metadata_conn = db_module.get_connection()

    def main_db_file(conn) -> str:
        return conn.execute("PRAGMA database_list").fetchone()[2]

    assert main_db_file(checkpointer_conn) == str(expected_db)
    assert main_db_file(metadata_conn) == str(expected_db)
    assert checkpointer_conn is not metadata_conn

    # 真实 data/ 在本轮操作后未新增任何文件；历史残留（如异常中止遗留的
    # 0 字节 sidecar）不算失败，等应用下次正常启停由 SQLite 自行清理。
    names_after = {p.name for p in data_dir.iterdir()} if data_dir.exists() else set()
    assert not (names_after - names_before), f"真实 data/ 出现新文件：{names_after - names_before}"


# ---------------------------------------------------------------------------
# P12：来源链接协议限制（输出边界）
# ---------------------------------------------------------------------------


def test_web_source_payload_sanitizes_unsafe_urls() -> None:
    payload = main_module._web_source_payload(
        [
            {"title": "安全", "url": "https://example.com/a", "site": "站A", "date": "2026-09-16"},
            {"title": "脚本", "url": "javascript:alert(1)", "site": "站B"},
            {"title": "数据", "url": "data:text/html,x", "site": "站C"},
            {"title": "相对", "url": "/relative", "site": "站D"},
            {"url": "https://ok.example.com", "site": "站E"},
        ]
    )
    assert payload == [
        {"title": "安全", "url": "https://example.com/a", "site": "站A", "date": "2026-09-16"},
        {"title": "脚本", "url": "", "site": "站B", "date": ""},
        {"title": "数据", "url": "", "site": "站C", "date": ""},
        {"title": "相对", "url": "", "site": "站D", "date": ""},
        {"title": "", "url": "https://ok.example.com", "site": "站E", "date": ""},
    ]
    # 非 dict 项继续忽略；空列表返回空
    assert main_module._web_source_payload(["invalid", {"title": "x", "url": "ftp://x"}]) == [
        {"title": "x", "url": "", "site": "", "date": ""}
    ]
    assert main_module._web_source_payload([]) == []


# ---------------------------------------------------------------------------
# P12-R1：同步 /api/chat 与流式 done 实际路径清理 URL
# ---------------------------------------------------------------------------


def test_chat_api_sanitizes_web_source_urls(monkeypatch) -> None:
    """P12-R1-T02：实际 POST /api/chat，Checkpoint 中非法 URL 被输出边界清空。"""
    set_ready(monkeypatch)

    class FakeGraph:
        def get_state(self, config):
            return SimpleNamespace(
                values={
                    "web_sources": [
                        {"title": "安全", "url": "https://example.com/a", "site": "站A", "date": "2026-09-21"},
                        {"title": "脚本", "url": "javascript:alert(1)", "site": "站B", "date": "2026-09-21"},
                        {"title": "缺主机", "url": "https://user@:443", "site": "站C", "date": "2026-09-21"},
                        {"title": "畸形", "url": "https://[broken", "site": "站D", "date": "2026-09-21"},
                    ]
                }
            )

    monkeypatch.setattr(
        main_module, "run_turn", lambda session_id, question, use_web_search: ("回答", [])
    )
    monkeypatch.setattr(main_module, "get_graph", lambda: FakeGraph())

    with TestClient(main_module.app) as client:
        response = client.post("/api/chat", json={"question": "问题", "use_web_search": True})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "回答"
    assert body["citations"] == []
    assert body["session_id"]  # 会话结构不回归
    assert body["web_sources"] == [
        {"title": "安全", "url": "https://example.com/a", "site": "站A", "date": "2026-09-21"},
        {"title": "脚本", "url": "", "site": "站B", "date": "2026-09-21"},
        {"title": "缺主机", "url": "", "site": "站C", "date": "2026-09-21"},
        {"title": "畸形", "url": "", "site": "站D", "date": "2026-09-21"},
    ]


def test_chat_event_stream_done_sanitizes_web_source_urls(monkeypatch) -> None:
    """P12-R1-T03：实际运行 chat_event_stream，done 事件中非法 URL 被清空。"""
    set_ready(monkeypatch)

    def fake_stream_turn(session_id, question, use_web_search):
        assert use_web_search is True
        yield {
            "type": "done",
            "citations": [],
            "web_sources": [
                {"title": "安全", "url": "https://example.com/a", "site": "站A", "date": "2026-09-21"},
                {"title": "脚本", "url": "javascript:alert(1)", "site": "站B", "date": "2026-09-21"},
                {"title": "缺主机", "url": "https://:80", "site": "站C", "date": "2026-09-21"},
                {"title": "畸形", "url": "https://[broken", "site": "站D", "date": "2026-09-21"},
            ],
        }

    monkeypatch.setattr(main_module, "stream_turn", fake_stream_turn)

    blocks = parse_sse_events("".join(main_module.chat_event_stream("s1", "问题", True)))

    assert blocks[0] == {"type": "session", "session_id": "s1"}
    done = [block for block in blocks if block["type"] == "done"]
    errors = [block for block in blocks if block["type"] == "error"]
    assert len(done) == 1  # 只有一个 done，P01/P11 语义不变
    assert errors == []
    assert done[0]["citations"] == []
    assert done[0]["web_sources"] == [
        {"title": "安全", "url": "https://example.com/a", "site": "站A", "date": "2026-09-21"},
        {"title": "脚本", "url": "", "site": "站B", "date": "2026-09-21"},
        {"title": "缺主机", "url": "", "site": "站C", "date": "2026-09-21"},
        {"title": "畸形", "url": "", "site": "站D", "date": "2026-09-21"},
    ]
