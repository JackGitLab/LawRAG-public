"""Langfuse 可观测性单元测试。

遵守 E-074：全部 monkeypatch，不依赖真实密钥、不联网、不发送真实 trace。
"""

import logging
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import config as lc_config
from llama_index.core.schema import NodeWithScore, TextNode

import app.db as db
from app import config, graph, observability


class StubModel:
    """记录 invoke 调用的假模型，返回固定文本。"""

    def __init__(self, reply: str = "模拟回答"):
        self.reply = reply
        self.calls: list = []

    def invoke(self, prompt, config=None):
        self.calls.append({"prompt": prompt, "config": config})
        return SimpleNamespace(content=self.reply, tool_calls=None)


def _enable_langfuse(monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "sk-lf-test")


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "")
    assert observability.get_langfuse_callback() is None


def test_enabled_returns_handler(monkeypatch):
    _enable_langfuse(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(observability, "_new_handler", lambda: sentinel)
    assert observability.get_langfuse_callback() is sentinel


def test_init_failure_returns_none(monkeypatch):
    _enable_langfuse(monkeypatch)

    def _boom():
        raise RuntimeError("langfuse 不可用")

    monkeypatch.setattr(observability, "_new_handler", _boom)
    assert observability.get_langfuse_callback() is None


def test_init_failure_log_redacts_sentinel(monkeypatch, caplog):
    """E-076：初始化异常日志必须脱敏——完整格式化输出含 [REDACTED] 且不含哨兵原文。"""
    sentinel = "sk-sentinel-abc123"
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", sentinel)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", sentinel)

    def _boom():
        raise RuntimeError(f"invalid key {sentinel}")

    monkeypatch.setattr(observability, "_new_handler", _boom)
    with caplog.at_level(logging.WARNING, logger="app.observability"):
        assert observability.get_langfuse_callback() is None

    output = caplog.text
    assert sentinel not in output  # 哨兵值绝不出现在任何输出行
    assert "[REDACTED]" in output  # 脱敏占位符存在
    assert "RuntimeError" in output  # 异常类型摘要仍保留，方便排障
    assert "Traceback" not in output  # 已移除 exc_info，不输出未脱敏堆栈


def test_traced_invoke_without_callback(monkeypatch):
    monkeypatch.setattr(observability, "get_langfuse_callback", lambda: None)
    model = StubModel()
    result = observability.traced_invoke(model, "问题")
    assert result.content == "模拟回答"
    assert len(model.calls) == 1
    assert model.calls[0]["prompt"] == "问题"
    assert model.calls[0]["config"] is None


def test_traced_invoke_with_callback(monkeypatch):
    """启用 Langfuse 时对继承回调追加而非替换（E-075）。"""
    sentinel = object()
    monkeypatch.setattr(observability, "get_langfuse_callback", lambda: sentinel)
    inherited_handler = object()
    token = lc_config.var_child_runnable_config.set(
        {"callbacks": [inherited_handler], "metadata": {"keep": "me"}}
    )
    try:
        model = StubModel()
        observability.traced_invoke(model, "问题", session_id="sess-1", tags=["generate"])
    finally:
        lc_config.var_child_runnable_config.reset(token)

    assert len(model.calls) == 1
    invoke_config = model.calls[0]["config"]
    # 追加而非替换：继承 handler 在前，Langfuse handler 在后
    assert invoke_config["callbacks"] == [inherited_handler, sentinel]
    metadata = invoke_config["metadata"]
    # LangGraph 可能注入的元数据必须保留，不覆盖
    assert metadata["keep"] == "me"
    assert metadata["langfuse_session_id"] == "sess-1"
    assert metadata["langfuse_tags"] == ["generate"]


def test_traced_invoke_without_tags_omits_key(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(observability, "get_langfuse_callback", lambda: sentinel)
    model = StubModel()
    observability.traced_invoke(model, "问题", session_id="sess-2")
    metadata = model.calls[0]["config"]["metadata"]
    assert metadata == {"langfuse_session_id": "sess-2"}


def test_prepare_passes_session_to_trace(monkeypatch):
    """graph.prepare 应把 thread_id 作为 langfuse_session_id 传入调用。"""
    _enable_langfuse(monkeypatch)
    recorded = {}
    monkeypatch.setattr(
        observability,
        "_new_handler",
        lambda: SimpleNamespace(handler=True),
    )
    model = StubModel(reply="改写后的完整问题")
    monkeypatch.setattr(graph, "get_chat_model", lambda: model)

    state = {
        "question": "那赔偿呢",
        "messages": [HumanMessage(content="公司违法解除怎么赔"), HumanMessage(content="那赔偿呢")],
    }
    result = graph.prepare(state, config={"configurable": {"thread_id": "sess-42"}})

    assert result["search_query"] == "改写后的完整问题"
    invoke_config = model.calls[0]["config"]
    assert invoke_config["metadata"]["langfuse_session_id"] == "sess-42"
    assert invoke_config["metadata"]["langfuse_tags"] == ["rewrite"]
    assert invoke_config["callbacks"]


def test_prepare_without_config_still_works(monkeypatch):
    """直接调用（无 LangGraph config）时不应报错，trace 无会话归属。"""
    _enable_langfuse(monkeypatch)
    monkeypatch.setattr(
        observability, "_new_handler", lambda: SimpleNamespace(handler=True)
    )
    model = StubModel(reply="改写后的完整问题")
    monkeypatch.setattr(graph, "get_chat_model", lambda: model)

    state = {
        "question": "那赔偿呢",
        "messages": [HumanMessage(content="公司违法解除怎么赔")],
    }
    result = graph.prepare(state)
    assert result["search_query"] == "改写后的完整问题"
    metadata = model.calls[0]["config"]["metadata"]
    assert metadata["langfuse_session_id"] is None


def test_hyde_passes_session_downstream(monkeypatch):
    """generate_hyde_query 应把 session_id 与 hyde 标签传给底层文本生成。"""
    from app import ask

    captured = {}

    def fake_short_text(prompt, max_tokens, session_id=None, tags=None):
        captured["session_id"] = session_id
        captured["tags"] = tags
        return "假想条文文本"

    monkeypatch.setattr(ask, "generate_short_retrieval_text", fake_short_text)
    assert ask.generate_hyde_query("问题", session_id="sess-7") == "假想条文文本"
    assert captured["session_id"] == "sess-7"
    assert captured["tags"] == ["hyde"]


def test_redact_covers_langfuse_secret_key(monkeypatch):
    """日志脱敏应覆盖 _SECRET_KEY 后缀（Langfuse 私钥）。"""
    monkeypatch.setenv("FAKE_LANGFUSE_SECRET_KEY", "sk-lf-super-secret")
    text = "初始化失败 key=sk-lf-super-secret"
    assert config._redact_secrets(text) == "初始化失败 key=[REDACTED]"


# ---------------------------------------------------------------------------
# E-075：真实编译图离线流式回归——开启 Langfuse 回调不得丢失 token
# ---------------------------------------------------------------------------


class RecordingHandler(BaseCallbackHandler):
    """记录 on_chat_model_start 收到的 metadata，验证 trace 归属。"""

    def __init__(self):
        self.metadata = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.metadata.append(kwargs.get("metadata"))


@pytest.mark.parametrize("enabled", [False, True])
def test_compiled_graph_stream_preserves_tokens(monkeypatch, tmp_path, enabled):
    """E-075 回归：真实编译图 stream_mode="messages" 在两种开关状态下 token 都不丢失。

    复用 test_graph.py 的临时 checkpoint 隔离模式与 FakeListChatModel 离线流式桩；
    全程不联网、不触真实密钥（conftest autouse 已禁用真实 Langfuse）。
    """
    graph.get_graph.cache_clear()
    if db.get_checkpointer_connection.cache_info().currsize:
        db.get_checkpointer_connection().close()
    db.get_checkpointer_connection.cache_clear()
    if db.get_connection.cache_info().currsize:
        db.get_connection().close()
    db.get_connection.cache_clear()
    monkeypatch.setattr(db, "DATABASE_PATH", tmp_path / "stream.sqlite3")
    monkeypatch.setattr(
        graph, "get_checkpointer_connection", db.get_checkpointer_connection
    )
    node = TextNode(text="测试法条", metadata={"source": "劳动合同法", "article": "第一条"})
    monkeypatch.setattr(
        graph,
        "retrieve",
        lambda question, top_k=3, session_id=None: [NodeWithScore(node=node, score=0.9)],
    )
    monkeypatch.setattr(graph, "get_chat_model", lambda: FakeListChatModel(responses=["流式回答"]))
    handler = RecordingHandler()
    monkeypatch.setattr(
        observability, "get_langfuse_callback", lambda: handler if enabled else None
    )

    try:
        events = list(graph.stream_turn("stream-session", "普通问题"))
        token_text = "".join(event["text"] for event in events if event["type"] == "token")
        saved = graph.get_graph().get_state(
            graph.graph_config("stream-session")
        ).values["answer"]
        assert saved == "流式回答"
        # token 顺序拼接必须等于 checkpoint 中的最终回答（E-075 核心回归点）
        assert token_text == saved
        assert sum(event["type"] == "done" for event in events) == 1
        if enabled:
            assert handler.metadata  # 伪造 handler 确实收到了模型调用
            first = handler.metadata[0]
            assert first["langfuse_session_id"] == "stream-session"
            assert first["langfuse_tags"] == ["generate"]
    finally:
        graph.get_graph.cache_clear()
        if db.get_checkpointer_connection.cache_info().currsize:
            db.get_checkpointer_connection().close()
        db.get_checkpointer_connection.cache_clear()
        if db.get_connection.cache_info().currsize:
            db.get_connection().close()
        db.get_connection.cache_clear()
