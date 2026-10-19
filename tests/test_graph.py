import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from llama_index.core.schema import NodeWithScore, TextNode

import app.db as db_module
import app.graph as graph_module
from app.ask import ExternalServiceError


class RouterStub:
    """离线占位：不真实调用模型，也不写入该用例的 prompts 列表。"""

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        return SimpleNamespace(tool_calls=[])


def test_two_turn_graph_reuses_history(monkeypatch, tmp_path: Path) -> None:
    graph_module.get_graph.cache_clear()
    if db_module.get_checkpointer_connection.cache_info().currsize:
        db_module.get_checkpointer_connection().close()
    db_module.get_checkpointer_connection.cache_clear()
    if db_module.get_connection.cache_info().currsize:
        db_module.get_connection().close()
    db_module.get_connection.cache_clear()
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path / "t.sqlite3")
    monkeypatch.setattr(
        graph_module, "get_checkpointer_connection", db_module.get_checkpointer_connection
    )

    nodes = [
        NodeWithScore(
            node=TextNode(text=f"法条 {index}", metadata={"source": "劳动合同法", "article": f"第{index}条"}),
            score=0.9 - index / 10,
        )
        for index in range(1, 3)
    ]
    queries = []
    prompts = []

    def fake_retrieve(query, top_k=3, session_id=None):
        queries.append(query)
        return nodes

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            if isinstance(prompt, str) and "改写成可独立检索" in prompt:
                return SimpleNamespace(content="劳动合同经济补偿如何计算？")
            return SimpleNamespace(content="测试回答")

    monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())
    monkeypatch.setattr(graph_module, "get_router_model", lambda: RouterStub())

    try:
        answer, citations = graph_module.run_turn("session-1", "劳动合同到期有补偿吗？")
        assert answer == "测试回答"
        assert citations == nodes
        assert queries == ["劳动合同到期有补偿吗？"]
        assert len(prompts) == 1

        graph_module.run_turn("session-1", "那补偿怎么算？")
        assert queries[1] != "那补偿怎么算？"
        assert len(prompts) == 3
        snapshot = graph_module.get_graph().get_state(
            config=graph_module.graph_config("session-1")
        )
        assert len(snapshot.values["messages"]) == 4
        assert snapshot.values["messages"][-1].additional_kwargs["citations"] == [
            {"source": "劳动合同法", "article": "第1条", "score": 0.8},
            {"source": "劳动合同法", "article": "第2条", "score": 0.7},
        ]
    finally:
        graph_module.get_graph.cache_clear()
        if db_module.get_checkpointer_connection.cache_info().currsize:
            db_module.get_checkpointer_connection().close()
        db_module.get_checkpointer_connection.cache_clear()
        if db_module.get_connection.cache_info().currsize:
            db_module.get_connection().close()
        db_module.get_connection.cache_clear()


def test_graph_uses_dedicated_checkpointer_connection(monkeypatch, tmp_path: Path) -> None:
    graph_module.get_graph.cache_clear()
    if db_module.get_checkpointer_connection.cache_info().currsize:
        db_module.get_checkpointer_connection().close()
    db_module.get_checkpointer_connection.cache_clear()
    if db_module.get_connection.cache_info().currsize:
        db_module.get_connection().close()
    db_module.get_connection.cache_clear()
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path / "t.sqlite3")

    try:
        graph = graph_module.get_graph()
        metadata_conn = db_module.get_connection()
        checkpointer_conn = db_module.get_checkpointer_connection()

        # 运行期断言：图确实使用专用 Checkpointer 连接，而非元数据连接
        assert graph.checkpointer.conn is checkpointer_conn
        assert graph.checkpointer.conn is not metadata_conn
    finally:
        graph_module.get_graph.cache_clear()
        if db_module.get_checkpointer_connection.cache_info().currsize:
            db_module.get_checkpointer_connection().close()
        db_module.get_checkpointer_connection.cache_clear()
        if db_module.get_connection.cache_info().currsize:
            db_module.get_connection().close()
        db_module.get_connection.cache_clear()


def test_stream_turn_filters_non_generate_events(monkeypatch) -> None:
    calls = {}

    class FakeGraph:
        def stream(self, input, config, stream_mode):
            calls.update(input=input, config=config, stream_mode=stream_mode)
            yield SimpleNamespace(content="噪声"), {"langgraph_node": "prepare"}
            # 模拟 prepare 自己的模型流，必须由节点名过滤。
            yield AIMessageChunk(content="改写后的问题"), {"langgraph_node": "prepare"}
            yield AIMessageChunk(content="第一段"), {"langgraph_node": "generate"}
            # 模拟 generate 节点返回的整条 AIMessage，必须由消息类型过滤。
            yield AIMessage(content="第一段第二段"), {"langgraph_node": "generate"}
            yield AIMessageChunk(content=""), {"langgraph_node": "generate"}
            yield AIMessageChunk(content="第二段"), {"langgraph_node": "generate"}

        def get_state(self, config):
            assert config == {"configurable": {"thread_id": "s1"}}
            return SimpleNamespace(
                values={"answer": "第一段第二段", "citations": ["stub"]}
            )

    monkeypatch.setattr(graph_module, "get_graph", lambda: FakeGraph())

    events = list(graph_module.stream_turn("s1", "问题", use_web_search=True))

    assert events == [
        {"type": "token", "text": "第一段"},
        {"type": "token", "text": "第二段"},
        {"type": "done", "citations": ["stub"], "web_sources": []},
    ]
    assert "".join(event["text"] for event in events if event["type"] == "token") == (
        graph_module.get_graph().get_state(graph_module.graph_config("s1")).values["answer"]
    )
    assert sum(event["type"] == "done" for event in events) == 1
    assert not any(event["type"] == "error" for event in events)
    assert calls == {
        "input": {"question": "问题", "use_web_search": True},
        "config": {"configurable": {"thread_id": "s1"}},
        "stream_mode": "messages",
    }


def test_call_tools_continues_when_tool_raises(monkeypatch) -> None:
    class FailingTool:
        name = "broken"

        async def ainvoke(self, args):
            raise RuntimeError("boom")

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [FailingTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": "broken", "args": {}}]}
    )

    assert result == {
        "tool_text": "[工具 broken 调用失败：RuntimeError]",
        "web_sources": [],
    }


@pytest.mark.parametrize("tool_name", ["web_search", "verify_statute_status"])
def test_call_tools_marks_empty_web_results_as_insufficient(monkeypatch, tool_name) -> None:
    class WebTool:
        name = tool_name

        async def ainvoke(self, args):
            return [{"type": "text", "text": '{"ok": true, "results": []}'}]

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [WebTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": tool_name, "args": {"query": "问题"}}]}
    )

    assert f"[工具 {tool_name} 联网资料不足，本轮无法完成联网核验。]" in result["tool_text"]
    assert "无法完成联网核验" in result["tool_text"]
    assert result["web_sources"] == []
    assert "results" not in result["tool_text"]  # 原始空结果不被当作成功核验资料


@pytest.mark.parametrize(
    "text",
    [
        '{"ok": false, "error": "额度不足"}',
        '{"error": "失败"}',
        "{}",
        "[]",  # JSON 数组
        '"字符串"',  # JSON 字符串
        "not json",  # 非 JSON
        "",  # 空字符串
    ],
)
def test_call_tools_marks_failed_payload_as_insufficient(monkeypatch, text) -> None:
    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            return [{"type": "text", "text": text}]

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [WebTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": "web_search", "args": {"query": "问题"}}]}
    )

    assert "联网资料不足，本轮无法完成联网核验" in result["tool_text"]
    assert result["web_sources"] == []
    # 不泄漏内部细节或原文
    assert "ConnectionError" not in result["tool_text"]
    assert "traceback" not in result["tool_text"].lower()


@pytest.mark.parametrize(
    "text",
    [
        '{"ok": true}',
        '{"ok": true, "results": null}',
        '{"ok": true, "results": {}}',
        '{"ok": true, "results": [null, "文本"]}',
        '{"ok": true, "results": [{}]}',
        '{"ok": true, "results": [{"title": "  ", "url": "", "site": "", "date": "", "summary": ""}]}',
    ],
)
def test_call_tools_marks_invalid_results_as_insufficient(monkeypatch, text) -> None:
    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            return [{"type": "text", "text": text}]

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [WebTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": "web_search", "args": {"query": "问题"}}]}
    )

    assert "联网资料不足，本轮无法完成联网核验" in result["tool_text"]
    assert result["web_sources"] == []


@pytest.mark.parametrize(
    ("field", "url"),
    [
        ("title", "https://example.com/t"),
        ("url", "https://example.com/u"),
        ("site", "站点"),
        ("date", "2026-09-18"),
        ("summary", "摘要内容"),
    ],
)
def test_call_tools_single_minimum_field_counts_as_usable(monkeypatch, field, url) -> None:
    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            item = {"title": "", "url": "", "site": "", "date": "", "summary": ""}
            item[field] = url
            return [{"type": "text", "text": json.dumps({"ok": True, "results": [item]})}]

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [WebTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": "web_search", "args": {"query": "问题"}}]}
    )

    assert "联网资料不足" not in result["tool_text"]
    assert "web_search 返回" in result["tool_text"]
    # summary-only 不得被误判为无资料（已作为可用资料进入成功路径）；
    # web_sources_from_text 的既有解析范围不变，不对 summary 强制断言空 sources
    if field == "summary":
        assert not any(
            s["title"] or s["url"] or s["site"] or s["date"] for s in result["web_sources"]
        )
    else:
        assert len(result["web_sources"]) == 1


def test_call_tools_mcp_init_failure_reaches_generation_context_unchanged(
    monkeypatch,
) -> None:
    from app.ask import build_prompt_with_tools

    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            return [{"type": "text", "text": '{"ok": true, "results": []}'}]

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: "MCP 工具加载失败")
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [WebTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": "web_search", "args": {"query": "问题"}}]}
    )

    assert "联网搜索初始化失败，本轮无法完成联网核验" in result["tool_text"]
    assert result["web_sources"] == []
    assert "MCP 工具加载失败" not in result["tool_text"]  # 不泄漏内部错误


def test_local_calculator_does_not_initialize_mcp(monkeypatch) -> None:
    class CalcTool:
        name = "calculate_labor_compensation"

        async def ainvoke(self, args):
            return "最终金额：8000.00 元"

    calls = {"n": 0}

    def spy_ensure():
        calls["n"] += 1
        return None

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", spy_ensure)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [CalcTool()])

    result = graph_module.call_tools(
        {
            "planned_tool_calls": [
                {
                    "name": "calculate_labor_compensation",
                    "args": {"monthly_wage": 8000, "years": 1},
                }
            ]
        }
    )

    assert calls["n"] == 0  # 纯本地计算器不触发 MCP 初始化
    assert "最终金额：8000.00 元" in result["tool_text"]
    assert result["web_sources"] == []
    assert "联网" not in result["tool_text"]


@pytest.mark.parametrize("name", ["web_search", "verify_statute_status"])
def test_mcp_initialization_failure_reaches_generation_context(monkeypatch, name) -> None:
    from app.ask import build_prompt_with_tools

    class WebTool:
        invoked = []

        async def ainvoke(self, args):
            WebTool.invoked.append(args)
            return [{"type": "text", "text": '{"ok": true, "results": []}'}]

    WebTool.name = name
    WebTool.invoked.clear()
    calls = {"n": 0}

    def failing_ensure():
        calls["n"] += 1
        return "MCP 工具加载失败：ConnectionError: 连接被拒绝"

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", failing_ensure)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [WebTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": name, "args": {"query": "劳动合同法修订"}}]}
    )

    assert calls["n"] == 1
    assert WebTool.invoked == []  # 对应工具不会被调用
    assert "联网搜索初始化失败，本轮无法完成联网核验，请稍后重试" in result["tool_text"]
    assert result["web_sources"] == []  # 不能伪装成已完成联网核验
    # 不泄漏内部错误细节
    assert "ConnectionError" not in result["tool_text"]
    assert "连接被拒绝" not in result["tool_text"]

    # 该 tool_text 原样进入生成阶段使用的提示
    prompt = build_prompt_with_tools("劳动合同法最近修订了吗？", [], result["tool_text"])
    assert "联网搜索初始化失败，本轮无法完成联网核验，请稍后重试" in prompt


def test_mcp_tool_success_returns_text_and_sources(monkeypatch) -> None:
    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            # MCP adapter 的真实返回形态为 list[dict] 内容块（E-036）
            return [
                {
                    "type": "text",
                    "text": (
                        '{"ok": true, "results": [{"title": "标题", '
                        '"url": "https://example.com", "site": "站点", '
                        '"date": "2026-09-16"}]}'
                    ),
                }
            ]

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [WebTool()])

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": "web_search", "args": {"query": "问题"}}]}
    )

    assert "web_search 返回" in result["tool_text"]
    assert result["web_sources"] == [
        {
            "title": "标题",
            "url": "https://example.com",
            "site": "站点",
            "date": "2026-09-16",
        }
    ]


def _extract_history_text(prompt: str) -> str:
    """按提示固定分隔提取“会话历史”段。"""
    start = prompt.index("会话历史：\n") + len("会话历史：\n")
    end = prompt.index("\n本轮追问：", start)
    return prompt[start:end]


def test_prepare_uses_only_recent_six_history_messages(monkeypatch) -> None:
    prompts = []

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="改写后的问题")

    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())

    history = []
    for index in range(8):
        history.append(HumanMessage(content=f"用户问题{index}"))
        history.append(AIMessage(content=f"回答{index}"))

    result = graph_module.prepare({"question": "最新追问", "messages": history})

    prompt = prompts[0]
    history_text = _extract_history_text(prompt)
    # 最近 6 条（H5,A5,H6,A6,H7,A7）出现，更早的 H0..A4 不出现
    assert "用户问题5" in history_text
    assert "回答7" in history_text
    assert "用户问题0" not in history_text
    assert "回答0" not in history_text
    # 最近 6 条顺序保持
    assert history_text.index("用户问题5") < history_text.index("回答7")
    # 当前问题完整出现在本轮追问
    assert prompt.endswith("本轮追问：最新追问")
    # 只调用改写模型一次
    assert len(prompts) == 1
    assert result["search_query"] == "改写后的问题"
    assert result["messages"] == [HumanMessage(content="最新追问")]


def test_prepare_history_respects_char_budget(monkeypatch) -> None:
    prompts = []

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="改写结果")

    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())

    history = [
        HumanMessage(content="很旧内容" * 3000),
        AIMessage(content="中间内容" * 500),
        HumanMessage(content="最新内容" * 500),
    ]
    result = graph_module.prepare({"question": "本轮问题", "messages": history})

    history_text = _extract_history_text(prompts[0])
    assert len(history_text) <= graph_module.MAX_REWRITE_HISTORY_CHARS
    # 最新消息内容优先保留，较旧部分被丢弃
    assert "最新内容" in history_text
    assert "很旧内容" not in history_text
    assert "中间内容" in history_text
    assert history_text.index("中间内容") < history_text.index("最新内容")
    # 当前问题不被历史字符预算截断
    assert prompts[0].endswith("本轮追问：本轮问题")
    assert result["search_query"] == "改写结果"


def test_prepare_truncates_single_overlong_history_message(monkeypatch) -> None:
    prompts = []

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="改写结果")

    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())

    history = [
        HumanMessage(content="旧问题"),
        AIMessage(content="超长内容" * 2500),  # 10000 字符
    ]
    result = graph_module.prepare({"question": "本轮问题", "messages": history})

    history_text = _extract_history_text(prompts[0])
    assert len(history_text) <= graph_module.MAX_REWRITE_HISTORY_CHARS
    assert history_text.startswith("ai: ")  # role/type 标识仍可辨认
    assert "超长内容" in history_text
    # 原始 message.content 未被修改
    assert len(history[1].content) == 10000
    # 模型只收到裁剪后的 prompt
    assert prompts[0].endswith("本轮追问：本轮问题")
    assert result["search_query"] == "改写结果"


def test_prepare_history_budget_counts_newline_separators(monkeypatch) -> None:
    prompts = []

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="改写结果")

    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())

    # 单字符单元构造：两条序列化行各自恰好 3000 字符，两行合计 6000
    first = "h" * 2993  # "human: " 前缀 7 字符
    second = "a" * 2996  # "ai: " 前缀 4 字符
    assert len(f"human: {first}") == 3000
    assert len(f"ai: {second}") == 3000

    history = [HumanMessage(content=first), AIMessage(content=second)]
    result = graph_module.prepare({"question": "本轮问题", "messages": history})

    history_text = _extract_history_text(prompts[0])
    # 两行合计 6000，若都保留则 join 后为 6001，超预算：
    # 较旧消息必须被舍弃，最终不超过 6000
    assert len(history_text) <= graph_module.MAX_REWRITE_HISTORY_CHARS
    # 最新一条（AI）被保留
    assert history_text.startswith("ai: a")
    # 会导致超限的较旧消息被舍弃
    assert "human: " not in history_text
    # 不以无意义的截断 role 前缀开始
    assert history_text.startswith("ai: ")
    # 当前问题不受影响
    assert prompts[0].endswith("本轮追问：本轮问题")
    assert result["search_query"] == "改写结果"


def test_prepare_first_turn_skips_rewrite_model(monkeypatch) -> None:
    calls = {"n": 0}

    def fail_model():
        calls["n"] += 1
        raise AssertionError("第一轮不应调用改写模型")

    monkeypatch.setattr(graph_module, "get_chat_model", fail_model)

    result = graph_module.prepare({"question": "第一个问题", "messages": []})

    assert result["search_query"] == "第一个问题"
    assert calls["n"] == 0
    assert result["messages"] == [HumanMessage(content="第一个问题")]


def test_prepare_rewrite_empty_content_falls_back_to_question(monkeypatch) -> None:
    prompts = []

    class EmptyModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="")

    monkeypatch.setattr(graph_module, "get_chat_model", lambda: EmptyModel())

    history = [HumanMessage(content="旧问题"), AIMessage(content="旧回答")]
    result = graph_module.prepare({"question": "本轮追问", "messages": history})

    assert result["search_query"] == "本轮追问"
    assert len(prompts) == 1


def test_prepare_rewrite_exception_becomes_external_error(monkeypatch) -> None:
    class FailingModel:
        def invoke(self, prompt):
            raise OSError("上游中断")

    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FailingModel())

    history = [HumanMessage(content="旧问题"), AIMessage(content="旧回答")]
    with pytest.raises(ExternalServiceError, match="生成服务调用失败") as caught:
        graph_module.prepare({"question": "本轮追问", "messages": history})

    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize("text", ["not json", ""])
def test_has_usable_web_results_tolerates_bad_json(text) -> None:
    # 非 JSON/空字符是合法的外部输入形态：返回 False，不抛异常
    assert graph_module._has_usable_web_results(text) is False


def test_has_usable_web_results_does_not_swallow_unexpected_error(monkeypatch) -> None:
    def broken_loads(*args, **kwargs):
        raise RuntimeError("模拟内部错误")

    monkeypatch.setattr(graph_module.json, "loads", broken_loads)

    # 意外异常（实现错误）必须继续上抛，不得被转换成 False
    with pytest.raises(RuntimeError, match="模拟内部错误"):
        graph_module._has_usable_web_results('{"ok": true, "results": []}')


def test_has_usable_web_results_url_protocol_boundary() -> None:
    # 仅合法 https/http URL → 可用
    assert graph_module._has_usable_web_results(
        '{"ok": true, "results": [{"url": "https://example.com/a"}]}'
    ) is True
    assert graph_module._has_usable_web_results(
        '{"ok": true, "results": [{"url": "http://example.com/a"}]}'
    ) is True

    # 仅非法 URL、其他字段为空 → 不可用（P12）
    for bad_url in (
        "javascript:alert(1)",
        "data:text/html,x",
        "file:///etc/passwd",
        "ftp://example.com/f",
        "//example.com/path",
        "/relative",
        "example.com",
        "https://",
        "",
    ):
        assert graph_module._has_usable_web_results(
            '{"ok": true, "results": [{"url": "%s"}]}' % bad_url
        ) is False, bad_url

    # 非法 URL + 任一有效字段 → 仍可用（不破坏其他证据字段）
    assert graph_module._has_usable_web_results(
        '{"ok": true, "results": [{"url": "javascript:x", "title": "标题"}]}'
    ) is True
    assert graph_module._has_usable_web_results(
        '{"ok": true, "results": [{"url": "data:x", "site": "站点"}]}'
    ) is True
    assert graph_module._has_usable_web_results(
        '{"ok": true, "results": [{"url": "ftp:x", "date": "2026-09-16"}]}'
    ) is True
    assert graph_module._has_usable_web_results(
        '{"ok": true, "results": [{"url": "bad", "summary": "摘要"}]}'
    ) is True

    # E-041 组合：summary 空、title/site/date 有效、合法 URL → 仍可用
    assert graph_module._has_usable_web_results(
        '{"ok": true, "results": [{"title": "标题", "site": "站点", "date": "2026-09-16", '
        '"url": "https://example.com/a", "summary": ""}]}'
    ) is True


# ---------------------------------------------------------------------------
# P09：联网工具与本地计算器共同执行
# ---------------------------------------------------------------------------


def _combined_calc_web_tools(calls_per_tool, calc_result="最终金额：16000.00 元", web_payload=None):
    """构造记录调用次数的计算器/联网工具替身。返回 (tools, ensure_calls, assets)。"""
    if web_payload is None:
        web_payload = [{"type": "text", "text": '{"ok": true, "results": [{"title": "标题", "url": "https://example.com", "site": "站点", "date": "2026-09-18"}]}'}]

    class CalcTool:
        name = "calculate_labor_compensation"

        async def ainvoke(self, args):
            calls_per_tool["calc"] += 1
            return calc_result

    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            calls_per_tool["web"] += 1
            return web_payload

    ensure_state = {"n": 0}

    def ensure():
        ensure_state["n"] += 1
        return None

    ensure.calls = ensure_state
    return [CalcTool(), WebTool()], ensure


def _calc_web_plan():
    return [
        {"name": "calculate_labor_compensation", "args": {"monthly_wage": 8000, "years": 3.5}},
        {"name": "web_search", "args": {"query": "问题", "count": 5}},
    ]


def test_combined_calculation_and_web_success(monkeypatch) -> None:
    calls = {"calc": 0, "web": 0}
    tools, ensure_calls = _combined_calc_web_tools(calls)
    monkeypatch.setattr(graph_module, "ensure_mcp_tools", ensure_calls)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: tools)

    result = graph_module.call_tools({"planned_tool_calls": _calc_web_plan()})

    # 两个工具各执行一次，MCP 初始化一次
    assert calls == {"calc": 1, "web": 1}
    assert ensure_calls.calls["n"] == 1
    assert "最终金额：16000.00 元" in result["tool_text"]
    assert "web_search 返回" in result["tool_text"]
    # 本地结果不产生伪来源；web_sources 只来自联网结果
    assert result["web_sources"] == [
        {"title": "标题", "url": "https://example.com", "site": "站点", "date": "2026-09-18"}
    ]


def test_combined_mcp_init_failure_keeps_calculator(monkeypatch) -> None:
    calls = {"calc": 0, "web": 0}
    tools, _ = _combined_calc_web_tools(calls)

    def failing_ensure():
        return "MCP 工具加载失败（内部）"

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", failing_ensure)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: tools)

    result = graph_module.call_tools({"planned_tool_calls": _calc_web_plan()})

    # 计算器仍执行并保留结果；MCP 工具不执行
    assert calls == {"calc": 1, "web": 0}
    assert "最终金额：16000.00 元" in result["tool_text"]
    assert "联网搜索初始化失败，本轮无法完成联网核验" in result["tool_text"]
    assert result["web_sources"] == []
    # 不泄漏内部错误串
    assert "（内部）" not in result["tool_text"]


def test_combined_web_tool_exception_keeps_calculator(monkeypatch) -> None:
    calls = {"calc": 0, "web": 0}

    class CalcTool:
        name = "calculate_labor_compensation"

        async def ainvoke(self, args):
            calls["calc"] += 1
            return "最终金额：16000.00 元"

    class FailingWebTool:
        name = "web_search"

        async def ainvoke(self, args):
            calls["web"] += 1
            raise ConnectionError("连接中断")

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [CalcTool(), FailingWebTool()])

    result = graph_module.call_tools({"planned_tool_calls": _calc_web_plan()})

    assert calls == {"calc": 1, "web": 1}
    assert "最终金额：16000.00 元" in result["tool_text"]
    # 联网失败提示存在且脱敏（不泄漏 ConnectionError 细节）
    assert "[工具 web_search 调用失败：ConnectionError]" in result["tool_text"]
    assert "连接中断" not in result["tool_text"]


def test_combined_calc_exception_keeps_web(monkeypatch) -> None:
    calls = {"calc": 0, "web": 0}

    class FailingCalcTool:
        name = "calculate_labor_compensation"

        async def ainvoke(self, args):
            calls["calc"] += 1
            raise RuntimeError("内部计算错误")

    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            calls["web"] += 1
            return [{"type": "text", "text": '{"ok": true, "results": [{"title": "标题", "url": "https://example.com", "site": "站点", "date": "2026-09-18"}]}'}]

    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [FailingCalcTool(), WebTool()])

    result = graph_module.call_tools({"planned_tool_calls": _calc_web_plan()})

    assert calls == {"calc": 1, "web": 1}
    assert "[工具 calculate_labor_compensation 调用失败：RuntimeError]" in result["tool_text"]
    assert "web_search 返回" in result["tool_text"]
    assert "内部计算错误" not in result["tool_text"]
    assert len(result["web_sources"]) == 1


def test_combined_calc_invalid_plan_degrades_to_web_only(monkeypatch) -> None:
    calls = {"calc": 0, "web": 0}
    # 只有联网计划（计算调用因参数无效未进入 planned）
    tools, ensure_calls = _combined_calc_web_tools(calls)
    monkeypatch.setattr(graph_module, "ensure_mcp_tools", ensure_calls)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: tools)

    result = graph_module.call_tools(
        {"planned_tool_calls": [{"name": "web_search", "args": {"query": "问题", "count": 5}}]}
    )

    assert calls == {"calc": 0, "web": 1}
    assert result["web_sources"]

def test_combined_calc_success_with_empty_web_results(monkeypatch) -> None:
    calls = {"calc": 0, "web": 0}
    tools, _ = _combined_calc_web_tools(
        calls, web_payload=[{"type": "text", "text": '{"ok": true, "results": []}'}]
    )
    monkeypatch.setattr(graph_module, "ensure_mcp_tools", lambda: None)
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: tools)

    result = graph_module.call_tools({"planned_tool_calls": _calc_web_plan()})

    # 计算结果保留；联网空结果沿用 P08 资料不足提示；web_sources 为空
    assert calls == {"calc": 1, "web": 1}
    assert "最终金额：16000.00 元" in result["tool_text"]
    assert "联网资料不足，本轮无法完成联网核验" in result["tool_text"]
    assert result["web_sources"] == []
    assert "返回" not in result["tool_text"] or "web_search 返回" not in result["tool_text"]


# ---------------------------------------------------------------------------
# 第三阶段第二步：OPT-06 检索不可用的错误边界 / OPT-07 观测如实反映
# ---------------------------------------------------------------------------


def _fresh_graph_env(monkeypatch, tmp_path: Path, db_name: str = "opt.sqlite3"):
    graph_module.get_graph.cache_clear()
    if db_module.get_checkpointer_connection.cache_info().currsize:
        db_module.get_checkpointer_connection().close()
    db_module.get_checkpointer_connection.cache_clear()
    if db_module.get_connection.cache_info().currsize:
        db_module.get_connection().close()
    db_module.get_connection.cache_clear()
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path / db_name)
    monkeypatch.setattr(
        graph_module, "get_checkpointer_connection", db_module.get_checkpointer_connection
    )


def _cleanup_graph_env():
    graph_module.get_graph.cache_clear()
    if db_module.get_checkpointer_connection.cache_info().currsize:
        db_module.get_checkpointer_connection().close()
    db_module.get_checkpointer_connection.cache_clear()
    if db_module.get_connection.cache_info().currsize:
        db_module.get_connection().close()
    db_module.get_connection.cache_clear()


class CalcRouterStub:
    """按固定工具调用应答的路由桩；每次 invoke 记录一次外部调用。"""

    def __init__(self, log: list, tool_calls: list):
        self.log = log
        self.tool_calls = tool_calls
        self.prompts: list = []

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        self.log.append("router")
        self.prompts.append(prompt)
        return SimpleNamespace(tool_calls=list(self.tool_calls))


class CalcToolStub:
    """本地计算器占位（进入路由规划所需）；clarify 轮不应执行它。"""

    name = graph_module.CALC_TOOL_NAME

    async def ainvoke(self, args):
        raise AssertionError("clarify 轮不应执行计算工具")


def test_opt6_retrieval_unavailable_clarify_asks_and_needed_branch_errors(
    monkeypatch, tmp_path: Path
) -> None:
    """OPT-06：检索不可用时 clarify 分支直接追问；需要检索的分支仍走现有错误处理。"""
    _fresh_graph_env(monkeypatch, tmp_path)
    log: list = []
    prompts: list = []

    def failing_retrieve(query, top_k=3, session_id=None):
        log.append("retrieve")
        raise ExternalServiceError("检索服务调用失败")

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="普通回答")

    monkeypatch.setattr(graph_module, "retrieve", failing_retrieve)
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [CalcToolStub()])
    try:
        # 缺参（已确定 clarify）：检索不可用也直接追问，未触达检索
        monkeypatch.setattr(
            graph_module,
            "get_router_model",
            lambda: CalcRouterStub(log, [
                {"name": graph_module.CALC_TOOL_NAME, "args": {"monthly_wage": 8000}}
            ]),
        )
        answer, citations = graph_module.run_turn(
            "opt6-ask", "帮我按 N 计算经济补偿，月平均工资 8000 元。"
        )
        assert graph_module.CALC_CLARIFY_YEARS in answer
        assert citations == []
        assert log == ["router"]  # 外部调用：路由 1 次、检索 0 次、生成 0 次

        # 需要检索的普通问答：仍走现有错误处理，不产出无依据成功回答
        with pytest.raises(ExternalServiceError):
            graph_module.run_turn("opt6-need", "试用期最长可以约定多久？")
        assert log[-1] == "retrieve"  # 路由提前后：本轮不进计算规划，无 router 调用，检索失败即错
        assert prompts == []  # 失败路径没有生成模型调用

        # 需要检索的完整计算：新顺序为 router→retrieve（失败），工具与生成均未执行
        monkeypatch.setattr(
            graph_module,
            "get_router_model",
            lambda: CalcRouterStub(log, [
                {"name": graph_module.CALC_TOOL_NAME, "args": {"monthly_wage": 8000, "years": 4}}
            ]),
        )
        before = len(log)
        with pytest.raises(ExternalServiceError):
            graph_module.run_turn(
                "opt6-calc", "月工资 8000 元，工作 4 年，帮我按 N 计算经济补偿金。"
            )
        assert log[before:] == ["router", "retrieve"]  # 实际外部调用顺序与次数
    finally:
        _cleanup_graph_env()


def test_opt7_clarify_turn_obs_have_no_retrieval_records(monkeypatch, tmp_path: Path) -> None:
    """OPT-07：clarify 轮观测真实缺少 retrieve 节点与 vector/BM25/rerank 记录。"""
    _fresh_graph_env(monkeypatch, tmp_path)
    obs_path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("LAWRAG_PERF_OBS_FILE", str(obs_path))

    def retrieval_stub(query, top_k=3, session_id=None):
        # 若被调用：像 ask.retrieve 一样写检索观测后失败（确保不是执行后丢弃/假零耗时）
        for step in ("vector", "bm25"):
            with graph_module.perf_span("retrieval", step):
                pass
        with graph_module.perf_span("retrieval", "rerank", extra={"attempts": 1}):
            pass
        raise AssertionError("clarify 轮不应调用检索")

    class FakeModel:
        def invoke(self, prompt):
            return SimpleNamespace(content="普通回答")

    monkeypatch.setattr(graph_module, "retrieve", retrieval_stub)
    monkeypatch.setattr(
        graph_module,
        "get_router_model",
        lambda: CalcRouterStub([], [
            {"name": graph_module.CALC_TOOL_NAME, "args": {"monthly_wage": 8000}}
        ]),
    )
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: FakeModel())
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [CalcToolStub()])
    try:
        graph_module.run_turn("opt7", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
        records = [
            json.loads(line)
            for line in obs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        nodes = [r["name"] for r in records if r.get("kind") == "node"]
        assert nodes == ["prepare", "route", "clarify"]  # 真实缺少 retrieve 节点
        assert [r["name"] for r in records if r.get("kind") == "retrieval"] == []  # 无检索观测
        assert [r["name"] for r in records if r.get("kind") == "llm"] == ["router"]
    finally:
        _cleanup_graph_env()
