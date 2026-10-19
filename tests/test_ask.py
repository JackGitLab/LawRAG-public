import json
import ssl
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

import app.ask as ask_module
from app.ask import (
    ExternalServiceError,
    bm25_retrieve,
    build_prompt,
    build_prompt_with_tools,
    generate_short_retrieval_text,
    get_bm25_index,
    parse_query_rewrites,
    reciprocal_rank_fusion,
    rerank_candidates,
    retrieve_with_strategy,
    tokenize_for_bm25,
)


def test_prompt_contains_grounding_and_citation_rules() -> None:
    node = SimpleNamespace(
        node=SimpleNamespace(
            text="第四十七条 经济补偿按工作年限计算。",
            metadata={
                "source": "中华人民共和国劳动合同法",
                "article": "第四十七条",
            },
        )
    )

    prompt = build_prompt("如何计算？", [node])

    assert "只能根据“检索资料”回答" in prompt
    assert "现有资料无法回答" in prompt
    assert "《中华人民共和国劳动合同法》" in prompt
    assert "每个法律结论句末必须标注对应条号" in prompt
    assert "（第四十六条）" in prompt
    assert "不要在回答正文中重复文档名" in prompt
    assert "不要自行生成“引用来源”清单" in prompt
    assert "文档名不含扩展名" not in prompt
    assert prompt.startswith("你是法律顾问问答助手")
    assert prompt.count("\n\n检索资料：\n") == 1
    assert prompt.endswith("用户问题：如何计算？\n")


def test_rerank_candidates_uses_api_order_and_scores(monkeypatch) -> None:
    candidates = [
        NodeWithScore(node=TextNode(text=f"候选法条 {index}"), score=0.5)
        for index in range(3)
    ]
    captured = {}
    calls = {"n": 0}
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "results": [
                        {"index": 2, "relevance_score": 0.95},
                        {"index": 0, "relevance_score": 0.75},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ask_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(ask_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ask_module, "require_env", lambda name: "test-key")

    reranked = rerank_candidates("测试问题", candidates, top_k=2)

    assert [item.node.text for item in reranked] == ["候选法条 2", "候选法条 0"]
    assert [item.score for item in reranked] == [0.95, 0.75]
    assert calls["n"] == 1  # 成功路径只请求一次
    assert sleeps == []  # 成功路径不等待
    assert captured == {
        "url": "https://api.siliconflow.cn/v1/rerank",
        "payload": {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "测试问题",
            "documents": ["候选法条 0", "候选法条 1", "候选法条 2"],
            "top_n": 2,
            "return_documents": False,
        },
        "timeout": 30,
    }


def test_rerank_candidates_immediately_wraps_non_retryable_error(monkeypatch) -> None:
    candidates = [NodeWithScore(node=TextNode(text="候选法条"), score=0.5)]
    calls = {"n": 0}
    sleeps = []
    oserror = OSError("连接中断")

    def fail_urlopen(request, timeout):
        calls["n"] += 1
        raise oserror

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ask_module, "urlopen", fail_urlopen)
    monkeypatch.setattr(ask_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ask_module, "require_env", lambda name: "test-key")

    with pytest.raises(ExternalServiceError, match="重排服务调用失败") as caught:
        rerank_candidates("测试问题", candidates, top_k=1)

    assert calls["n"] == 1  # 非可重试错误立即包装，不重试
    assert sleeps == []
    assert caught.value.__cause__ is oserror


def test_short_generation_preserves_original_exception(monkeypatch) -> None:
    class FailingModel:
        def bind(self, **kwargs):
            return self

        def invoke(self, prompt):
            raise OSError("上游连接中断")

    monkeypatch.setattr(ask_module, "get_chat_model", lambda: FailingModel())

    with pytest.raises(ExternalServiceError, match="生成服务调用失败") as caught:
        generate_short_retrieval_text("检索问题", max_tokens=160)

    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize(
    ("strategy", "generated"),
    [("hyde", ""), ("multi_query", "只有一个查询")],
)
def test_invalid_hyde_or_multi_query_output_uses_external_error(
    monkeypatch, strategy, generated
) -> None:
    monkeypatch.setattr(ask_module, "load_index", lambda: object())
    monkeypatch.setattr(
        ask_module,
        "generate_short_retrieval_text",
        lambda prompt, max_tokens, session_id=None, tags=None: generated,
    )

    with pytest.raises(ExternalServiceError):
        retrieve_with_strategy("测试问题", strategy, top_k=3)


def test_retrieve_uses_hybrid_rerank_as_default(monkeypatch) -> None:
    captured = {}

    def fake_retrieve(question, strategy, top_k, session_id=None):
        captured.update(question=question, strategy=strategy, top_k=top_k)
        return ["result"]

    monkeypatch.setattr(ask_module, "retrieve_with_strategy", fake_retrieve)

    assert ask_module.retrieve("测试问题", top_k=2) == ["result"]
    assert captured == {
        "question": "测试问题",
        "strategy": "hybrid_rerank",
        "top_k": 2,
    }


def test_bm25_retrieval_is_offline_and_ranks_keyword_match(monkeypatch) -> None:
    nodes = [
        TextNode(id_="labor", text="第十九条 试用期不得超过六个月"),
        TextNode(id_="consumer", text="第二十五条 七日无理由退货"),
    ]
    bm25 = ask_module.BM25Okapi([tokenize_for_bm25(node.text) for node in nodes])
    monkeypatch.setattr(ask_module, "get_bm25_index", lambda: (bm25, nodes))

    results = bm25_retrieve("试用期最长多久", top_k=1)

    assert [item.node.node_id for item in results] == ["labor"]


def test_bm25_index_loads_chroma_nodes_once(tmp_path, monkeypatch) -> None:
    calls = 0

    class FakeCollection:
        def get(self, include):
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["article-1", "article-2"],
                "documents": ["第一条 劳动合同", "第二条 消费者权益"],
                "metadatas": [
                    {"source": "劳动合同法", "article": "第一条"},
                    {"source": "消费者权益保护法", "article": "第二条"},
                ],
            }

    class FakeClient:
        def get_collection(self, name):
            assert name == "lawrag"
            return FakeCollection()

    def fake_client(path):
        nonlocal calls
        calls += 1
        assert path == str(tmp_path)
        return FakeClient()

    monkeypatch.setattr(ask_module, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(ask_module.chromadb, "PersistentClient", fake_client)
    get_bm25_index.cache_clear()
    try:
        first = get_bm25_index()
        second = get_bm25_index()
    finally:
        get_bm25_index.cache_clear()

    assert first is second
    assert calls == 1
    assert [node.node_id for node in first[1]] == ["article-1", "article-2"]


def test_bm25_strategy_does_not_load_vector_index(monkeypatch) -> None:
    monkeypatch.setattr(ask_module, "bm25_retrieve", lambda query, top_k: ["bm25"])

    def fail_load_index():
        raise AssertionError("纯 BM25 不应加载向量索引")

    monkeypatch.setattr(ask_module, "load_index", fail_load_index)

    assert retrieve_with_strategy("试用期", "bm25", top_k=3) == ["bm25"]


def test_hybrid_fuses_vector_and_bm25_results(monkeypatch) -> None:
    first = TextNode(id_="first", text="第一条")
    second = TextNode(id_="second", text="第二条")
    third = TextNode(id_="third", text="第三条")
    vector_results = [
        NodeWithScore(node=first, score=0.9),
        NodeWithScore(node=second, score=0.8),
    ]
    bm25_results = [
        NodeWithScore(node=second, score=2.0),
        NodeWithScore(node=third, score=1.0),
    ]
    monkeypatch.setattr(ask_module, "load_index", lambda: object())
    monkeypatch.setattr(
        ask_module, "vector_retrieve", lambda index, query, top_k: vector_results
    )
    monkeypatch.setattr(
        ask_module, "bm25_retrieve", lambda query, top_k: bm25_results
    )

    results = retrieve_with_strategy("测试", "hybrid", top_k=2)

    assert [item.node.node_id for item in results] == ["second", "first"]


def test_hybrid_rerank_reranks_fused_candidates(monkeypatch) -> None:
    first = TextNode(id_="first", text="第一条")
    second = TextNode(id_="second", text="第二条")
    vector_results = [NodeWithScore(node=first, score=0.9)]
    bm25_results = [NodeWithScore(node=second, score=2.0)]
    captured = {}

    monkeypatch.setattr(ask_module, "load_index", lambda: object())
    monkeypatch.setattr(
        ask_module, "vector_retrieve", lambda index, query, top_k: vector_results
    )
    monkeypatch.setattr(
        ask_module, "bm25_retrieve", lambda query, top_k: bm25_results
    )

    def fake_rerank(question, candidates, top_k):
        captured.update(question=question, candidates=candidates, top_k=top_k)
        return ["reranked"]

    monkeypatch.setattr(ask_module, "rerank_candidates", fake_rerank)

    assert retrieve_with_strategy("测试", "hybrid_rerank", top_k=1) == ["reranked"]
    assert captured["question"] == "测试"
    assert captured["top_k"] == 1
    assert {item.node.node_id for item in captured["candidates"]} == {
        "first",
        "second",
    }


def test_parse_query_rewrites_removes_numbering_and_limits_count() -> None:
    text = "1. 格式条款效力\n2、经营者限制消费者权利\n- 售出概不退换是否无效\n额外查询"

    assert parse_query_rewrites(text) == [
        "格式条款效力",
        "经营者限制消费者权利",
        "售出概不退换是否无效",
    ]


def test_reciprocal_rank_fusion_combines_duplicate_nodes() -> None:
    first = TextNode(text="第一条")
    second = TextNode(text="第二条")
    third = TextNode(text="第三条")
    result_sets = [
        [NodeWithScore(node=first, score=0.9), NodeWithScore(node=second, score=0.8)],
        [NodeWithScore(node=second, score=0.9), NodeWithScore(node=third, score=0.8)],
    ]

    fused = reciprocal_rank_fusion(result_sets, limit=3)

    assert [item.node.text for item in fused] == ["第二条", "第一条", "第三条"]


def test_tool_prompt_declares_tool_materials_usable() -> None:
    node = SimpleNamespace(
        node=SimpleNamespace(
            text="第四十七条 经济补偿按工作年限计算。",
            metadata={
                "source": "中华人民共和国劳动合同法",
                "article": "第四十七条",
            },
        )
    )
    question = "第四十七条最近有没有修订？"
    tool_text = "[工具 verify_statute_status 返回]\n某网页标题"

    prompt = build_prompt_with_tools(question, [node], tool_text)

    assert tool_text in prompt
    assert "[联网核验资料]" in prompt
    # 存在最低可用证据：标题/站点/日期/摘要是有限可用证据，只能陈述直接支持的事实
    assert "属于有限可用证据" in prompt
    assert "只能据此陈述这些资料直接支持的事实" in prompt
    # 不再无条件要求明确结论、不再无条件禁止现有资料无法回答
    assert "必须据此给出明确结论" not in prompt
    assert "不得回复“现有资料无法回答”" not in prompt
    assert "应明确说明该部分无法核实，不得编造" in prompt
    # 旧的“仅当失败或额度不足才允许无法核实”限制句已删除
    assert "仅当联网核验资料明确表示检索失败或额度不足时" not in prompt
    # 统一规则覆盖失败、额度不足、未返回可用结果、证据不足以支持结论
    assert "当联网工具失败、额度不足、未返回可用结果" in prompt
    assert "或现有联网证据不足以支持具体结论时" in prompt
    assert prompt.count("\n\n检索资料：\n") == 1
    assert prompt.endswith(f"用户问题：{question}\n")


def test_tool_prompt_allows_failure_and_insufficient_material() -> None:
    node = SimpleNamespace(
        node=SimpleNamespace(
            text="第四十七条 经济补偿按工作年限计算。",
            metadata={"source": "中华人民共和国劳动合同法", "article": "第四十七条"},
        )
    )
    question = "第四十七条最近有没有修订？"
    tool_text = "[工具 web_search 联网资料不足，本轮无法完成联网核验。]"

    prompt = build_prompt_with_tools(question, [node], tool_text)

    assert tool_text in prompt  # 稳定资料不足标记原样进入 prompt
    # 允许说明无法核实；不编造；可回答本地资料支持部分；禁止冒充完成联网核验
    assert "应明确说明该部分无法核实，不得编造" in prompt
    assert "不得编造来源、日期" in prompt
    assert "继续回答本地法条资料能够独立支持的部分" in prompt
    assert "不得把本地回答冒充成已完成联网核验" in prompt


def test_tool_prompt_falls_back_to_build_prompt_without_tools() -> None:
    node = SimpleNamespace(
        node=SimpleNamespace(
            text="第四十七条 经济补偿按工作年限计算。",
            metadata={
                "source": "中华人民共和国劳动合同法",
                "article": "第四十七条",
            },
        )
    )
    question = "如何计算经济补偿？"

    assert (
        build_prompt_with_tools(question, [node])
        == build_prompt_with_tools(question, [node], "")
        == build_prompt(question, [node])
    )


def test_rerank_retries_once_after_transient_tls_failure(monkeypatch) -> None:
    candidates = [
        NodeWithScore(node=TextNode(text=f"候选法条 {index}"), score=0.5)
        for index in range(3)
    ]
    calls = {"n": 0}
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "results": [
                        {"index": 2, "relevance_score": 0.95},
                        {"index": 0, "relevance_score": 0.75},
                    ]
                }
            ).encode("utf-8")

    def flaky_urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            # 贴近历史 E-008：URLError 包装 SSLEOFError
            raise URLError(ssl.SSLEOFError("EOF occurred in violation of protocol"))
        return FakeResponse()

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ask_module, "urlopen", flaky_urlopen)
    monkeypatch.setattr(ask_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ask_module, "require_env", lambda name: "test-key")

    reranked = rerank_candidates("测试问题", candidates, top_k=2)

    assert calls["n"] == 2  # 第一次失败后重试一次
    assert sleeps == [ask_module.RERANK_RETRY_DELAY_SECONDS]
    assert [item.node.text for item in reranked] == ["候选法条 2", "候选法条 0"]
    assert [item.score for item in reranked] == [0.95, 0.75]
    assert all(isinstance(item, NodeWithScore) for item in reranked)


def test_rerank_stops_after_second_transient_failure(monkeypatch) -> None:
    candidates = [NodeWithScore(node=TextNode(text="候选法条"), score=0.5)]
    calls = {"n": 0}
    sleeps = []
    first_error = URLError("temporary")
    second_error = URLError("still down")

    def always_fail(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise first_error
        raise second_error

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ask_module, "urlopen", always_fail)
    monkeypatch.setattr(ask_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ask_module, "require_env", lambda name: "test-key")

    with pytest.raises(ExternalServiceError, match="重排服务调用失败") as caught:
        rerank_candidates("测试问题", candidates, top_k=1)

    assert calls["n"] == 2  # 恰好尝试两次，不发生第三次调用
    assert sleeps == [ask_module.RERANK_RETRY_DELAY_SECONDS]
    assert caught.value.__cause__ is second_error  # cause 是最后一次实际异常


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_rerank_retries_retryable_http_status(monkeypatch, status) -> None:
    candidates = [NodeWithScore(node=TextNode(text="候选法条"), score=0.5)]
    calls = {"n": 0}
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps({"results": [{"index": 0, "relevance_score": 0.9}]}).encode(
                "utf-8"
            )

    def flaky(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(
                "https://api.siliconflow.cn/v1/rerank", status, "err", {}, BytesIO(b"")
            )
        return FakeResponse()

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ask_module, "urlopen", flaky)
    monkeypatch.setattr(ask_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ask_module, "require_env", lambda name: "test-key")

    reranked = rerank_candidates("测试问题", candidates, top_k=1)

    assert calls["n"] == 2
    assert sleeps == [ask_module.RERANK_RETRY_DELAY_SECONDS]
    assert [item.score for item in reranked] == [0.9]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_rerank_does_not_retry_permanent_http_status(monkeypatch, status) -> None:
    candidates = [NodeWithScore(node=TextNode(text="候选法条"), score=0.5)]
    calls = {"n": 0}
    sleeps = []
    http_error = HTTPError(
        "https://api.siliconflow.cn/v1/rerank", status, "err", {}, BytesIO(b"")
    )

    def fail(request, timeout):
        calls["n"] += 1
        raise http_error

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ask_module, "urlopen", fail)
    monkeypatch.setattr(ask_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ask_module, "require_env", lambda name: "test-key")

    with pytest.raises(ExternalServiceError, match="重排服务调用失败") as caught:
        rerank_candidates("测试问题", candidates, top_k=1)

    assert calls["n"] == 1  # 非暂时性 HTTP 状态不重试
    assert sleeps == []
    assert caught.value.__cause__ is http_error


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        (b"not json", json.JSONDecodeError),
        (json.dumps({"foo": "bar"}).encode("utf-8"), KeyError),
        (
            json.dumps({"results": [{"index": 5, "relevance_score": 0.9}]}).encode("utf-8"),
            IndexError,
        ),
        (
            json.dumps({"results": [{"index": 0, "relevance_score": "abc"}]}).encode("utf-8"),
            ValueError,
        ),
    ],
)
def test_rerank_does_not_retry_invalid_response(monkeypatch, payload, error_type) -> None:
    candidates = [NodeWithScore(node=TextNode(text="候选法条"), score=0.5)]
    calls = {"n": 0}
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return payload

    def respond(request, timeout):
        calls["n"] += 1
        return FakeResponse()

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ask_module, "urlopen", respond)
    monkeypatch.setattr(ask_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ask_module, "require_env", lambda name: "test-key")

    with pytest.raises(ExternalServiceError, match="重排服务调用失败") as caught:
        rerank_candidates("测试问题", candidates, top_k=1)

    assert calls["n"] == 1  # 响应解析/本地数据错误不重试
    assert sleeps == []
    assert isinstance(caught.value.__cause__, error_type)
