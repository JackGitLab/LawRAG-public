from types import SimpleNamespace

import pytest

import app.graph as graph_module
from app.ask import ExternalServiceError


class RouterModel:
    def __init__(self, calls):
        self.calls = calls
        self.invocations = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, prompt):
        self.invocations.append(prompt)
        return SimpleNamespace(tool_calls=self.calls)


@pytest.mark.parametrize(
    ("question", "expected_name"),
    [
        ("劳动合同法最近修订了吗？", "verify_statute_status"),
        ("今年发布了哪些劳动法规？", "web_search"),
    ],
)
def test_rule_routes_do_not_call_model(monkeypatch, question, expected_name) -> None:
    monkeypatch.setattr(
        graph_module,
        "get_router_model",
        lambda: pytest.fail("规则短路不应调用路由模型"),
    )
    result = graph_module.route({"search_query": question, "use_web_search": True})
    assert result["planned_tool_calls"][0]["name"] == expected_name
    assert result["web_sources"] == []
    assert result["tool_text"] == ""


@pytest.mark.parametrize(
    ("question", "calls"),
    [
        (
            "月薪八千工作三年半，违法解除赔偿多少？",
            [{"name": "calculate_labor_compensation", "args": {"monthly_wage": 8000, "years": 3.5}}],
        ),
    ],
)
def test_model_routes_calculation(monkeypatch, question, calls) -> None:
    model = RouterModel(calls)
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)
    external = SimpleNamespace(name="web_search")
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: [*graph_module.LOCAL_TOOLS, external])

    result = graph_module.route({"search_query": question})

    assert result["planned_tool_calls"] == calls
    assert len(model.invocations) == 1
    assert {tool.name for tool in model.bound_tools} == graph_module.LOCAL_TOOL_NAMES
    assert result["web_sources"] == []
    assert result["tool_text"] == ""


def test_router_model_failure_preserves_original_exception(monkeypatch) -> None:
    class FailingModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, prompt):
            raise OSError("路由上游中断")

    monkeypatch.setattr(graph_module, "get_router_model", lambda: FailingModel())
    monkeypatch.setattr(graph_module, "get_all_tools", lambda: graph_module.LOCAL_TOOLS)

    with pytest.raises(ExternalServiceError, match="工具路由服务调用失败") as caught:
        graph_module._model_tool_calls("赔偿多少？", graph_module.CALC_HINT)

    assert isinstance(caught.value.__cause__, OSError)


def test_unmatched_question_skips_router_model(monkeypatch) -> None:
    model = RouterModel([])
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)

    result = graph_module.route({"search_query": "试用期最长可以约定多久？"})

    assert result["planned_tool_calls"] == []
    assert len(model.invocations) == 0
    assert result["web_sources"] == []
    assert result["tool_text"] == ""


def test_external_search_requires_user_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_module,
        "get_router_model",
        lambda: pytest.fail("关闭联网搜索时不应调用路由模型"),
    )

    result = graph_module.route(
        {"search_query": "劳动合同法最近修订了吗？", "use_web_search": False}
    )

    assert result["planned_tool_calls"] == []


def test_user_opt_in_routes_ordinary_question_to_web_search(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_module,
        "get_router_model",
        lambda: pytest.fail("开启联网搜索后应由规则直接路由"),
    )

    result = graph_module.route(
        {"search_query": "试用期最长可以约定多久？", "use_web_search": True}
    )

    assert result["planned_tool_calls"] == [
        {"name": "web_search", "args": {"query": "试用期最长可以约定多久？", "count": 5}}
    ]


def test_model_tool_calls_are_filtered_by_local_whitelist(monkeypatch) -> None:
    model = RouterModel([{"name": "web_search", "args": {"query": "问题"}}])
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)

    planned = graph_module._model_tool_calls("赔偿多少？", graph_module.CALC_HINT)

    assert planned == []
    assert {tool.name for tool in model.bound_tools} == graph_module.LOCAL_TOOL_NAMES


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"monthly_wage": 0, "years": 0}, []),
        (
            {"monthly_wage": 8000, "years": 3.5, "kind": "2N"},
            [
                {
                    "name": "calculate_labor_compensation",
                    "args": {"monthly_wage": 8000, "years": 3.5, "kind": "2N"},
                }
            ],
        ),
    ],
)
def test_calculation_calls_require_usable_positive_args(monkeypatch, args, expected) -> None:
    model = RouterModel([{"name": "calculate_labor_compensation", "args": args}])
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)

    result = graph_module.route({"search_query": "月薪八千工作三年半，赔偿多少钱？"})

    # 参数不可用的计算调用不会进入可执行规划（改为进入缺参追问，见 test_calculation_clarification.py）
    assert result["planned_tool_calls"] == expected


def test_unmatched_route_is_independent_of_mcp_load_state(monkeypatch) -> None:
    import app.mcp_tools as mcp_tools_module

    model = RouterModel([])
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)
    question = {"search_query": "试用期最长可以约定多久？"}
    try:
        mcp_tools_module.set_mcp_tools([])
        before = graph_module.route(question)
        mcp_tools_module.set_mcp_tools([SimpleNamespace(name="verify_statute_status")])
        after = graph_module.route(question)
    finally:
        mcp_tools_module.set_mcp_tools([])

    assert before["planned_tool_calls"] == []
    assert after["planned_tool_calls"] == []
    assert len(model.invocations) == 0


def _calc_calls():
    return [
        {
            "name": "calculate_labor_compensation",
            "args": {"monthly_wage": 8000, "years": 3.5},
        }
    ]


def test_web_search_and_calculation_planned_together(monkeypatch) -> None:
    model = RouterModel(_calc_calls())
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)

    result = graph_module.route(
        {"search_query": "月薪八千工作三年半，赔偿多少钱？", "use_web_search": True}
    )

    assert result["planned_tool_calls"] == [
        {
            "name": "calculate_labor_compensation",
            "args": {"monthly_wage": 8000, "years": 3.5},
        },
        {"name": "web_search", "args": {"query": "月薪八千工作三年半，赔偿多少钱？", "count": 5}},
    ]
    assert result["web_sources"] == []
    assert result["tool_text"] == ""


def test_statute_verification_and_calculation_planned_together(monkeypatch) -> None:
    model = RouterModel(_calc_calls())
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)

    result = graph_module.route(
        {"search_query": "月薪八千工作三年半，赔偿多少钱？劳动合同法最近修订了吗？", "use_web_search": True}
    )

    names = [c["name"] for c in result["planned_tool_calls"]]
    assert names.count("calculate_labor_compensation") == 1
    assert names.count("verify_statute_status") == 1
    assert "web_search" not in names
    assert result["planned_tool_calls"][0]["name"] == "calculate_labor_compensation"
    assert result["planned_tool_calls"][1]["name"] == "verify_statute_status"


def test_web_search_off_never_plans_mcp_tool(monkeypatch) -> None:
    model = RouterModel(_calc_calls())
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)

    result = graph_module.route(
        {"search_query": "月薪八千工作三年半，赔偿多少钱？", "use_web_search": False}
    )

    assert result["planned_tool_calls"] == [
        {
            "name": "calculate_labor_compensation",
            "args": {"monthly_wage": 8000, "years": 3.5},
        }
    ]
    assert not any(c["name"] in graph_module.MCP_TOOL_NAMES for c in result["planned_tool_calls"])


def test_web_only_routes_without_router_model(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_module,
        "get_router_model",
        lambda: pytest.fail("仅联网的问题不应调用路由模型"),
    )

    result = graph_module.route(
        {"search_query": "劳动合同法最近修订了吗？", "use_web_search": True}
    )

    assert result["planned_tool_calls"] == [
        {"name": "verify_statute_status", "args": {"query": "劳动合同法最近修订了吗？"}}
    ]


def test_invalid_calc_parameters_leave_only_web_tool(monkeypatch) -> None:
    model = RouterModel([{"name": "calculate_labor_compensation", "args": {}}])
    monkeypatch.setattr(graph_module, "get_router_model", lambda: model)

    result = graph_module.route(
        {"search_query": "赔偿多少钱？", "use_web_search": True}
    )

    # 计算参数无效：不保留无效调用，仅保留确定性联网调用
    assert result["planned_tool_calls"] == [
        {"name": "web_search", "args": {"query": "赔偿多少钱？", "count": 5}}
    ]
