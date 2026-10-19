"""第二阶段：计算工具缺参追问与多轮参数补齐（C2-01～C2-12）。

覆盖两层能力中的“确定性流程”：使用实际 LangGraph、隔离 SQLite、模型与检索桩，
核对状态、工具实际调用次数及参数、历史消息与 SSE 事件契约。
真实模型抽取能力由 logs/stage2 下的冒烟验证单独记录，不能用本文件的桩测试代替。
"""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

import app.db as db_module
import app.graph as graph_module
import app.tools as tools_module

CALC_NAME = graph_module.CALC_TOOL_NAME


def calc_call(**args) -> dict:
    return {"name": CALC_NAME, "args": args}


def node(index: int) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=f"法条 {index}",
            metadata={"source": "劳动合同法", "article": f"第{index}条"},
        ),
        score=0.9 - index / 10,
    )


def independent_amount(monthly_wage: float, years: float, kind: str = "N") -> float:
    """按现有折算规则独立复算金额，不调用被测工具本身。"""
    base = tools_module.fold_years(years) * monthly_wage
    if kind == "2N":
        return 2 * base
    if kind == "N+1":
        return base + monthly_wage
    return base


class ScriptedRouter:
    """按调用顺序返回预设工具调用的路由模型替身（每次调用消费一个脚本回合）。"""

    def __init__(self):
        self.script: list[list[dict]] = []
        self.prompts: list[str] = []
        self.bound_tools: list = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(tool_calls=self.script.pop(0) if self.script else [])


class ScriptedChatModel:
    """按提示内容区分追问改写与最终回答的生成模型替身。"""

    def __init__(self, harness: "Harness"):
        self.harness = harness

    def bind(self, **kwargs):
        return self

    def invoke(self, prompt):
        if "改写成可独立检索" in prompt:
            self.harness.rewrite_prompts.append(prompt)
            return SimpleNamespace(content=self.harness.rewrite_text)
        self.harness.generate_prompts.append(prompt)
        return SimpleNamespace(content=self.harness.answer_text)


class CalcSpy:
    """记录实际调用参数的本地计算器替身，返回真实工具的确定性结果。"""

    name = CALC_NAME

    def __init__(self, calls: list[dict], fail: bool = False):
        self.calls = calls
        self.fail = fail

    async def ainvoke(self, args):
        self.calls.append(dict(args))
        if self.fail:
            raise RuntimeError("controlled tool failure")
        return tools_module.calculate_labor_compensation.invoke(args)


class Harness:
    def __init__(self, monkeypatch, tmp_path: Path):
        self.monkeypatch = monkeypatch
        self.calc_calls: list[dict] = []
        self.retrieve_queries: list[str] = []
        self.rewrite_prompts: list[str] = []
        self.generate_prompts: list[str] = []
        self.mcp_init = {"n": 0}
        self.router = ScriptedRouter()
        self.answer_text = "生成回答：按现有规则计算。"
        # 改写文本故意带上“9000”，用于验证计算参数只从用户原话抽取
        self.rewrite_text = "经济补偿 月工资 9000 元 工作年限 3 年 计算"
        self.chat_model_factory = lambda: ScriptedChatModel(self)
        self.fail_calc = False
        self.extra_tools: list = []

        graph_module.get_graph.cache_clear()
        if db_module.get_checkpointer_connection.cache_info().currsize:
            db_module.get_checkpointer_connection().close()
        db_module.get_checkpointer_connection.cache_clear()
        if db_module.get_connection.cache_info().currsize:
            db_module.get_connection().close()
        db_module.get_connection.cache_clear()
        monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path / "clarify.sqlite3")
        monkeypatch.setattr(
            graph_module, "get_checkpointer_connection", db_module.get_checkpointer_connection
        )
        monkeypatch.setattr(graph_module, "retrieve", self._retrieve)
        monkeypatch.setattr(graph_module, "get_chat_model", lambda: self.chat_model_factory())
        monkeypatch.setattr(graph_module, "get_router_model", lambda: self.router)
        monkeypatch.setattr(graph_module, "get_all_tools", self._all_tools)
        monkeypatch.setattr(graph_module, "ensure_mcp_tools", self._ensure_mcp_tools)

    def _all_tools(self):
        return [CalcSpy(self.calc_calls, self.fail_calc), *self.extra_tools]

    def _retrieve(self, query, top_k=3, session_id=None):
        self.retrieve_queries.append(query)
        return [node(1), node(2)]

    def _ensure_mcp_tools(self):
        self.mcp_init["n"] += 1
        return None

    def script(self, *turns):
        self.router.script.extend(turns)

    def reopen(self):
        """关闭连接并清空图缓存（模拟进程/连接重建），随后仍指向同一隔离数据库。"""
        graph_module.get_graph.cache_clear()
        for getter in (db_module.get_checkpointer_connection, db_module.get_connection):
            if getter.cache_info().currsize:
                getter().close()
            getter.cache_clear()
        self.monkeypatch.setattr(
            graph_module, "get_checkpointer_connection", db_module.get_checkpointer_connection
        )

    def run(self, session_id: str, question: str, use_web_search: bool = False):
        return graph_module.run_turn(session_id, question, use_web_search)

    def stream(self, session_id: str, question: str, use_web_search: bool = False):
        return list(graph_module.stream_turn(session_id, question, use_web_search))

    def state(self, session_id: str) -> dict:
        return graph_module.get_graph().get_state(
            config=graph_module.graph_config(session_id)
        ).values

    def messages(self, session_id: str) -> list:
        return self.state(session_id).get("messages", [])

    def teardown(self):
        graph_module.get_graph.cache_clear()
        if db_module.get_checkpointer_connection.cache_info().currsize:
            db_module.get_checkpointer_connection().close()
        db_module.get_checkpointer_connection.cache_clear()
        if db_module.get_connection.cache_info().currsize:
            db_module.get_connection().close()
        db_module.get_connection.cache_clear()


@pytest.fixture
def harness(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    yield harness
    harness.teardown()


def clarify_text(context: dict) -> str:
    return graph_module._clarify_text(context)


# ---------------------------------------------------------------------------
# C2-01 首轮只给工资；次轮补年限
# ---------------------------------------------------------------------------


def test_c2_01_first_turn_asks_years_then_calculates(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(years=3.5, kind="N")],
    )

    answer, citations = harness.run("s1", "帮我按 N 计算经济补偿，月平均工资 8000 元。")

    assert harness.calc_calls == []  # 首轮不调用计算器
    state = harness.state("s1")
    assert state["planned_tool_calls"] == []
    assert state["calc_context"]["status"] == "awaiting"
    assert state["calc_context"]["monthly_wage"] == 8000
    assert state["calc_context"]["years"] is None
    assert graph_module.CALC_CLARIFY_YEARS in answer
    assert "8000" not in answer  # 不输出金额，也不复述猜测数字
    assert "还需" in answer and "工作年限" in answer
    assert citations == []
    assert [message.type for message in state["messages"]] == ["human", "ai"]
    assert state["messages"][-1].additional_kwargs["citations"] == []
    # 抽取只从用户原话出发，追问改写文本里的 9000 不进入参数
    assert harness.router.prompts[0].endswith("问题：帮我按 N 计算经济补偿，月平均工资 8000 元。")

    answer, citations = harness.run("s1", "三年半。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    state = harness.state("s1")
    assert state["calc_context"]["status"] == "done"
    assert state["planned_tool_calls"] == [
        {"name": CALC_NAME, "args": {"monthly_wage": 8000, "years": 3.5, "kind": "N"}}
    ]
    amount = independent_amount(8000, 3.5, "N")
    assert f"{amount:.2f} 元" in state["tool_text"]
    assert answer == harness.answer_text
    assert len(citations) == 2
    assert [message.type for message in state["messages"]] == ["human", "ai", "human", "ai"]
    assert harness.mcp_init["n"] == 0  # 未开启联网


# ---------------------------------------------------------------------------
# C2-02 首轮只给年限；只追问工资
# ---------------------------------------------------------------------------


def test_c2_02_first_turn_only_asks_missing_wage(harness) -> None:
    harness.script([calc_call(years=4, kind="N")], [calc_call(monthly_wage=9000, kind="N")])

    answer, _ = harness.run("s2", "我在公司干了 4 年，帮我按 N 计算经济补偿。")

    assert harness.calc_calls == []
    assert answer.startswith(graph_module.CALC_CLARIFY_WAGE)
    assert graph_module.CALC_CLARIFY_YEARS not in answer  # 已知的年限不重复追问
    assert harness.state("s2")["calc_context"]["years"] == 4

    harness.run("s2", "月工资 9000 元。")

    assert harness.calc_calls == [{"monthly_wage": 9000, "years": 4, "kind": "N"}]
    assert harness.state("s2")["calc_context"]["status"] == "done"


# ---------------------------------------------------------------------------
# C2-03 两项都缺；分两轮补齐后只调用一次
# ---------------------------------------------------------------------------


def test_c2_03_both_missing_then_filled_in_two_turns(harness) -> None:
    harness.script(
        [calc_call(kind="N")],
        [calc_call(monthly_wage=8000)],
        [calc_call(years=3.5)],
    )

    answer, _ = harness.run("s3", "帮我按 N 计算经济补偿。")
    assert answer.startswith(graph_module.CALC_CLARIFY_BOTH)
    assert harness.calc_calls == []

    answer, _ = harness.run("s3", "月平均工资 8000 元。")
    assert graph_module.CALC_CLARIFY_YEARS in answer
    assert graph_module.CALC_CLARIFY_WAGE not in answer
    assert harness.calc_calls == []

    harness.run("s3", "三年半。")
    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    assert harness.state("s3")["calc_context"]["status"] == "done"


# ---------------------------------------------------------------------------
# C2-04 等待年限时同时更正工资并补齐年限
# ---------------------------------------------------------------------------


def test_c2_04_correction_and_completion_in_one_reply(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=9000, years=3.5, kind="N")],
    )

    harness.run("s4", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    assert harness.calc_calls == []

    harness.run("s4", "工资改成 9000，工龄三年半。")

    assert harness.calc_calls == [{"monthly_wage": 9000, "years": 3.5, "kind": "N"}]
    context = harness.state("s4")["calc_context"]
    assert context["monthly_wage"] == 9000
    assert context["status"] == "done"


# ---------------------------------------------------------------------------
# C2-05 计算完成后只更正参数：不重复调用
# ---------------------------------------------------------------------------


def test_c2_05_correction_after_completion_does_not_recalculate(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=20000, years=3, kind="2N")],
        [calc_call(monthly_wage=9000)],
    )

    harness.run("s5", "我工作 3 年，月工资 20000 元，帮我按 2N 计算赔偿金是多少？")
    assert len(harness.calc_calls) == 1

    answer, citations = harness.run("s5", "工资说错了，是 9000。")

    assert len(harness.calc_calls) == 1  # 该轮不重复调用
    context = harness.state("s5")["calc_context"]
    assert context["monthly_wage"] == 9000
    assert context["years"] == 3
    assert context["kind"] == "2N"
    assert context["status"] == "done"
    assert context["reply"] == "correction_ack"
    assert "9000" in answer and "3" in answer  # 针对当前计算更正
    assert graph_module.CALC_CORRECTION_ACK.split("{")[0] in answer
    assert citations == []


# ---------------------------------------------------------------------------
# C2-06 零、负数或无法解释的补参输入
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "model_args", "has_validity_note"),
    [
        ("0", {"years": 0}, True),
        ("-3", {"years": -3}, True),
        ("工龄负三年", {"years": -3}, True),
    ],
)
def test_c2_06_invalid_reply_keeps_asking_without_calculating(
    harness, reply, model_args, has_validity_note
) -> None:
    harness.script([calc_call(monthly_wage=8000, kind="N")], [calc_call(**model_args)])

    harness.run("s6", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    assert harness.calc_calls == []

    answer, citations = harness.run("s6", reply)

    assert harness.calc_calls == []
    context = harness.state("s6")["calc_context"]
    assert context["status"] == "awaiting"
    assert context["years"] is None  # 零/负数不写入已知参数
    assert ("大于 0" in answer) is has_validity_note
    assert "工作年限" in answer
    assert "最终金额" not in answer
    assert citations == []
    assert harness.state("s6")["planned_tool_calls"] == []


def test_c2_06_no_response_from_extractor_still_requires_valid_value(harness) -> None:
    """模型没有抽出参数但用户确实给了一个数字：保留任务并说明需要有效信息。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")], [])

    harness.run("s6b", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    answer, _ = harness.run("s6b", "工龄负三年")

    assert harness.calc_calls == []
    assert harness.state("s6b")["calc_context"]["status"] == "awaiting"
    assert "大于 0" in answer
    assert "工作年限" in answer


# ---------------------------------------------------------------------------
# C2-07 取消后换话题：不残留参数
# ---------------------------------------------------------------------------


def test_c2_07_cancel_then_unrelated_question_leaves_no_state(harness) -> None:
    harness.script([calc_call(monthly_wage=8000, kind="N")], [], [])

    harness.run("s7", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    assert harness.state("s7")["calc_context"]["status"] == "awaiting"

    answer, _ = harness.run("s7", "算了，不问了。")
    assert harness.state("s7")["calc_context"] is None
    assert harness.calc_calls == []
    assert answer == harness.answer_text

    harness.run("s7", "那劳动合同到期有经济补偿吗？")
    assert harness.state("s7")["calc_context"] is None
    assert harness.calc_calls == []
    # 旧任务参数不再出现在抽取提示中
    assert "已确认参数" not in harness.router.prompts[-1]


# ---------------------------------------------------------------------------
# C2-08 等待补参时切换消费维权问题
# ---------------------------------------------------------------------------


def test_c2_08_topic_switch_to_consumer_question(harness) -> None:
    harness.script([calc_call(monthly_wage=8000, kind="N")], [])

    harness.run("s8", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    answer, citations = harness.run("s8", "七天无理由退货哪些商品不适用？")

    assert harness.calc_calls == []
    assert harness.state("s8")["calc_context"] is None
    assert answer == harness.answer_text
    assert "工作年限" not in answer  # 不再索要工资或年限
    assert len(citations) == 2  # 走正常检索问答


# ---------------------------------------------------------------------------
# C2-09 普通咨询与消费退款不触发追问
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["经济补偿有哪些条件？", "网购商品退款金额是多少？"],
)
def test_c2_09_ordinary_questions_do_not_start_clarification(harness, question) -> None:
    harness.script([])

    answer, _ = harness.run("s9", question)

    assert harness.calc_calls == []
    assert harness.state("s9")["calc_context"] is None
    assert harness.state("s9")["planned_tool_calls"] == []
    assert answer == harness.answer_text
    assert harness.mcp_init["n"] == 0


def test_c2_09_conditions_question_without_usable_params_is_not_a_task(harness) -> None:
    """模型即使误调了计算器，只要没有可追溯到用户原话的参数也不进入追问。"""
    harness.script([calc_call()])

    answer, _ = harness.run("s9b", "经济补偿有哪些条件？")

    assert harness.state("s9b")["calc_context"] is None
    assert harness.calc_calls == []
    assert answer == harness.answer_text


# ---------------------------------------------------------------------------
# C2-10 首轮参数完整的既有请求
# ---------------------------------------------------------------------------


def test_c2_10_complete_request_keeps_existing_capability(harness) -> None:
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])

    answer, citations = harness.run("s10", "我连续工作 4 年，月工资 8000 元，经济补偿金是多少？")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]
    assert "还需" not in answer  # 不增加多余追问
    assert answer == harness.answer_text
    amount = independent_amount(8000, 4, "N")
    assert f"{amount:.2f} 元" in harness.state("s10")["tool_text"]
    assert harness.state("s10")["calc_context"]["status"] == "done"
    assert len(citations) == 2


# ---------------------------------------------------------------------------
# C2-11 会话隔离与 checkpoint 恢复
# ---------------------------------------------------------------------------


def test_c2_11_sessions_are_isolated_and_resumable(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=5000, kind="N")],
        [calc_call(years=3.5)],
        [calc_call(years=4)],
    )

    harness.run("a", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    harness.run("b", "帮我按 N 计算经济补偿，月平均工资 5000 元。")

    harness.run("a", "三年半。")
    harness.run("b", "四年。")

    assert harness.calc_calls == [
        {"monthly_wage": 8000, "years": 3.5, "kind": "N"},
        {"monthly_wage": 5000, "years": 4, "kind": "N"},
    ]
    state_a = harness.state("a")
    assert state_a["calc_context"]["monthly_wage"] == 8000
    assert state_a["calc_context"]["years"] == 3.5
    assert len(state_a["messages"]) == 4  # 同一任务状态可恢复


# ---------------------------------------------------------------------------
# C2-12 同步与 SSE 行为一致
# ---------------------------------------------------------------------------


def test_c2_12_sync_and_stream_share_text_and_history(harness) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    harness.script([calc_call(monthly_wage=8000, kind="N")])
    sync_answer, _ = harness.run("sync", "帮我按 N 计算经济补偿，月平均工资 8000 元。")

    harness.chat_model_factory = lambda: FakeListChatModel(
        responses=["追问改写文本", "生成回答：按现有规则计算。"]
    )
    harness.script([calc_call(monthly_wage=8000, kind="N")])
    first = harness.stream("sse", "帮我按 N 计算经济补偿，月平均工资 8000 元。")

    assert first[-1]["type"] == "done"
    assert sum(event["type"] == "done" for event in first) == 1
    assert not any(event["type"] == "error" for event in first)
    tokens = [event["text"] for event in first if event["type"] == "token"]
    assert "".join(tokens) == sync_answer == harness.state("sse")["answer"]
    assert graph_module.CALC_CLARIFY_YEARS in "".join(tokens)
    assert "追问改写文本" not in "".join(tokens)  # 不泄漏参数收集过程的模型输出
    assert [message.type for message in harness.messages("sse")] == ["human", "ai"]
    assert len([m for m in harness.messages("sse") if m.type == "ai"]) == 1

    harness.script([calc_call(years=3.5, kind="N")])
    second = harness.stream("sse", "三年半。")

    assert sum(event["type"] == "done" for event in second) == 1
    assert not any(event["type"] == "error" for event in second)
    streamed = "".join(event["text"] for event in second if event["type"] == "token")
    assert streamed == harness.state("sse")["answer"]
    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    assert [message.type for message in harness.messages("sse")] == [
        "human",
        "ai",
        "human",
        "ai",
    ]


def test_c2_12_clarify_turn_never_emits_duplicate_assistant_message(harness) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    harness.script([calc_call(monthly_wage=8000, kind="N")])
    harness.chat_model_factory = lambda: FakeListChatModel(responses=[])

    events = harness.stream("sse2", "帮我按 N 计算经济补偿，月平均工资 8000 元。")

    assert [event["type"] for event in events] == ["token", "done"]
    assistant_messages = [m for m in harness.messages("sse2") if m.type == "ai"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == events[0]["text"] == harness.state("sse2")["answer"]
    assert harness.generate_prompts == []  # 追问不经过生成模型


# ---------------------------------------------------------------------------
# 回归要求
# ---------------------------------------------------------------------------


def test_regression_no_external_tools_when_web_disabled(harness) -> None:
    harness.script([calc_call(monthly_wage=8000, kind="N")])

    harness.run("r1", "帮我按 N 计算经济补偿，月平均工资 8000 元。")

    planned = harness.state("r1")["planned_tool_calls"]
    assert not any(call["name"] in graph_module.MCP_TOOL_NAMES for call in planned)
    assert harness.mcp_init["n"] == 0


def test_regression_old_checkpoint_without_new_field(harness) -> None:
    """旧 checkpoint 没有 calc_context 字段时按“无任务”处理，不报错。"""
    result = graph_module.route(
        {"question": "试用期最长可以约定多久？", "search_query": "试用期最长可以约定多久？"}
    )

    assert result["calc_context"] is None
    assert result["planned_tool_calls"] == []
    assert graph_module._next_node({}) == "generate"
    assert graph_module._next_node({"planned_tool_calls": []}) == "generate"
    # 命中计算关键词但模型未调用计算器时同样不报错
    assert graph_module._next_node({"planned_tool_calls": [{"name": "web_search"}]}) == "tools"


def test_regression_plain_question_after_calculation_does_not_recalculate(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, years=4, kind="N")],
        [],
    )

    harness.run("r2", "我连续工作 4 年，月工资 8000 元，按 N 计算经济补偿金。")
    assert len(harness.calc_calls) == 1

    answer, _ = harness.run("r2", "试用期最长可以约定多久？")

    assert len(harness.calc_calls) == 1
    assert harness.state("r2")["calc_context"] is None
    assert answer == harness.answer_text


def test_regression_new_task_does_not_inherit_previous_params(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=20000, years=3, kind="2N")],
        [calc_call(monthly_wage=5000, years=5, kind="N")],
    )

    harness.run("r3", "我工作 3 年，月工资 20000 元，按 2N 计算赔偿金。")
    harness.run("r3", "再帮我算一个：月工资 5000 元，工作 5 年。")

    assert harness.calc_calls[-1] == {"monthly_wage": 5000, "years": 5, "kind": "N"}
    context = harness.state("r3")["calc_context"]
    assert context["years"] == 5  # 不继承上一个任务的 3 年


def test_regression_clarify_turn_keeps_citations_consistent(harness) -> None:
    """追问轮次不残留上一轮引用，下一轮计算仍带本轮检索来源。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")], [calc_call(years=4)])

    _, citations = harness.run("r4", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    assert citations == []
    assert harness.state("r4")["citations"] == []

    _, citations = harness.run("r4", "四年。")
    assert len(citations) == 2
    payload = harness.state("r4")["messages"][-1].additional_kwargs["citations"]
    assert [item["article"] for item in payload] == ["第1条", "第2条"]


def test_rewrite_model_output_never_becomes_calc_parameter(harness) -> None:
    """追问改写文本里的数字不得被当成用户事实（改写桩固定带 9000）。"""
    harness.script([calc_call(monthly_wage=6000, kind="N")])

    harness.run("r5", "帮我按 N 计算经济补偿，月平均工资 6000 元。")

    assert harness.rewrite_prompts == []  # 首轮不触发改写模型
    assert harness.state("r5")["calc_context"]["monthly_wage"] == 6000

    harness.script([calc_call(years=3)])
    harness.run("r5", "三年。")

    assert harness.rewrite_prompts  # 第二轮确实经过了改写模型
    assert harness.state("r5")["calc_context"]["monthly_wage"] == 6000
    assert harness.calc_calls == [{"monthly_wage": 6000, "years": 3, "kind": "N"}]


# ---------------------------------------------------------------------------
# C2-R1-01 逐字段核验参数来源（F-C2-01）
# ---------------------------------------------------------------------------


def test_c2_r1_01_fabricated_years_is_not_accepted(harness) -> None:
    """用户只给工资 8000，模型额外给出 years=20：不计算，追问年限。"""
    harness.script([calc_call(monthly_wage=8000, years=20, kind="N")])

    answer, _ = harness.run("f1", "按 N 计算经济补偿，月工资8000元。")

    assert harness.calc_calls == []
    context = harness.state("f1")["calc_context"]
    assert context["monthly_wage"] == 8000
    assert context["years"] is None  # 无依据的 20 不得进入参数
    assert graph_module.CALC_CLARIFY_YEARS in answer
    assert harness.state("f1")["planned_tool_calls"] == []


def test_c2_r1_01_unmentioned_wage_does_not_overwrite(harness) -> None:
    """已确认工资 8000，补“三年半”时模型回带 9000：工资保持 8000，只调用一次。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=9000, years=3.5, kind="N")],
    )

    harness.run("f2", "按 N 计算经济补偿，月工资8000元。")
    harness.run("f2", "三年半。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    assert harness.state("f2")["calc_context"]["monthly_wage"] == 8000


def test_c2_r1_01_kind_only_request_yields_no_numbers(harness) -> None:
    """只说“按 2N 计算”：不接受模型补出的工资与年限，且 2N 里的 2 不能当年限。"""
    harness.script([calc_call(monthly_wage=8000, years=2, kind="2N")])

    answer, _ = harness.run("f3", "按 2N 计算。")

    assert harness.calc_calls == []
    context = harness.state("f3")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["years"] is None
    assert context["kind"] == "2N"
    assert graph_module.CALC_CLARIFY_BOTH in answer


def test_c2_r1_01_valid_correction_and_chinese_fill_still_work(harness) -> None:
    """合法明确纠正与原有中文补参继续通过。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=9000, years=4.5, kind="N")],
    )

    harness.run("f4", "按 N 计算经济补偿，月工资8000元。")
    harness.run("f4", "工资改成 9000 元，工龄四年半。")

    assert harness.calc_calls == [{"monthly_wage": 9000, "years": 4.5, "kind": "N"}]


def test_c2_r1_01_bare_value_binds_only_to_single_missing_field(harness) -> None:
    """待补字段唯一时，裸数字可结合当前待补字段解释；两个字段都缺时不解释。"""
    harness.script(
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=9000, kind="N")],
    )

    harness.run("f5", "按 N 计算经济补偿，我工作了 4 年。")
    harness.run("f5", "9000")

    assert harness.calc_calls == [{"monthly_wage": 9000, "years": 4, "kind": "N"}]


# ---------------------------------------------------------------------------
# C2-R1-02 明确新任务从空参数开始（F-C2-02）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state_before", ["awaiting", "done", "corrected"])
def test_c2_r1_02_new_task_does_not_inherit_old_params(harness, state_before) -> None:
    """旧任务处于 awaiting / done / 更正确认后时，明确新任务只给工资都必须追问年限。"""
    harness.script([calc_call(monthly_wage=20000, years=3, kind="2N", social_average_wage=8000)])
    harness.run("n1", "按 2N 计算，月工资20000元，工作3年，当地社平工资8000元。")
    if state_before == "awaiting":
        harness.script([calc_call(monthly_wage=20000, kind="2N")])
        harness.run("n1", "重新开一个：按 N 计算经济补偿，月工资20000元。")
        assert harness.state("n1")["calc_context"]["status"] == "awaiting"
    elif state_before == "corrected":
        harness.script([calc_call(monthly_wage=25000)])
        harness.run("n1", "工资说错了，是 25000 元。")
        assert harness.state("n1")["calc_context"]["monthly_wage"] == 25000

    harness.script([calc_call(monthly_wage=5000, kind="N")])
    answer, _ = harness.run("n1", "另一个同事：按 N 计算经济补偿，月工资5000元。")

    context = harness.state("n1")["calc_context"]
    assert context["years"] is None  # 不继承旧任务的 3 年
    assert context["social_average_wage"] is None  # 不继承旧任务的社平工资
    assert context["monthly_wage"] == 5000
    assert graph_module.CALC_CLARIFY_YEARS in answer
    assert harness.calc_calls == [
        {"monthly_wage": 20000, "years": 3, "kind": "2N", "social_average_wage": 8000}
    ]


def test_c2_r1_02_new_task_with_only_years_asks_for_wage(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=20000, years=3, kind="2N")],
        [calc_call(years=5)],
    )

    harness.run("n2", "按 2N 计算，月工资20000元，工作3年。")
    answer, _ = harness.run("n2", "另外一个人的情况：工作 5 年。")

    context = harness.state("n2")["calc_context"]
    assert context["monthly_wage"] is None  # 不继承旧工资
    assert context["kind_explicit"] is False  # 不继承旧类型
    assert graph_module.CALC_CLARIFY_WAGE in answer
    assert len(harness.calc_calls) == 1


def test_c2_r1_02_current_task_fill_and_correction_keep_fields(harness) -> None:
    """当前任务的正常补参与明确纠正仍保留合法旧字段。"""
    harness.script(
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=8000)],
    )

    harness.run("n3", "按 N 计算经济补偿，我工作了 4 年。")
    harness.run("n3", "月工资改成 8000 元。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]


# ---------------------------------------------------------------------------
# C2-R1-03 无效纠正必须使旧值失效（F-C2-03）
# ---------------------------------------------------------------------------


def test_c2_r1_03_zero_wage_correction_invalidates_confirmed_value(harness) -> None:
    """已知工资 8000、待年限；“工资改成0，工龄4年”零次调用且工资待确认。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=0, years=4, kind="N")],
    )

    harness.run("i1", "按 N 计算经济补偿，月工资8000元。")
    answer, _ = harness.run("i1", "工资改成0，工龄4年。")

    assert harness.calc_calls == []
    context = harness.state("i1")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["years"] == 4
    assert context["status"] == "awaiting"
    assert context["invalid_fields"] == ["monthly_wage"]
    assert "大于 0" in answer and "工资" in answer
    assert "最终金额" not in answer


def test_c2_r1_03_negative_wage_correction_blocks_recompute(harness) -> None:
    """已完成 8000/4/N 后“工资改成-1，重新计算”：零次新调用，不重用旧工资。"""
    harness.script(
        [calc_call(monthly_wage=8000, years=4, kind="N")],
        [calc_call(monthly_wage=-1, kind="N")],
    )

    harness.run("i2", "按 N 计算经济补偿，月工资8000元，工龄4年。")
    answer, _ = harness.run("i2", "工资改成-1，重新计算。")

    assert len(harness.calc_calls) == 1  # 只有首轮那一次
    context = harness.state("i2")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["status"] == "awaiting"
    assert "大于 0" in answer

    harness.script([calc_call(monthly_wage=9000)])
    harness.run("i2", "月工资 9000 元。")

    assert len(harness.calc_calls) == 2
    assert harness.calc_calls[-1] == {"monthly_wage": 9000, "years": 4, "kind": "N"}
    assert harness.state("i2")["calc_context"]["status"] == "done"


@pytest.mark.parametrize(
    ("reply", "model_args"),
    [("工龄改成0", {"years": 0}), ("工龄改成-2", {"years": -2})],
)
def test_c2_r1_03_years_invalid_correction_invalidates(harness, reply, model_args) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, years=3, kind="N")],
        [calc_call(**model_args)],
    )

    harness.run("i3", "按 N 计算经济补偿，月工资8000元，工龄3年。")
    answer, _ = harness.run("i3", reply)

    assert len(harness.calc_calls) == 1
    context = harness.state("i3")["calc_context"]
    assert context["years"] is None
    assert context["monthly_wage"] == 8000
    assert "大于 0" in answer and "工作年限" in answer


def test_c2_r1_03_unsourced_zero_does_not_clear_confirmed_value(harness) -> None:
    """用户没有更正工资、只是模型无依据返回零：已确认工资不得被误删。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=0, years=3.5, kind="N")],
    )

    harness.run("i4", "按 N 计算经济补偿，月工资8000元。")
    harness.run("i4", "三年半。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    assert harness.state("i4")["calc_context"]["monthly_wage"] == 8000


# ---------------------------------------------------------------------------
# C2-R1-04 明确取消优先于模型抽参（F-C2-04）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_output",
    [
        [],
        [{"name": CALC_NAME, "args": {"kind": "N"}}],
        [{"name": CALC_NAME, "args": {"monthly_wage": 8000, "years": 4, "kind": "N"}}],
    ],
)
def test_c2_r1_04_cancel_wins_over_model_output(harness, model_output) -> None:
    harness.script([calc_call(monthly_wage=8000, kind="N")], model_output)

    harness.run("c1", "按 N 计算经济补偿，月工资8000元。")
    assert harness.state("c1")["calc_context"]["status"] == "awaiting"

    answer, _ = harness.run("c1", "别算了。")

    assert harness.state("c1")["calc_context"] is None
    assert harness.calc_calls == []
    assert answer == harness.answer_text


def test_c2_r1_04_cancelled_task_is_not_inherited(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, years=4, kind="N")],
        [calc_call(monthly_wage=5000, kind="N")],
    )

    harness.run("c2", "按 N 计算经济补偿，月工资8000元，工龄4年。")
    harness.run("c2", "取消这个任务。")
    answer, _ = harness.run("c2", "按 N 计算经济补偿，月工资5000元。")

    context = harness.state("c2")["calc_context"]
    assert context["years"] is None
    assert context["monthly_wage"] == 5000
    assert graph_module.CALC_CLARIFY_YEARS in answer


# ---------------------------------------------------------------------------
# C2-R1-05 相关澄清保留待办（F-C2-05）
# ---------------------------------------------------------------------------


def test_c2_r1_05_field_question_keeps_task_and_is_answered(harness) -> None:
    """已知年限 4、待工资；询问口径问题后年限与待办仍在，并能正常回答。"""
    harness.script([calc_call(years=4, kind="N")], [])

    harness.run("q1", "按 N 计算经济补偿，我工作了 4 年。")
    answer, citations = harness.run("q1", "月平均工资是税前还是税后？")

    context = harness.state("q1")["calc_context"]
    assert context["years"] == 4
    assert context["status"] == "awaiting"
    assert answer == harness.answer_text  # 正常回答澄清问题，不只是重复索要
    assert len(citations) == 2
    assert harness.calc_calls == []

    harness.script([calc_call(monthly_wage=9000)])
    harness.run("q1", "月工资 9000 元。")

    assert harness.calc_calls == [{"monthly_wage": 9000, "years": 4, "kind": "N"}]


def test_c2_r1_05_unusable_reply_keeps_task_and_asks(harness) -> None:
    """无效的字段更正：保留任务、作废该字段并说明需要什么样的数值。"""
    harness.script([calc_call(years=4, kind="N")], [])

    harness.run("q2", "按 N 计算经济补偿，我工作了 4 年。")
    answer, _ = harness.run("q2", "工龄负三年")

    context = harness.state("q2")["calc_context"]
    assert context["years"] is None  # 负值更正使旧工龄失效，不再作为有效值
    assert context["status"] == "awaiting"
    assert context["invalid_fields"] == ["years"]
    assert "大于 0" in answer and "工作年限" in answer
    assert harness.calc_calls == []

    harness.script([calc_call(years=4, monthly_wage=8000, kind="N")])
    harness.run("q2", "工龄是 4 年，月工资 8000 元。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]


def test_c2_r1_05_topic_switch_still_exits_task(harness) -> None:
    harness.script([calc_call(monthly_wage=8000, kind="N")], [])

    harness.run("q3", "按 N 计算经济补偿，月工资8000元。")
    answer, _ = harness.run("q3", "七天无理由退货哪些商品不适用？")

    assert harness.state("q3")["calc_context"] is None
    assert "还需" not in answer
    assert harness.calc_calls == []


def test_c2_r1_05_plain_question_after_completion_stops_asking(harness) -> None:
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")], [])

    harness.run("q4", "按 N 计算经济补偿，月工资8000元，工龄4年。")
    answer, _ = harness.run("q4", "试用期最长可以约定多久？")

    assert harness.state("q4")["calc_context"] is None
    assert "还需" not in answer
    assert len(harness.calc_calls) == 1


# ---------------------------------------------------------------------------
# C2-R1-06 实际执行成功后才完成（F-C2-06）
# ---------------------------------------------------------------------------


def test_c2_r1_06_tool_failure_is_not_done(harness) -> None:
    harness.fail_calc = True
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])

    harness.run("e1", "按 N 计算经济补偿，月工资8000元，工龄4年。")

    context = harness.state("e1")["calc_context"]
    assert context["status"] == "failed"
    assert context["status"] != "done"
    assert context["monthly_wage"] == 8000  # 失败保留已确认上下文
    assert "调用失败" in harness.state("e1")["tool_text"]
    assert "最终金额" not in harness.state("e1")["tool_text"]


def test_c2_r1_06_success_marks_done_once(harness) -> None:
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])

    harness.run("e2", "按 N 计算经济补偿，月工资8000元，工龄4年。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]
    assert harness.state("e2")["calc_context"]["status"] == "done"


def test_c2_r1_06_plain_question_after_failure_does_not_retry(harness) -> None:
    harness.fail_calc = True
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])
    harness.run("e3", "按 N 计算经济补偿，月工资8000元，工龄4年。")

    harness.fail_calc = False
    harness.script([])
    harness.run("e3", "试用期最长可以约定多久？")

    assert len(harness.calc_calls) == 1


def test_c2_r1_06_explicit_retry_after_failure_uses_confirmed_params(harness) -> None:
    harness.fail_calc = True
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])
    harness.run("e4", "按 N 计算经济补偿，月工资8000元，工龄4年。")
    assert harness.state("e4")["calc_context"]["status"] == "failed"

    harness.fail_calc = False
    harness.script([])
    harness.run("e4", "请重新计算一次。")

    assert harness.calc_calls == [
        {"monthly_wage": 8000, "years": 4, "kind": "N"},
        {"monthly_wage": 8000, "years": 4, "kind": "N"},
    ]
    assert harness.state("e4")["calc_context"]["status"] == "done"


def test_c2_r1_06_calc_failure_not_masked_by_other_tool(harness) -> None:
    """计算器失败不能被同一轮其他工具的成功掩盖成计算成功。"""

    class WebTool:
        name = "web_search"

        async def ainvoke(self, args):
            return [
                {
                    "type": "text",
                    "text": '{"ok": true, "results": [{"title": "标题", "url": "https://example.com"}]}',
                }
            ]

    harness.fail_calc = True
    harness.extra_tools = [WebTool()]
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])

    harness.run("e5", "按 N 计算经济补偿，月工资8000元，工龄4年。", use_web_search=True)

    assert harness.state("e5")["calc_context"]["status"] == "failed"
    assert len(harness.state("e5")["web_sources"]) == 1  # 联网结果照常保留


def test_state_survives_checkpointer_reopen(harness) -> None:
    """连接关闭并与图缓存一起重建后，同一会话仍能继续补齐参数。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")])
    harness.run("o1", "按 N 计算经济补偿，月工资8000元。")

    harness.reopen()

    harness.script([calc_call(years=3.5, kind="N")])
    harness.run("o1", "三年半。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    assert len(harness.messages("o1")) == 4


# ---------------------------------------------------------------------------
# R2-01 无效更正不依赖模型是否报出（R1-F01）
# ---------------------------------------------------------------------------


def test_r2_01_invalid_wage_omitted_by_model(harness) -> None:
    """模型漏报无效工资，仍须按用户原话作废工资并追问。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(years=4, kind="N")],
    )

    harness.run("w1", "按N计算经济补偿，月工资8000元。")
    answer, _ = harness.run("w1", "工资改成0，工龄4年。")

    assert harness.calc_calls == []
    context = harness.state("w1")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["years"] == 4
    assert context["status"] == "awaiting"
    assert context["invalid_fields"] == ["monthly_wage"]
    assert "大于 0" in answer


def test_r2_01_invalid_wage_empty_model_then_fill(harness) -> None:
    """更正轮模型空调用也要作废旧工资，后续只补年限不得使用旧工资。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [],
        [calc_call(years=4, kind="N")],
    )

    harness.run("w2", "按N计算经济补偿，月工资8000元。")
    harness.run("w2", "工资改成0，工龄4年。")
    harness.run("w2", "工龄4年，请计算。")

    assert harness.calc_calls == []
    context = harness.state("w2")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["years"] == 4


def test_r2_01_invalid_years_omitted_by_model(harness) -> None:
    harness.script(
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=8000, kind="N")],
    )

    harness.run("w3", "按N计算经济补偿，工龄4年。")
    answer, _ = harness.run("w3", "工龄改成0，月工资8000元。")

    assert harness.calc_calls == []
    context = harness.state("w3")["calc_context"]
    assert context["years"] is None
    assert context["monthly_wage"] == 8000
    assert context["invalid_fields"] == ["years"]
    assert "大于 0" in answer


@pytest.mark.parametrize(
    "model_output",
    [
        [calc_call(kind="N")],
        [calc_call(monthly_wage=8000, years=4, kind="N")],
    ],
)
def test_r2_01_completed_task_invalid_recompute_blocked(harness, model_output) -> None:
    """完成后“工资改成-1，重新计算”，模型仅给 kind 或回带旧值都不得按旧值重算。"""
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")], model_output)

    harness.run("w4", "按N计算经济补偿，月工资8000元，工龄4年。")
    answer, _ = harness.run("w4", "工资改成-1，重新计算。")

    assert len(harness.calc_calls) == 1  # 只有首轮那一次
    context = harness.state("w4")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["status"] == "awaiting"
    assert "大于 0" in answer


def test_r2_01_invalid_correction_positive_control(harness) -> None:
    """正向对照：模型忠实报出零值时同样作废并阻止调用。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=0, years=4, kind="N")],
    )

    harness.run("w5", "按N计算经济补偿，月工资8000元。")
    harness.run("w5", "工资改成0，工龄4年。")

    assert harness.calc_calls == []
    assert harness.state("w5")["calc_context"]["monthly_wage"] is None


def test_r2_01_valid_refill_after_invalidation_calculates_once(harness) -> None:
    """作废后提供有效工资：保留合法年限并恰好计算一次。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=9000)],
    )

    harness.run("w6", "按N计算经济补偿，月工资8000元。")
    harness.run("w6", "工资改成0，工龄4年。")
    harness.run("w6", "月工资 9000 元。")

    assert harness.calc_calls == [{"monthly_wage": 9000, "years": 4, "kind": "N"}]
    assert harness.state("w6")["calc_context"]["status"] == "done"


# ---------------------------------------------------------------------------
# R2-02 按明确字段绑定金额（R1-F02）
# ---------------------------------------------------------------------------


def test_r2_02_other_money_not_used_as_wage(harness) -> None:
    """赔偿金额不能当作月工资依据。"""
    harness.script([calc_call(monthly_wage=30000, years=4, kind="N")])

    harness.run("m1", "按N计算经济补偿，公司说赔偿30000元，我月工资8000元，工龄4年。")

    assert harness.calc_calls == []
    assert harness.state("m1")["calc_context"]["monthly_wage"] is None


def test_r2_02_other_money_positive_control(harness) -> None:
    """正向对照：同一句话里正确抽取月工资 8000 时正常调用。"""
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])

    harness.run("m2", "按N计算经济补偿，公司说赔偿30000元，我月工资8000元，工龄4年。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]


def test_r2_02_social_wage_swap_rejected(harness) -> None:
    """月工资与社平工资互换时，两个错配值都不得进入调用。"""
    harness.script(
        [calc_call(monthly_wage=5000, social_average_wage=8000, years=4, kind="N")]
    )

    harness.run("m3", "按N计算经济补偿，月工资8000元，当地社平工资5000元，工龄4年。")

    assert harness.calc_calls == []
    context = harness.state("m3")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["social_average_wage"] is None


def test_r2_02_social_wage_correct_pair_accepted(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, social_average_wage=5000, years=4, kind="N")]
    )

    harness.run("m4", "按N计算经济补偿，月工资8000元，当地社平工资5000元，工龄4年。")

    assert harness.calc_calls == [
        {"monthly_wage": 8000, "years": 4, "kind": "N", "social_average_wage": 5000}
    ]


def test_r2_02_compensation_amount_rejected_when_wage_is_only_missing(harness) -> None:
    """只缺工资时，“赔偿30000元”仍不能当作月工资。"""
    harness.script(
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=30000)],
    )

    harness.run("m5", "按N计算经济补偿，工龄4年。")
    harness.run("m5", "公司说赔偿30000元")

    assert harness.calc_calls == []
    assert harness.state("m5")["calc_context"]["monthly_wage"] is None


def test_r2_02_bare_money_binds_unique_missing_wage(harness) -> None:
    """只缺工资时的无字段词短补参仍按既有唯一待补字段规则可用。"""
    harness.script(
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=8000)],
    )

    harness.run("m6", "按N计算经济补偿，工龄4年。")
    harness.run("m6", "8000元")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]


# ---------------------------------------------------------------------------
# R2-03 复合年限沿用“不支持则追问”（R1-F03）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_years", [3, 0.5, 3.5])
def test_r2_03_compound_years_blocks_partial_values(harness, model_years) -> None:
    """“3年6个月”不得被截成部分数值执行，须要求统一表达确认。"""
    harness.script([calc_call(monthly_wage=8000, years=model_years, kind="N")])

    answer, _ = harness.run("y1", "按N计算经济补偿，月工资8000元，工龄3年6个月。")

    assert harness.calc_calls == []
    context = harness.state("y1")["calc_context"]
    assert context["years"] is None
    assert context["status"] == "awaiting"
    assert context["compound_years"] is True
    assert graph_module.CALC_CLARIFY_COMPOUND_YEARS in answer


def test_r2_03_compound_years_confirmed_by_supported_form(harness) -> None:
    harness.script(
        [calc_call(monthly_wage=8000, years=3, kind="N")],
        [calc_call(years=3.5, kind="N")],
    )

    harness.run("y2", "按N计算经济补偿，月工资8000元，工龄3年6个月。")
    harness.run("y2", "工龄是 3.5 年。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    assert harness.state("y2")["calc_context"]["status"] == "done"


def test_r2_03_compound_years_invalidates_confirmed_years(harness) -> None:
    """已有工龄后改成复合表达：旧工龄不得继续参与计算。"""
    harness.script(
        [calc_call(monthly_wage=8000, years=3, kind="N")],
        [calc_call(years=3)],
    )

    harness.run("y3", "按N计算经济补偿，月工资8000元，工龄3年。")
    harness.run("y3", "工龄改成3年6个月。")

    assert len(harness.calc_calls) == 1  # 只有首轮那一次
    context = harness.state("y3")["calc_context"]
    assert context["years"] is None
    assert context["status"] == "awaiting"


def test_r2_03_half_year_and_three_and_half_still_supported(harness) -> None:
    """原有“三年半”“半年”等表达不因阻断复合表达而回退。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(years=3.5, kind="N")],
    )

    harness.run("y4", "按N计算经济补偿，月工资8000元。")
    harness.run("y4", "三年半。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]

    harness.script([calc_call(monthly_wage=8000, kind="N")], [calc_call(years=0.5, kind="N")])
    harness.run("y5", "按N计算经济补偿，月工资8000元。")
    harness.run("y5", "半年。")

    assert harness.calc_calls[-1] == {"monthly_wage": 8000, "years": 0.5, "kind": "N"}


# ---------------------------------------------------------------------------
# R2-04 相关澄清不受模型调用形态影响（R1-F04）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_output",
    [
        [],
        [calc_call(kind="N")],
        [calc_call(monthly_wage=8000)],
        [calc_call(years=4)],
    ],
)
def test_r2_04_related_clarification_unaffected_by_model_shape(harness, model_output) -> None:
    """空调用、仅 kind、无依据的历史参数都必须走正常回答路径并保留待办。"""
    harness.script([calc_call(years=4, kind="N")], model_output)

    harness.run("c3", "按N计算经济补偿，工龄4年。")
    answer, citations = harness.run("c3", "月平均工资是税前还是税后？")

    context = harness.state("c3")["calc_context"]
    assert context["years"] == 4
    assert context["status"] == "awaiting"
    assert context["reply"] is None
    assert harness.calc_calls == []
    assert answer == harness.answer_text  # 实际进入生成节点，而不是确定性追问
    assert "还需" not in answer
    assert len(citations) == 2  # 仍然走检索资料
    assert harness.generate_prompts  # 生成节点确实被调用

    harness.script([calc_call(monthly_wage=8000)])
    harness.run("c3", "月工资 8000 元。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]


# ---------------------------------------------------------------------------
# R3-01 字段边界：按当前片段与字段含义绑定，停止跨片段抢占（R2-F01）
# ---------------------------------------------------------------------------


def test_r3_01_scan_each_literal_binds_to_its_own_fragment() -> None:
    """扫描器层：赔偿金额只归属其他金额，不跨片段进入社平或个人工资依据。"""
    groups = graph_module._scan_calc_literals(
        "按N计算经济补偿，月工资20000元，工龄4年，社平5000元，赔偿30000元。"
    )
    assert groups["monthly_wage"] == {20000.0}
    assert groups["social_average_wage"] == {5000.0}
    assert groups["years"] == {4.0}
    assert 30000.0 in groups["other_money"]
    assert 30000.0 not in groups["monthly_wage"]
    assert 30000.0 not in groups["social_average_wage"]
    assert 30000.0 not in groups["unattributed"]


def test_r3_01_scan_social_before_personal_keeps_each_field() -> None:
    """扫描器层：社平在前时不得吸走后面的个人月工资，反之亦然。"""
    groups = graph_module._scan_calc_literals(
        "按N计算经济补偿，社平5000元，月工资8000元，工龄4年。"
    )
    assert groups["social_average_wage"] == {5000.0}
    assert groups["monthly_wage"] == {8000.0}
    assert 5000.0 not in groups["monthly_wage"]
    assert 8000.0 not in groups["social_average_wage"]


def test_r3_01_scan_social_anchor_containing_wage_word_is_still_social() -> None:
    """扫描器层：社平称谓本身含“工资”，仍按完整的社平字段处理，不拆成个人工资。"""
    groups = graph_module._scan_calc_literals("按N计算经济补偿，当地社平工资5000元，月工资8000元。")
    assert groups["social_average_wage"] == {5000.0}
    assert groups["monthly_wage"] == {8000.0}


def test_r3_01_other_money_after_social_cannot_be_social(harness) -> None:
    """受控模型把赔偿 30000 报成社平：不得通过社平校验，也不进入调用。"""
    harness.script([calc_call(monthly_wage=20000, years=4, social_average_wage=30000, kind="N")])

    harness.run("g1", "按N计算经济补偿，月工资20000元，工龄4年，社平5000元，赔偿30000元。")

    assert all(call.get("social_average_wage") != 30000 for call in harness.calc_calls)
    context = harness.state("g1")["calc_context"]
    assert context["social_average_wage"] != 30000
    assert context["monthly_wage"] == 20000


def test_r3_01_social_correct_positive_control(harness) -> None:
    """正向对照：同句正确返回社平 5000 时唯一调用，赔偿金额不参与者。"""
    harness.script([calc_call(monthly_wage=20000, years=4, social_average_wage=5000, kind="N")])

    harness.run("g2", "按N计算经济补偿，月工资20000元，工龄4年，社平5000元，赔偿30000元。")

    assert harness.calc_calls == [
        {"monthly_wage": 20000, "years": 4, "kind": "N", "social_average_wage": 5000}
    ]


def test_r3_01_social_before_personal_correct_input_no_extra_clarification(harness) -> None:
    """社平在月工资之前：完整正确抽取应唯一调用，不得反过来再追问工资。"""
    harness.script([calc_call(monthly_wage=8000, social_average_wage=5000, years=4, kind="N")])

    harness.run("g3", "按N计算经济补偿，社平5000元，月工资8000元，工龄4年。")

    assert harness.calc_calls == [
        {"monthly_wage": 8000, "years": 4, "kind": "N", "social_average_wage": 5000}
    ]
    assert harness.state("g3")["calc_context"]["status"] == "done"


def test_r3_01_personal_before_social_correct_input_no_extra_clarification(harness) -> None:
    """个人工资在社平之前：同样由完整正确抽取唯一调用。"""
    harness.script([calc_call(monthly_wage=8000, social_average_wage=5000, years=4, kind="N")])

    harness.run("g4", "按N计算经济补偿，月工资8000元，社平5000元，工龄4年。")

    assert harness.calc_calls == [
        {"monthly_wage": 8000, "years": 4, "kind": "N", "social_average_wage": 5000}
    ]


def test_r3_01_swapped_wage_fields_rejected(harness) -> None:
    """模型互换两个月工资字段：两个错配值都不能通过对应字段校验。"""
    harness.script([calc_call(monthly_wage=5000, social_average_wage=8000, years=4, kind="N")])

    harness.run("g5", "按N计算经济补偿，月工资8000元，社平5000元，工龄4年。")

    assert harness.calc_calls == []
    context = harness.state("g5")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["social_average_wage"] is None


@pytest.mark.parametrize(
    "order",
    list(itertools.permutations(["月工资8000元", "社平5000元", "赔偿30000元"])),
)
def test_r3_01_field_fragment_order_permutation(harness, order) -> None:
    """月工资、社平、赔偿三个明确片段交换顺序，正确归属保持一致。"""
    question = "按N计算经济补偿，" + "，".join([*order, "工龄4年"]) + "。"
    harness.script([calc_call(monthly_wage=8000, social_average_wage=5000, years=4, kind="N")])

    harness.run("p1", question)

    assert harness.calc_calls == [
        {"monthly_wage": 8000, "years": 4, "kind": "N", "social_average_wage": 5000}
    ]


# ---------------------------------------------------------------------------
# R3-02 时间单位排除金额归属（R2-F02）
# ---------------------------------------------------------------------------


def test_r3_02_scan_years_unit_never_money() -> None:
    """扫描器层：紧跟时间单位的数值只作年限，绝不同时成为金额依据。"""
    groups = graph_module._scan_calc_literals("工资0，4年。")
    assert groups["monthly_wage"] == {0.0}
    assert groups["years"] == {4.0}
    assert 4.0 not in groups["monthly_wage"]
    assert 4.0 not in groups["unattributed"]
    assert 0.0 not in groups["years"]


@pytest.mark.parametrize(
    "model_output",
    [
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=8000, years=4, kind="N")],
        [calc_call(kind="N")],
        [],
    ],
)
def test_r3_02_invalid_wage_not_masked_by_adjacent_years(harness, model_output) -> None:
    """“工资0，4年”里的 4 年不得进入工资集合，零工资必须失效且不计算。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")], model_output)

    harness.run("z1", "按N计算经济补偿，月工资8000元。")
    answer, _ = harness.run("z1", "工资0，4年。")

    assert harness.calc_calls == []
    context = harness.state("z1")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["status"] == "awaiting"
    assert "大于 0" in answer


def test_r3_02_invalid_wage_then_valid_refill_calculates_once(harness) -> None:
    """失效后的旧工资不得恢复；后续补有效工资才恰好计算一次。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(years=4, kind="N")],
        [calc_call(monthly_wage=9000)],
    )

    harness.run("z2", "按N计算经济补偿，月工资8000元。")
    harness.run("z2", "工资0，4年。")
    harness.run("z2", "月工资 9000 元。")

    assert harness.calc_calls == [{"monthly_wage": 9000, "years": 4, "kind": "N"}]
    assert harness.state("z2")["calc_context"]["status"] == "done"


def test_r3_02_years_literal_cannot_be_wage(harness) -> None:
    """首轮“工资0，4年”，模型把 4 年当作 4 元工资：4 无工资依据，不得执行。"""
    harness.script([calc_call(monthly_wage=4, years=4, kind="N")])

    harness.run("z3", "按N计算经济补偿，工资0，4年。")

    assert harness.calc_calls == []
    context = harness.state("z3")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["years"] == 4


def test_r3_02_negative_wage_with_adjacent_years_blocks(harness) -> None:
    """负工资更正与相邻年限同轮出现：工资失效并按有效值追问。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")], [calc_call(years=4, kind="N")])

    harness.run("z4", "按N计算经济补偿，月工资8000元。")
    answer, _ = harness.run("z4", "工资-3，工龄4年。")

    assert harness.calc_calls == []
    context = harness.state("z4")["calc_context"]
    assert context["monthly_wage"] is None
    assert context["years"] == 4
    assert "大于 0" in answer


def test_r3_02_invalid_normal_positive_control(harness) -> None:
    """正向对照：原“工资改成0，工龄4年”路径仍作废工资且不计算。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")], [calc_call(years=4, kind="N")])

    harness.run("z5", "按N计算经济补偿，月工资8000元。")
    harness.run("z5", "工资改成0，工龄4年。")

    assert harness.calc_calls == []
    assert harness.state("z5")["calc_context"]["monthly_wage"] is None


def test_r3_02_unsourced_model_zero_does_not_clear_confirmed_wage(harness) -> None:
    """模型无依据返回零时不得误删已确认的合法工资。"""
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],
        [calc_call(monthly_wage=0, years=4, kind="N")],
    )

    harness.run("z6", "按N计算经济补偿，月工资8000元。")
    harness.run("z6", "工龄4年。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]


def test_r3_02_same_value_in_distinct_fields_is_kept(harness) -> None:
    """工资与年限碰巧相等时按各自字段处理，不能因全局集合相减被误拒绝。"""
    harness.script([calc_call(monthly_wage=4, years=4, kind="N")])

    harness.run("z7", "按N计算经济补偿，月工资4元，工龄4年。")

    assert harness.calc_calls == [{"monthly_wage": 4, "years": 4, "kind": "N"}]


def test_r3_02_valid_positive_short_fill_passes(harness) -> None:
    """合法正工资的短补参仍正常通过。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")], [calc_call(years=4, kind="N")])

    harness.run("z8", "按N计算经济补偿，月工资8000元。")
    harness.run("z8", "工龄4年。")

    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]


# ---------------------------------------------------------------------------
# 第三阶段第二步：clarify 分支跳过检索（OPT-01～OPT-05）
# ---------------------------------------------------------------------------


def failing_retrieve_stub():
    """检索桩：被调用即失败，确保 clarify 轮不是“执行后丢弃结果”。"""
    calls = []

    def _retrieve(query, top_k=3, session_id=None):
        calls.append(query)
        raise AssertionError("clarify 轮不应调用检索")

    return calls, _retrieve


def build_legacy_graph():
    """按优化前节点顺序（prepare→retrieve→route→…）编译对照图；节点函数与生产相同。"""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(graph_module.ChatState)
    builder.add_node("prepare", graph_module.prepare)
    builder.add_node("retrieve", graph_module.retrieve_node)
    builder.add_node("route", graph_module.route)
    builder.add_node("tools", graph_module.call_tools)
    builder.add_node("clarify", graph_module.clarify)
    builder.add_node("generate", graph_module.generate)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "retrieve")
    builder.add_edge("retrieve", "route")
    builder.add_conditional_edges(
        "route",
        graph_module._next_node,
        {"clarify": "clarify", "tools": "tools", "generate": "generate"},
    )
    builder.add_edge("tools", "generate")
    builder.add_edge("clarify", END)
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=SqliteSaver(db_module.get_checkpointer_connection()))


def message_shape(messages) -> list:
    return [
        {
            "type": message.type,
            "content": message.content,
            "citations": message.additional_kwargs.get("citations"),
        }
        for message in messages
    ]


def test_opt1_clarify_turn_skips_retrieval_entirely(harness, monkeypatch) -> None:
    """OPT-01：缺参请求进入 clarify，检索桩被调用即失败但全程未被调用。"""
    calls, stub = failing_retrieve_stub()
    monkeypatch.setattr(graph_module, "retrieve", stub)
    harness.script([calc_call(monthly_wage=8000, kind="N")])

    answer, citations = harness.run("opt1", "帮我按 N 计算经济补偿，月平均工资 8000 元。")

    assert calls == []  # 检索桩从未被调用（不是执行后丢弃结果）
    assert answer == clarify_text(harness.state("opt1")["calc_context"])
    assert citations == [] and harness.state("opt1")["citations"] == []
    assert harness.generate_prompts == []  # 不调用生成模型
    assert harness.calc_calls == []  # 不执行计算工具
    assert len(harness.router.prompts) == 1  # route 仅一次
    assert "月平均工资 8000 元" in harness.router.prompts[0]  # 原有 router 行为不变


def test_opt1_correction_ack_branch_skips_retrieval(harness, monkeypatch) -> None:
    """OPT-01：既有 correction_ack 进入 clarify 的分支同样跳过检索。"""
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])
    harness.run("opt1b", "我连续工作 4 年，月工资 8000 元，经济补偿金是多少？")
    generates_before = len(harness.generate_prompts)

    calls, stub = failing_retrieve_stub()
    monkeypatch.setattr(graph_module, "retrieve", stub)  # 只对 clarify 追问轮启用失败桩
    harness.script([calc_call(monthly_wage=9000, kind="N")])
    answer, _ = harness.run("opt1b", "工资说错了，是 9000。")

    assert calls == []
    assert answer.startswith("已按你的更正更新当前计算参数")
    assert harness.state("opt1b")["calc_context"]["reply"] == "correction_ack"
    assert len(harness.generate_prompts) == generates_before  # 追问轮不调用生成模型


def test_opt2_clarify_output_state_and_sse_match_legacy_graph(harness) -> None:
    """OPT-02：相同模型桩与初始状态下，优化前后对外结果一致（缺参追问）。"""
    script = [calc_call(monthly_wage=8000, kind="N")]

    harness.script([dict(call) for call in script])
    new_answer, _ = harness.run("opt2-new", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    new_state = harness.state("opt2-new")

    harness.script([dict(call) for call in script])
    legacy = build_legacy_graph()
    legacy_config = graph_module.graph_config("opt2-legacy")
    legacy_state = legacy.invoke(
        {"question": "帮我按 N 计算经济补偿，月平均工资 8000 元。", "use_web_search": False},
        config=legacy_config,
    )

    assert new_answer == legacy_state["answer"]  # 追问内容一致
    assert new_state["calc_context"] == legacy_state["calc_context"]  # 任务状态一致
    assert message_shape(new_state["messages"]) == message_shape(legacy_state["messages"])
    for key in ("citations", "web_sources", "tool_text", "planned_tool_calls"):
        assert new_state[key] == legacy_state[key], key

    # 新图 SSE：追问整条文本 + 单个 done 终止事件，与同步回答一致
    harness.script([dict(call) for call in script])
    events = harness.stream("opt2-sse", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    assert [event["type"] for event in events] == ["token", "done"]
    assert events[0]["text"] == new_answer
    assert events[1]["citations"] == [] and events[1]["web_sources"] == []


def test_opt2_correction_ack_matches_legacy_graph(harness) -> None:
    """OPT-02：correction_ack 分支优化前后对外结果一致。"""
    turns = [
        ("我连续工作 4 年，月工资 8000 元，经济补偿金是多少？", [calc_call(monthly_wage=8000, years=4, kind="N")]),
        ("工资说错了，是 9000。", [calc_call(monthly_wage=9000, kind="N")]),
    ]
    results = {}
    for tag, build in (("new", None), ("legacy", build_legacy_graph)):
        harness.script(*[list(turn[1]) for turn in turns])
        if build is None:
            answers = []
            for question, _ in turns:
                answers.append(harness.run(f"opt2c-{tag}", question)[0])
            state = harness.state(f"opt2c-{tag}")
        else:
            legacy = build()
            config = graph_module.graph_config(f"opt2c-{tag}")
            state = None
            answers = []
            for question, _ in turns:
                state = legacy.invoke(
                    {"question": question, "use_web_search": False}, config=config
                )
                answers.append(state["answer"])
        results[tag] = (answers, state)

    new_answers, new_state = results["new"]
    legacy_answers, legacy_state = results["legacy"]
    assert new_answers == legacy_answers
    assert new_state["calc_context"] == legacy_state["calc_context"]
    assert message_shape(new_state["messages"]) == message_shape(legacy_state["messages"])
    for key in ("citations", "web_sources", "tool_text"):
        assert new_state[key] == legacy_state[key], key


def test_opt3_fill_after_clarify_retrieves_once_and_calculates(harness) -> None:
    """OPT-03：第一轮零检索，第二轮检索一次、计算一次、正常生成。"""
    harness.script([calc_call(monthly_wage=8000, kind="N")], [calc_call(years=3.5, kind="N")])

    harness.run("opt3", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    assert harness.retrieve_queries == []  # 第一轮：零检索
    assert harness.calc_calls == []

    answer, citations = harness.run("opt3", "三年半。")

    assert len(harness.retrieve_queries) == 1  # 第二轮：检索一次
    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 3.5, "kind": "N"}]
    # 独立固定预期（E-105 修正）：不调用 fold_years / calculate_labor_compensation /
    # independent_amount 等生产折算或计算 helper。原始工龄 3.5 年是计算器输入；
    # 按项目既有折算规则（剩余年限达 0.5 计 1）折算系数为 4 = 3 + 1；
    # 金额 = 8000 × (3 + 1) = 32000.00。断言真实工具结果里的折算年限与金额。
    expected_folded_years = 3 + 1
    expected_amount = 8000 * (3 + 1)
    tool_text = harness.state("opt3")["tool_text"]
    assert "原始工作年限：3.5 年" in tool_text  # 原始年限（输入）与折算系数区分
    assert f"折算后年限：{expected_folded_years} 年" in tool_text
    assert f"计算公式：{expected_folded_years} × 8000.00" in tool_text
    assert f"最终金额：{expected_amount:.2f} 元。" in tool_text
    assert answer == harness.answer_text
    assert [(c.score, c.node.metadata["article"]) for c in citations] == [
        (0.8, "第1条"),
        (0.7, "第2条"),
    ]
    state = harness.state("opt3")
    assert state["calc_context"]["status"] == graph_module.CALC_STATUS_DONE
    assert [message.type for message in harness.messages("opt3")] == ["human", "ai", "human", "ai"]
    assert state["messages"][-1].additional_kwargs["citations"] == [
        {"source": "劳动合同法", "article": "第1条", "score": 0.8},
        {"source": "劳动合同法", "article": "第2条", "score": 0.7},
    ]


def test_opt5_history_citations_do_not_leak_into_clarify(harness, monkeypatch) -> None:
    """OPT-05：先完成一次带引用回答，再进入缺参追问；旧引用不泄漏，补齐后只用新检索结果。"""
    old_nodes = [node(1), node(2)]
    new_nodes = [node(3)]
    rounds = {"n": 0}

    def switching_retrieve(query, top_k=3, session_id=None):
        rounds["n"] += 1
        harness.retrieve_queries.append(query)
        return old_nodes if rounds["n"] == 1 else new_nodes

    monkeypatch.setattr(graph_module, "retrieve", switching_retrieve)
    # 普通问答轮不进入计算规划（不消耗脚本）；脚本只供缺参轮与补齐轮
    harness.script(
        [calc_call(monthly_wage=8000, kind="N")],  # 第二轮缺参追问
        [calc_call(years=3.5, kind="N")],  # 第三轮补齐
    )

    _, citations = harness.run("opt5", "试用期最长可以约定多久？")
    assert citations == old_nodes

    events = harness.stream("opt5", "帮我按 N 计算经济补偿，月平均工资 8000 元。")
    state = harness.state("opt5")
    assert state["citations"] == []  # checkpoint 中即使有旧引用，追问也清空
    assert state["messages"][-1].additional_kwargs["citations"] == []
    assert events[1]["type"] == "done" and events[1]["citations"] == []
    assert events[1]["web_sources"] == []

    _, citations = harness.run("opt5", "三年半。")
    assert citations == new_nodes  # 补齐后的回答只使用该轮新检索结果


def test_opt4_non_clarify_paths_retrieve_exactly_once(harness) -> None:
    """OPT-04：非 clarify 路径（普通问答/参数齐全计算/显式联网）仍各检索一次。"""
    # 普通问答轮不进入计算规划（不消耗脚本）
    harness.script([calc_call(monthly_wage=8000, years=4, kind="N")])  # 参数齐全的计算

    harness.run("opt4", "试用期最长可以约定多久？")
    harness.run("opt4", "我连续工作 4 年，月工资 8000 元，经济补偿金是多少？")
    assert len(harness.retrieve_queries) == 2  # 每轮恰好一次
    assert harness.calc_calls == [{"monthly_wage": 8000, "years": 4, "kind": "N"}]

    # 显式联网：确定性规划联网工具 + 检索一次 + 工具执行一次
    class WebStub:
        name = "verify_statute_status"

        async def ainvoke(self, args):
            return json.dumps({
                "ok": True,
                "results": [{"title": "劳动合同法", "url": "https://example.com/law"}],
            })

    harness.extra_tools = [WebStub()]
    answer, _ = harness.run("opt4web", "《劳动合同法》最近有没有被废止？", use_web_search=True)
    assert len(harness.retrieve_queries) == 3
    assert harness.mcp_init["n"] == 1
    assert "verify_statute_status" in harness.state("opt4web")["tool_text"]
    assert harness.state("opt4web")["web_sources"] == [
        {"title": "劳动合同法", "url": "https://example.com/law", "site": "", "date": ""}
    ]
    assert answer == harness.answer_text
