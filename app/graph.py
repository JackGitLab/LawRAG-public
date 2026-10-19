import json
import re
from functools import lru_cache, wraps
from inspect import signature
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.ask import ExternalServiceError, build_prompt_with_tools, retrieve
from app.config import get_chat_model, get_router_model
from app.db import get_checkpointer_connection
from app.observability import perf_enabled, perf_span, traced_invoke
from app.mcp_tools import (
    MCP_TOOL_NAMES,
    ensure_mcp_tools,
    get_all_tools,
    run_async,
    safe_web_source_url,
    tool_result_text,
    web_sources_from_text,
)
from app.mcp_tools import LOCAL_TOOLS


STATUTE_KEYWORDS = ("废止", "修订", "修正", "现行有效", "失效", "还有效", "改了", "变了吗")
CALC_KEYWORDS = (
    "多少", "计算", "赔偿", "补偿", "代通知金", "加班费", "几个月", "N+1", "2N", "工资",
    "赔", "离职", "月工资",
)
LOCAL_TOOL_NAMES = frozenset(tool.name for tool in LOCAL_TOOLS)

CALC_TOOL_NAME = "calculate_labor_compensation"
CALC_REQUIRED_FIELDS = ("monthly_wage", "years")
CALC_FIELD_FIELDS = ("monthly_wage", "years", "social_average_wage")
CALC_KINDS = ("N", "N+1", "2N")
# 计算任务状态：缺参待补 / 已规划待执行 / 实际执行成功 / 实际执行失败
CALC_STATUS_AWAITING = "awaiting"
CALC_STATUS_COMPUTING = "computing"
CALC_STATUS_DONE = "done"
CALC_STATUS_FAILED = "failed"
CALC_CANCEL_MARKERS = ("算了", "不用了", "不问了", "取消", "没事了", "别算了")
CALC_COMPUTE_MARKERS = ("计算", "算", "多少", "金额")
CALC_RETRY_MARKERS = ("重试", "重新算", "重新计算", "再算一次", "再计算", "再算一遍")
CALC_QUESTION_MARKERS = ("？", "?", "吗", "呢", "怎么", "如何", "哪些", "什么", "为什么", "是否")
CALC_NEW_TASK_MARKERS = (
    "另一个", "另外一个", "重新开一个", "重新开始", "新任务", "再帮我算", "再算一个",
)
CALC_WAGE_ANCHORS = ("工资", "月薪", "薪资", "薪金", "社平")
CALC_YEARS_ANCHORS = ("工龄", "年限", "工作")
# 当地社平工资的依据：必须来自这些词的上下文，避免与个人月工资互相充当依据
CALC_SOCIAL_ANCHORS = ("社平", "职工月平均", "职工平均工资", "上年度职工")
# 明确属于其他金额（赔偿、退款等）的上下文：不作为任何工资字段的依据
CALC_OTHER_MONEY_ANCHORS = ("赔偿", "赔款", "退款", "退货", "补偿金", "违约金", "罚款", "金额")
CALC_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
CALC_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
# 数字字面量：阿拉伯数字（含小数）或受限中文数词；“半”用于“三年半/半年”这类说法
CALC_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?|[零一二两三四五六七八九十百千万半]+")
# 复合年限表达（“3年6个月”）：本轮不支持可靠合成，识别后要求用户用统一表达确认
CALC_COMPOUND_YEARS_RE = re.compile(
    r"(?:\d+(?:\.\d+)?|[零一二两三四五六七八九十百千万]+)\s*年\s*(?:零\s*)?"
    r"(?:\d+(?:\.\d+)?|[零一二两三四五六七八九十百千万]+)\s*个?\s*月"
)
# 片段硬边界：逗号、分号、句号、问号、叹号、顿号、换行。字段归属只看当前片段。
CALC_FRAGMENT_SEP_RE = re.compile(r"[，,；;。.！!？?、\n\r]")
FIELD_LABELS = {
    "monthly_wage": "月工资",
    "years": "工作年限",
    "social_average_wage": "社平工资",
}
KIND_LABELS = {
    "N": "N（普通经济补偿）",
    "N+1": "N+1（经济补偿加一个月代通知金）",
    "2N": "2N（违法解除赔偿金）",
}

CALC_HINT = "本轮问题可能涉及金额或期限计算。\n"
CALC_PENDING_HINT = (
    "用户正在补充或更正一个尚未完成的劳动补偿计算任务；已确认参数：{known}。"
    "用户这句话里出现新的数字参数时就必须调用计算工具，只填写这句话里出现的参数，"
    "其他参数留空，不要填 0、不要用已知参数或示例值占位；"
    "这句话没有出现新的数字参数就不要调用工具。\n"
)
CALC_CLARIFY_BOTH = "还需要你提供月平均工资和在本单位的工作年限，才能按现有规则计算金额。"
CALC_CLARIFY_WAGE = "还需要你提供月平均工资（解除或终止前十二个月的平均工资）。"
CALC_CLARIFY_YEARS = "还需要你在本单位的工作年限。"
CALC_CLARIFY_INVALID = "你刚才提供的{fields}需要大于 0，请重新提供有效的数值。"
CALC_CLARIFY_COMPOUND_YEARS = "你提供的工龄是“年＋个月”的复合写法，请用统一的年数确认，例如 3.5 年。"
CALC_ASSUMPTION = "（当前按 {kind} 计算；如属其他情形请说明。）"
CALC_CORRECTION_ACK = (
    "已按你的更正更新当前计算参数：月工资 {wage} 元、工作年限 {years} 年、类型 {kind}。"
    "如需按更新后的参数重新计算，请明确说明。"
)

ROUTER_INSTRUCTION = (
    "你是法律问答系统的工具路由判断器。判断回答下面的问题是否需要调用工具，需要就调用，"
    "不需要就不要调用。不要输出解释，不要直接回答问题。\n"
    "本地计算工具只用于劳动关系中的经济补偿、代通知金、违法解除赔偿金等金额计算。"
    "用户明确要求计算这类金额时就必须调用它：即使只提供了月工资或工作年限中的一项，"
    "也要调用工具，并且只填写用户这句话里明确说出的数字；没有说出的参数留空，"
    "不要猜造、不要填 0、也不要填示例值。"
    "消费退款、商品价格及其他与劳动关系无关的金额问题不要调用计算工具。\n{hint}问题：{question}"
)


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    search_query: str
    citations: list
    answer: str
    planned_tool_calls: list
    web_sources: list
    tool_text: str
    use_web_search: bool
    # 进行中的劳动补偿计算任务；旧 checkpoint 没有该字段时按“无任务”处理。
    calc_context: dict | None


MAX_REWRITE_HISTORY_MESSAGES = 6
MAX_REWRITE_HISTORY_CHARS = 6000


def _bounded_rewrite_history(messages) -> str:
    """把最近有限条历史序列化为不超过字符预算的文本，优先保留最新消息。

    - 最多取最近 MAX_REWRITE_HISTORY_MESSAGES 条，保持原时间顺序；
    - 序列化为 "<type>: <content>" 行；content 非字符串时安全转换；
    - 从最新往旧累计，加入更旧一行时把分隔换行（1 字符）一并计入预算，
      超出字符预算即从较旧部分裁减；
    - 单条最新消息本身超预算时，裁剪该消息，但 type 标识保留在开头。
    """
    serialized = []
    for message in messages[-MAX_REWRITE_HISTORY_MESSAGES:]:
        content = message.content if isinstance(message.content, str) else str(message.content)
        serialized.append(f"{message.type}: {content}")

    kept = []
    total = 0
    for line in reversed(serialized):
        # 非首行需要多计 1 个换行字符；首行成本仅为行本身
        cost = len(line) if not kept else len(line) + 1
        if total + cost > MAX_REWRITE_HISTORY_CHARS:
            break
        kept.append(line)
        total += cost
    kept.reverse()
    if not kept and serialized:
        kept.append(serialized[-1][:MAX_REWRITE_HISTORY_CHARS])
    return "\n".join(kept)


def _session_id_from_config(config: RunnableConfig | None) -> str | None:
    """从 LangGraph 节点的 RunnableConfig 提取会话 ID（thread_id）。"""
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def prepare(state: ChatState, config: RunnableConfig | None = None) -> dict:
    question = state["question"]
    history = state.get("messages", [])
    if not any(isinstance(message, HumanMessage) for message in history):
        search_query = question
    else:
        history_text = _bounded_rewrite_history(history)
        prompt = (
            "请结合会话历史，把本轮追问改写成可独立检索的完整法律问题。"
            "只输出改写后的问题，不要解释。\n"
            f"会话历史：\n{history_text}\n本轮追问：{question}"
        )
        try:
            response = traced_invoke(
                get_chat_model(),
                prompt,
                session_id=_session_id_from_config(config),
                tags=["rewrite"],
            )
        except Exception as error:
            raise ExternalServiceError("生成服务调用失败") from error
        search_query = (response.content or question).strip()
    return {"messages": [HumanMessage(content=question)], "search_query": search_query}


def retrieve_node(state: ChatState, config: RunnableConfig | None = None) -> dict:
    return {
        "citations": retrieve(
            state["search_query"], top_k=3, session_id=_session_id_from_config(config)
        )
    }


def _model_tool_calls(
    question: str, hint: str = "", session_id: str | None = None
) -> list[dict]:
    """模型规划的白名单内工具调用；保留参数不完整的计算调用，供缺参追问使用。"""
    tools = [tool for tool in get_all_tools() if tool.name in LOCAL_TOOL_NAMES]
    if not tools:
        return []
    try:
        response = traced_invoke(
            get_router_model().bind_tools(tools),
            ROUTER_INSTRUCTION.format(hint=hint, question=question),
            session_id=session_id,
            tags=["router"],
        )
    except Exception as error:
        raise ExternalServiceError("工具路由服务调用失败") from error
    planned = []
    for call in getattr(response, "tool_calls", None) or []:
        name = call.get("name")
        args = call.get("args") or {}
        if name not in LOCAL_TOOL_NAMES:
            continue
        planned.append({"name": name, "args": args})
    return planned


def _cn_number(token: str) -> float | None:
    """受限中文数词转换（0~99999，支持 十/百/千/万 组合）；无法确认返回 None。"""
    if not token:
        return None
    total = section = number = 0
    for char in token:
        if char in CALC_CN_DIGITS:
            number = CALC_CN_DIGITS[char]
        elif char in CALC_CN_UNITS:
            unit = CALC_CN_UNITS[char]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    return float(total + section + number)


def _token_value(token: str) -> float | None:
    """把数字字面量转为数值；阿拉伯数字直接解析，中文数词走受限转换。"""
    if token[:1].isdigit():
        try:
            return float(token)
        except ValueError:
            return None
    if token == "半":
        return 0.5
    if token.endswith("半"):
        base = _cn_number(token[:-1])
        return None if base is None else base + 0.5
    return _cn_number(token)


def _adjacent_chars(text: str, index: int, count: int, forward: bool) -> str:
    """取 index 附近最多 count 个非空白字符（forward 为真时向右）。"""
    chars = []
    position = index
    while 0 <= position < len(text) and len(chars) < count:
        if not text[position].isspace():
            chars.append(text[position])
        position += 1 if forward else -1
    return "".join(chars if forward else list(reversed(chars)))


def _is_calc_kind_token(text: str, start: int, end: int) -> bool:
    """排除 2N / N+1 里的数字：它们是计算类型，不是工资或年限。"""
    if _adjacent_chars(text, end, 1, forward=True) in {"N", "n"}:
        return True
    before = _adjacent_chars(text, start - 1, 2, forward=False)
    return before == "N+" or before == "n+"


def _signed_token_value(text: str, start: int, end: int) -> float | None:
    value = _token_value(text[start:end])
    if value is None:
        return None
    before = _adjacent_chars(text, start - 1, 1, forward=False)
    return -value if before in {"-", "－", "负"} else value


def _fragment_prefix(text: str, start: int) -> str:
    """数值所在片段的片段内前缀：以硬分隔符切分，只保留当前片段中数值之前的部分。

    字段归属只看当前片段，已结束片段里的字段词不会抢占后面的数值（R3-01）。
    """
    segment_start = 0
    for sep in CALC_FRAGMENT_SEP_RE.finditer(text, 0, start):
        segment_start = sep.end()
    return text[segment_start:start]


def _money_field(text: str, start: int, end: int, following: str) -> str | None:
    """判断一个数值在工资类字段中的归属，只看当前片段内的字段词。

    - 明确属于赔偿、退款等其他金额的数值排除出所有工资依据；
    - 当地社平工资（社平/职工月平均/…）的字段词常含“工资”，按完整字段含义优先于个人工资；
    - 个人工资字段词（工资/月薪/…）；
    - 都没有字段线索时，带“元/块/万”的记为无法归属（unattributed），否则记 None。
    """
    prefix = _fragment_prefix(text, start)
    if any(anchor in prefix for anchor in CALC_OTHER_MONEY_ANCHORS):
        return "other_money"
    if any(anchor in prefix for anchor in CALC_SOCIAL_ANCHORS):
        return "social_average_wage"
    if any(anchor in prefix for anchor in CALC_WAGE_ANCHORS):
        return "monthly_wage"
    if following.startswith(("元", "块", "万")):
        return "unattributed"
    return None


def _scan_calc_literals(text: str) -> dict:
    """把用户原话里的数值按字段分组；其他金额与无法归属的金额单独记录。

    每个数值只归属一个组：紧跟 `年/个月` 的只作年限，带 `元/块/万` 的只作金额
    （再按当前片段字段词分流），其余才看年限字段词或记为裸数字（R3-01、R3-02）。
    """
    groups = {
        "monthly_wage": set(),
        "social_average_wage": set(),
        "years": set(),
        "unattributed": set(),
        "other_money": set(),
        "bare": set(),
    }
    for match in CALC_NUMBER_RE.finditer(text):
        start, end = match.start(), match.end()
        if _is_calc_kind_token(text, start, end):
            continue
        value = _signed_token_value(text, start, end)
        if value is None:
            continue
        following = _adjacent_chars(text, end, 2, forward=True)
        preceding = _adjacent_chars(text, start - 1, 5, forward=False)
        # 单位优先：紧跟时间单位的数值只作年限，绝不同时进入任何金额字段（R3-02）
        if following.startswith("年"):
            groups["years"].add(value + (0.5 if following[1:2] == "半" else 0.0))
            continue
        if following.startswith("个月"):
            groups["years"].add(value / 12)
            continue
        field = _money_field(text, start, end, following)
        if field == "other_money":
            groups["other_money"].add(value * (10000 if following.startswith("万") else 1))
            continue
        if field in ("monthly_wage", "social_average_wage"):
            groups[field].add(value * (10000 if following.startswith("万") else 1))
            continue
        if field == "unattributed":
            groups["unattributed"].add(value * (10000 if following.startswith("万") else 1))
            continue
        if any(anchor in preceding for anchor in CALC_YEARS_ANCHORS):
            groups["years"].add(value)
            continue
        groups["bare"].add(value)
    return groups


def _has_compound_duration(text: str) -> bool:
    """是否出现“N年M个月”这类复合年限表达。"""
    return bool(CALC_COMPOUND_YEARS_RE.search(text))


def _has_money_literal(scanned: dict) -> bool:
    """本轮原话是否出现任何金额字面量（含无法归属与其他金额）。"""
    return any(
        scanned.get(field)
        for field in ("monthly_wage", "social_average_wage", "unattributed", "other_money")
    )


def _field_literals(raw_question: str, missing_fields: list) -> dict:
    """各字段本轮可用的数值依据。

    - 个人月工资与当地社平工资各自只接受对应字段词上下文的数值，互不充当依据；
    - 带“元/万”但没有字段线索的金额只在字段含义唯一时可用：只剩一个待补工资字段，
      或原话只提到工资类字段（例如“工资说错了，是 25000 元”）；
    - 无单位的裸数字只在看起来像补参回复、且字段含义唯一时才绑定（疑问句不绑定）；
    - 明确属于赔偿、退款等其他金额的数值不进入任何字段依据。
    """
    groups = _scan_calc_literals(raw_question)
    wage = set(groups["monthly_wage"])
    years = set(groups["years"])
    targets = set()
    if any(anchor in raw_question for anchor in CALC_WAGE_ANCHORS):
        targets.add("monthly_wage")
    if any(anchor in raw_question for anchor in CALC_YEARS_ANCHORS):
        targets.add("years")
    if len(missing_fields) == 1:
        targets.add(missing_fields[0])
    if targets == {"monthly_wage"}:
        wage |= groups["unattributed"]
    if len(groups["bare"]) == 1 and not _is_calc_question(raw_question):
        if len(targets) == 1:
            target = targets.pop()
            (years if target == "years" else wage).update(groups["bare"])
    return {
        "monthly_wage": wage,
        "years": years,
        "social_average_wage": set(groups["social_average_wage"]),
    }


def _source_backed(number: float, allowed: set) -> bool:
    """该数值是否出现在本轮用户原话的对应字段依据里。"""
    return any(abs(number - candidate) < 1e-6 for candidate in allowed)


def _review_calc_args(args: dict, literals: dict) -> tuple[dict, list]:
    """逐字段核验模型参数。

    只有本轮原话里存在对应依据（单位匹配，或绑定到唯一待补字段）的字段才会被处理：
    正数被采用，零或负数视为用户对本字段的无效更正而被作废；没有依据的字段一律忽略，
    既不新增也不覆盖已确认值。kind 由模型依情形判断，不要求数字依据。
    """
    accepted = {}
    invalidated = []
    for field in CALC_FIELD_FIELDS:
        raw_value = args.get(field)
        if raw_value is None:
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            continue  # 非数值：不采用也不作废
        if not _source_backed(number, literals.get(field) or set()):
            continue
        if number > 0:
            accepted[field] = number
        else:
            invalidated.append(field)
    kind = args.get("kind")
    if isinstance(kind, str) and kind.upper() in CALC_KINDS:
        accepted["kind"] = kind.upper()
    return accepted, invalidated


def _confirmed_fields(context) -> dict:
    """当前任务已确认的字段；被无效更正作废的字段不会出现在这里。"""
    if not context:
        return {}
    fields = {
        field: context[field]
        for field in CALC_FIELD_FIELDS
        if context.get(field) is not None
    }
    if context.get("kind_explicit"):
        fields["kind"] = context.get("kind")
    return fields


def _calc_context_from(
    fields: dict, status: str, reply=None, invalid_fields=(), compound_years: bool = False
) -> dict:
    """按已确认字段构建计算上下文；每轮整轮重建，上一轮标记不会残留。"""
    return {
        "monthly_wage": fields.get("monthly_wage"),
        "years": fields.get("years"),
        "social_average_wage": fields.get("social_average_wage"),
        "kind": fields.get("kind") or "N",
        "kind_explicit": "kind" in fields,
        "status": status,
        "reply": reply,
        "invalid_fields": list(invalid_fields),
        "compound_years": bool(compound_years),
    }


def _calc_tool_args(context: dict) -> dict:
    """把已确认参数转成计算器调用参数；未明确说明的 kind 交给工具默认值。"""
    args = {
        "monthly_wage": context["monthly_wage"],
        "years": context["years"],
    }
    if context.get("kind_explicit"):
        args["kind"] = context["kind"]
    if context.get("social_average_wage") is not None:
        args["social_average_wage"] = context["social_average_wage"]
    return args


def _planned_calc(fields: dict, invalid_fields=()) -> tuple[list[dict], dict]:
    """参数齐全且本轮要计算：只规划一次调用，完成状态留到实际执行成功后写入。"""
    context = _calc_context_from(fields, CALC_STATUS_COMPUTING, None, invalid_fields)
    return [{"name": CALC_TOOL_NAME, "args": _calc_tool_args(context)}], context


def _known_params_text(context) -> str:
    if not context:
        return "无"
    parts = []
    if context.get("monthly_wage") is not None:
        parts.append(f"月工资 {context['monthly_wage']:g} 元")
    if context.get("years") is not None:
        parts.append(f"工作年限 {context['years']:g} 年")
    if not parts:
        return "无"
    parts.append(f"类型 {context.get('kind') or 'N'}")
    return "、".join(parts)


def _is_calc_cancel(text: str) -> bool:
    return any(marker in text for marker in CALC_CANCEL_MARKERS)


def _is_compute_request(text: str) -> bool:
    return any(marker in text for marker in CALC_COMPUTE_MARKERS)


def _is_calc_retry(text: str) -> bool:
    return any(marker in text for marker in CALC_RETRY_MARKERS)


def _is_calc_question(text: str) -> bool:
    return any(marker in text for marker in CALC_QUESTION_MARKERS)


def _is_new_calc_task(text: str) -> bool:
    return any(marker in text for marker in CALC_NEW_TASK_MARKERS)


def _starts_calc_task(text: str, has_kind: bool = False) -> bool:
    """能否凭本轮建立新的计算任务。

    要求路由模型确实按计算器语义给出了 `kind`（说明它把这句话判成计算请求），并且
    原话本身带有计算意图；否则普通法律追问（如“那补偿怎么算？”）不应变成参数追问。
    """
    if not has_kind:
        return False
    return _is_compute_request(text)


def _mentions_missing_field(text: str, missing_fields) -> bool:
    """本轮是否在问当前缺失字段本身（例如工资口径），用于保留待办并正常回答。"""
    for field in missing_fields:
        anchors = CALC_YEARS_ANCHORS if field == "years" else CALC_WAGE_ANCHORS
        if any(anchor in text for anchor in anchors):
            return True
    return False


def _invalid_literal_fields(literals: dict) -> list:
    """本轮原话里只给出零或负数的字段，用于提示需要有效数值。"""
    return [
        field
        for field in CALC_FIELD_FIELDS
        if literals.get(field) and all(value <= 0 for value in literals[field])
    ]


def _calc_without_plan(
    raw_question, previous, base, missing_before, literals, kind=None, has_money=False
) -> tuple[list, dict | None]:
    """本轮没有可采用的新字段值时的处理：补参、相关澄清、无效补充或换话题。"""
    if not previous:
        if _starts_calc_task(raw_question, kind is not None):
            # 明确要求计算但没有可核验的字段值：建立等待补参的任务
            fields = {**base, **({"kind": kind} if kind else {})}
            return [], _calc_context_from(
                fields, CALC_STATUS_AWAITING, "ask", _invalid_literal_fields(literals)
            )
        return [], None  # 没有进行中的任务：按普通问答继续
    if previous.get("status") != CALC_STATUS_AWAITING:
        if _is_calc_retry(raw_question) and all(
            base.get(field) is not None for field in CALC_REQUIRED_FIELDS
        ):
            # 明确要求重试：沿用仍有效的已确认参数（例如上一次执行失败之后）
            return _planned_calc(base)
        return [], None  # 已完成/失败任务后的普通追问：结束任务
    presented = [field for field in missing_before if literals.get(field)]
    if presented or has_money:
        # 用户给了金额或数值但无法归属：保留任务并说明缺什么、需要什么样的数值
        return [], _calc_context_from(
            base, CALC_STATUS_AWAITING, "ask", _invalid_literal_fields(literals)
        )
    if _mentions_missing_field(raw_question, missing_before):
        # 当前字段的相关澄清：保留任务，本轮按普通问答回答
        return [], _calc_context_from(base, CALC_STATUS_AWAITING)
    if any(literals.get(field) for field in CALC_FIELD_FIELDS):
        return [], _calc_context_from(
            base, CALC_STATUS_AWAITING, "ask", _invalid_literal_fields(literals)
        )
    return [], None  # 明确换话题：结束当前计算任务


def _plan_calculation_calls(
    raw_question: str, previous, session_id: str | None
) -> tuple[list[dict], dict | None]:
    """规划本地计算器调用，并给出本轮的计算任务状态。

    返回 (planned_calc_calls, calc_context)：calc_context 为 None 表示本轮没有进行中的
    计算任务（对话按普通问答继续）。处理顺序固定为：明确取消 → 新任务重置 →
    原话可证的无效更正与复合年限使旧值失效 → 核验模型新值 → 补齐、执行或保留待办。
    """
    scanned = _scan_calc_literals(raw_question)
    if _is_calc_cancel(raw_question) and not (
        scanned["monthly_wage"] or scanned["years"]
    ):
        # 明确取消：不依赖模型返回空调用、仅 kind 或旧参数，一律结束当前任务。
        return [], None

    reset = _is_new_calc_task(raw_question)
    base = {} if reset else _confirmed_fields(previous)
    # 原话可证的无效更正（零/负数）与复合年限：在调用模型之前就让旧值失效
    explicit_invalid = _invalid_literal_fields(_field_literals(raw_question, []))
    compound = _has_compound_duration(raw_question)
    if not reset:
        for field in explicit_invalid:
            base.pop(field, None)
        if compound:
            base.pop("years", None)

    missing_before = [field for field in CALC_REQUIRED_FIELDS if base.get(field) is None]
    literals = _field_literals(raw_question, missing_before)
    hint = (
        CALC_PENDING_HINT.format(known=_known_params_text(previous))
        if previous
        else CALC_HINT
    )
    calls = _model_tool_calls(raw_question, hint, session_id=session_id)
    calc_call = next((call for call in calls if call["name"] == CALC_TOOL_NAME), None)
    accepted, invalidated = (
        _review_calc_args(calc_call.get("args") or {}, literals)
        if calc_call
        else ({}, [])
    )
    if compound:
        # 复合年限本轮不作数：不接受任何片段值，也不把旧年限当有效值
        accepted.pop("years", None)
        invalidated = [field for field in invalidated if field != "years"]

    accepted_fields = {key: value for key, value in accepted.items() if key != "kind"}
    if not accepted_fields and not invalidated and not explicit_invalid and not compound:
        # 本轮没有任何可采用的新字段值：交给统一分流处理
        return _calc_without_plan(
            raw_question,
            previous,
            base,
            missing_before,
            literals,
            accepted.get("kind"),
            _has_money_literal(scanned),
        )

    fields = {**base, **accepted_fields}
    if "kind" in accepted:
        fields["kind"] = accepted["kind"]
    for field in [*explicit_invalid, *invalidated]:
        fields.pop(field, None)
    if compound:
        fields.pop("years", None)
    if any(fields.get(field) is None for field in CALC_REQUIRED_FIELDS):
        # 缺参或本轮无效更正：保留其余已确认字段并追问，绝不计算。
        invalid_note = [
            field
            for field in dict.fromkeys([*explicit_invalid, *invalidated])
            if field in CALC_FIELD_FIELDS
        ]
        return [], _calc_context_from(
            fields, CALC_STATUS_AWAITING, "ask", invalid_note, compound
        )

    previous_status = (previous or {}).get("status")
    if reset or previous is None or previous_status == CALC_STATUS_AWAITING:
        return _planned_calc(fields, invalidated)
    if _is_calc_retry(raw_question) or (accepted_fields and _is_compute_request(raw_question)):
        return _planned_calc(fields, invalidated)
    if accepted_fields:
        # 计算完成后只更正参数、未要求重算：本轮不重复调用计算器。
        return [], _calc_context_from(
            fields, previous_status or CALC_STATUS_DONE, "correction_ack", invalidated
        )
    return [], _calc_context_from(fields, previous_status or CALC_STATUS_DONE)


def route(state: ChatState, config: RunnableConfig | None = None) -> dict:
    search_query = state.get("search_query") or state.get("question") or ""
    # 计算参数只从用户本轮原话抽取：search_query 可能被追问改写模型加工过。
    raw_question = state.get("question") or search_query
    session_id = _session_id_from_config(config)
    previous = state.get("calc_context") or None
    awaiting = bool(previous) and previous.get("status") == CALC_STATUS_AWAITING

    # 本地计算器：命中计算关键词、正在等待补参，或对已有任务明确取消/重试时才进入。
    planned = []
    calc_context = None
    if (
        awaiting
        or any(keyword in search_query for keyword in CALC_KEYWORDS)
        or (previous is not None and (_is_calc_cancel(search_query) or _is_calc_retry(search_query)))
    ):
        planned, calc_context = _plan_calculation_calls(raw_question, previous, session_id)

    # 联网工具：只有显式开启 use_web_search 时才可能规划，且最多一个，由确定性规则决定。
    if state.get("use_web_search"):
        if any(keyword in search_query for keyword in STATUTE_KEYWORDS):
            planned = [*planned, {"name": "verify_statute_status", "args": {"query": search_query}}]
        else:
            planned = [*planned, {"name": "web_search", "args": {"query": search_query, "count": 5}}]

    return {
        "planned_tool_calls": planned,
        "web_sources": [],
        "tool_text": "",
        "calc_context": calc_context,
    }


def _has_usable_web_results(text: str) -> bool:
    """判断 MCP 联网工具返回是否有可用联网证据。

    满足全部条件才算“存在可用联网证据”：
    1. text 可解析为 JSON 对象；
    2. payload.ok 严格为 True；
    3. payload.results 是 list；
    4. 至少一个结果是 dict；
    5. 该结果的 title/site/date/summary 任一非空白，或 url 通过
       safe_web_source_url() 校验。
    只作证据是否足够的判断；非法 URL 不单独构成可用证据，
    但不破坏其他证据字段的价值。
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return False
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    for item in results:
        if not isinstance(item, dict):
            continue
        for field in ("title", "site", "date", "summary"):
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                return True
        if safe_web_source_url(item.get("url")):
            return True
    return False


def call_tools(state: ChatState) -> dict:
    planned = state.get("planned_tool_calls", [])
    question = str(state.get("question") or "")[:80]
    # 只有本轮确实计划调用联网工具时才初始化 MCP；
    # 纯本地计算器（如 calculate_labor_compensation）不触发 MCP 初始化。
    mcp_error = None
    if any((call.get("name") or "") in MCP_TOOL_NAMES for call in planned):
        with perf_span("tool", "mcp_init", question=question):
            mcp_error = ensure_mcp_tools()
    tools = {tool.name: tool for tool in get_all_tools()}
    calc_context = state.get("calc_context") or None
    calc_ok = None
    chunks = []
    sources = []
    for call in planned:
        name = call.get("name") or ""
        if name in MCP_TOOL_NAMES and mcp_error is not None:
            # 联网初始化失败：不调用对应工具，不产出 web_sources，
            # 用稳定、面向用户的提示把“本轮联网核验未完成”带入生成上下文。
            chunks.append(
                f"[工具 {name} 联网搜索初始化失败，本轮无法完成联网核验，请稍后重试。]"
            )
            continue
        tool = tools.get(name)
        if tool is None:
            chunks.append(f"[工具 {name} 本轮不可用]")
            if name == CALC_TOOL_NAME:
                calc_ok = False
            continue
        try:
            with perf_span("tool", name, question=question):
                result = run_async(tool.ainvoke(call.get("args") or {}))
        except Exception as error:
            chunks.append(f"[工具 {name} 调用失败：{type(error).__name__}]")
            if name == CALC_TOOL_NAME:
                calc_ok = False
            continue
        text = tool_result_text(result)
        if name in MCP_TOOL_NAMES and not _has_usable_web_results(text):
            # 联网调用成功但无可用证据：不伪装成已完成核验，
            # 用稳定、面向生成模型的标记标明资料不足。
            chunks.append(
                f"[工具 {name} 联网资料不足，本轮无法完成联网核验。]"
            )
            continue
        chunks.append(f"[工具 {name} 返回]\n{text}")
        sources.extend(web_sources_from_text(text))
        if name == CALC_TOOL_NAME:
            calc_ok = True
    payload = {"tool_text": "\n\n".join(chunks), "web_sources": sources}
    if calc_ok is not None and calc_context is not None:
        # 只有计算器实际执行成功才标记完成；失败保留上下文，供用户明确重试时复用。
        # 其他工具的成功或失败不参与该状态判定。
        payload["calc_context"] = {
            **calc_context,
            "status": CALC_STATUS_DONE if calc_ok else CALC_STATUS_FAILED,
            "reply": None,
        }
    return payload


def _clarify_text(context: dict) -> str:
    """缺参追问或更正确认的确定性文案。"""
    if context.get("reply") == "correction_ack":
        return CALC_CORRECTION_ACK.format(
            wage=f"{context['monthly_wage']:g}",
            years=f"{context['years']:g}",
            kind=context.get("kind") or "N",
        )
    parts = []
    if context.get("compound_years"):
        parts.append(CALC_CLARIFY_COMPOUND_YEARS)
    invalid = [
        field for field in (context.get("invalid_fields") or []) if field in FIELD_LABELS
    ]
    if invalid:
        parts.append(
            CALC_CLARIFY_INVALID.format(fields="、".join(FIELD_LABELS[field] for field in invalid))
        )
    missing_wage = context.get("monthly_wage") is None
    missing_years = context.get("years") is None
    if missing_wage and missing_years:
        parts.append(CALC_CLARIFY_BOTH)
    elif missing_wage:
        parts.append(CALC_CLARIFY_WAGE)
    else:
        parts.append(CALC_CLARIFY_YEARS)
    parts.append(
        CALC_ASSUMPTION.format(kind=KIND_LABELS.get(context.get("kind") or "N", "N"))
    )
    return "".join(parts)


def clarify(state: ChatState, config: RunnableConfig | None = None) -> dict:
    """缺参追问：输出确定性文案，不调用生成模型，也不带本轮检索引用。"""
    text = _clarify_text(state.get("calc_context") or {})
    return {
        "answer": text,
        "citations": [],
        "web_sources": [],
        "tool_text": "",
        "messages": [AIMessage(content=text, additional_kwargs={"citations": []})],
    }


def _next_node(state: ChatState) -> str:
    context = state.get("calc_context") or {}
    if context.get("reply") in {"ask", "correction_ack"}:
        return "clarify"
    return "tools" if state.get("planned_tool_calls") else "generate"


def generate(state: ChatState, config: RunnableConfig | None = None) -> dict:
    prompt = build_prompt_with_tools(
        state["search_query"], state["citations"], state.get("tool_text", "")
    )
    try:
        response = traced_invoke(
            get_chat_model(),
            prompt,
            session_id=_session_id_from_config(config),
            tags=["generate"],
        )
    except Exception as error:
        raise ExternalServiceError("生成服务调用失败") from error
    answer = response.content or "现有资料无法回答"
    citations = [
        {
            "source": item.node.metadata["source"],
            "article": item.node.metadata["article"],
            "score": float(item.score),
        }
        for item in state["citations"]
    ]
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer, additional_kwargs={"citations": citations})],
    }


def _perf_node(name: str, fn):
    """给图节点包一层计时（仅 LAWRAG_PERF_OBS_FILE 设置时写观测记录）。

    只做 try/finally 计时，参数、返回值与异常传播完全不变；
    记录按层级区分（node），不与节点内 LLM/工具/检索调用记录混加。
    """
    accepts_config = len(signature(fn).parameters) >= 2

    @wraps(fn)
    def wrapper(state, config: RunnableConfig | None = None):
        if not perf_enabled():
            return fn(state, config) if accepts_config else fn(state)
        question = str(state.get("question") or "")[:80]
        with perf_span(
            "node",
            name,
            session_id=_session_id_from_config(config),
            question=question,
        ):
            return fn(state, config) if accepts_config else fn(state)

    return wrapper


@lru_cache(maxsize=1)
def get_graph():
    builder = StateGraph(ChatState)
    builder.add_node("prepare", _perf_node("prepare", prepare))
    builder.add_node("retrieve", _perf_node("retrieve", retrieve_node))
    builder.add_node("route", _perf_node("route", route))
    builder.add_node("tools", _perf_node("tools", call_tools))
    builder.add_node("clarify", _perf_node("clarify", clarify))
    builder.add_node("generate", _perf_node("generate", generate))
    builder.add_edge(START, "prepare")
    # 第三阶段第二步（clarify 跳过检索）：route 提前到 retrieve 之前。
    # - 本轮路由判 clarify（缺参追问/更正确认）：直接 clarify → END，不执行向量/BM25/重排；
    # - 其他分支：retrieve 检索一次，再进 tools 或 generate。
    # 分支判断一律复用 _next_node（读取本轮 route 写入的状态），每轮 route 只执行一次；
    # 不按上一轮 awaiting 状态提前跳过检索。
    builder.add_edge("prepare", "route")
    builder.add_conditional_edges(
        "route",
        _next_node,
        {"clarify": "clarify", "tools": "retrieve", "generate": "retrieve"},
    )
    builder.add_conditional_edges(
        "retrieve",
        _next_node,
        {"clarify": "clarify", "tools": "tools", "generate": "generate"},
    )
    builder.add_edge("tools", "generate")
    builder.add_edge("clarify", END)
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=SqliteSaver(get_checkpointer_connection()))


def graph_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def run_turn(
    session_id: str, question: str, use_web_search: bool = False
) -> tuple[str, list]:
    result = get_graph().invoke(
        {"question": question, "use_web_search": use_web_search},
        config=graph_config(session_id),
    )
    return result["answer"], result["citations"]


def stream_turn(session_id: str, question: str, use_web_search: bool = False):
    """逐字流式执行一轮对话，产出 token 事件与结尾的 done 事件。"""
    config = graph_config(session_id)
    graph = get_graph()
    emitted = False
    for chunk, metadata in graph.stream(
        {"question": question, "use_web_search": use_web_search},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "generate":
            continue
        # generate 节点返回的整条 AIMessage 会被重复发射一次（新 id 挡不住按 id 去重），只接受流式 chunk（E-034 / E-032 复发）
        if not isinstance(chunk, AIMessageChunk):
            continue
        text = chunk.content if isinstance(chunk.content, str) else ""
        if not text:
            continue
        emitted = True
        yield {"type": "token", "text": text}
    snapshot = graph.get_state(config)
    answer = snapshot.values.get("answer") or ""
    # 缺参追问等确定性回答不经过生成模型，没有流式 chunk，需补发整条文本避免页面空白。
    if not emitted and answer:
        yield {"type": "token", "text": answer}
    yield {
        "type": "done",
        "citations": snapshot.values.get("citations", []),
        "web_sources": snapshot.values.get("web_sources", []),
    }


def delete_session_state(session_id: str) -> None:
    """清除该会话在 checkpointer 中的消息历史；sessions 元数据行由 app.db 负责。"""
    get_graph().checkpointer.delete_thread(session_id)
