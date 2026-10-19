"""LawRAG 回答质量与工具路由评测 CLI（R1 返工版）。

四种模式：
- validate：离线校验题集（结构、ID、分类数量、标注完整性、与旧 55 题重复），不访问网络。
- collect ：真实采集，复用现有 LangGraph 编排（app/graph.py），逐轮落盘 JSONL。
- rescore ：显式转换：把既有运行的采集结果转换到新目录并记录来源核验，用于修正规则重算。
- report  ：离线汇总，从已保存结果与评分记录生成报告，不调用模型。

R1 返工要点（对应 outputs/STAGE1_ACCEPTANCE_20261009.md F-01～F-08）：
- F-01 每个场景每次 attempt 使用独立会话（attempt + 随机 nonce）；按完整场景 attempt 组选取结果，
  不跨 attempt 拼接轮次；最新 attempt 不完整时如实显示不完整。
- F-02 复核记录绑定冻结题集、场景、轮次、attempt 与结果内容 hash；评分项完整性对照冻结定义校验，
  缺项/重复/替换/无依据 N/A 与绑定失配一律进入待答案复核；报告不自动修补旧评分。
- F-03 采集时冻结题集快照并按 attempt 记录 provenance，后续采集只追加不覆盖；报告只读冻结快照，
  版本不匹配时明确拒绝。
- F-04 needs_legal_review 标注的场景评分单列为“待标注复核”，不进入可信 PASS/FAIL；
  可保留明确标识的诊断试算。
- F-05 适用总数来自冻结定义；状态互斥且满足
  适用总数 = 通过 + 失败 + 运行错误 + 未采集 + 待标注复核 + 待答案复核；N/A 单列。
  graph 场景完成指标仅含 38 个场景，route_only 独立报告；任一必要轮次报错/缺失时场景不能 PASS。
- F-06 拒答指标分为“应拒答/需说明不足场景的处理”与“可答题场景的错误拒答”，适用性由冻结
  required_points 推导，不做场景 ID 特判，不以关键词程序替代语义判断（判定来自复核记录）。
- F-07 引用审核分别报告已提取需证据结论总数、已审核数、支持数、不支持数、待审数与
  结论提取完整性未确认的回答数；supported=None 计为待审；空结论列表须由审核者确认才算“无此类结论”。
- F-08 完成评分逐项对应冻结 completion_criteria；必要条件 FAIL/待审时整体不得 PASS；
  完成 PASS 与轮次 FAIL 并存时必须显式解释两者关系，否则评分无效。

R2 返工要点（对应 outputs/STAGE1_R1_ACCEPTANCE_20261009.md F-R1-01～03 与 A-01～02）：
- R2-01 结果身份按记录当前内容重算 hash，校验“保存 hash—实际内容 hash—复核绑定 hash”三方一致；
  内容失配进入无效/待复核，不沿用旧 PASS，不自动更新旧 hash；缺 hash 历史记录仅经 rescore 显式
  转换（记录来源）补算，失配记录不重新包装为可信记录。
- R2-02 评分前核验冻结文件当前字节 hash 与 manifest / conversion 绑定身份（原始题集文件 hash 与
  规范化冻结 hash 是不同字段，不判等）；快照身份失配拒绝生成报告（不改写证据、不覆盖已有报告）；
  仅 reviews 绑定失配时全部复核失效并写明原因。
- R2-03 needs_legal_review 题的引用结论全部分流为“待标注复核”（初评 True/False/None 均不进可信
  已审/支持/不支持，仅留诊断计数）；可核算关系：已提取 = 可信已审 + 可信标注下待答案复核 +
  待标注复核，可信已审 = 支持 + 不支持；覆盖率分母不排除待标注结论。

设计约束（见 docs/EVALUATION.md）：
- 预期答案、法条、评分规则只用于评分，绝不进入被测应用输入（仅传 question / use_web_search / session_id）。
- 端到端场景复用实际问答图，不另写评测专用问答链。
- 区分工具规划（planned_tool_calls）与实际执行（tool_text），不把规划当执行证据。
- 数据隔离：LAWRAG_DB_PATH 指向运行目录、Chroma 使用运行目录内的隔离副本、默认关闭 Langfuse。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CASES_PATH = ROOT / "tests" / "evaluation_cases.json"
REGRESSION_PATH = ROOT / "tests" / "regression_questions.json"
EVAL_ROOT = ROOT / "logs" / "evaluation"

# 真实业务目录（仅作为索引复制来源与污染监测对象；评测不得写入这些目录）
CHROMA_SOURCE_DIR = ROOT / "chroma_db"
REAL_DATA_DIR = ROOT / "data"
REAL_KB_DIR = ROOT / "kb"

CATEGORIES = ("library_qa", "boundary", "tool_routing", "multi_turn")
EXECUTION_SCOPES = ("graph", "route_only")
ANNOTATION_STATUSES = ("executor_annotated", "needs_legal_review")

CATEGORY_TOTALS = {"library_qa": 16, "boundary": 8, "tool_routing": 8, "multi_turn": 8}
BOUNDARY_SUBCATEGORY_TOTALS = {"out_of_scope": 4, "insufficient_evidence": 4}
TOOL_SUBCATEGORY_TOTALS = {"calc_complete": 4, "calc_missing_params": 2, "web_route": 2}
MULTI_TURN_SUBCATEGORY_TOTALS = {"reference": 2, "ellipsis": 2, "fact_correction": 2, "topic_switch": 2}
MISSING_PARAM_KEYS = {"years", "monthly_wage"}
NUMERIC_TOLERANCE = 0.01
SCENARIO_KEYS = {
    "id",
    "category",
    "subcategory",
    "execution_scope",
    "annotation_status",
    "turns",
    "completion_criteria",
    "evidence",
    "difference_from_regression",
}
TURN_KEYS = {
    "question",
    "use_web_search",
    "expected_articles",
    "required_points",
    "forbidden_claims",
    "expected_tools",
}
ARTICLE_LABEL_PATTERN = re.compile(r"^.+ 第[零〇一二三四五六七八九十百千0-9]+条$")

# 冻结 gold 中表示“应拒答/需说明不足”的措辞（仅用于从冻结定义推导指标适用性，
# 不用于对回答文本做语义判定；语义判定一律来自人工复核记录）。
REFUSAL_EXPECTATION_MARKERS = (
    "无法回答",
    "无法算出",
    "无法计算",
    "无法评估",
    "无法认定",
    "资料不足",
    "证据不足",
    "事实不足",
    "信息不足",
    "缺少",
    "缺失",
)

# 参与结果身份 hash 的字段（覆盖影响评分的答案、引用、工具规划及相关输出）。
RESULT_HASH_FIELDS = (
    "status",
    "question",
    "search_query",
    "answer",
    "citations",
    "planned_tool_calls",
    "tool_text",
    "web_sources",
    "error_type",
    "error_message",
)

RESULT_STATUSES = (
    "PASS",
    "FAIL",
    "ERROR",
    "NOT_VERIFIED",
    "PENDING_ANNOTATION",
    "PENDING_REVIEW",
)

REVIEW_FORMAT = "lawrag-quality-reviews/2.0"


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    """尽力取得当前 Git HEAD；拿不到时返回 unknown（测试环境不依赖 Git）。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def normalize_question(text: str) -> str:
    """去空白与常见标点后的小写化文本，用于明显重复检测。"""
    stripped = re.sub(r"[\s，。？！、,.?:;；:（）()\[\]《》\"'“”‘’]", "", str(text))
    return stripped.lower()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cases(path: Path = CASES_PATH) -> dict:
    return load_json(path)


def load_regression_questions(path: Path = REGRESSION_PATH) -> list:
    return load_json(path)


def load_kb_articles() -> dict:
    """读取 kb/ 法条原文，返回 {(source, article): text}。

    复用 app/ingest.py 的解析逻辑（与入库同源），只读 kb/，不访问网络。
    """
    from app.ingest import build_article_nodes

    articles = {}
    for node in build_article_nodes():
        key = (node.metadata["source"], node.metadata["article"])
        articles[key] = node.text
    return articles


def citation_label(citation) -> str:
    """把检索结果条目格式化为“法律名 第X条”标签（与旧回归口径一致）。"""
    if isinstance(citation, dict):
        source = citation.get("source") or ""
        article = citation.get("article") or ""
    else:
        metadata = getattr(getattr(citation, "node", None), "metadata", {}) or {}
        source = metadata.get("source") or ""
        article = metadata.get("article") or ""
    return f"{source} {article}".strip()


def cases_content_sha256(cases: dict) -> str:
    """题集内容 hash（规范 JSON：排序键、UTF-8、LF），用于冻结快照身份与版本比对。"""
    payload = json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_result_hash(record: dict) -> str:
    """结果身份 hash：覆盖影响评分的答案、引用、工具规划及相关输出。

    对参与字段做规范化 JSON 序列化（排序键、去空白差异）后取 SHA-256；
    缺失字段按 null 参与计算，避免“字段缺失”与“字段为空”被误认为同一结果。
    """
    payload = {field: record.get(field) for field in RESULT_HASH_FIELDS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def result_identity_problems(record: dict) -> list[str]:
    """R2-01：按记录当前内容重算 hash，校验“保存 hash—实际内容 hash”一致。

    内容失配必须进入无效/待复核状态，不得沿用旧 PASS，也不得自动更新旧 hash 掩盖失配；
    缺少保存 hash 的历史记录只能经 rescore 显式转换（记录来源）后复用评分。
    """
    actual_hash = compute_result_hash(record)
    saved_hash = record.get("result_hash")
    if saved_hash is None:
        return [
            "缺少保存的结果 hash：历史记录须经 rescore 显式转换补算并记录来源后才能复用评分"
        ]
    if saved_hash != actual_hash:
        return [
            f"内容失配：保存 hash {saved_hash} 与当前内容重算 hash {actual_hash} 不一致"
            "（记录内容已变化，旧评分不得沿用，也不得自动更新 hash 掩盖失配）"
        ]
    return []


# ---------------------------------------------------------------------------
# 离线校验
# ---------------------------------------------------------------------------


def _is_obvious_duplicate(left: str, right: str) -> bool:
    a, b = normalize_question(left), normalize_question(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 8 and shorter in longer


def validate_case_set(
    cases: dict,
    regression_questions: list | None = None,
    kb_articles: dict | None = None,
) -> list[str]:
    """校验题集，返回错误列表（空列表表示通过）。全程离线。"""
    errors: list[str] = []
    if not isinstance(cases, dict):
        return ["题集顶层必须是对象（含 version 与 scenarios）"]
    if not cases.get("version"):
        errors.append("缺少 version")
    scenarios = cases.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return [*errors, "缺少 scenarios 列表"]

    seen_ids: set[str] = set()
    category_counts: Counter = Counter()
    subcategory_counts: dict[str, Counter] = {key: Counter() for key in CATEGORIES}
    library_law_counts: Counter = Counter()
    questions_seen: dict[str, str] = {}
    total_turns = {"graph": 0, "route_only": 0}
    two_turn_scenarios = 0

    for index, scenario in enumerate(scenarios):
        prefix = f"scenarios[{index}]"
        if not isinstance(scenario, dict):
            errors.append(f"{prefix} 必须是对象")
            continue

        missing = SCENARIO_KEYS - set(scenario)
        if missing:
            errors.append(f"{prefix} 缺少字段：{sorted(missing)}")
        scenario_id = str(scenario.get("id") or "").strip()
        if not scenario_id:
            errors.append(f"{prefix} 缺少 id")
        elif scenario_id in seen_ids:
            errors.append(f"重复的场景 ID：{scenario_id}")
        else:
            seen_ids.add(scenario_id)
        label = f"{prefix}({scenario_id or '?'})"

        category = scenario.get("category")
        if category not in CATEGORIES:
            errors.append(f"{label} category 非法：{category}")
        else:
            category_counts[category] += 1
            subcategory_counts[category][scenario.get("subcategory")] += 1

        if scenario.get("annotation_status") not in ANNOTATION_STATUSES:
            errors.append(f"{label} annotation_status 非法：{scenario.get('annotation_status')}")
        if scenario.get("annotation_status") == "needs_legal_review" and not str(
            scenario.get("annotation_note") or ""
        ).strip():
            errors.append(f"{label} needs_legal_review 必须提供 annotation_note")

        if not str(scenario.get("difference_from_regression") or "").strip():
            errors.append(f"{label} 缺少 difference_from_regression")

        criteria = scenario.get("completion_criteria")
        if not isinstance(criteria, list) or not criteria or not all(
            str(item or "").strip() for item in criteria
        ):
            errors.append(f"{label} completion_criteria 必须是非空字符串列表")

        evidence = scenario.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{label} evidence 必须是非空列表")
        else:
            for ev_index, entry in enumerate(evidence):
                ev_label = f"{label}.evidence[{ev_index}]"
                if not isinstance(entry, dict):
                    errors.append(f"{ev_label} 必须是对象")
                    continue
                if str(entry.get("note") or "").strip():
                    continue
                source = str(entry.get("source") or "")
                article = str(entry.get("article") or "")
                quote = str(entry.get("quote") or "")
                if not (source and article and quote.strip()):
                    errors.append(f"{ev_label} 必须提供 note 或 source+article+quote")
                elif kb_articles is not None:
                    text = kb_articles.get((source, article))
                    if text is None:
                        errors.append(f"{ev_label} 来源条文不存在于 kb/：{source} {article}")
                    elif quote not in text:
                        errors.append(f"{ev_label} quote 不是 kb/ 原文子串：{source} {article}")

        turns = scenario.get("turns")
        expected_turn_count = 2 if category == "multi_turn" else 1
        if category == "multi_turn" and isinstance(turns, list) and len(turns) == 2:
            two_turn_scenarios += 1
        if not isinstance(turns, list) or len(turns) != expected_turn_count:
            errors.append(f"{label} turns 数量应为 {expected_turn_count}，实际：{type(turns).__name__}/{len(turns) if isinstance(turns, list) else '?'}")
            continue

        scope = scenario.get("execution_scope")
        if scope not in EXECUTION_SCOPES:
            errors.append(f"{label} execution_scope 非法：{scope}")
        is_route_only = scope == "route_only"
        if is_route_only != (scenario.get("subcategory") == "web_route"):
            errors.append(f"{label} execution_scope=route_only 必须与 subcategory=web_route 对应")
        total_turns["route_only" if is_route_only else "graph"] += len(turns)

        scenario_laws: set[str] = set()
        for turn_index, turn in enumerate(turns):
            turn_label = f"{label}.turns[{turn_index}]"
            if not isinstance(turn, dict):
                errors.append(f"{turn_label} 必须是对象")
                continue
            missing_turn_keys = TURN_KEYS - set(turn)
            if missing_turn_keys:
                errors.append(f"{turn_label} 缺少字段：{sorted(missing_turn_keys)}")
            question = str(turn.get("question") or "").strip()
            if not question:
                errors.append(f"{turn_label} question 不能为空")
            else:
                normalized = normalize_question(question)
                if normalized in questions_seen:
                    errors.append(
                        f"{turn_label} 问题与 {questions_seen[normalized]} 重复"
                    )
                else:
                    questions_seen[normalized] = turn_label
            if not isinstance(turn.get("use_web_search"), bool):
                errors.append(f"{turn_label} use_web_search 必须是布尔值")
            elif turn["use_web_search"] != is_route_only:
                errors.append(
                    f"{turn_label} use_web_search 应为 {is_route_only}（仅 route_only 场景允许联网）"
                )

            expected_articles = turn.get("expected_articles")
            if not isinstance(expected_articles, list):
                errors.append(f"{turn_label} expected_articles 必须是列表")
            else:
                for article_label in expected_articles:
                    if not isinstance(article_label, str) or not ARTICLE_LABEL_PATTERN.match(article_label):
                        errors.append(f"{turn_label} expected_articles 格式非法：{article_label}")
                        continue
                    law_name, article = article_label.rsplit(" ", 1)
                    scenario_laws.add(law_name)
                    if kb_articles is not None and (law_name, article) not in kb_articles:
                        errors.append(f"{turn_label} 目标法条不存在于 kb/：{article_label}")
                if is_route_only and expected_articles:
                    errors.append(f"{turn_label} route_only 场景不得设置 expected_articles")

            required_points = turn.get("required_points")
            forbidden_claims = turn.get("forbidden_claims")
            for field_name, values in (
                ("required_points", required_points),
                ("forbidden_claims", forbidden_claims),
            ):
                if not isinstance(values, list) or not all(
                    isinstance(item, str) and item.strip() for item in values
                ):
                    errors.append(f"{turn_label} {field_name} 必须是字符串列表")
                elif not values and not is_route_only:
                    errors.append(f"{turn_label} {field_name} 不得为空（route_only 除外）")

            expected_tools = turn.get("expected_tools")
            if not isinstance(expected_tools, list):
                errors.append(f"{turn_label} expected_tools 必须是列表")
            else:
                for tool_entry in expected_tools:
                    if not isinstance(tool_entry, dict) or not str(tool_entry.get("name") or "").strip():
                        errors.append(f"{turn_label} expected_tools 项必须含 name")
                        continue
                    args = tool_entry.get("args", {})
                    if not isinstance(args, dict):
                        errors.append(f"{turn_label} expected_tools 项的 args 必须是对象")

            missing_params = turn.get("missing_params")
            if missing_params is not None:
                if scenario.get("subcategory") != "calc_missing_params":
                    errors.append(f"{turn_label} 只有 calc_missing_params 场景可设置 missing_params")
                elif not isinstance(missing_params, list) or not missing_params or not set(
                    missing_params
                ) <= MISSING_PARAM_KEYS:
                    errors.append(f"{turn_label} missing_params 非法：{missing_params}")
                elif expected_tools not in ([], None):
                    errors.append(f"{turn_label} 缺参场景不得预期工具调用")
            if scenario.get("subcategory") == "calc_missing_params" and missing_params is None:
                errors.append(f"{turn_label} calc_missing_params 场景必须声明 missing_params")

        if category == "library_qa":
            if len(scenario_laws) != 1:
                errors.append(f"{label} 库内问答场景应归属唯一法律，实际：{sorted(scenario_laws)}")
            else:
                library_law_counts[next(iter(scenario_laws))] += 1

    for category, expected_total in CATEGORY_TOTALS.items():
        if category_counts[category] != expected_total:
            errors.append(f"分类 {category} 数量应为 {expected_total}，实际 {category_counts[category]}")
    for subcategory, expected_total in BOUNDARY_SUBCATEGORY_TOTALS.items():
        if subcategory_counts["boundary"][subcategory] != expected_total:
            errors.append(f"boundary/{subcategory} 数量应为 {expected_total}，实际 {subcategory_counts['boundary'][subcategory]}")
    for subcategory, expected_total in TOOL_SUBCATEGORY_TOTALS.items():
        if subcategory_counts["tool_routing"][subcategory] != expected_total:
            errors.append(f"tool_routing/{subcategory} 数量应为 {expected_total}，实际 {subcategory_counts['tool_routing'][subcategory]}")
    for subcategory, expected_total in MULTI_TURN_SUBCATEGORY_TOTALS.items():
        if subcategory_counts["multi_turn"][subcategory] != expected_total:
            errors.append(f"multi_turn/{subcategory} 数量应为 {expected_total}，实际 {subcategory_counts['multi_turn'][subcategory]}")
    if two_turn_scenarios != 8:
        errors.append(f"两轮场景应为 8 个，实际 {two_turn_scenarios}")
    if total_turns["graph"] != 46:
        errors.append(f"问答图执行轮数应为 46，实际 {total_turns['graph']}")
    if total_turns["route_only"] != 2:
        errors.append(f"route_only 轮数应为 2，实际 {total_turns['route_only']}")
    if len(library_law_counts) != 2 or any(count != 8 for count in library_law_counts.values()):
        errors.append(f"库内问答应两部法律各 8 题，实际：{dict(library_law_counts)}")

    if regression_questions is not None:
        old_questions = [str(item.get("question") or "") for item in regression_questions]
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            for turn_index, turn in enumerate(scenario.get("turns") or []):
                if not isinstance(turn, dict):
                    continue
                question = str(turn.get("question") or "")
                for old_index, old_question in enumerate(old_questions, start=1):
                    if _is_obvious_duplicate(question, old_question):
                        errors.append(
                            f"{scenario.get('id')}.turns[{turn_index}] 与旧题 {old_index} 明显重复"
                        )
    return errors


# ---------------------------------------------------------------------------
# 自动评分（确定性部分）
# ---------------------------------------------------------------------------


def _norm_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[元年月天个\s,，]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _norm_kind(value) -> str:
    return str(value).strip().upper().replace("＋", "+").replace(" ", "")


def score_retrieval_turn(turn: dict, record: dict) -> dict:
    """Hit@3：前三条中至少一条命中允许法条。无目标/仅路由场景不进入分母。"""
    expected = list(turn.get("expected_articles") or [])
    if record.get("execution_scope") == "route_only":
        return {"applicable": False, "result": "N/A", "reason": "route_only 不计入检索分母"}
    if not expected:
        return {"applicable": False, "result": "N/A", "reason": "无检索目标题"}
    if record.get("status") != "ok":
        return {"applicable": True, "result": "ERROR", "reason": "采集运行错误"}
    top3 = [citation_label(item) for item in (record.get("citations") or [])][:3]
    hit = [label for label in expected if label in top3]
    missing = [label for label in expected if label not in top3]
    return {
        "applicable": True,
        "result": "PASS" if hit else "FAIL",
        "top3": top3,
        "hit": hit,
        "missing": missing,
    }


def _match_tool_args(expected_args: dict, actual_args: dict, question: str, tool_name: str) -> list[str]:
    problems = []
    for key, expected_value in expected_args.items():
        actual_value = actual_args.get(key)
        if actual_value is None and key == "kind" and tool_name == "calculate_labor_compensation":
            actual_value = "N"  # 工具默认值
        if isinstance(expected_value, dict) and expected_value.get("match") == "question":
            if str(actual_value or "").strip() != question.strip():
                problems.append(f"参数 {key} 应取自本轮问题，实际：{actual_value!r}")
            continue
        if key == "kind":
            if _norm_kind(actual_value) != _norm_kind(expected_value):
                problems.append(f"参数 kind 期望 {expected_value}，实际 {actual_value!r}")
            continue
        expected_number = _norm_number(expected_value)
        actual_number = _norm_number(actual_value)
        if expected_number is None or actual_number is None:
            problems.append(f"参数 {key} 期望 {expected_value!r}，实际 {actual_value!r}（无法归一化为数值）")
        elif abs(expected_number - actual_number) > NUMERIC_TOLERANCE:
            problems.append(f"参数 {key} 期望 {expected_number:g}，实际 {actual_number:g}")
    for key in sorted(set(actual_args) - set(expected_args)):
        problems.append(f"编造/多余参数 {key}={actual_args[key]!r}")
    return problems


def score_tool_selection(turn: dict, record: dict) -> dict:
    """工具选择：规划工具与预期一致（含预期不调用的场景）。"""
    expected = list(turn.get("expected_tools") or [])
    if record.get("status") != "ok":
        return {"applicable": True, "result": "ERROR", "reason": "采集运行错误"}
    planned = list(record.get("planned_tool_calls") or [])
    planned_names = [str(call.get("name") or "") for call in planned]
    if not expected:
        if planned:
            return {
                "applicable": True,
                "result": "FAIL",
                "reason": f"预期不调用工具，实际规划：{planned_names}",
                "planned": planned_names,
            }
        return {"applicable": True, "result": "PASS", "reason": "预期不调用且未规划工具", "planned": []}
    expected_names = sorted(str(entry.get("name") or "") for entry in expected)
    if sorted(planned_names) != expected_names:
        return {
            "applicable": True,
            "result": "FAIL",
            "reason": f"预期工具 {expected_names}，实际规划 {sorted(planned_names)}",
            "planned": planned_names,
        }
    return {"applicable": True, "result": "PASS", "reason": "规划工具与预期一致", "planned": planned_names}


def score_tool_params(turn: dict, record: dict) -> dict:
    """参数正确率：关键参数全部正确才算通过；缺失调用、编造参数均不正确。"""
    expected = [entry for entry in (turn.get("expected_tools") or []) if entry.get("args")]
    if not expected:
        return {"applicable": False, "result": "N/A", "reason": "本场景不考核工具参数"}
    if record.get("status") != "ok":
        return {"applicable": True, "result": "ERROR", "reason": "采集运行错误"}
    planned = list(record.get("planned_tool_calls") or [])
    problems: list[str] = []
    for entry in expected:
        name = str(entry.get("name") or "")
        matches = [call for call in planned if str(call.get("name") or "") == name]
        if not matches:
            problems.append(f"缺少工具调用：{name}")
            continue
        if len(matches) > 1:
            problems.append(f"工具 {name} 被规划 {len(matches)} 次")
            continue
        actual_args = matches[0].get("args") or {}
        if not isinstance(actual_args, dict):
            problems.append(f"工具 {name} 参数格式非法：{actual_args!r}")
            continue
        problems.extend(
            _match_tool_args(entry.get("args") or {}, actual_args, str(record.get("question") or ""), name)
        )
    return {
        "applicable": True,
        "result": "PASS" if not problems else "FAIL",
        "reason": "；".join(problems) if problems else "关键参数全部正确",
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# 真实采集
# ---------------------------------------------------------------------------


def snapshot_dir(path: Path) -> dict:
    """对真实业务目录做只读快照（文件名、大小、mtime、SHA-256）。目录不存在视为空集合。"""
    entries = {}
    if not path.exists():
        return {"exists": False, "entries": {}}
    for child in sorted(path.rglob("*")):
        if child.is_file():
            relative = child.relative_to(path).as_posix()
            stat = child.stat()
            entries[relative] = {
                "size": stat.st_size,
                "mtime": round(stat.st_mtime, 3),
                "sha256": sha256_file(child),
            }
    return {"exists": True, "entries": entries}


def _real_dirs_snapshot() -> dict:
    return {
        "chroma_db": snapshot_dir(CHROMA_SOURCE_DIR),
        "data": snapshot_dir(REAL_DATA_DIR),
        "kb": snapshot_dir(REAL_KB_DIR),
    }


def setup_isolation(run_dir: Path) -> dict:
    """在评测进程内建立数据隔离。

    - LAWRAG_DB_PATH 指向运行目录（必须在导入 app 前设置）；
    - Langfuse 上报默认关闭（进程内清空密钥，不改用户系统环境）；
    - 把当前索引复制到运行目录（复制前后比对源快照，检测并发入库）；
    - app.db.DATABASE_PATH / app.ask.CHROMA_DIR 改绑到隔离路径并清空各缓存。
    返回隔离信息（写入 manifest）。失败抛 RuntimeError，真实评测标为未执行。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LAWRAG_DB_PATH"] = str(run_dir / "lawrag.sqlite3")
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""

    source_before = snapshot_dir(CHROMA_SOURCE_DIR)
    if not source_before["exists"]:
        raise RuntimeError("未找到真实 chroma_db，无法建立隔离索引副本（真实评测未执行）")
    index_copy = run_dir / "chroma_copy"
    if index_copy.exists():
        shutil.rmtree(index_copy)
    shutil.copytree(CHROMA_SOURCE_DIR, index_copy)
    source_after = snapshot_dir(CHROMA_SOURCE_DIR)
    index_copy_consistent = source_before["entries"] == source_after["entries"]
    if not index_copy_consistent:
        shutil.rmtree(index_copy, ignore_errors=True)
        raise RuntimeError("复制索引期间源目录发生变化，疑似并发入库，无法取得一致副本（真实评测未执行）")

    import app.ask
    import app.config
    import app.db
    import app.graph

    app.db.DATABASE_PATH = Path(os.environ["LAWRAG_DB_PATH"])
    app.ask.CHROMA_DIR = index_copy
    app.config.CHROMA_DIR = index_copy
    app.graph.get_graph.cache_clear()
    app.db.get_connection.cache_clear()
    app.db.get_checkpointer_connection.cache_clear()
    app.ask.load_index.cache_clear()
    app.ask.get_bm25_index.cache_clear()

    return {
        "db_path": str(run_dir / "lawrag.sqlite3"),
        "chroma_source": str(CHROMA_SOURCE_DIR),
        "chroma_copy": str(index_copy),
        "index_copy_consistent": index_copy_consistent,
        "langfuse_disabled": True,
    }


def teardown_isolation() -> None:
    """关闭本次连接并清空缓存；保留运行证据，不清理用户原有数据。"""
    try:
        import app.ask
        import app.db
        import app.graph
    except Exception:
        return
    app.graph.get_graph.cache_clear()
    app.ask.load_index.cache_clear()
    app.ask.get_bm25_index.cache_clear()
    for getter in (app.db.get_connection, app.db.get_checkpointer_connection):
        try:
            getter().close()
        except Exception:
            pass
        getter.cache_clear()


def _invoke_graph(session_id: str, question: str, use_web_search: bool) -> dict:
    """实际问答图入口（评测专用包装，不改动应用逻辑）。测试可整体替身。"""
    from app.graph import get_graph, graph_config

    return get_graph().invoke(
        {"question": question, "use_web_search": use_web_search},
        config=graph_config(session_id),
    )


def _plan_route(question: str, use_web_search: bool) -> list:
    """仅调用 route() 节点做工具规划，不执行工具与真实搜索。"""
    from app.graph import route

    result = route({"question": question, "use_web_search": use_web_search})
    return list(result.get("planned_tool_calls") or [])


def _serialize_citations(citations) -> list:
    serialized = []
    for item in citations or []:
        if isinstance(item, dict):
            serialized.append(
                {
                    "source": item.get("source"),
                    "article": item.get("article"),
                    "score": item.get("score"),
                    "text": item.get("text"),
                }
            )
            continue
        node = getattr(item, "node", None)
        metadata = getattr(node, "metadata", {}) or {}
        serialized.append(
            {
                "source": metadata.get("source"),
                "article": metadata.get("article"),
                "score": float(getattr(item, "score", 0.0) or 0.0),
                "text": getattr(node, "text", None),
            }
        )
    return serialized


def read_existing_results(run_dir: Path) -> list:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return []
    records = []
    with open(results_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_result(run_dir: Path, record: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _next_attempt(existing: list, scenario_id: str) -> int:
    attempts = [int(item.get("attempt") or 0) for item in existing if item.get("scenario_id") == scenario_id]
    return (max(attempts) if attempts else 0) + 1


def new_session_id(run_id: str, scenario_id: str, attempt: int, nonce: str | None = None) -> str:
    """每次 attempt 的独立会话身份（F-01）。

    attempt 编号 + 随机 nonce：即使上一 attempt 中断在首条结果落盘前（attempt 编号会被复用），
    会话名仍不同，不会复用旧会话历史。
    """
    token = nonce or uuid.uuid4().hex[:8]
    return f"eval-{run_id}-{scenario_id}-a{attempt}-{token}"


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return load_json(manifest_path)
    return {
        "run_id": run_dir.name,
        "created_at": now_iso(),
        "git_head": git_head(),
        "code_sha256": {},
        "retrieval_strategy": "hybrid_rerank（默认策略，仅测默认策略）",
        "attempt_selection_policy": (
            "report 按完整场景 attempt 组选取最新 attempt，再读取该 attempt 各轮；"
            "不跨 attempt 拼接轮次，不取最高分"
        ),
        "collection_passes": [],
    }


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_frozen_cases_if_absent(run_dir: Path, cases: dict, provenance: dict) -> dict:
    """把本次采集使用的题集冻结到运行目录（F-03）。

    已存在时校验哈希一致（同一运行目录必须同一冻结题集），绝不覆盖旧快照。
    返回 {"path", "cases_sha256", "created": bool}。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = run_dir / "cases_frozen.json"
    payload_bytes = json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    cases_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if frozen_path.exists():
        existing_sha = sha256_file(frozen_path)
        if existing_sha != cases_sha256:
            raise RuntimeError(
                f"运行目录已有冻结题集（{existing_sha}），与本次题集（{cases_sha256}）不一致；"
                "请使用新的运行目录，provenance 不允许覆盖"
            )
        return {"path": str(frozen_path), "cases_sha256": cases_sha256, "created": False}
    frozen_path.write_bytes(payload_bytes)  # 按字节写入，避免换行翻译导致 hash 失配
    (run_dir / "cases_frozen.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"path": str(frozen_path), "cases_sha256": cases_sha256, "created": True}


def run_collection(
    run_dir: Path,
    scenarios: list,
    run_id: str,
    cases: dict | None = None,
    cases_sha256: str = "",
    scorer_sha256: str = "",
    with_isolation: bool = True,
    invoke_graph=None,
    plan_route=None,
) -> dict:
    """逐场景真实采集并即时落盘。返回本次采集 pass 的汇总。

    F-01：每个场景每次 attempt 使用独立会话（attempt + nonce），同 attempt 内轮次共享会话。
    F-03：manifest 以 collection_passes 追加记录本次采集的题集/评分规则 provenance，不覆盖历史；
    每条结果记录携带本次采集的 cases/scorer SHA 与 result_hash。
    """
    invoke_graph = invoke_graph or _invoke_graph
    plan_route = plan_route or _plan_route
    cases = cases if cases is not None else load_cases()
    frozen = write_frozen_cases_if_absent(
        run_dir,
        cases,
        provenance={
            "created_at": now_iso(),
            "git_head": git_head(),
            "source": "workspace tests/evaluation_cases.json（collect 冻结）",
            "cases_version": cases.get("version"),
        },
    )
    cases_sha256 = cases_sha256 or frozen["cases_sha256"]
    scorer_sha256 = scorer_sha256 or sha256_file(Path(__file__))

    run_dir.mkdir(parents=True, exist_ok=True)
    isolation = setup_isolation(run_dir) if with_isolation else {"skipped": "with_isolation=False"}
    real_before = _real_dirs_snapshot()
    existing = read_existing_results(run_dir)

    manifest = _load_manifest(run_dir)
    manifest.setdefault("collection_passes", [])
    pass_record = {
        "pass_id": f"pass-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        "started_at": now_iso(),
        "git_head": git_head(),
        "run_mode": "collect",
        "cases_sha256_at_collection": cases_sha256,
        "scorer_sha256_at_collection": scorer_sha256,
        "cases_version": cases.get("version"),
        "retrieval_strategy": "hybrid_rerank（默认策略，仅测默认策略）",
        "isolation": isolation,
        "real_dirs_before": real_before,
        "scenarios_requested": [scenario["id"] for scenario in scenarios],
        "counts": {"ok": 0, "error": 0},
        "attempts": {},
    }
    manifest["attempt_selection_policy"] = (
        "report 按完整场景 attempt 组选取最新 attempt，再读取该 attempt 各轮；"
        "不跨 attempt 拼接轮次，不取最高分"
    )
    manifest["code_sha256"] = {
        "tests/run_quality_evaluation.py": scorer_sha256,
        "tests/evaluation_cases.json": cases_sha256,
    }
    try:
        import app.config

        manifest["models"] = {
            "chat": app.config.CHAT_MODEL,
            "router": app.config.ROUTER_MODEL,
            "embedding": app.config.EMBEDDING_MODEL,
            "rerank": app.config.RERANK_MODEL,
        }
        manifest["langfuse_disabled"] = not app.config.langfuse_enabled()
    except Exception as error:  # 模型信息缺失不阻断采集，记录原因
        manifest["models"] = {"error": f"{type(error).__name__}: {error}"}
    manifest["collection_passes"].append(pass_record)
    _write_manifest(run_dir, manifest)

    try:
        for scenario in scenarios:
            scenario_id = scenario["id"]
            attempt = _next_attempt(existing, scenario_id)
            pass_record["attempts"][scenario_id] = attempt
            if scenario.get("execution_scope") == "graph":
                session_id = new_session_id(run_id, scenario_id, attempt)
            else:
                session_id = None
            for turn_index, turn in enumerate(scenario["turns"]):
                base_record = {
                    "record_type": "turn_result",
                    "run_id": run_id,
                    "pass_id": pass_record["pass_id"],
                    "scenario_id": scenario_id,
                    "category": scenario.get("category"),
                    "subcategory": scenario.get("subcategory"),
                    "turn_index": turn_index,
                    "attempt": attempt,
                    "collected_at": now_iso(),
                    "question": turn["question"],
                    "use_web_search": turn["use_web_search"],
                    "cases_sha256_at_collection": cases_sha256,
                    "scorer_sha256_at_collection": scorer_sha256,
                }
                if scenario.get("execution_scope") == "route_only":
                    base_record["execution_scope"] = "route_only"
                    try:
                        planned = plan_route(turn["question"], turn["use_web_search"])
                        base_record.update(
                            {
                                "status": "ok",
                                "planned_tool_calls": planned,
                                "note": "route_only：仅采集 route() 规划结果，未执行真实搜索",
                            }
                        )
                    except Exception as error:
                        base_record.update(
                            {
                                "status": "error",
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                            }
                        )
                else:
                    base_record["execution_scope"] = "graph"
                    base_record["session_id"] = session_id
                    try:
                        result = invoke_graph(session_id, turn["question"], turn["use_web_search"])
                        result = result or {}
                        base_record.update(
                            {
                                "status": "ok",
                                "search_query": result.get("search_query"),
                                "citations": _serialize_citations(result.get("citations")),
                                "answer": result.get("answer"),
                                "planned_tool_calls": list(result.get("planned_tool_calls") or []),
                                "tool_text": result.get("tool_text") or "",
                                "web_sources": list(result.get("web_sources") or []),
                            }
                        )
                    except Exception as error:
                        base_record.update(
                            {
                                "status": "error",
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                                "traceback": traceback.format_exc(limit=8),
                            }
                        )
                base_record["result_hash"] = compute_result_hash(base_record)
                append_result(run_dir, base_record)
                existing.append(base_record)
                pass_record["counts"]["ok" if base_record["status"] == "ok" else "error"] += 1
                _write_manifest(run_dir, manifest)
    finally:
        if with_isolation:
            teardown_isolation()
        pass_record["completed_at"] = now_iso()
        pass_record["real_dirs_after"] = _real_dirs_snapshot()
        pass_record["real_dirs_unchanged"] = (
            pass_record["real_dirs_before"] == pass_record["real_dirs_after"]
        )
        _write_manifest(run_dir, manifest)

    return pass_record


# ---------------------------------------------------------------------------
# 冻结版本与转换（F-03）
# ---------------------------------------------------------------------------


def load_frozen_cases(run_dir: Path) -> tuple[dict, str]:
    """读取运行目录的冻结题集快照；缺失时明确报错（不回退到工作区题集）。"""
    frozen_path = run_dir / "cases_frozen.json"
    if not frozen_path.exists():
        raise RuntimeError(
            f"{run_dir} 缺少 cases_frozen.json：历史运行请先执行 rescore 显式转换，"
            "报告不会改读当前工作区题集"
        )
    return load_json(frozen_path), sha256_file(frozen_path)


def verify_frozen_identity(manifest: dict, conversion: dict | None, frozen_sha256: str) -> list[str]:
    """R2-02：评分前核验冻结题集身份，返回失配问题列表（空 = 一致）。

    比较对象（各字段含义见 docs/EVALUATION.md）：
    - cases_frozen.json 当前字节 SHA-256（= 规范 JSON 内容 hash，即运行目录冻结身份）；
    - manifest.code_sha256['tests/evaluation_cases.json']（采集/转换时记录的冻结身份）；
    - conversion.frozen_cases_sha256（重算目录的冻结身份）。
    原始题集文件 hash（工作区原始字节）与冻结 hash 是不同字段，不在此判等。
    """
    problems = []
    manifest_frozen = (manifest.get("code_sha256") or {}).get("tests/evaluation_cases.json")
    if manifest_frozen is None:
        problems.append(
            "manifest 缺少冻结题集身份（code_sha256['tests/evaluation_cases.json']），无法核验快照身份"
        )
    elif manifest_frozen != frozen_sha256:
        problems.append(
            f"冻结题集失配：cases_frozen.json 当前 hash {frozen_sha256} ≠ manifest 记录 {manifest_frozen}"
        )
    if conversion is not None:
        conversion_frozen = conversion.get("frozen_cases_sha256")
        if conversion_frozen != frozen_sha256:
            problems.append(
                f"冻结题集失配：cases_frozen.json 当前 hash {frozen_sha256} ≠ conversion.frozen_cases_sha256 {conversion_frozen}"
            )
    return problems


def rescore_convert(source_run_dir: Path, target_run_dir: Path, change_reason: str = "") -> dict:
    """显式转换：把既有运行的采集结果复制到新目录并记录来源核验（修正规则重算用）。

    - 核验来源：记录源 results/manifest 的 SHA-256；若源 manifest 记录了采集时题集 hash，
      必须与当前工作区题集一致才能把工作区题集作为冻结快照，否则拒绝（不静默换 gold）。
    - 转换过程：为每条结果补算 result_hash（原产物缺字段），保留原记录内容不动；
      无法还原的信息（如逐条 scorer SHA）标记 unknown，不补造历史事实。
    """
    source_manifest_path = source_run_dir / "manifest.json"
    source_results_path = source_run_dir / "results.jsonl"
    if not source_results_path.exists():
        raise RuntimeError(f"源运行缺少 results.jsonl：{source_results_path}")
    source_manifest = load_json(source_manifest_path) if source_manifest_path.exists() else {}
    source_cases_sha = (
        (source_manifest.get("code_sha256") or {}).get("tests/evaluation_cases.json") or ""
    )
    workspace_cases = load_cases()
    workspace_cases_sha = cases_content_sha256(workspace_cases)
    workspace_file_sha = sha256_file(CASES_PATH) if CASES_PATH.exists() else "unknown"
    if source_cases_sha and source_cases_sha not in {workspace_cases_sha, workspace_file_sha}:
        raise RuntimeError(
            "源运行的采集题集 hash 与当前工作区题集不一致，拒绝转换："
            f"source={source_cases_sha} workspace_content={workspace_cases_sha} "
            f"workspace_file={workspace_file_sha}"
        )

    target_run_dir.mkdir(parents=True, exist_ok=True)
    records = read_existing_results(source_run_dir)
    converted = []
    hash_derived = 0
    identity_mismatches = []
    for record in records:
        new_record = dict(record)
        actual_hash = compute_result_hash(record)
        saved_hash = record.get("result_hash")
        if saved_hash is None:
            # 缺少保存 hash 的历史记录：仅在本次显式转换中按内容补算，来源记录于 conversion.json
            new_record["result_hash"] = actual_hash
            hash_derived += 1
        elif saved_hash != actual_hash:
            # 已有失配记录保留原 hash，不得重新包装为可信记录（R2-01）
            identity_mismatches.append(
                {
                    "scenario_id": record.get("scenario_id"),
                    "turn_index": record.get("turn_index"),
                    "attempt": record.get("attempt"),
                    "saved_result_hash": saved_hash,
                    "content_result_hash": actual_hash,
                }
            )
        new_record.setdefault("cases_sha256_at_collection", source_cases_sha or "unknown")
        new_record.setdefault("scorer_sha256_at_collection", "unknown")
        converted.append(new_record)
    with open(target_run_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for record in converted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    frozen_info = write_frozen_cases_if_absent(
        target_run_dir,
        workspace_cases,
        provenance={
            "created_at": now_iso(),
            "source": "rescore 显式转换：工作区题集（已与源 manifest 记录 hash 核验一致）",
            "source_cases_sha256": source_cases_sha or "unknown",
            "workspace_cases_sha256": workspace_cases_sha,
        },
    )
    conversion = {
        "conversion_id": f"rescore-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        "created_at": now_iso(),
        "report_mode": "rescore_with_repaired_rules",
        "source_run": str(source_run_dir),
        "source_results_sha256": sha256_file(source_results_path),
        "source_manifest_sha256": sha256_file(source_manifest_path) if source_manifest_path.exists() else "unknown",
        "source_cases_sha256": source_cases_sha or "unknown",
        "source_scorer_sha256": (source_manifest.get("code_sha256") or {}).get(
            "tests/run_quality_evaluation.py", "unknown"
        ),
        "new_scorer_sha256": sha256_file(Path(__file__)),
        "frozen_cases_sha256": frozen_info["cases_sha256"],
        "records_converted": len(converted),
        "result_hash_missing_derived": hash_derived,
        "identity_mismatches": identity_mismatches,
        "result_hash_derivation": (
            "缺少保存 hash 的记录由源记录字段（answer/citations/planned_tool_calls/tool_text 等）按内容补算，"
            "来源记录于本文件；已有保存 hash 的记录保留原值，与内容 hash 失配的列于 identity_mismatches，"
            "不重新包装为可信记录；原记录内容未改动"
        ),
        "change_reason": change_reason
        or "修复 F-01～F-08 评分绑定、冻结版本、状态分母与复核口径（修正规则重算，非原版成绩的无变化重放）",
        "note": "本目录报告为修正规则重算结果：原始采集来自 source_run，评分代码为 new_scorer_sha256，"
        "不等同于原版成绩的无变化重放",
    }
    (target_run_dir / "conversion.json").write_text(
        json.dumps(conversion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = _load_manifest(target_run_dir)
    manifest["run_id"] = target_run_dir.name
    manifest["report_mode"] = "rescore_with_repaired_rules"
    manifest["original_run"] = str(source_run_dir)
    manifest["original_cases_sha256"] = source_cases_sha or "unknown"
    manifest["original_scorer_sha256"] = conversion["source_scorer_sha256"]
    manifest["new_scorer_sha256"] = conversion["new_scorer_sha256"]
    manifest["change_reason"] = conversion["change_reason"]
    if source_manifest.get("models"):
        manifest["models"] = source_manifest["models"]
    if "langfuse_disabled" in source_manifest:
        manifest["langfuse_disabled"] = source_manifest["langfuse_disabled"]
    if source_manifest.get("retrieval_strategy"):
        manifest["retrieval_strategy"] = source_manifest["retrieval_strategy"]
    manifest["code_sha256"] = {
        "tests/run_quality_evaluation.py": conversion["new_scorer_sha256"],
        "tests/evaluation_cases.json": frozen_info["cases_sha256"],
    }
    _write_manifest(target_run_dir, manifest)
    return conversion


# ---------------------------------------------------------------------------
# attempt 组选择（F-01/F-05）
# ---------------------------------------------------------------------------


def select_attempt_groups(records: list, scenarios: list) -> dict:
    """按完整场景 attempt 组选取结果（F-01）。

    返回 {scenario_id: {"attempt", "turns": {turn_index: record}, "missing_turns": [...]}}：
    - 每个 (scenario, attempt) 是一个 attempt 组；选取该场景**最新 attempt** 的组；
    - 组内每个轮次取最后落盘的一条记录；缺失轮次如实列为 missing_turns；
    - 绝不回退旧 attempt、绝不跨 attempt 拼接轮次。
    """
    groups: dict[tuple, dict] = {}
    for order, record in enumerate(records):
        if record.get("record_type") != "turn_result":
            continue
        key = (record.get("scenario_id"), int(record.get("attempt") or 0))
        group = groups.setdefault(key, {"turns": {}})
        group["turns"][int(record.get("turn_index") or 0)] = record  # 后落盘覆盖先落盘
        group.setdefault("last_order", order)

    selected: dict[str, dict] = {}
    for scenario in scenarios:
        scenario_id = scenario["id"]
        attempts = sorted(attempt for (sid, attempt) in groups if sid == scenario_id)
        if not attempts:
            selected[scenario_id] = {
                "attempt": None,
                "turns": {},
                "missing_turns": list(range(len(scenario["turns"]))),
            }
            continue
        attempt = attempts[-1]
        turns = groups[(scenario_id, attempt)]["turns"]
        missing = [index for index in range(len(scenario["turns"])) if index not in turns]
        selected[scenario_id] = {"attempt": attempt, "turns": turns, "missing_turns": missing}
    return selected


# ---------------------------------------------------------------------------
# 冻结期望的适用性推导（F-06）
# ---------------------------------------------------------------------------


def frozen_expects_refusal(turn: dict) -> bool:
    """从冻结 required_points 推导该轮是否“应拒答/需说明不足”（不做回答文本判定）。"""
    for point in turn.get("required_points") or []:
        text = str(point)
        if any(marker in text for marker in REFUSAL_EXPECTATION_MARKERS):
            return True
    return False


# ---------------------------------------------------------------------------
# 复核记录（v2：绑定结果身份 + 评分项完整性校验）
# ---------------------------------------------------------------------------


def _frozen_item_texts(items) -> list[str]:
    return [str(item) for item in items or []]


def _check_item_list(review_items, frozen_texts: list[str], label: str) -> list[str]:
    """对照冻结定义核对评分项：缺项、重复项、替换项均返回问题描述。"""
    problems = []
    review_texts = [str(item.get("text") or item.get("point") or item.get("claim") or item.get("criterion") or "") for item in review_items or []]
    review_counter = Counter(review_texts)
    frozen_counter = Counter(frozen_texts)
    for text, count in sorted(frozen_counter.items()):
        if review_counter[text] < count:
            problems.append(f"{label} 缺项或被替换：{text!r}（冻结 {count} 项，复核 {review_counter[text]} 项）")
    for text, count in sorted(review_counter.items()):
        if text not in frozen_counter:
            problems.append(f"{label} 出现冻结定义之外的项：{text!r}")
        elif count > frozen_counter[text]:
            problems.append(f"{label} 重复项：{text!r}（冻结 {frozen_counter[text]} 项，复核 {count} 项）")
    return problems


def validate_turn_review(
    entry: dict, frozen_turn: dict, record: dict | None, is_route_only: bool = False
) -> list[str]:
    """复核条目有效性校验；返回问题列表（空 = 有效）。"""
    problems = []
    if record is None:
        problems.append("无对应的采集结果记录")
        return problems
    if entry.get("attempt_used") != record.get("attempt"):
        problems.append(
            f"绑定失配：复核 attempt={entry.get('attempt_used')}，结果 attempt={record.get('attempt')}"
        )
    # R2-01：三方核验（保存 hash — 实际内容 hash — 复核绑定 hash），身份以当前内容为准
    problems.extend(result_identity_problems(record))
    actual_hash = compute_result_hash(record)
    if entry.get("result_hash") != actual_hash:
        problems.append("绑定失配：复核记录的 result_hash 与结果当前内容 hash 不一致（结果或证据已变化）")

    problems.extend(
        _check_item_list(entry.get("required_points"), _frozen_item_texts(frozen_turn.get("required_points")), "required_points")
    )
    problems.extend(
        _check_item_list(entry.get("forbidden_claims"), _frozen_item_texts(frozen_turn.get("forbidden_claims")), "forbidden_claims")
    )
    for item in (entry.get("required_points") or []) + (entry.get("forbidden_claims") or []):
        result = item.get("result")
        label = item.get("point") or item.get("claim")
        if result not in {"PASS", "FAIL", "N/A", "PENDING"}:
            problems.append(f"评分项结果非法：{item!r}")
        if result == "N/A" and not str(item.get("na_basis") or "").strip():
            problems.append(
                f"无依据的 N/A：{label!r} 缺少 na_basis（冻结规则或上位任务依据）"
            )
    conclusions = entry.get("conclusions")
    if conclusions is None:
        problems.append("缺少 conclusions 字段（无法区分“无此类结论”与“未完成提取”）")
    for conclusion in conclusions or []:
        if conclusion.get("needs_evidence") is True and conclusion.get("supported") not in {True, False, None}:
            problems.append(f"结论 supported 取值非法：{conclusion!r}")
    if entry.get("conclusions_extraction_complete") not in {True, False, None}:
        problems.append("conclusions_extraction_complete 取值非法")

    # 拒答两项的适用性来自冻结定义，复核不得自行改写（F-06）
    expected_insufficiency = frozen_expects_refusal(frozen_turn)
    insufficiency = entry.get("insufficiency_handling") or {}
    if insufficiency.get("applicable") is not expected_insufficiency:
        problems.append("insufficiency_handling.applicable 与冻结定义不一致")
    expected_wrong_refusal = (not expected_insufficiency) and not is_route_only
    wrong_refusal = entry.get("wrong_refusal") or {}
    if wrong_refusal.get("applicable") is not expected_wrong_refusal:
        problems.append("wrong_refusal.applicable 与冻结定义不一致")
    return problems


def validate_scenario_review(entry: dict, scenario: dict, group: dict) -> list[str]:
    """场景完成复核有效性校验；返回问题列表（空 = 有效）。"""
    problems = []
    if entry.get("attempt_used") != group.get("attempt"):
        problems.append(
            f"绑定失配：完成复核 attempt={entry.get('attempt_used')}，选中 attempt={group.get('attempt')}"
        )
    result_hashes = entry.get("result_hashes") or {}
    for turn_index, record in sorted(group.get("turns", {}).items()):
        # R2-01：完成绑定同样以当前内容重算的 hash 为准
        problems.extend(result_identity_problems(record))
        actual_hash = compute_result_hash(record)
        if result_hashes.get(str(turn_index)) != actual_hash and result_hashes.get(turn_index) != actual_hash:
            problems.append(f"绑定失配：完成复核缺 turn {turn_index} 的结果 hash 或 hash 不一致")
    problems.extend(
        _check_item_list(
            entry.get("criteria_results"),
            _frozen_item_texts(scenario.get("completion_criteria")),
            "completion_criteria",
        )
    )
    for item in entry.get("criteria_results") or []:
        if item.get("result") not in {"PASS", "FAIL", "PENDING", "N/A"}:
            problems.append(f"完成条件结果非法：{item!r}")
        if item.get("result") == "N/A" and not str(item.get("na_basis") or "").strip():
            problems.append(f"完成条件无依据的 N/A：{item.get('criterion')!r}")
    return problems


def completion_verdict(criteria_results: list[dict]) -> str:
    """完成判定：必要条件 FAIL/待审时整体不得 PASS（F-08）。"""
    results = [item.get("result") for item in criteria_results or []]
    if any(result == "FAIL" for result in results):
        return "FAIL"
    if any(result == "PENDING" for result in results):
        return "PENDING"
    if results and all(result in {"PASS", "N/A"} for result in results):
        return "PASS" if "PASS" in results else "N/A"
    return "PENDING"


def build_review_template(cases: dict, selected: dict, run_id: str = "<run_id>") -> dict:
    """生成待填写的语义复核模板（v2，不覆盖已有 reviews.json）。"""
    turn_reviews = []
    scenario_reviews = []
    for scenario in cases["scenarios"]:
        scenario_id = scenario["id"]
        for turn_index, turn in enumerate(scenario["turns"]):
            group = selected.get(scenario_id) or {"attempt": None, "turns": {}}
            record = (group.get("turns") or {}).get(turn_index) or {}
            turn_reviews.append(
                {
                    "scenario_id": scenario_id,
                    "turn_index": turn_index,
                    "attempt_used": record.get("attempt"),
                    "result_hash": record.get("result_hash") or (
                        compute_result_hash(record) if record else None
                    ),
                    "response_type": "PENDING",
                    "required_points": [
                        {"point": point, "result": "PENDING", "reason": ""}
                        for point in turn.get("required_points") or []
                    ],
                    "forbidden_claims": [
                        {"claim": claim, "result": "PENDING", "reason": ""}
                        for claim in turn.get("forbidden_claims") or []
                    ],
                    "conclusions": [],
                    "conclusions_extraction_complete": None,
                    "insufficiency_handling": {
                        "applicable": frozen_expects_refusal(turn),
                        "result": "PENDING" if frozen_expects_refusal(turn) else "N/A",
                        "reason": "",
                    },
                    "wrong_refusal": {
                        "applicable": not frozen_expects_refusal(turn)
                        and scenario.get("execution_scope") == "graph",
                        "result": (
                            "PENDING"
                            if not frozen_expects_refusal(turn) and scenario.get("execution_scope") == "graph"
                            else "N/A"
                        ),
                        "reason": "",
                    },
                    "evidence_location": (
                        f"logs/evaluation/{run_id}/results.jsonl :: scenario={scenario_id} "
                        f"turn={turn_index} attempt={record.get('attempt')}"
                    ),
                    "reviewer": "PENDING",
                    "reviewer_role": "PENDING",
                }
            )
        group = selected.get(scenario_id) or {"attempt": None, "turns": {}}
        scenario_reviews.append(
            {
                "scenario_id": scenario_id,
                "attempt_used": group.get("attempt"),
                "result_hashes": {
                    str(index): (record.get("result_hash") or compute_result_hash(record))
                    for index, record in (group.get("turns") or {}).items()
                },
                "criteria_results": [
                    {"criterion": criterion, "result": "PENDING", "reason": ""}
                    for criterion in scenario.get("completion_criteria") or []
                ],
                "reconciles_turn_failures": "",
                "evidence_location": f"logs/evaluation/{run_id}/results.jsonl :: scenario={scenario_id}",
                "reviewer": "PENDING",
                "reviewer_role": "PENDING",
            }
        )
    return {
        "format": REVIEW_FORMAT,
        "cases_sha256": None,
        "reviewer": "PENDING",
        "reviewer_role": "PENDING",
        "reviewed_at": None,
        "turn_reviews": turn_reviews,
        "scenario_reviews": scenario_reviews,
    }


# ---------------------------------------------------------------------------
# 状态统计（F-05）
# ---------------------------------------------------------------------------


def _tally() -> dict:
    return {
        "applicable": 0,
        "pass": 0,
        "fail": 0,
        "error": 0,
        "not_verified": 0,
        "pending_annotation": 0,
        "pending_review": 0,
        "not_applicable": 0,
    }


def _record_tally(tally: dict, result: str) -> None:
    """计入分项统计（F-05）。

    适用总数 = 通过 + 失败 + 运行错误 + 未采集 + 待标注复核 + 待答案复核；
    N/A 单列，不进适用总数。各状态互斥。
    """
    if result == "N/A":
        tally["not_applicable"] += 1
        return
    key = {
        "PASS": "pass",
        "FAIL": "fail",
        "ERROR": "error",
        "NOT_VERIFIED": "not_verified",
        "PENDING_ANNOTATION": "pending_annotation",
        "PENDING_REVIEW": "pending_review",
    }.get(result)
    if key is None:
        raise ValueError(f"未知状态：{result}")
    tally["applicable"] += 1
    tally[key] += 1


def tally_identity_ok(tally: dict) -> bool:
    """核对：适用总数 = 通过 + 失败 + 运行错误 + 未采集 + 待标注复核 + 待答案复核。"""
    return tally["applicable"] == (
        tally["pass"]
        + tally["fail"]
        + tally["error"]
        + tally["not_verified"]
        + tally["pending_annotation"]
        + tally["pending_review"]
    )


# ---------------------------------------------------------------------------
# 离线汇总
# ---------------------------------------------------------------------------


def _annotation_pending(scenario: dict) -> bool:
    return scenario.get("annotation_status") == "needs_legal_review"


def aggregate(cases: dict, selected: dict, reviews: dict | None, frozen_cases_sha256: str | None = None) -> dict:
    """统一汇总自动指标与语义指标（F-02/F-04/F-05/F-06/F-07；R2-01/02/03）。

    frozen_cases_sha256 提供时校验 reviews.cases_sha256 绑定同一冻结身份；
    失配则全部复核失效并记录原因（R2-02）。
    """
    auto = {
        "retrieval": _tally(),
        "tool_selection": _tally(),
        "tool_params": _tally(),
        "route_tool_selection": _tally(),
        "route_tool_params": _tally(),
    }
    semantic = {
        "required_points": _tally(),
        "forbidden_claims": _tally(),
        "insufficiency_handling": _tally(),
        "wrong_refusal": _tally(),
        "completion_graph": _tally(),
        "completion_route_only": _tally(),
    }
    citation = {
        "extracted_needing_evidence": 0,
        "extracted_trusted_annotation": 0,
        "reviewed": 0,
        "supported": 0,
        "unsupported": 0,
        "pending": 0,
        "pending_annotation_conclusions": 0,
        "pending_annotation_executor_preliminary": {
            "supported_true": 0,
            "supported_false": 0,
            "supported_none": 0,
        },
        "extraction_unconfirmed_answers": 0,
    }
    per_turn_rows = []
    route_rows = []
    failure_cases = []
    error_rows = []
    invalid_reviews = []
    reviewers = set()
    turn_review_state: dict[tuple, dict] = {}

    turn_reviews = {
        (item.get("scenario_id"), item.get("turn_index")): item
        for item in (reviews or {}).get("turn_reviews", [])
    }
    scenario_reviews = {
        item.get("scenario_id"): item for item in (reviews or {}).get("scenario_reviews", [])
    }

    # R2-02：已有 reviews 必须绑定同一冻结身份；失配则全部复核失效并明确记录原因
    reviews_identity_problem = None
    if reviews is not None and frozen_cases_sha256 is not None:
        bound_sha = reviews.get("cases_sha256")
        if bound_sha != frozen_cases_sha256:
            reviews_identity_problem = (
                f"reviews 绑定冻结身份失配：reviews.cases_sha256={bound_sha!r} ≠ 冻结题集 "
                f"{frozen_cases_sha256}（全部复核失效，不采信任何旧评分）"
            )

    for scenario in cases["scenarios"]:
        scenario_id = scenario["id"]
        is_route_only = scenario.get("execution_scope") == "route_only"
        annotation_pending = _annotation_pending(scenario)
        group = selected.get(scenario_id) or {"attempt": None, "turns": {}, "missing_turns": []}

        for turn_index, turn in enumerate(scenario["turns"]):
            record = (group.get("turns") or {}).get(turn_index)

            # --- 自动指标 ---
            if is_route_only:
                if record is None:
                    _record_tally(auto["route_tool_selection"], "NOT_VERIFIED")
                    if any(entry.get("args") for entry in turn.get("expected_tools") or []):
                        _record_tally(auto["route_tool_params"], "NOT_VERIFIED")
                else:
                    selection_result = score_tool_selection(turn, record)
                    params_result = score_tool_params(turn, record)
                    for tally, computed in (
                        (auto["route_tool_selection"], selection_result),
                        (auto["route_tool_params"], params_result),
                    ):
                        if computed["result"] == "N/A":
                            _record_tally(tally, "N/A")
                        elif annotation_pending:
                            _record_tally(tally, "PENDING_ANNOTATION")
                        else:
                            _record_tally(tally, computed["result"])
                    route_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "turn_index": turn_index,
                            "question": turn["question"],
                            "attempt": record.get("attempt"),
                            "planned": record.get("planned_tool_calls"),
                            "selection": selection_result,
                            "params": params_result,
                        }
                    )
            else:
                if record is None:
                    if turn.get("expected_articles"):
                        _record_tally(auto["retrieval"], "NOT_VERIFIED")
                    _record_tally(auto["tool_selection"], "NOT_VERIFIED")
                    if any(entry.get("args") for entry in turn.get("expected_tools") or []):
                        _record_tally(auto["tool_params"], "NOT_VERIFIED")
                    per_turn_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "turn_index": turn_index,
                            "result": "NOT_VERIFIED",
                            "reason": f"attempt {group.get('attempt')} 无该轮采集记录",
                        }
                    )
                else:
                    retrieval_result = score_retrieval_turn(turn, record)
                    selection_result = score_tool_selection(turn, record)
                    params_result = score_tool_params(turn, record)
                    for tally, computed in (
                        (auto["retrieval"], retrieval_result),
                        (auto["tool_selection"], selection_result),
                        (auto["tool_params"], params_result),
                    ):
                        if computed["result"] == "N/A":
                            _record_tally(tally, "N/A")
                        elif record.get("status") != "ok":
                            _record_tally(tally, "ERROR")
                        elif annotation_pending:
                            _record_tally(tally, "PENDING_ANNOTATION")
                        else:
                            _record_tally(tally, computed["result"])
                    row = {
                        "scenario_id": scenario_id,
                        "turn_index": turn_index,
                        "attempt": record.get("attempt"),
                        "result_hash": record.get("result_hash"),
                        "question": turn["question"],
                        "annotation_pending": annotation_pending,
                        "retrieval": retrieval_result,
                        "tool_selection": selection_result,
                        "tool_params": params_result,
                    }
                    per_turn_rows.append(row)
                    if record.get("status") != "ok":
                        error_rows.append(row)
                    elif not annotation_pending and "FAIL" in (
                        retrieval_result["result"],
                        selection_result["result"],
                        params_result["result"],
                    ):
                        failure_cases.append(row)

            # --- 语义指标（复核记录驱动） ---
            entry = turn_reviews.get((scenario_id, turn_index))
            record_for_review = record
            if entry and entry.get("reviewer_role") not in (None, "PENDING"):
                reviewers.add(f"{entry.get('reviewer')}({entry.get('reviewer_role')})")

            review_valid = False
            if entry is not None:
                review_problems = validate_turn_review(
                    entry, turn, record_for_review, is_route_only=is_route_only
                )
                if reviews_identity_problem:
                    review_problems = [reviews_identity_problem, *review_problems]
                review_valid = not review_problems
                if review_problems:
                    invalid_reviews.append(
                        {"scenario_id": scenario_id, "turn_index": turn_index, "problems": review_problems}
                    )

            item_results = {}
            for key, frozen_texts in (
                ("required_points", _frozen_item_texts(turn.get("required_points"))),
                ("forbidden_claims", _frozen_item_texts(turn.get("forbidden_claims"))),
            ):
                if not frozen_texts:
                    item_results[key] = "N/A"
                elif record_for_review is None:
                    item_results[key] = "NOT_VERIFIED"
                elif record_for_review.get("status") != "ok":
                    item_results[key] = "ERROR"
                elif not review_valid:
                    item_results[key] = "PENDING_REVIEW"
                else:
                    results = [item.get("result") for item in entry.get(key) or []]
                    if any(result == "PENDING" for result in results):
                        item_results[key] = "PENDING_REVIEW"
                    elif any(result == "FAIL" for result in results):
                        item_results[key] = "FAIL"
                    elif all(result in {"PASS", "N/A"} for result in results):
                        item_results[key] = "PASS" if "PASS" in results else "N/A"
                    else:
                        item_results[key] = "PENDING_REVIEW"

            turn_review_state[(scenario_id, turn_index)] = {
                "valid": review_valid,
                "has_fail": review_valid
                and any(
                    item.get("result") == "FAIL"
                    for item in (entry or {}).get("required_points", [])
                    + (entry or {}).get("forbidden_claims", [])
                ),
            }

            for key in ("required_points", "forbidden_claims"):
                result = item_results.get(key, "N/A")
                if annotation_pending and result in {"PASS", "FAIL"}:
                    _record_tally(semantic[key], "PENDING_ANNOTATION")
                else:
                    _record_tally(semantic[key], result)

            # 拒答两类指标（适用性来自冻结定义）
            for key, applicable in (
                ("insufficiency_handling", frozen_expects_refusal(turn)),
                ("wrong_refusal", not frozen_expects_refusal(turn) and not is_route_only),
            ):
                if not applicable:
                    _record_tally(semantic[key], "N/A")
                    continue
                if record_for_review is None:
                    _record_tally(semantic[key], "NOT_VERIFIED")
                    continue
                if record_for_review.get("status") != "ok":
                    _record_tally(semantic[key], "ERROR")
                    continue
                if not review_valid:
                    _record_tally(semantic[key], "PENDING_REVIEW")
                    continue
                review_item = entry.get(key) or {}
                result = review_item.get("result") or "PENDING"
                if result == "PENDING":
                    result = "PENDING_REVIEW"
                if annotation_pending and result in {"PASS", "FAIL"}:
                    _record_tally(semantic[key], "PENDING_ANNOTATION")
                else:
                    _record_tally(semantic[key], result)

            # 引用结论计数（F-07；R2-03：待标注题结论全部分流）
            if record_for_review is not None and record_for_review.get("status") == "ok":
                if review_valid:
                    for conclusion in entry.get("conclusions") or []:
                        if conclusion.get("needs_evidence") is not True:
                            continue
                        citation["extracted_needing_evidence"] += 1
                        supported = conclusion.get("supported")
                        if annotation_pending:
                            # 待标注题结论无论执行者初评 True/False/None，均不进可信已审/支持/不支持
                            citation["pending_annotation_conclusions"] += 1
                            diagnostic_key = (
                                "supported_true"
                                if supported is True
                                else "supported_false"
                                if supported is False
                                else "supported_none"
                            )
                            citation["pending_annotation_executor_preliminary"][diagnostic_key] += 1
                            continue
                        citation["extracted_trusted_annotation"] += 1
                        if supported is True:
                            citation["reviewed"] += 1
                            citation["supported"] += 1
                        elif supported is False:
                            citation["reviewed"] += 1
                            citation["unsupported"] += 1
                        else:
                            citation["pending"] += 1
                    if entry.get("conclusions_extraction_complete") is not True:
                        citation["extraction_unconfirmed_answers"] += 1
                else:
                    citation["extraction_unconfirmed_answers"] += 1

        # --- 场景完成（graph 与 route_only 分开，F-05/F-08） ---
        completion_tally = semantic["completion_route_only"] if is_route_only else semantic["completion_graph"]
        turns_map = group.get("turns") or {}
        expected_turns = list(range(len(scenario["turns"])))
        if not turns_map:
            _record_tally(completion_tally, "NOT_VERIFIED")
        elif any(index not in turns_map for index in expected_turns):
            _record_tally(completion_tally, "NOT_VERIFIED")
        elif any(turns_map[index].get("status") != "ok" for index in expected_turns):
            _record_tally(completion_tally, "ERROR")
        else:
            entry = scenario_reviews.get(scenario_id)
            if entry is None:
                _record_tally(completion_tally, "PENDING_REVIEW")
            else:
                problems = validate_scenario_review(entry, scenario, group)
                if reviews_identity_problem:
                    problems = [reviews_identity_problem, *problems]
                verdict = completion_verdict(entry.get("criteria_results"))
                if verdict == "PASS":
                    invalid_turn_review = any(
                        not (turn_review_state.get((scenario_id, index)) or {}).get("valid", False)
                        for index in expected_turns
                    )
                    if invalid_turn_review:
                        problems.append("存在无效或缺失的轮次复核，完成判定不得 PASS")
                    # 完成 PASS 与轮次 FAIL 并存时必须显式解释（F-08）
                    has_turn_fail = any(
                        (turn_review_state.get((scenario_id, index)) or {}).get("has_fail")
                        for index in expected_turns
                    )
                    if has_turn_fail and not str(entry.get("reconciles_turn_failures") or "").strip():
                        problems.append(
                            "完成判 PASS 但存在轮次 FAIL 评分项，且未提供 reconciles_turn_failures 解释"
                        )
                if problems:
                    invalid_reviews.append(
                        {"scenario_id": scenario_id, "turn_index": "completion", "problems": problems}
                    )
                    _record_tally(completion_tally, "PENDING_REVIEW")
                elif annotation_pending and verdict in {"PASS", "FAIL"}:
                    _record_tally(completion_tally, "PENDING_ANNOTATION")
                else:
                    _record_tally(completion_tally, "PASS" if verdict == "PASS" else "FAIL" if verdict == "FAIL" else "PENDING_REVIEW")

    totals = {**auto, **semantic}
    identity_checks = {key: tally_identity_ok(value) for key, value in totals.items()}
    citation_reviewed = citation["reviewed"]
    # R2-03 可核算关系：已提取 = 可信已审 + 可信标注下待答案复核 + 待标注复核；可信已审 = 支持 + 不支持
    citation["reviewed_identity_ok"] = citation["reviewed"] == (
        citation["supported"] + citation["unsupported"]
    )
    citation["extracted_identity_ok"] = citation["extracted_needing_evidence"] == (
        citation["reviewed"] + citation["pending"] + citation["pending_annotation_conclusions"]
    )
    citation["support_rate"] = (
        round(citation["supported"] / citation_reviewed, 4) if citation_reviewed else None
    )
    citation["review_coverage_rate"] = (
        round(citation_reviewed / citation["extracted_needing_evidence"], 4)
        if citation["extracted_needing_evidence"]
        else None
    )

    semantic_items_total = sum(
        totals[key]["applicable"]
        for key in ("required_points", "forbidden_claims", "insufficiency_handling", "wrong_refusal")
    )
    semantic_items_reviewed = sum(
        totals[key]["pass"] + totals[key]["fail"]
        for key in ("required_points", "forbidden_claims", "insufficiency_handling", "wrong_refusal")
    )
    completion_total = semantic["completion_graph"]["applicable"] + semantic["completion_route_only"]["applicable"]
    completion_reviewed = (
        semantic["completion_graph"]["pass"]
        + semantic["completion_graph"]["fail"]
        + semantic["completion_route_only"]["pass"]
        + semantic["completion_route_only"]["fail"]
    )

    def observed(tally: dict) -> dict:
        decided = tally["pass"] + tally["fail"]
        return {
            "decided": decided,
            "pass": tally["pass"],
            "rate": round(tally["pass"] / decided, 4) if decided else None,
            "coverage": round(decided / tally["applicable"], 4) if tally["applicable"] else None,
        }

    return {
        "auto": auto,
        "semantic": semantic,
        "citation": citation,
        "identity_checks": identity_checks,
        "reviews_identity_problem": reviews_identity_problem,
        "coverage": {
            "semantic_items_reviewed": semantic_items_reviewed,
            "semantic_items_total": semantic_items_total,
            "semantic_rate": (
                round(semantic_items_reviewed / semantic_items_total, 4)
                if semantic_items_total
                else None
            ),
            "completion_reviewed": completion_reviewed,
            "completion_total": completion_total,
            "completion_rate": (
                round(completion_reviewed / completion_total, 4) if completion_total else None
            ),
            "reviewers": sorted(reviewers),
        },
        "observed_subset": {
            "retrieval": observed(auto["retrieval"]),
            "tool_selection": observed(auto["tool_selection"]),
            "tool_params": observed(auto["tool_params"]),
            "completion_graph": observed(semantic["completion_graph"]),
        },
        "per_turn_rows": per_turn_rows,
        "route_rows": route_rows,
        "failure_cases": failure_cases,
        "error_rows": error_rows,
        "invalid_reviews": invalid_reviews,
    }


def _tally_line(tally: dict) -> str:
    return (
        f"{tally['pass']}/{tally['applicable']}"
        f"（失败 {tally['fail']}、运行错误 {tally['error']}、未采集 {tally['not_verified']}、"
        f"待标注复核 {tally['pending_annotation']}、待答案复核 {tally['pending_review']}、"
        f"不适用 {tally['not_applicable']}）"
    )


def _observed_line(name: str, observed: dict) -> str:
    return (
        f"{name}：已判定 {observed['decided']} 项，其中通过 {observed['pass']} 项"
        f"（已观察子集通过率 {observed['rate']}，覆盖率 {observed['coverage']}）"
    )


def _rate_text(value) -> str:
    """分母为零时显示 N/A。"""
    return "N/A" if value is None else str(value)


def build_report_md(
    run_dir: Path,
    manifest: dict,
    cases: dict,
    cases_sha256: str,
    result: dict,
    conversion: dict | None,
) -> str:
    auto = result["auto"]
    semantic = result["semantic"]
    citation = result["citation"]
    coverage = result["coverage"]
    selected = result["selected"]

    lines = [
        "# LawRAG 回答质量与工具路由评测报告（修正规则重算）" if conversion else "# LawRAG 回答质量与工具路由评测报告",
        "",
        f"- run_id：{manifest.get('run_id')}",
        f"- 生成时间：{now_iso()}",
        f"- 报告模式：{manifest.get('report_mode', 'collect_original')}",
    ]
    if conversion:
        lines += [
            f"- 原始运行：{conversion.get('source_run')}（原始采集结果 hash：{conversion.get('source_results_sha256')}）",
            f"- 原始采集题集 SHA-256：{conversion.get('source_cases_sha256')}",
            f"- 原始评分脚本 SHA-256：{conversion.get('source_scorer_sha256')}",
            f"- 本次评分脚本 SHA-256：{conversion.get('new_scorer_sha256')}",
            f"- 变更原因：{conversion.get('change_reason')}",
            "- 性质：修正规则重算，**不是**原版成绩的无变化重放；原始成绩见 source_run 目录。",
        ]
    lines += [
        f"- Git HEAD：{manifest.get('git_head')}",
        f"- 冻结题集版本：{cases.get('version')}，SHA-256：{cases_sha256}（来自运行目录 cases_frozen.json，非工作区）",
        f"- 模型标识：{json.dumps(manifest.get('models', {}), ensure_ascii=False)}",
        f"- 检索策略：{manifest.get('retrieval_strategy', 'hybrid_rerank')}",
        f"- attempt 选择：{manifest.get('attempt_selection_policy', '')}",
        "",
        "## attempt 选择与完整性",
        "",
        "| 场景 | 选中 attempt | 缺失轮次 | 完整性 |",
        "| --- | --- | --- | --- |",
    ]
    for scenario in cases["scenarios"]:
        group = selected.get(scenario["id"]) or {}
        missing = group.get("missing_turns") or []
        lines.append(
            f"| {scenario['id']} | {group.get('attempt') if group.get('attempt') is not None else '—'} | "
            f"{'、'.join(f'turn {index}' for index in missing) or '—'} | "
            f"{'完整' if not missing and group.get('turns') else '不完整（如实显示，不回退旧 attempt）'} |"
        )

    lines += [
        "",
        "## 自动指标（确定性判定；冻结分母，状态互斥）",
        "",
        "计数关系：适用总数 = 通过 + 失败 + 运行错误 + 未采集 + 待标注复核 + 待答案复核；N/A 单列。",
        "",
        "### 检索 Hit@3",
        "",
        f"- {_tally_line(auto['retrieval'])}",
        f"- {_observed_line('已观察子集（仅已判定项，诊断用）', result['observed_subset']['retrieval'])}",
        "- 口径：适用轮次 = 冻结题集中有 expected_articles 的问答图轮次（37）；库外题、无检索目标题、",
        "  route_only 不计入分母；needs_legal_review 标注题全部计入“待标注复核”。",
        "",
        "| 场景 | 轮次 | 命中法条 | 缺失法条 | 判定 | 标注状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["per_turn_rows"]:
        retrieval = row.get("retrieval") or {}
        if not retrieval.get("applicable"):
            continue
        verdict = "待标注复核（试算：" + retrieval.get("result", "?") + "）" if row.get("annotation_pending") else retrieval.get("result")
        lines.append(
            f"| {row['scenario_id']} | {row['turn_index']} | "
            f"{'、'.join(retrieval.get('hit') or []) or '—'} | "
            f"{'、'.join(retrieval.get('missing') or []) or '—'} | {verdict} | "
            f"{'待法律复核' if row.get('annotation_pending') else '可信'} |"
        )

    lines += [
        "",
        "### 工具规划与参数（问答图轮次）",
        "",
        f"- 工具选择正确率：{_tally_line(auto['tool_selection'])}",
        f"- 参数正确率：{_tally_line(auto['tool_params'])}",
        f"- {_observed_line('工具选择已观察子集（诊断用）', result['observed_subset']['tool_selection'])}",
        f"- {_observed_line('参数已观察子集（诊断用）', result['observed_subset']['tool_params'])}",
        "- 口径：适用总数来自冻结定义（工具选择 46 轮、参数 6 轮），不随已采集数量缩小；",
        "  缺失调用、错误工具、缺失参数或编造参数均不计为正确。无实际执行证据，故不输出“工具执行成功率”。",
        "",
        "### 联网路由（route_only 2 题，独立报告）",
        "",
        f"- 工具选择：{_tally_line(auto['route_tool_selection'])}",
        f"- 路由参数：{_tally_line(auto['route_tool_params'])}",
        "- 口径：仅测 route() 规划结果，不执行真实搜索，不混入端到端任务成功率。",
        "",
        "| 场景 | 问题 | 规划结果 | 判定 |",
        "| --- | --- | --- | --- |",
    ]
    for row in result["route_rows"]:
        planned = json.dumps(row.get("planned"), ensure_ascii=False)
        lines.append(
            f"| {row['scenario_id']} | {row['question']} | {planned} | "
            f"{row['selection']['result']} / {row['params']['result']} |"
        )

    lines += [
        "",
        "## 语义复核（人工评分；结果绑定 + 评分项完整性校验）",
        "",
        f"- 复核者：{coverage['reviewers'] or ['尚无已复核记录']}",
        f"- 语义评分项复核覆盖率：{coverage['semantic_items_reviewed']}/{coverage['semantic_items_total']}"
        f"（{coverage['semantic_rate']}）",
        f"- 场景完成复核覆盖率：{coverage['completion_reviewed']}/{coverage['completion_total']}"
        f"（{coverage['completion_rate']}）",
        "- 引用审核覆盖率（与上述语义覆盖率分开统计）："
        f"{citation['reviewed']}/{citation['extracted_needing_evidence']}"
        f"（{citation['review_coverage_rate']}）",
        "",
        "### 引用审核明细（F-07 口径；R2-03 待标注结论分流）",
        "",
        f"- 已提取的需证据结论总数：{citation['extracted_needing_evidence']}"
        f"（可信标注题 {citation['extracted_trusted_annotation']} + 待标注题 {citation['pending_annotation_conclusions']}）",
        f"- 可信已审结论数：{citation['reviewed']}",
        f"- 有支持结论数（可信）：{citation['supported']}",
        f"- 无支持结论数（可信）：{citation['unsupported']}",
        f"- 可信标注下待答案复核结论数（supported=None）：{citation['pending']}",
        f"- 待标注复核结论数（needs_legal_review 题，不进可信已审/支持/不支持）：{citation['pending_annotation_conclusions']}"
        f"（执行者初评仅供诊断：支持 {citation['pending_annotation_executor_preliminary']['supported_true']}、"
        f"不支持 {citation['pending_annotation_executor_preliminary']['supported_false']}、"
        f"未知 {citation['pending_annotation_executor_preliminary']['supported_none']}）",
        f"- 结论提取完整性尚未确认的回答数：{citation['extraction_unconfirmed_answers']}",
        f"- 已审核引用支持率 = 有支持 / 可信已审 = {_rate_text(citation['support_rate'])}",
        f"- 已提取结论审核覆盖率 = 可信已审 / 已提取 = {_rate_text(citation['review_coverage_rate'])}",
        "- 可核算关系：已提取 = 可信已审 + 可信标注下待答案复核 + 待标注复核；可信已审 = 支持 + 不支持"
        f"（核验：{'一致' if citation['extracted_identity_ok'] and citation['reviewed_identity_ok'] else '不一致'}）",
        "- 三类未完成原因（互斥，不合并计数）：标注待审 = 待标注复核结论数；",
        "  结论支持待审 = 可信标注下待答案复核结论数；结论提取未确认按回答数另外报告，不与结论数相加。",
        "- 口径：待标注题结论无论执行者初评 supported=True/False/None 均分流为待标注复核，不进可信成绩；",
        "  空结论列表只有在审核者确认（conclusions_extraction_complete=true）时才算“无此类结论”；",
        "  覆盖率分母不排除待标注结论（否则会误称“全部结论已审”）。分母为零显示 N/A。",
        "",
        "### 分项状态",
        "",
        f"- 要点覆盖：{_tally_line(semantic['required_points'])}",
        f"- 禁止结论（编造检查）：{_tally_line(semantic['forbidden_claims'])}",
        f"- 应拒答/需说明不足场景的处理：{_tally_line(semantic['insufficiency_handling'])}",
        f"- 可答题场景的错误拒答：{_tally_line(semantic['wrong_refusal'])}",
        f"- 场景完成（graph 38 题）：{_tally_line(semantic['completion_graph'])}",
        f"- 场景完成（route_only 2 题，独立）：{_tally_line(semantic['completion_route_only'])}",
        f"- {_observed_line('graph 场景完成已观察子集（诊断用）', result['observed_subset']['completion_graph'])}",
        "- 口径：拒答两类指标的适用性由冻结 required_points 推导；判定来自人工复核记录，",
        "  不以关键词匹配回答文本替代语义判断。绑定失配/评分项不完整 → 待答案复核。",
    ]

    if result.get("reviews_identity_problem"):
        lines += [
            "",
            "### 复核冻结身份失配（全部复核失效，R2-02）",
            "",
            f"- {result['reviews_identity_problem']}",
        ]

    if result["invalid_reviews"]:
        lines += ["", "### 无效评分记录（进入待答案复核）", ""]
        for item in result["invalid_reviews"][:20]:
            lines.append(
                f"- {item['scenario_id']} turn {item['turn_index']}：{'；'.join(item['problems'])}"
            )

    lines += ["", "## 待标注复核（needs_legal_review，不进入可信 PASS/FAIL）", ""]
    pending_ids = [
        scenario["id"]
        for scenario in cases["scenarios"]
        if scenario.get("annotation_status") == "needs_legal_review"
    ]
    lines.append(f"- 待法律复核标注题：{len(pending_ids)} 个（{'、'.join(pending_ids)}）")
    lines += [
        "- 影响指标：上述场景的检索、工具、参数与全部语义项均计入“待标注复核”状态；",
        "  明细表中的“试算”仅诊断用途，不是可信成绩。",
        f"- 可信子集覆盖：检索适用 {auto['retrieval']['applicable']} 项中可信判定 "
        f"{auto['retrieval']['pass'] + auto['retrieval']['fail'] + auto['retrieval']['error'] + auto['retrieval']['not_verified']} 项，"
        f"其余 {auto['retrieval']['pending_annotation']} 项待标注复核。",
    ]

    lines += ["", "## 失败案例（可信子集）", ""]
    if not result["failure_cases"]:
        lines.append("（无）")
    for row in result["failure_cases"][:10]:
        lines.append(f"- {row['scenario_id']} turn {row.get('turn_index')}：{json.dumps(row, ensure_ascii=False)[:400]}")

    lines += ["", "## 运行错误", ""]
    if not result["error_rows"]:
        lines.append("（无运行错误）")
    for row in result["error_rows"]:
        lines.append(f"- {row['scenario_id']} turn {row['turn_index']}：{json.dumps(row, ensure_ascii=False)[:400]}")

    lines += ["", "## 未验证项与已知限制", ""]
    not_verified = [
        row for row in result["per_turn_rows"] if row.get("result") == "NOT_VERIFIED"
    ]
    lines.append(f"- 未采集/不完整 attempt 轮次：{len(not_verified)}")
    for row in not_verified[:20]:
        lines.append(f"  - {row['scenario_id']} turn {row['turn_index']}：{row.get('reason')}")
    if coverage["semantic_rate"] is not None and coverage["semantic_rate"] < 1:
        lines.append("- 语义复核未完成：最终回答质量结论待独立复核（REVIEW_PENDING）。")
    lines.append(
        "- 语义评分为执行者初评（executor_preliminary）；“待答案复核为 0”仅表示程序校验与初评覆盖完成，"
        "不代表独立法律复核完成。"
    )
    lines.append("- route_only 场景只验证工具规划，不代表联网检索或端到端成功率。")
    lines.append("- 本报告由离线汇总生成，生成过程不调用模型；报告不修改任何复核记录。")
    lines += [
        "",
        "## 产物路径",
        "",
        f"- 运行目录：{run_dir}",
        f"- 冻结题集：{run_dir / 'cases_frozen.json'}",
        f"- 原始结果：{run_dir / 'results.jsonl'}",
        f"- 复核记录：{run_dir / 'reviews.json'}",
        f"- 运行清单：{run_dir / 'manifest.json'}",
    ]
    if conversion:
        lines.append(f"- 转换记录：{run_dir / 'conversion.json'}")
    return "\n".join(lines) + "\n"


def run_report(run_dir: Path) -> dict:
    """离线汇总：读取冻结题集、results.jsonl、reviews.json，输出 report.md。

    - 只读运行目录的冻结题集（F-03），绝不改读工作区题集；
    - R2-02：评分前核验冻结文件当前 hash 与 manifest/conversion 绑定身份，失配拒绝生成报告
      （不改写任何证据文件、不覆盖已有报告）；reviews 绑定失配时全部复核失效并写明原因；
    - reviews.json 已存在时只读不写（F-02：不自动补 hash、不伪装已审核）。
    """
    manifest = load_json(run_dir / "manifest.json")
    cases, cases_sha256 = load_frozen_cases(run_dir)

    conversion = None
    conversion_path = run_dir / "conversion.json"
    if conversion_path.exists():
        conversion = load_json(conversion_path)

    identity_problems = verify_frozen_identity(manifest, conversion, cases_sha256)
    if identity_problems:
        raise RuntimeError(
            "冻结题集身份失配，拒绝生成报告（不改写任何证据文件、不覆盖已有报告）："
            + "；".join(identity_problems)
        )

    records = read_existing_results(run_dir)
    selected = select_attempt_groups(records, cases["scenarios"])

    reviews_path = run_dir / "reviews.json"
    if reviews_path.exists():
        reviews = load_json(reviews_path)
        reviews_created = False
    else:
        reviews = build_review_template(
            cases, selected, run_id=str(manifest.get("run_id") or run_dir.name)
        )
        reviews["cases_sha256"] = cases_sha256
        reviews_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
        reviews_created = True

    result = aggregate(cases, selected, reviews, frozen_cases_sha256=cases_sha256)
    result["selected"] = selected
    report_md = build_report_md(run_dir, manifest, cases, cases_sha256, result, conversion)
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")
    return {
        "manifest": manifest,
        "cases_sha256": cases_sha256,
        "result": result,
        "report_path": run_dir / "report.md",
        "reviews_path": reviews_path,
        "reviews_created": reviews_created,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_validate(args) -> int:
    cases = load_cases(Path(args.cases_file) if args.cases_file else CASES_PATH)
    regression = load_regression_questions()
    kb_articles = load_kb_articles()
    errors = validate_case_set(cases, regression, kb_articles)
    cases_path = Path(args.cases_file) if args.cases_file else CASES_PATH
    print(f"cases_sha256={sha256_file(cases_path)}")
    print(f"scorer_sha256={sha256_file(Path(__file__))}")
    if errors:
        print(f"校验失败：{len(errors)} 项")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"校验通过：{len(cases['scenarios'])} 个场景（版本 {cases.get('version')}）")
    return 0


def cmd_collect(args) -> int:
    cases = load_cases()
    requested_ids = [item.strip() for token in (args.case_id or []) for item in token.split(",") if item.strip()]
    if requested_ids:
        known = {scenario["id"]: scenario for scenario in cases["scenarios"]}
        unknown = [scenario_id for scenario_id in requested_ids if scenario_id not in known]
        if unknown:
            print(f"未知场景 ID：{unknown}")
            return 2
        scenarios = [known[scenario_id] for scenario_id in requested_ids]
    else:
        scenarios = cases["scenarios"]

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = EVAL_ROOT / run_id
    pass_record = run_collection(
        run_dir=run_dir,
        scenarios=scenarios,
        run_id=run_id,
        cases=cases,
        cases_sha256=cases_content_sha256(cases),
        scorer_sha256=sha256_file(Path(__file__)),
        with_isolation=True,
    )
    print(f"采集完成：run_id={run_id}，ok={pass_record['counts']['ok']}，error={pass_record['counts']['error']}")
    print(f"产物目录：{run_dir}")
    print(f"真实目录未被改动：{pass_record['real_dirs_unchanged']}")
    return 0 if pass_record["counts"]["error"] == 0 else 1


def cmd_rescore(args) -> int:
    source_dir = EVAL_ROOT / args.source_run
    target_dir = EVAL_ROOT / args.run_id
    if not source_dir.exists():
        print(f"源运行目录不存在：{source_dir}")
        return 2
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"目标运行目录已存在且非空，拒绝覆盖：{target_dir}")
        return 2
    conversion = rescore_convert(source_dir, target_dir, change_reason=args.reason)
    print(f"转换完成：{conversion['conversion_id']}")
    print(f"来源核验：source_results_sha256={conversion['source_results_sha256']}")
    print(f"冻结题集：{conversion['frozen_cases_sha256']}")
    print(f"新评分脚本：{conversion['new_scorer_sha256']}")
    print(f"产物目录：{target_dir}")
    return 0


def cmd_report(args) -> int:
    run_dir = EVAL_ROOT / args.run_id
    if not run_dir.exists():
        print(f"运行目录不存在：{run_dir}")
        return 2
    try:
        result = run_report(run_dir)
    except RuntimeError as error:
        print(f"报告生成被拒绝：{error}")
        return 2
    print(f"报告已生成：{result['report_path']}")
    print(f"复核记录：{result['reviews_path']}{'（本次新建模板）' if result['reviews_created'] else '（已存在，未修改）'}")
    coverage = result["result"]["coverage"]
    print(f"语义评分项复核覆盖率：{coverage['semantic_items_reviewed']}/{coverage['semantic_items_total']}")
    citation = result["result"]["citation"]
    print(f"引用审核覆盖率：{citation['reviewed']}/{citation['extracted_needing_evidence']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LawRAG 回答质量与工具路由评测 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="离线校验题集（不访问网络）")
    validate_parser.add_argument("--cases-file", default="", help="替代题集路径（默认 tests/evaluation_cases.json）")
    validate_parser.set_defaults(func=cmd_validate)

    collect_parser = subparsers.add_parser("collect", help="真实采集（复用实际问答图）")
    collect_parser.add_argument("--run-id", default="", help="运行 ID（默认时间戳）")
    collect_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="定向执行指定场景（可重复或逗号分隔）；默认全集",
    )
    collect_parser.set_defaults(func=cmd_collect)

    rescore_parser = subparsers.add_parser(
        "rescore", help="显式转换既有运行到新目录（修正规则重算，记录来源核验）"
    )
    rescore_parser.add_argument("--source-run", required=True, help="源运行 ID（原目录只读）")
    rescore_parser.add_argument("--run-id", required=True, help="目标运行 ID（新目录）")
    rescore_parser.add_argument("--reason", default="", help="变更原因")
    rescore_parser.set_defaults(func=cmd_rescore)

    report_parser = subparsers.add_parser("report", help="离线汇总（不调用模型，不修改复核记录）")
    report_parser.add_argument("--run-id", required=True, help="要汇总的运行 ID")
    report_parser.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
