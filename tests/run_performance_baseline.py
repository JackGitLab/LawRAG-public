"""LawRAG 第三阶段第一步：请求性能与调用成本基线测量 CLI（R1 返工版）。

四种模式：
- validate：离线校验性能场景（tests/performance_cases.json）并冻结其 SHA-256，不访问网络。
- collect ：真实采集：启动本次专用的隔离服务进程，用真实 HTTP/SSE 执行预热与正式样本，
            逐条落盘 requests.jsonl（含原始 SSE 时间线）；服务端观测由 app 侧记录器写入
            observations.jsonl。
- report  ：离线汇总：只读运行目录已保存数据，在**新的输出目录**生成 report.md 与派生请求清单，
            不调用服务、不联网、绝不回写运行目录原始文件。
- serve   ：内部子命令：在隔离数据（LAWRAG_DB_PATH + Chroma 副本）上启动本次专用 uvicorn，
            由 collect 以子进程方式拉起，不占用、不接触用户已有服务。

R1 返工口径（对应 outputs/STAGE3_STEP1_ACCEPTANCE_20261010.md F-01～F-07）：
- 成功指标只属于满足成功条件的请求；失败请求的时间观察值单独保留，不进入成功统计。
- 一个完整场景执行（多轮共一个任务）才算一个成功任务；部分成功轮次不计完整场景耗时。
- 服务端观测按阶段（预热/正式）分流；性能表与瓶颈占比只用正式样本，预热单列，
  调用/费用可另给“含预热”的整次采集总账。
- 费用只用该次运行 manifest 冻结的单价与模型映射；usage 按字段判已知/未知；
  每成功任务费用 = 正式已知费用 ÷ 正式完整成功任务数（为 0 时 N/A）。
- 新采集保存原始 SSE 时间线（相对单调时钟偏移），报告从原始时间线重算指标；
  缺时间线的旧样本标 NOT_VERIFIED，不倒造。
- 服务启动失败（超时/提前退出/异常）都清理本次进程；隔离与清理保存实际证据快照。

R2 返工口径（对应 R1 独立复验 outputs/STAGE3_STEP1_R1_ACCEPTANCE_20261010.md R1-F01～F05）：
- 费用逐调用判断可计费/未知，再按模型汇总；同模型未知调用不抹掉已知费用，
  一次调用缺多个字段只计一个未知调用；缓存未知上界只针对符合条件的未知调用。
- 计划任务/计划请求来自 manifest 冻结计划（场景、预热次数、正式重复次数、必需轮次），
  未执行任务保留展示；每成功任务费用分母 = 实际完整成功任务数（不是计划数）。
- 完整任务检查轮次身份与唯一性：计划每轮恰好一条成功且路径匹配的记录。
- done 后文本按原始事件接收顺序判断，同刻事件按保存顺序定先后，不排序改写时间线。
- 缺单项冻结费率产生明确未知原因（含缺失字段名），不回退默认价、不静默记零、不抛 TypeError。

R3 返工口径（对应 outputs/STAGE3_STEP1_R2_ACCEPTANCE_20261010.md R2-F01）：
- 计划/实际/异常三类分离：计划数字只来自 manifest 冻结槽位，未执行 = 空槽位数；
  实际记录按 (scenario_id, phase, rep, turn_index) 匹配槽位，恰好一条才算计划内成功；
  同槽位多条为重复异常、身份不在计划内为计划外记录，全部单列保留身份，
  不填补缺失槽位、不进计划内成功耗时与完整成功任务分母，其可核实费用仍计费用分子。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = Path(__file__).resolve()
CASES_PATH = ROOT / "tests" / "performance_cases.json"
PERF_ROOT = ROOT / "logs" / "performance"

# 真实业务目录：只做只读快照与副本来源，测量绝不写入
CHROMA_SOURCE_DIR = ROOT / "chroma_db"
REAL_DATA_DIR = ROOT / "data"
REAL_KB_DIR = ROOT / "kb"

APP_FILES_FOR_HASH = (
    "app/observability.py",
    "app/graph.py",
    "app/ask.py",
    "app/main.py",
    "app/config.py",
    "tests/run_performance_baseline.py",
    "tests/performance_cases.json",
)

WARMUP_PASSES = 1
FORMAL_REPS = 3
HTTP_TIMEOUT_SECONDS = 600
HEALTHZ_TIMEOUT_SECONDS = 180
CORRELATE_FALLBACK_SECONDS = 1.0

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens")

# 检索步骤 → 供应商模型（费用审计用；BM25 为进程内计算，无外部调用）
RETRIEVAL_MODEL_BY_STEP = {
    "vector": "BAAI/bge-m3",
    "rerank": "BAAI/bge-reranker-v2-m3",
}

# 公开单价（来源链接、抓取时间、摘录、适用条件、核验状态随 manifest 冻结）。
# 均为“按公开单价估算”，不代表实际账单扣费；未核实事项在 note/conditions 写明。
PRICING = [
    {
        "service": "DeepSeek",
        "model": "deepseek-chat",
        "applies_to_models": ["deepseek-chat", "deepseek-flash"],
        "currency": "CNY",
        "unit": "元/百万tokens",
        "input_cache_miss_per_m": 1.0,
        "input_cache_hit_per_m": 0.02,
        "output_per_m": 2.0,
        "source_url": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
        "checked_on": "2026-10-10",
        "source_excerpt": (
            "deepseek-v4-flash 价格行：百万tokens输入（缓存命中）0.02元、（缓存未命中）1元、输出 2元；"
            "脚注：deepseek-chat 与 deepseek-reasoner 于 2026/07/24 弃用，分别对应 deepseek-v4-flash 的非思考与思考模式。"
        ),
        "conditions": (
            "按 deepseek-v4-flash（Flash 档）计价。同一 URL 在 2026-10-10 不同抓取中曾显示分时定价"
            "（输出空闲 4 元 / 高峰 8 元）版本，页面存在多版本快照差异；采用值为 2026-10-10（R1 复核）"
            "直接抓取页面所列 0.02/1/2 元。"
        ),
        "price_status": "assumed",
        "note": "页面价格可能变动，估算非实际账单；历史采集时的页面快照缺失，不能证明历史费率。",
    },
    {
        "service": "SiliconFlow",
        "model": "BAAI/bge-m3",
        "applies_to_models": ["BAAI/bge-m3"],
        "currency": "CNY",
        "unit": "元/百万tokens",
        "price_per_m": 0.0,
        "source_url": "https://siliconflow.cn/pricing",
        "checked_on": "2026-10-10",
        "source_excerpt": "价格中心列表：BAAI/bge-m3 免费；Pro/BAAI/bge-m3 ¥0.07。",
        "conditions": "免费档为非 Pro 档；Pro 档另计。供应商未返回 token 用量。",
        "price_status": "assumed",
        "note": "公开免费档不能证明本账号实际费用为零；账户档位/抵扣未核实。",
    },
    {
        "service": "SiliconFlow",
        "model": "BAAI/bge-reranker-v2-m3",
        "applies_to_models": ["BAAI/bge-reranker-v2-m3"],
        "currency": "CNY",
        "unit": "元/百万tokens",
        "price_per_m": 0.0,
        "source_url": "https://siliconflow.cn/pricing",
        "checked_on": "2026-10-10",
        "source_excerpt": "价格中心列表：BAAI/bge-reranker-v2-m3 免费；Pro/BAAI/bge-reranker-v2-m3 ¥0.07。",
        "conditions": "免费档为非 Pro 档；Pro 档另计。供应商未返回 token 用量。",
        "price_status": "assumed",
        "note": "公开免费档不能证明本账号实际费用为零；账户档位/抵扣未核实。",
    },
    {
        "service": "博查 Web Search",
        "model": "web-search",
        "applies_to_models": ["web_search", "verify_statute_status"],
        "currency": "CNY",
        "unit": "元/次",
        "price_per_call": None,
        "source_url": "https://open.bochaai.com/pricing",
        "checked_on": "2026-10-10",
        "source_excerpt": (
            "页面显示：'No fees, No subscriptions… Individual Free / web searches… "
            "Enterprise: Custom pricing based on usage.'（2026-10-10 R1 复核抓取；同日早前该 URL 曾返回 404，页面不稳定。）"
        ),
        "conditions": "个人档公开免费、企业档自定义定价；本账号实际档位未知。",
        "price_status": "unknown",
        "note": "单次调用价格无法核验，费用按未知处理，不以 0 表示。",
    },
]


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
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True, timeout=30
        )
        return bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return True


def snapshot_dir(path: Path) -> dict:
    """只读快照（文件名、大小、mtime、SHA-256）；目录不存在视为空集合。"""
    entries = {}
    if not path.exists():
        return {"exists": False, "entries": {}}
    for child in sorted(path.rglob("*")):
        if child.is_file():
            stat = child.stat()
            entries[child.relative_to(path).as_posix()] = {
                "size": stat.st_size,
                "mtime": round(stat.st_mtime, 3),
                "sha256": sha256_file(child),
            }
    return {"exists": True, "entries": entries}


def snapshot_real_dirs() -> dict:
    return {
        "chroma_db": snapshot_dir(CHROMA_SOURCE_DIR),
        "data": snapshot_dir(REAL_DATA_DIR),
        "kb": snapshot_dir(REAL_KB_DIR),
    }


# ---------------------------------------------------------------------------
# 场景冻结与校验
# ---------------------------------------------------------------------------


def load_cases(path: Path = CASES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cases_sha256(cases: dict) -> str:
    payload = json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


SCENARIO_KEYS = {"id", "name", "measurement_focus", "turns"}
TURN_KEYS = {"question", "use_web_search", "expected_path", "note"}
PATH_KEYS = {
    "nodes",
    "llm_calls",
    "tool_executions",
    "tool_inits",
    "retrieval_steps",
    "answer_mode",
}
ANSWER_MODES = {"generated", "deterministic_clarify"}


def validate_cases(cases: dict) -> list[str]:
    problems = []
    if cases.get("format") != "lawrag-performance-cases/1.0.0":
        problems.append(f"format 非法：{cases.get('format')!r}")
    scenarios = cases.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return problems + ["scenarios 缺失或为空"]
    ids = set()
    for scenario in scenarios:
        sid = scenario.get("id")
        if not sid:
            problems.append("场景缺少 id")
            continue
        if sid in ids:
            problems.append(f"场景 id 重复：{sid}")
        ids.add(sid)
        for key in SCENARIO_KEYS - set(scenario):
            problems.append(f"{sid} 缺少字段 {key}")
        for turn_index, turn in enumerate(scenario.get("turns") or [], start=1):
            for key in TURN_KEYS - set(turn):
                problems.append(f"{sid} 第 {turn_index} 轮缺少字段 {key}")
            path = turn.get("expected_path") or {}
            for key in PATH_KEYS - set(path):
                problems.append(f"{sid} 第 {turn_index} 轮 expected_path 缺少 {key}")
            if path.get("answer_mode") not in ANSWER_MODES:
                problems.append(f"{sid} 第 {turn_index} 轮 answer_mode 非法：{path.get('answer_mode')!r}")
            if not str(turn.get("question") or "").strip():
                problems.append(f"{sid} 第 {turn_index} 轮 question 为空")
    return problems


# ---------------------------------------------------------------------------
# 客户端指标（纯函数，可控时钟：事件时间由调用方给出）
# ---------------------------------------------------------------------------


def derive_metrics(items: list, t_start: float, t_end: float, failure: dict | None = None) -> dict:
    """从事件时间线推导单请求指标（R1-01 核心，原始观察与成功指标分离）。

    items: [{"t": 秒（单调时钟同一基准）, "type": 事件类型, "text_nonempty": 是否非空可见文本}]
    failure: {"t": 秒, "error": str}，连接异常时由调用方给出。

    成功条件（全部满足才 status=ok，否则 failed）：
    无连接异常、无 error 事件、恰好 1 个 done、至少 1 个非空文本且首个非空文本在首个
    done 之前、done 之后没有非空文本。成功指标（first_text_ms / done_ms）只属于满足
    成功条件的请求；失败请求的时间观察值保留在 observed_* / failure_ms，不进入成功统计。

    “done 之后”一律按 items 的接收顺序判断（R2-04）：时间戳只用于计时，
    不替代事件先后；同刻事件也按保存顺序定先后，绝不对时间线重新排序。
    """
    first_text_t = None
    first_text_after_done = False
    done_times: list[float] = []
    error_times: list[float] = []
    text_event_count = 0
    text_after_done = False
    for item in items:
        t = float(item["t"])
        if item.get("type") == "token" and item.get("text_nonempty"):
            text_event_count += 1
            if first_text_t is None:
                first_text_t = t
                first_text_after_done = bool(done_times)
            if done_times:
                text_after_done = True
        elif item.get("type") == "done":
            done_times.append(t)
        elif item.get("type") == "error":
            error_times.append(t)

    failure_t = float(failure["t"]) if failure else None
    failure_reasons = []
    if failure:
        failure_reasons.append(f"连接异常：{failure.get('error')}")
    if error_times:
        failure_reasons.append("收到错误事件")

    notes = list(failure_reasons)
    if len(done_times) == 0:
        notes.append("缺少 done 事件")
    elif len(done_times) > 1:
        notes.append(f"done 事件数 {len(done_times)} > 1")
    if first_text_t is None:
        notes.append("缺少非空回答文本")
    if text_after_done:
        notes.append("done 之后仍有非空文本事件")
    if first_text_after_done:
        notes.append("首段回答出现在首个 done 之后")

    success = not notes
    observed_first_text = round((first_text_t - t_start) * 1000, 3) if first_text_t is not None else None
    observed_done = round((done_times[0] - t_start) * 1000, 3) if done_times else None
    failure_ms = None
    candidates = [t for t in (*error_times, failure_t) if t is not None]
    if candidates:
        failure_ms = round((min(candidates) - t_start) * 1000, 3)
    return {
        "status": "ok" if success else "failed",
        "duration_ms": round((t_end - t_start) * 1000, 3),
        # 成功指标：仅成功请求有值；失败请求为 None（不填零、不进成功统计）
        "first_text_ms": observed_first_text if success else None,
        "done_ms": observed_done if success else None,
        # 原始时间观察：无论成败都保留，永不进入成功统计
        "observed_first_text_ms": observed_first_text,
        "observed_done_ms": observed_done,
        "failure_ms": failure_ms,
        "done_count": len(done_times),
        "text_event_count": text_event_count,
        "notes": notes,
    }


def compute_request_metrics(events: list, t_start: float, t_end: float, failure: dict | None = None) -> dict:
    """从客户端 SSE 事件序列计算单请求指标（兼容入口，内部走 derive_metrics）。"""
    items = [
        {
            "t": float(item["t"]),
            "type": (item.get("event") or {}).get("type"),
            "text_nonempty": bool(str((item.get("event") or {}).get("text") or "").strip()),
        }
        for item in events
    ]
    return derive_metrics(items, t_start, t_end, failure)


def metrics_from_raw(record: dict) -> dict | None:
    """从记录里的原始 SSE 时间线重算指标；缺原始时间线返回 None（不得倒造）。"""
    events = record.get("events")
    t_end = record.get("t_end")
    if not isinstance(events, list) or t_end is None:
        return None
    items = [
        {"t": e.get("t"), "type": e.get("type"), "text_nonempty": bool(e.get("text_nonempty"))}
        for e in events
    ]
    return derive_metrics(items, 0.0, float(t_end), record.get("failure"))


def summarize(values: list) -> dict:
    """逐样本汇总：缺失值不填零、不参与 min/median/max，缺失数单列。"""
    present = [float(value) for value in values if value is not None]
    return {
        "n": len(present),
        "missing": len(values) - len(present),
        "min": round(min(present), 3) if present else None,
        "median": round(statistics.median(present), 3) if present else None,
        "max": round(max(present), 3) if present else None,
    }


# ---------------------------------------------------------------------------
# 任务口径（R1-01/R1-04）：一次完整场景执行 = 一个任务
# ---------------------------------------------------------------------------


def evaluate_tasks(requests: list[dict], manifest: dict) -> list[dict]:
    """从 manifest 冻结计划展开任务，再匹配实际记录（R2-02/R2-03）。

    - 计划任务 = 场景 ×（warmup 1..warmup_passes ∪ formal 1..formal_reps），
      计划轮次 = 场景 turns；计划任务/计划请求不随实际记录减少，未执行任务保留为
      not_executed。实际记录中超出冻结计划的 (scenario, phase, rep) 组合也保留，
      标记为计划外，不静默丢弃。
    - 完整成功 = 计划规定的每一轮**恰好一条**记录（轮次身份与唯一性），且该记录
      请求成功（status=ok）且路径符合预期；仅“记录条数相等”不算完整。
      重复轮次、缺轮、越界轮次均使任务无效（不进完整场景耗时与每成功任务费用分母），
      但实际发生且可核实的调用费用仍由费用口径按观测计入分子。
    - duration_ms 只按实际记录合计，仅为事实记录；未执行任务为 None。
    """
    scenarios = manifest.get("scenarios") or []
    warmup_passes = int(manifest.get("warmup_passes") or 0)
    formal_reps = int(manifest.get("formal_reps") or 0)
    expected_turns = {entry["id"]: int(entry.get("turns") or 0) for entry in scenarios}
    groups: dict[tuple, list] = {}
    for record in requests:
        key = (record.get("scenario_id"), record.get("phase"), record.get("rep"))
        groups.setdefault(key, []).append(record)

    plan: list[tuple] = []  # (scenario_id, phase, rep, planned_turns, planned)
    for entry in scenarios:
        sid = entry["id"]
        for phase, passes in (("warmup", warmup_passes), ("formal", formal_reps)):
            for rep in range(1, passes + 1):
                plan.append((sid, phase, rep, expected_turns.get(sid, 0), True))
    planned_keys = {(sid, phase, rep) for sid, phase, rep, _, _ in plan}
    for key in sorted(groups, key=str):
        if key not in planned_keys:
            sid, phase, rep = key
            plan.append((sid, phase, rep, expected_turns.get(sid, 0), False))

    tasks = []
    for sid, phase, rep, planned_turns, is_planned in sorted(
        plan, key=lambda item: (str(item[0]), str(item[1]), item[2] if isinstance(item[2], int) else 0)
    ):
        records = groups.get((sid, phase, rep), [])
        problems = []
        if not is_planned:
            problems.append("实际记录不在冻结计划内（scenario/phase/rep 计划外）")
        by_turn: dict = {}
        for record in records:
            by_turn.setdefault(record.get("turn_index"), []).append(record)
        for turn in sorted(by_turn, key=lambda value: (value is None, value if isinstance(value, int) else 0)):
            if not isinstance(turn, int) or not (1 <= turn <= planned_turns):
                problems.append(f"越界/非法轮次 T{turn}（计划轮次 1~{planned_turns}）")
        for turn in range(1, planned_turns + 1):
            rows = by_turn.get(turn, [])
            if not rows:
                problems.append(f"缺少轮次 T{turn}")
            elif len(rows) > 1:
                problems.append(f"轮次 T{turn} 重复 {len(rows)} 条（计划每轮恰好一条）")
        for record in records:
            if record.get("metrics", {}).get("status") != "ok":
                problems.append(f"{record.get('request_id')} 请求未成功")
            if not record.get("path_match"):
                problems.append(f"{record.get('request_id')} 路径不符")
        executed = bool(records)
        complete_success = not problems
        tasks.append(
            {
                "task_id": f"{sid}-{phase}{rep}",
                "scenario_id": sid,
                "phase": phase,
                "rep": rep,
                "planned": is_planned,
                "planned_turns": planned_turns,
                "turns": len(records),
                "executed": executed,
                "status": (
                    "complete_success" if complete_success
                    else ("executed_incomplete" if executed else "not_executed")
                ),
                "complete_success": complete_success,
                "problems": problems,
                "records": [
                    {
                        "request_id": record.get("request_id"),
                        "turn_index": record.get("turn_index"),
                        "status": (record.get("metrics") or {}).get("status"),
                    }
                    for record in sorted(records, key=lambda r: (r.get("turn_index") or 0, str(r.get("request_id"))))
                ],
                "duration_ms": (
                    round(sum(float(record.get("metrics", {}).get("duration_ms") or 0.0) for record in records), 3)
                    if records else None
                ),
            }
        )
    return tasks


def classify_request_slots(requests: list[dict], manifest: dict) -> dict:
    """按冻结计划槽位匹配实际记录（R3：计划 / 唯一匹配 / 异常三类分离）。

    请求槽位身份 = (scenario_id, phase, rep, turn_index)，全部来自 manifest 冻结计划：
    - 槽位恰好 1 条记录 → 唯一匹配（成功/失败按该记录 metrics.status）；
    - 槽位 0 条 → 未执行；槽位多条 → 重复异常槽位（不任选一条算成功）；
    - 身份不在计划槽位内（计划外 rep/场景/阶段、越界轮次）→ 计划外异常记录。
    计划数字 = 槽位数（不按实际条数反推）；未执行 = 空槽位数（不用计划数－实际条数，
    不用 max(...) 掩盖错配）；计划外/重复记录不填补任何缺失槽位，身份全部保留单列。
    """
    scenarios = manifest.get("scenarios") or []
    warmup_passes = int(manifest.get("warmup_passes") or 0)
    formal_reps = int(manifest.get("formal_reps") or 0)
    turns_of = {entry["id"]: int(entry.get("turns") or 0) for entry in scenarios}
    slots: dict[tuple, list] = {}
    for entry in scenarios:
        sid = entry["id"]
        for phase, passes in (("warmup", warmup_passes), ("formal", formal_reps)):
            for rep in range(1, passes + 1):
                for turn in range(1, turns_of.get(sid, 0) + 1):
                    slots[(sid, phase, rep, turn)] = []
    out_of_plan = []
    for record in requests:
        key = (record.get("scenario_id"), record.get("phase"), record.get("rep"), record.get("turn_index"))
        if key in slots:
            slots[key].append(record)
        else:
            out_of_plan.append(record)
    matched: list[dict] = []
    duplicate_records: list[dict] = []
    slot_status: dict[tuple, str] = {}
    for key, rows in slots.items():
        if not rows:
            slot_status[key] = "not_executed"
        elif len(rows) == 1:
            slot_status[key] = "success" if rows[0].get("metrics", {}).get("status") == "ok" else "failed"
            matched.append(rows[0])
        else:
            slot_status[key] = "duplicate"
            duplicate_records.extend(rows)
    anomalies = [
        (record, "计划外记录（身份不在冻结计划槽位内）") for record in out_of_plan
    ] + [
        (record, "重复记录（同槽位多条，该槽位判重复异常，不计成功）") for record in duplicate_records
    ]
    status_values = list(slot_status.values())
    return {
        "slots": slots,
        "slot_status": slot_status,
        "matched_records": matched,
        "anomaly_records": anomalies,
        "counts": {
            "planned_slots": len(slots),
            "success_slots": status_values.count("success"),
            "failed_slots": status_values.count("failed"),
            "duplicate_slots": status_values.count("duplicate"),
            "empty_slots": status_values.count("not_executed"),
            "matched_records": len(matched),
            "out_of_plan_records": len(out_of_plan),
            "duplicate_records": len(duplicate_records),
            "anomaly_records": len(anomalies),
            "physical_ok": sum(1 for r in requests if r.get("metrics", {}).get("status") == "ok"),
            "physical_failed": sum(1 for r in requests if r.get("metrics", {}).get("status") != "ok"),
        },
    }


# ---------------------------------------------------------------------------
# 服务端观测：加载、归属、分层统计
# ---------------------------------------------------------------------------


def load_observations(path: Path) -> list[dict]:
    """读取观测 JSONL；按 seq 去重（同一条记录重复出现只计一次）。"""
    if not path.exists():
        return []
    records = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            seq = record.get("seq")
            if seq in seen:
                continue
            seen.add(seq)
            records.append(record)
    records.sort(key=lambda item: item.get("seq", 0))
    return records


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def correlate_observations(observations: list[dict], requests: list[dict]) -> dict:
    """按墙钟时间把观测归属到请求（串行执行，精确窗口互不重叠）。

    先按精确时间窗 [wall_start, wall_end] 归属；落在所有窗口之外的记录，
    仅当距最近窗口不超过 1 秒时兜底归属（避免漏掉紧邻边界的服务端记录），
    否则计入 unassigned。窗口不加外扩边距：外扩会让上一请求的 +margin 窗口
    吞掉下一请求早期结束的记录（E-097 预防）。
    """
    windows = [
        (record["request_id"], _parse_ts(record["wall_start"]), _parse_ts(record["wall_end"]))
        for record in requests
    ]
    grouped: dict[str, list] = {request_id: [] for request_id, _, _ in windows}
    unassigned_records = []
    for obs in observations:
        ts = _parse_ts(obs["ts_utc"])
        matched = None
        for request_id, start, end in windows:
            if start <= ts <= end:
                matched = request_id
                break
        if matched is None:
            best: tuple[float, str] | None = None
            for request_id, start, end in windows:
                if start <= ts <= end:
                    continue
                distance = min(abs((ts - start).total_seconds()), abs((ts - end).total_seconds()))
                if distance <= CORRELATE_FALLBACK_SECONDS and (best is None or distance < best[0]):
                    best = (distance, request_id)
            if best is not None:
                matched = best[1]
        if matched is None:
            unassigned_records.append(obs)
        else:
            grouped[matched].append(obs)
    return {
        "by_request": grouped,
        "unassigned": len(unassigned_records),
        "unassigned_records": unassigned_records,
    }


def split_observations_by_phase(observations: list[dict], requests: list[dict]) -> dict:
    """把观测按请求阶段分流（R1-05）：warmup / formal / unassigned（如服务启动期）。"""
    correlation = correlate_observations(observations, requests)
    phase_of = {record["request_id"]: record.get("phase") for record in requests}
    split: dict[str, list] = {"warmup": [], "formal": [], "unassigned": list(correlation["unassigned_records"])}
    for request_id, records in correlation["by_request"].items():
        bucket = "warmup" if phase_of.get(request_id) == "warmup" else "formal"
        split[bucket].extend(records)
    return split


def build_actual_path(obs_records: list[dict]) -> dict:
    """从观测记录还原实际执行路径（按 seq 排序）。"""
    nodes, llm_calls, tool_executions, tool_inits, retrieval_steps = [], [], [], [], []
    for record in obs_records:
        kind, name = record.get("kind"), record.get("name")
        if kind == "node":
            nodes.append(name)
        elif kind == "llm":
            llm_calls.append(name)
        elif kind == "tool":
            (tool_inits if name == "mcp_init" else tool_executions).append(name)
        elif kind == "retrieval":
            retrieval_steps.append(name)
    return {
        "nodes": nodes,
        "llm_calls": llm_calls,
        "tool_executions": tool_executions,
        "tool_inits": tool_inits,
        "retrieval_steps": retrieval_steps,
    }


def path_matches(expected: dict, actual: dict) -> bool:
    """按多重集比较（顺序不敏感）；五类路径全部一致才算匹配。"""
    for key in ("nodes", "llm_calls", "tool_executions", "tool_inits", "retrieval_steps"):
        if sorted(expected.get(key) or []) != sorted(actual.get(key) or []):
            return False
    return True


def annotate_requests(requests: list[dict], observations: list[dict]) -> tuple[list[dict], int]:
    """派生标注（纯变换）：原始时间线重算指标 + 实际路径 + 确定性输出口径。

    - 有原始时间线的样本：metrics 一律从时间线重算，保存的派生 metrics 不被信任；
    - 无原始时间线的样本（旧数据）：保留保存的 metrics 并标 raw_recomputable=False
      （报告标 NOT_VERIFIED），绝不从旧 metrics 反向伪造事件时间线。
    """
    correlation = correlate_observations(observations, requests)
    annotated = []
    for record in requests:
        updated = dict(record)
        recomputed = metrics_from_raw(updated)
        if recomputed is not None:
            updated["metrics"] = recomputed
            updated["raw_recomputable"] = True
        else:
            updated["raw_recomputable"] = False
        obs_records = correlation["by_request"].get(record["request_id"], [])
        actual = build_actual_path(obs_records)
        expected = record.get("expected_path") or {}
        updated["actual_path"] = actual
        updated["path_match"] = path_matches(expected, actual)
        deterministic_actual = "clarify" in actual["nodes"] and "generate" not in actual["llm_calls"]
        expected_mode = expected.get("answer_mode")
        if expected_mode == "deterministic_clarify" or deterministic_actual:
            updated["first_text_kind"] = "deterministic"
        elif updated.get("metrics", {}).get("first_text_ms") is not None:
            updated["first_text_kind"] = "generated"
        else:
            updated["first_text_kind"] = "unknown"
        annotated.append(updated)
    return annotated, correlation["unassigned"]


def aggregate_observations(observations: list[dict]) -> dict:
    """分层汇总：node 层与 call 层分开统计，父子包含关系不跨层相加。

    重排 attempts 为实际可观察次数；缺 attempts 字段的记录计 attempts_unknown
    （历史记录不可倒造为 0，见 E-098/F-02）。
    """
    nodes: dict[str, dict] = {}
    llm_by_purpose: dict[str, dict] = {}
    llm_records = []
    retrieval: dict[str, dict] = {}
    tools: dict[str, dict] = {}
    for record in observations:
        kind, name = record.get("kind"), record.get("name")
        duration = float(record.get("duration_ms") or 0.0)
        bucket = None
        if kind == "node":
            bucket = nodes.setdefault(name, {"count": 0, "failed": 0, "total_ms": 0.0})
        elif kind == "llm":
            bucket = llm_by_purpose.setdefault(
                name, {"count": 0, "failed": 0, "total_ms": 0.0, "usage_known": 0, "usage_missing": 0}
            )
            llm_records.append(record)
        elif kind == "retrieval":
            bucket = retrieval.setdefault(
                name,
                {"count": 0, "failed": 0, "total_ms": 0.0, "attempts": 0, "attempts_unknown": 0},
            )
            if isinstance(record.get("attempts"), (int, float)):
                bucket["attempts"] += int(record["attempts"])
            else:
                bucket["attempts_unknown"] += 1
        elif kind == "tool":
            bucket = tools.setdefault(name, {"count": 0, "failed": 0, "total_ms": 0.0})
        else:
            continue
        bucket["count"] += 1
        if not record.get("ok", True):
            bucket["failed"] += 1
        bucket["total_ms"] = round(bucket["total_ms"] + duration, 3)
        if kind == "llm":
            if record.get("usage") is None:
                bucket["usage_missing"] += 1
            else:
                bucket["usage_known"] += 1
    usage_by_model = usage_totals(llm_records)
    return {
        "node_layer_ms": nodes,
        "llm_by_purpose": llm_by_purpose,
        "retrieval": retrieval,
        "tools": tools,
        "usage_by_model": usage_by_model,
        "llm_records": llm_records,
    }


def usage_totals(llm_records: list[dict]) -> dict:
    """按模型汇总 usage；按字段区分已知/未知，缺失字段绝不填 0。"""
    totals: dict[str, dict] = {}
    for record in llm_records:
        model = record.get("model") or "unknown"
        bucket = totals.setdefault(
            model,
            {
                "calls": 0,
                "calls_with_usage": 0,
                "calls_without_usage": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "input_unknown_calls": 0,
                "output_unknown_calls": 0,
                "cache_unknown_calls": 0,
            },
        )
        bucket["calls"] += 1
        usage = record.get("usage")
        if not isinstance(usage, dict) or not usage:
            bucket["calls_without_usage"] += 1
            for key in ("input_unknown_calls", "output_unknown_calls", "cache_unknown_calls"):
                bucket[key] += 1
            continue
        bucket["calls_with_usage"] += 1
        for field, unknown_key in (
            ("input_tokens", "input_unknown_calls"),
            ("output_tokens", "output_unknown_calls"),
            ("cache_read_tokens", "cache_unknown_calls"),
        ):
            value = usage.get(field)
            if isinstance(value, (int, float)):
                bucket[field] += int(value)
            else:
                bucket[unknown_key] += 1
    return totals


# ---------------------------------------------------------------------------
# 费用（R1-03/R1-04）：只用冻结价格，字段级 usage，未知不填零
# ---------------------------------------------------------------------------


def price_entry_for(pricing: list[dict] | None, model: str) -> dict | None:
    for entry in pricing or []:
        applies = entry.get("applies_to_models") or [entry.get("model")]
        if model in applies:
            return entry
    return None


def price_status_of(entry: dict | None) -> str:
    """价格依据状态：verified（有来源摘录）/ assumed（估算假设）/ unknown（不可核验）。"""
    if not entry:
        return "missing"
    for key in ("source_url", "checked_on", "currency", "unit"):
        if not entry.get(key):
            return "unknown"
    status = str(entry.get("price_status") or entry.get("verification_status") or "")
    if status in {"verified", "frozen_verified"}:
        return "frozen_verified"
    if status == "unknown":
        return "unknown"
    return "assumed"


def _price_rates(entry: dict) -> tuple[str, dict, list[str]]:
    """返回 (计价方式, 费率, 缺失费率字段)（R2-05）。

    当前计价方式实际所需的每个费率都要检查：缺字段或值为 None 一律列入缺失清单，
    调用方必须先处理缺失清单，绝不把 None 送进 float*None 算式；
    显式数值 0 是有效费率，不视为缺失。
    """
    cache_fields = ("input_cache_miss_per_m", "input_cache_hit_per_m", "output_per_m")
    if any(entry.get(field) is not None for field in cache_fields):
        return (
            "token_cache_split",
            {field: entry.get(field) for field in cache_fields},
            [field for field in cache_fields if entry.get(field) is None],
        )
    if entry.get("price_per_m") is not None:
        return "token_flat", {"price_per_m": entry.get("price_per_m")}, []
    if entry.get("price_per_call") is not None:
        return "per_call", {"price_per_call": entry.get("price_per_call")}, []
    return "none", {}, ["计价费率字段全部缺失或为 None（input_cache_miss_per_m/input_cache_hit_per_m/output_per_m/price_per_m/price_per_call）"]


def _cost_one_call(record: dict, model: str, entry: dict | None) -> dict:
    """逐调用判断该次调用可否计费（R2-01/R2-05）。

    返回 {"cost": float|None, "reason": str|None, "upper_bound": float|None}：
    - cost 为 None 即该次调用费用未知，reason 给出可追溯原因（单价/费率/usage 字段）；
    - 仅“缓存拆分缺失而输入输出已知”的未知调用给标注上界估算，不进已知合计。
    """
    status = price_status_of(entry)
    if not entry or status in ("missing", "unknown"):
        return {
            "cost": None,
            "reason": f"模型 {model} 缺少可核验的单价（价格状态 {status}）",
            "upper_bound": None,
        }
    kind, rates, missing_rates = _price_rates(entry)
    if missing_rates:
        return {
            "cost": None,
            "reason": f"模型 {model} 缺少计价所需费率：{'、'.join(missing_rates)}（不回退默认价、不按 0 计）",
            "upper_bound": None,
        }
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else None
    if kind == "token_cache_split":
        required = ("input_tokens", "output_tokens", "cache_read_tokens")
    elif kind == "token_flat":
        required = ("input_tokens", "output_tokens")
    else:
        return {
            "cost": None,
            "reason": f"模型 {model} 当前计价方式 {kind} 无法按 token 计费",
            "upper_bound": None,
        }
    missing_usage = [field for field in required if not isinstance((usage or {}).get(field), (int, float))]
    if missing_usage:
        result = {
            "cost": None,
            "reason": f"模型 {model} 调用 usage 缺字段：{'、'.join(missing_usage)}",
            "upper_bound": None,
        }
        if kind == "token_cache_split" and missing_usage == ["cache_read_tokens"]:
            result["upper_bound"] = (
                usage["input_tokens"] / 1_000_000 * rates["input_cache_miss_per_m"]
                + usage["output_tokens"] / 1_000_000 * rates["output_per_m"]
            )
        return result
    if kind == "token_cache_split":
        cost = (
            (usage["input_tokens"] - usage["cache_read_tokens"]) / 1_000_000 * rates["input_cache_miss_per_m"]
            + usage["cache_read_tokens"] / 1_000_000 * rates["input_cache_hit_per_m"]
            + usage["output_tokens"] / 1_000_000 * rates["output_per_m"]
        )
    else:
        cost = (usage["input_tokens"] + usage["output_tokens"]) / 1_000_000 * rates["price_per_m"]
    return {"cost": cost, "reason": None, "upper_bound": None}


def compute_cost(agg: dict, pricing: list[dict] | None) -> dict:
    """按冻结单价估算“已知部分费用”（R2-01：逐调用判断，再按模型汇总）。

    - 每条 llm 调用独立判断该次计价所需 usage 与费率是否完整：可计费调用计入
      known_total（含失败但 usage 可核实的调用）；不可计费调用单列为未知调用
      （一次调用缺多个字段只计一个未知调用），同模型的未知调用不抹掉已知费用；
    - calls_costed、未知调用数、缓存未知上界各自只对应自己的调用集合；
      缓存拆分缺失的上界估算不进已知合计，也不包含已计费调用；
    - 缺单价/缺费率一律保留未知原因（含缺失字段名），不回退默认价、不静默记零。
    """
    items = []
    unknowns = []
    known_total = 0.0
    usage_field_unknowns = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    if not pricing:
        unknowns.append("manifest 中缺少冻结价格（pricing），全部费用未知，不使用当前代码默认价格")

    usage_by_model = agg.get("usage_by_model") or {}
    results_by_model: dict[str, list] = {}
    for record in agg.get("llm_records") or []:
        model = record.get("model") or "unknown"
        results_by_model.setdefault(model, []).append(
            _cost_one_call(record, model, price_entry_for(pricing, model))
        )

    for model in sorted(set(usage_by_model) | set(results_by_model)):
        totals = usage_by_model.get(
            model,
            {
                "calls": 0, "calls_with_usage": 0, "calls_without_usage": 0,
                "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                "input_unknown_calls": 0, "output_unknown_calls": 0, "cache_unknown_calls": 0,
            },
        )
        results = results_by_model.get(model, [])
        entry = price_entry_for(pricing, model)
        status = price_status_of(entry)
        for key, unknown_key in (
            ("input_tokens", "input_unknown_calls"),
            ("output_tokens", "output_unknown_calls"),
            ("cache_read_tokens", "cache_unknown_calls"),
        ):
            usage_field_unknowns[key] += totals[unknown_key]
        costed = [result for result in results if result["cost"] is not None]
        unknown_calls = [result for result in results if result["cost"] is None]
        item = {
            "model": model,
            "calls": totals["calls"],
            "calls_with_usage": totals["calls_with_usage"],
            "calls_without_usage": totals["calls_without_usage"],
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "cache_read_tokens": totals["cache_read_tokens"],
            "input_unknown_calls": totals["input_unknown_calls"],
            "output_unknown_calls": totals["output_unknown_calls"],
            "cache_unknown_calls": totals["cache_unknown_calls"],
            "price_status": status,
            "currency": (entry or {}).get("currency"),
            "source_url": (entry or {}).get("source_url"),
            "checked_on": (entry or {}).get("checked_on"),
            "calls_costed": len(costed),
            "calls_unknown": len(unknown_calls),
            "estimated_cost": round(sum(result["cost"] for result in costed), 6) if costed else None,
        }
        known_total += sum(result["cost"] for result in costed)
        upper_bounds = [result["upper_bound"] for result in unknown_calls if result["upper_bound"] is not None]
        if upper_bounds:
            item["assumed_upper_bound"] = round(sum(upper_bounds), 6)
            item["assumed_upper_bound_note"] = (
                "上界假设：仅缓存拆分缺失、输入输出已知的未知调用按全部未命中计；非确数，不进已知合计"
            )
        if unknown_calls:
            reasons: dict[str, int] = {}
            for result in unknown_calls:
                reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
            reason_text = "；".join(
                f"{reason}（{count} 次）" if count > 1 else reason for reason, count in sorted(reasons.items())
            )
            unknowns.append(f"模型 {model}：{len(unknown_calls)} 次调用费用未知（单次调用缺多个字段只计一次）——{reason_text}")
        items.append(item)

    # 检索与工具的按次/按量计价（向量、重排、联网搜索不得从调用清单消失）
    for step, model in RETRIEVAL_MODEL_BY_STEP.items():
        bucket = (agg.get("retrieval") or {}).get(step)
        if not bucket:
            continue
        entry = price_entry_for(pricing, model)
        status = price_status_of(entry)
        kind, rates, missing_rates = _price_rates(entry) if entry else ("none", {}, ["缺少价格条目"])
        item = {
            "model": f"{model}（{step}）",
            "calls": bucket["count"],
            "calls_costed": 0,
            "estimated_cost": None,
            "price_status": status,
            "currency": (entry or {}).get("currency"),
            "source_url": (entry or {}).get("source_url"),
            "checked_on": (entry or {}).get("checked_on"),
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
        }
        if kind == "per_call" and rates.get("price_per_call") is not None and status != "unknown":
            cost = bucket["count"] * rates["price_per_call"]
            item.update({"calls_costed": bucket["count"], "estimated_cost": round(cost, 6)})
            known_total += cost
        else:
            unknowns.append(
                f"{model}（{step}，{bucket['count']} 次）：供应商未返回 token 用量，"
                f"按量费用未知（公开免费档仅是估算假设，不证明账户费用为零）"
            )
        items.append(item)
    for tool_name in sorted((agg.get("tools") or {}).keys()):
        if tool_name == "mcp_init":
            continue
        bucket = agg["tools"][tool_name]
        entry = price_entry_for(pricing, tool_name)
        status = price_status_of(entry)
        kind, rates, missing_rates = _price_rates(entry) if entry else ("none", {}, ["缺少价格条目"])
        item = {
            "model": f"{tool_name}（按次）",
            "calls": bucket["count"],
            "calls_costed": 0,
            "estimated_cost": None,
            "price_status": status,
            "currency": (entry or {}).get("currency"),
            "source_url": (entry or {}).get("source_url"),
            "checked_on": (entry or {}).get("checked_on"),
        }
        if kind == "per_call" and rates.get("price_per_call") is not None and status != "unknown":
            cost = bucket["count"] * rates["price_per_call"]
            item.update({"calls_costed": bucket["count"], "estimated_cost": round(cost, 6)})
            known_total += cost
        else:
            unknowns.append(
                f"{tool_name}（{bucket['count']} 次）：单次价格不可核验（price_status={status}），费用未知，不以 0 表示"
            )
        items.append(item)

    unknowns.append("所有已计费用均为按冻结公开单价估算，未与实际账单核对；折扣、套餐抵扣不计")
    return {
        "currency": "CNY",
        "basis": "按冻结公开单价估算（非实际扣费）",
        "known_total": round(known_total, 6),
        "items": items,
        "unknowns": unknowns,
        "usage_field_unknowns": usage_field_unknowns,
    }


def compute_scoped_costs(split: dict, pricing: list[dict] | None, tasks: list[dict]) -> dict:
    """分口径费用（R1-04/R1-05）：预热 / 正式 / 整次采集 + 正式每成功任务已知费用。

    每成功任务口径：任务 = 一次完整场景执行（多轮场景共一个任务）；
    分子 = 正式样本全部已知费用（含失败尝试已发生的费用）；
    分母 = 正式完整成功任务数；为 0 时 N/A。
    """
    formal_cost = compute_cost(aggregate_observations(split.get("formal") or []), pricing)
    warmup_cost = compute_cost(aggregate_observations(split.get("warmup") or []), pricing)
    whole_cost = compute_cost(
        aggregate_observations(
            (split.get("warmup") or []) + (split.get("formal") or []) + (split.get("unassigned") or [])
        ),
        pricing,
    )
    # 计划任务数只来自冻结计划（planned）；计划外任务单列，不进计划口径（R3）
    formal_all = [t for t in tasks if t.get("phase") == "formal"]
    formal_planned = [t for t in formal_all if t.get("planned", True)]
    warmup_planned = [t for t in tasks if t.get("phase") == "warmup" and t.get("planned", True)]
    success_tasks = [t for t in formal_planned if t.get("complete_success")]
    per_task = "N/A" if not success_tasks else round(formal_cost["known_total"] / len(success_tasks), 8)
    return {
        "warmup": warmup_cost,
        "formal": formal_cost,
        "whole_run": whole_cost,
        "unassigned_records": len(split.get("unassigned") or []),
        "formal_tasks": len(formal_planned),
        "formal_unplanned_tasks": len(formal_all) - len(formal_planned),
        "formal_complete_success_tasks": len(success_tasks),
        "warmup_tasks": len(warmup_planned),
        "per_success_task_formal": per_task,
        "per_success_task_note": (
            "正式每成功任务已知费用 = 正式范围全部已知费用（含失败及计划外请求已发生的可核实费用）"
            "÷ 正式计划内完整成功任务数；仅为已知费用部分，不等同实际扣费；成功任务数为 0 时为 N/A。"
        ),
    }


# ---------------------------------------------------------------------------
# 报告（离线重建：只依赖运行目录已保存数据；绝不回写原始文件）
# ---------------------------------------------------------------------------


def _fmt(value) -> str:
    return "缺失" if value is None else str(value)


def _usage_text(known: int | None, unknown_calls: int) -> str:
    if unknown_calls and known:
        return f"{known}（另 {unknown_calls} 次未知）"
    if unknown_calls:
        return f"未知（{unknown_calls} 次）"
    return str(known)


def build_report_md(requests: list[dict], observations: list[dict], manifest: dict) -> str:
    requests, unassigned = annotate_requests(requests, observations)
    split = split_observations_by_phase(observations, requests)
    tasks = evaluate_tasks(requests, manifest)
    agg_formal = aggregate_observations(split["formal"])
    agg_warmup = aggregate_observations(split["warmup"])
    agg_whole = aggregate_observations(split["warmup"] + split["formal"] + split["unassigned"])
    costs = compute_scoped_costs(split, manifest.get("pricing"), tasks)
    formal = [record for record in requests if record["phase"] == "formal"]
    warmup = [record for record in requests if record["phase"] == "warmup"]

    lines: list[str] = []
    lines.append("# LawRAG 请求性能与调用成本基线报告（R1 口径）")
    lines.append("")
    lines.append("本报告由 `tests/run_performance_baseline.py report` 从运行目录已保存数据离线重建，不调用任何服务。")
    lines.append("指标一律从原始 SSE 时间线重算；缺时间线的样本标 NOT_VERIFIED（不倒造事件时间线）。")
    lines.append("")
    lines.append("## 1. 运行标识与版本绑定")
    lines.append("")
    lines.append(f"- run_id：{manifest.get('run_id')}")
    lines.append(
        f"- 采集代码版本：git HEAD `{manifest.get('git_head')}`"
        f"（采集时工作区{'有' if manifest.get('git_dirty') else '无'}未提交改动）"
    )
    for name, digest in sorted((manifest.get("code_hashes") or {}).items()):
        lines.append(f"- 采集文件哈希：{name} = `{digest}`")
    lines.append(f"- 报告生成代码版本：tests/run_performance_baseline.py = `{sha256_file(SCRIPT_PATH)}`")
    lines.append("- 说明：采集代码版本与报告生成代码版本独立记录；本报告不含重建时刻（保证同输入重建字节一致）。")
    lines.append(f"- 场景文件哈希：`{manifest.get('cases_sha256')}`")
    lines.append(f"- 场景冻结日期：{manifest.get('cases_frozen_date')}")
    lines.append(f"- 采集时间：{manifest.get('started_utc')} ~ {manifest.get('finished_utc')}")
    lines.append(
        f"- 模型：chat={manifest.get('models', {}).get('chat')}，router={manifest.get('models', {}).get('router')}，"
        f"embedding={manifest.get('models', {}).get('embedding')}，rerank={manifest.get('models', {}).get('rerank')}"
    )
    lines.append(f"- 检索策略：{manifest.get('retrieval_strategy')}")
    lines.append(f"- Langfuse：{manifest.get('langfuse')}（测量期间统一关闭外部上报）")
    lines.append(f"- 服务启动耗时（另列，不属请求指标）：{_fmt(manifest.get('server_start_seconds'))} 秒")
    lines.append("")
    lines.append("## 2. 测量方法与指标定义")
    lines.append("")
    lines.append("- 首段可见回答耗时：客户端开始发送请求 → 第一个非空回答文本事件（token）。`session` 事件不是回答首字。")
    lines.append("- 请求完成耗时：客户端开始发送请求 → 唯一 `done`。")
    lines.append("- 失败耗时：请求开始 → 错误事件或连接异常。")
    lines.append("- 成功条件：无连接异常、无 error 事件、恰好 1 个 done、有非空回答且首段在 done 之前、done 后无非空文本；"
                 "“done 之后”按原始事件接收顺序判断，同刻事件按保存顺序定先后，时间戳只用于计时。")
    lines.append("- **成功指标（first_text_ms / done_ms）只属于满足成功条件的请求**；失败请求即使已收到 token 或 done，"
                 "其时间只保留为原始观察值（observed_*），不进入成功统计。HTTP 200 不等于任务成功。")
    lines.append("- 场景耗时：一次完整场景执行（多轮共一个任务）各轮请求耗时之和；只有计划规定的每一轮恰好出现一次、"
                 "全部必需轮次成功且路径符合预期才计为完整成功任务（轮次身份与唯一性），"
                 "部分成功轮次、重复轮次、缺轮或越界轮次均不相加为完整场景耗时。")
    lines.append("- 计划任务/计划请求来自 manifest 冻结计划（场景、warmup_passes、formal_reps、必需轮次）；"
                 "整体表、逐轮表、场景表与费用说明使用同一份计划口径，未执行任务保留展示，不自动补跑。"
                 "实际记录按槽位身份 (scenario_id, phase, rep, turn_index) 与冻结计划匹配：恰好一条才算计划内成功，"
                 "无记录为未执行，同槽位多条为重复异常；计划外/重复/越界记录单列保留身份，不填补缺失槽位，"
                 "不进计划内成功耗时与完整成功任务分母（其可核实费用仍计费用分子）。")
    lines.append("- 服务端节点耗时：prepare/retrieve/route/tools/clarify/generate 各节点实测耗时（服务端时钟）。")
    lines.append("- 工具执行耗时：每次实际工具调用开始→结束（含失败调用）；mcp_init 为联网工具初始化。")
    lines.append("- 确定性追问（clarify）整段输出的首段耗时是确定性输出耗时，不是模型首 token 时间；逐样本标注 first_text_kind。")
    lines.append("- SSE 文本事件数 ≠ 模型 token 数；token 用量只采用供应商返回的 usage，按字段判已知/未知。")
    lines.append("- 服务端分段与客户端时间分别统计，两者时钟不相减。node 层与 llm/tool/retrieval 调用层为包含关系，不跨层相加。")
    lines.append("- 统计口径：客户端成功耗时表只含成功请求；节点/调用/瓶颈表只用**正式样本**（含失败请求的服务端事实，标注于表头）；"
                 "预热单列；调用与费用另给“含预热”的整次采集总账。")
    lines.append("")

    lines.append("## 3. 客户端指标：逐样本（成功指标与原始观察分列）")
    lines.append("")
    lines.append("| 请求 | 阶段 | 轮 | HTTP | 状态 | 首段(ms) | 完成(ms) | 失败(ms) | 原始观察首段/完成(ms) | done数 | 文本事件数 | 原始时间线 |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |")
    for record in requests:
        metrics = record["metrics"]
        observed = f"{_fmt(metrics.get('observed_first_text_ms'))}/{_fmt(metrics.get('observed_done_ms'))}"
        raw_flag = "可重算" if record.get("raw_recomputable") else "NOT_VERIFIED"
        lines.append(
            f"| {record['request_id']} | {record['phase']} | {record['turn_index']} "
            f"| {record.get('http_status')} | {metrics['status']} "
            f"| {_fmt(metrics['first_text_ms'])} | {_fmt(metrics['done_ms'])} | {_fmt(metrics.get('failure_ms'))} "
            f"| {observed} | {metrics.get('done_count', 0)} | {metrics.get('text_event_count', 0)} | {raw_flag} |"
        )
    lines.append("")
    anomaly_notes = [
        f"- {record['request_id']}：{'；'.join(record['metrics'].get('notes') or []) or record['metrics']['status']}"
        for record in requests
        if record["metrics"].get("notes")
    ]
    if anomaly_notes:
        lines.append("异常与失败原因（可追溯）：")
        lines.extend(anomaly_notes)
        lines.append("")

    lines.append("## 4. 样本与任务口径")
    lines.append("")
    expected_turns_total = sum(int(entry.get("turns") or 0) for entry in manifest.get("scenarios") or [])
    warmup_planned = int(manifest.get("warmup_passes") or 0) * expected_turns_total
    formal_planned = int(manifest.get("formal_reps") or 0) * expected_turns_total
    planned_total = warmup_planned + formal_planned
    slot_info = classify_request_slots(requests, manifest)
    sc = slot_info["counts"]
    rate_text = f"{sc['success_slots']}/{planned_total}" if planned_total else "N/A（计划槽位 0）"
    lines.append(
        f"- 计划样本：预热 {warmup_planned} + 正式 {formal_planned} = {planned_total}"
        f"（冻结计划槽位，**计划分母不因失败或计划外记录改变**）；"
        f"计划槽位结果：成功 {sc['success_slots']}，失败 {sc['failed_slots']}，"
        f"重复异常 {sc['duplicate_slots']}，未执行 {sc['empty_slots']}，成功率 {rate_text}。"
    )
    lines.append(
        f"- 实际请求 {len(requests)}，成功 {sc['physical_ok']}，失败 {sc['physical_failed']}（物理计数）；"
        f"计划内唯一匹配 {sc['matched_records']} 条，计划外/重复异常记录 {sc['anomaly_records']} 条"
        f"（计划外 {sc['out_of_plan_records']}、重复 {sc['duplicate_records']}），单列如下，不填补缺失槽位。"
    )
    if slot_info["anomaly_records"]:
        lines.append("")
        lines.append("计划外/重复异常记录（单列，保留身份；不填补缺失槽位，不进计划内成功耗时与完整成功任务分母）：")
        lines.append("")
        lines.append("| 请求 | 身份（scenario/phase/rep/turn） | 说明 |")
        lines.append("| --- | --- | --- |")
        for record, reason in slot_info["anomaly_records"]:
            identity = (
                f"{record.get('scenario_id')}/{record.get('phase')}/{record.get('rep')}/T{record.get('turn_index')}"
            )
            lines.append(f"| {record.get('request_id')} | {identity} | {reason} |")
    else:
        lines.append("- 计划外/重复异常记录：无。")
    lines.append("")
    warmup_tasks = [t for t in tasks if t["phase"] == "warmup" and t.get("planned", True)]
    formal_tasks = [t for t in tasks if t["phase"] == "formal" and t.get("planned", True)]
    unplanned_tasks = [t for t in tasks if not t.get("planned", True)]

    def task_counts(task_rows: list[dict]) -> str:
        planned = len(task_rows)
        executed = sum(1 for t in task_rows if t["executed"])
        complete = sum(1 for t in task_rows if t["complete_success"])
        return (
            f"计划 {planned}，实际执行 {executed}，完整成功 {complete}，"
            f"未执行 {planned - executed}，已执行未完成或失败 {executed - complete}"
        )

    lines.append(
        f"- 任务口径（一次完整场景执行 = 一个任务，多轮共一个任务；**计划任务来自 manifest 冻结计划**，"
        "不随实际记录增减）："
        f"预热任务：{task_counts(warmup_tasks)}；正式任务：{task_counts(formal_tasks)}；"
        f"计划外任务 {len(unplanned_tasks)} 个（单列，不进计划口径）。"
    )
    lines.append("")
    lines.append("| 任务 | 阶段 | 计划轮次 | 实际轮次 | 状态 | 请求耗时合计(ms，仅事实) | 未完成原因与实际记录 |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | --- |")
    status_text = {"complete_success": "完整成功", "executed_incomplete": "已执行未完成或失败", "not_executed": "未执行"}
    for task in tasks:
        detail = "；".join(task["problems"]) if task["problems"] else "-"
        if task["records"]:
            actual = [
                "{}(T{},{})".format(r["request_id"], r["turn_index"], r["status"]) for r in task["records"]
            ]
            detail += f"；实际记录：{actual}"
        status = status_text.get(task["status"], task["status"])
        if not task.get("planned", True):
            status = "计划外（不进计划口径）"
        lines.append(
            f"| {task['task_id']} | {task['phase']} | {task['planned_turns']} | {task['turns']} "
            f"| {status} "
            f"| {_fmt(task['duration_ms'])} | {detail} |"
        )
    lines.append("")
    lines.append("### 4.1 正式场景耗时与首段/完成（仅计划内完整成功任务与计划槽位成功请求；计划列来自冻结计划）")
    lines.append("")
    lines.append("| 场景×轮次 | 指标 | 成功n | 计划 | 失败/异常 | 未执行 | min | median | max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    def _stat(value) -> str:
        return "N/A" if value is None else str(value)

    for scenario in manifest.get("scenarios") or []:
        sid = scenario["id"]
        planned_turns = int(scenario.get("turns") or 0)
        for turn_index in range(1, planned_turns + 1):
            keys = [k for k in slot_info["slots"] if k[0] == sid and k[1] == "formal" and k[3] == turn_index]
            statuses = [slot_info["slot_status"][k] for k in keys]
            success_rows = [slot_info["slots"][k][0] for k in keys if slot_info["slot_status"][k] == "success"]
            turn_planned = len(keys)
            turn_failed = statuses.count("failed") + statuses.count("duplicate")
            turn_not_executed = statuses.count("not_executed")
            for label, key in (
                ("首段可见回答耗时(ms)", "first_text_ms"),
                ("请求完成耗时(ms)", "done_ms"),
            ):
                summary = summarize([record["metrics"].get(key) for record in success_rows])
                lines.append(
                    f"| {sid} T{turn_index} | {label} | {summary['n']} | {turn_planned} | {turn_failed} "
                    f"| {turn_not_executed} "
                    f"| {_stat(summary['min'])} | {_stat(summary['median'])} | {_stat(summary['max'])} |"
                )
        scenario_tasks = [t for t in formal_tasks if t["scenario_id"] == sid]
        complete = [t for t in scenario_tasks if t["complete_success"]]
        incomplete = [t for t in scenario_tasks if t["executed"] and not t["complete_success"]]
        not_executed = [t for t in scenario_tasks if not t["executed"]]
        summary = summarize([t["duration_ms"] for t in complete])
        suffix = f"（未计入不完整任务：{[t['task_id'] for t in incomplete]}）" if incomplete else ""
        lines.append(
            f"| {sid} | 场景耗时(ms){suffix} | {summary['n']} | {len(scenario_tasks)} "
            f"| {len(incomplete)} | {len(not_executed)} "
            f"| {_stat(summary['min'])} | {_stat(summary['median'])} | {_stat(summary['max'])} |"
        )
    lines.append("")
    lines.append("样本量说明：每场景正式重复 3 次，仅用于初步基线；**不发布 P95（样本不足）**，尾延迟未覆盖。")
    lines.append("预热样本单列（下表），不进入上表任何统计，也不称为严格冷启动测试：")
    lines.append("")
    lines.append("| 请求 | 首段(ms) | 完成(ms) | 状态 |")
    lines.append("| --- | ---: | ---: | --- |")
    for record in warmup:
        metrics = record["metrics"]
        lines.append(
            f"| {record['request_id']} | {_fmt(metrics['first_text_ms'])} "
            f"| {_fmt(metrics['done_ms'])} | {metrics['status']} |"
        )
    lines.append("")

    def render_layers(title: str, agg: dict, scope_note: str) -> None:
        lines.append(f"### {title}（{scope_note}）")
        lines.append("")
        lines.append("#### 节点层（node，含其内部调用，不与调用层相加）")
        lines.append("")
        lines.append("| 节点 | 次数 | 失败 | 总耗时(ms) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, bucket in sorted(agg["node_layer_ms"].items()):
            lines.append(f"| {name} | {bucket['count']} | {bucket['failed']} | {bucket['total_ms']} |")
        lines.append("")
        lines.append("#### 调用层（LLM 逻辑调用，单点计量不重复计费）")
        lines.append("")
        lines.append("| 用途 | 次数 | 失败 | 总耗时(ms) | usage已知 | usage缺失 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, bucket in sorted(agg["llm_by_purpose"].items()):
            lines.append(
                f"| {name} | {bucket['count']} | {bucket['failed']} | {bucket['total_ms']} "
                f"| {bucket['usage_known']} | {bucket['usage_missing']} |"
            )
        lines.append("")
        lines.append("#### 检索调用层（向量/BM25/重排；BM25 为进程内计算，无外部服务调用）")
        lines.append("")
        lines.append("| 步骤 | 次数 | 失败 | 总耗时(ms) | 尝试次数 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name, bucket in sorted(agg["retrieval"].items()):
            if name == "rerank":
                if bucket.get("attempts_unknown"):
                    attempts_text = f"UNKNOWN（{bucket['attempts_unknown']} 条记录无 attempts 字段）"
                else:
                    attempts_text = str(bucket.get("attempts", 0))
            else:
                attempts_text = "-"
            lines.append(
                f"| {name} | {bucket['count']} | {bucket['failed']} | {bucket['total_ms']} | {attempts_text} |"
            )
        lines.append("")
        lines.append("#### 工具执行层（含失败调用与联网初始化）")
        lines.append("")
        lines.append("| 工具 | 次数 | 失败 | 总耗时(ms) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, bucket in sorted(agg["tools"].items()):
            lines.append(f"| {name} | {bucket['count']} | {bucket['failed']} | {bucket['total_ms']} |")
        lines.append("")

    lines.append("## 5. 服务端分段与调用量")
    lines.append("")
    render_layers("5.1 正式样本", agg_formal, "只用正式阶段观测；含失败请求的服务端事实，与客户端成功表口径不同，不混用")
    render_layers("5.2 预热样本", agg_warmup, "仅预热，不进入任何正式统计")
    lines.append("### 5.3 整次采集审计总账（含预热，仅供对账）")
    lines.append("")
    lines.append(
        f"- 调用总数（含预热）：LLM {len(agg_whole['llm_records'])} 次"
        f"（正式 {len(agg_formal['llm_records'])} + 预热 {len(agg_warmup['llm_records'])}"
        f" + 未归属 {len(split['unassigned'])} 条观测）；"
        f"检索 {sum(b['count'] for b in agg_whole['retrieval'].values())} 次；"
        f"工具 {sum(b['count'] for b in agg_whole['tools'].values())} 次。"
    )
    lines.append("- 预热＋正式＝整次采集总账可核对；上表正式/预热口径之和与本总账一致。")
    lines.append("- 统计口径：改写（rewrite）、路由（router）、生成（generate）分别统计；向量化、重排、联网搜索均在调用清单中；"
                 "库内部不可观测的网络尝试不猜测；重排尝试次数为实际可观察值，历史记录缺 attempts 记 UNKNOWN，不按 0 处理。")
    lines.append("")

    lines.append("## 6. token 用量与费用覆盖")
    lines.append("")
    lines.append("口径：费用只使用本次运行 manifest 冻结的单价与模型映射（不使用当前代码默认价格）；"
                 "**计费逐调用判断**（该次调用所需 usage 与费率完整才计费），再按模型汇总，"
                 "同模型的未知调用不抹掉其他调用的已知费用；usage 按字段判已知/未知，缺失不填零；"
                 "费用合计只是**已知部分费用**，存在未知调用时不等于完整费用。")
    lines.append("")

    def render_cost_table(title: str, cost: dict) -> None:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| 模型 | 调用 | 计费调用 | 输入tokens | 输出tokens | 缓存命中tokens | 估算费用 | 价格依据 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for item in cost["items"]:
            if item.get("estimated_cost") is not None:
                cost_text = f"{item.get('currency') or 'CNY'} {item['estimated_cost']}"
                if item.get("calls_unknown"):
                    cost_text += f"（已知部分，另 {item['calls_unknown']} 次调用费用未知）"
            else:
                cost_text = "未知"
            lines.append(
                f"| {item['model']} | {item.get('calls', 0)} | {item.get('calls_costed', 0)} "
                f"| {_usage_text(item.get('input_tokens'), item.get('input_unknown_calls', 0))} "
                f"| {_usage_text(item.get('output_tokens'), item.get('output_unknown_calls', 0))} "
                f"| {_usage_text(item.get('cache_read_tokens'), item.get('cache_unknown_calls', 0))} "
                f"| {cost_text} | {item.get('price_status')} |"
            )
        lines.append("")
        lines.append(f"- 已知部分费用合计：**CNY {cost['known_total']}**（{cost['basis']}；仅为可核实的已知部分，不等于完整费用）")
        lines.append("- 未知项（不填零）：")
        for entry in cost["unknowns"]:
            lines.append(f"  - {entry}")
        lines.append("")

    render_cost_table("6.1 正式样本费用", costs["formal"])
    render_cost_table("6.2 预热样本费用", costs["warmup"])
    render_cost_table("6.3 整次采集费用（含预热）", costs["whole_run"])
    lines.append("### 6.4 每成功任务费用（正式）")
    lines.append("")
    lines.append(
        f"- 正式完整成功任务数：{costs['formal_complete_success_tasks']}（分母；计划正式任务 {costs['formal_tasks']}）；"
        f"预热任务 {costs['warmup_tasks']} 另计；计划外正式任务 {costs['formal_unplanned_tasks']} 个单列"
        "（不进计划口径，其已发生、可核实的费用仍计入分子）。"
    )
    lines.append(f"- 正式每成功任务已知费用：**{costs['per_success_task_formal']}**。{costs['per_success_task_note']}")
    lines.append("")
    lines.append("### 6.5 单价依据（冻结于 manifest，含来源摘录与适用条件）")
    lines.append("")
    for entry in manifest.get("pricing") or []:
        lines.append(
            f"- {entry.get('service')} / {entry.get('model')}：{entry.get('unit') or '单位缺失'}，"
            f"币种 {entry.get('currency') or '缺失'}，状态 {price_status_of(entry)}，"
            f"来源 {entry.get('source_url') or '缺失'}（{entry.get('checked_on') or '未记录'} 抓取）。"
        )
        if entry.get("source_excerpt"):
            lines.append(f"  - 来源摘录：{entry['source_excerpt']}")
        if entry.get("conditions"):
            lines.append(f"  - 适用条件：{entry['conditions']}")
        if entry.get("note"):
            lines.append(f"  - 备注：{entry['note']}")
    if not manifest.get("pricing"):
        lines.append("- manifest 中缺少冻结价格：全部费用未知（不回退到当前代码默认价格）。")
    lines.append("- Langfuse 可见的 LLM token 不能代表全部服务成本（不含向量、重排、联网搜索与本地工具）。")
    lines.append("")

    lines.append("## 7. 预期路径与实际路径核对")
    lines.append("")
    lines.append("路径符合预期只表示完成了指定技术流程，不代表独立法律质量审核通过。")
    lines.append("")
    for scenario in manifest.get("scenarios") or []:
        for record in requests:
            if record["scenario_id"] != scenario["id"]:
                continue
            lines.append(
                f"- {record['request_id']}（{record['phase']}）：路径{'一致' if record.get('path_match') else '不一致'}；"
                f"预期 nodes={record['expected_path'].get('nodes')}，实际 nodes={record.get('actual_path', {}).get('nodes')}；"
                f"预期 llm={record['expected_path'].get('llm_calls')}，实际 llm={record.get('actual_path', {}).get('llm_calls')}；"
                f"预期 tools={record['expected_path'].get('tool_executions')}，实际 tools={record.get('actual_path', {}).get('tool_executions')}"
            )
    lines.append("")
    lines.append("## 8. 失败、样本量与限制")
    lines.append("")
    failures = [record for record in requests if record["metrics"]["status"] != "ok"]
    lines.append(
        f"- 共 {len(requests)} 个 HTTP 请求：预热 {len(warmup)} 个，正式 {len(formal)} 个；"
        f"失败/异常请求 {len(failures)} 个（全部保留，未补跑、未剔除；计划分母 {planned_total} 不因失败减少）。"
    )
    for record in failures:
        lines.append(f"  - {record['request_id']}：{'；'.join(record['metrics'].get('notes') or []) or record['metrics']['status']}")
    lines.append(
        f"- 观测记录共 {len(observations)} 条（按 seq 去重后），未归属到任何请求窗口的 {unassigned} 条"
        f"（服务启动/索引加载等，不进入正式/预热统计）。"
    )
    not_verified = [record["request_id"] for record in requests if not record.get("raw_recomputable")]
    if not_verified:
        lines.append(
            f"- **NOT_VERIFIED（缺原始 SSE 时间线，无法独立重算客户端计时）**：{not_verified}；"
            "这些样本的耗时来自保存的派生 metrics，未从旧 metrics 反向伪造事件时间线。"
        )
    lines.append("- 服务端分段计时来自应用内观测（env 门控、fail-open）；异常路径同样记录，失败调用保留。")
    lines.append("- 客户端与服务端时钟分别统计，未做跨机器时钟相减；时间归属按墙钟窗口，串行执行下无重叠。")
    lines.append("- 检索内部（向量召回内部步骤、BM25 分词）未进一步拆分。")
    lines.append("- 3 次重复不足以稳定描述尾延迟，不发布 P95；不开展并发压测，不推断吞吐量或生产容量。")
    lines.append("- 费用为按冻结公开单价估算，未核实账单、折扣与套餐；公开免费档不证明账户费用为零。")
    lines.append("- 本报告只陈述测量事实与统计；推断与优化候选见 docs/PERFORMANCE.md，未验证项均已标注。")
    lines.append("")
    lines.append("## 9. 瓶颈分析（正式样本，事实排序，不含优化承诺）")
    lines.append("")
    node_total = sum(bucket["total_ms"] for bucket in agg_formal["node_layer_ms"].values())
    if node_total > 0:
        lines.append("正式样本节点层耗时占比（与 5.1 同一集合；预热不参与）：")
        lines.append("")
        lines.append("| 节点 | 占节点层总耗时 |")
        lines.append("| --- | ---: |")
        for name, bucket in sorted(agg_formal["node_layer_ms"].items(), key=lambda kv: -kv[1]["total_ms"]):
            lines.append(f"| {name} | {bucket['total_ms'] / node_total * 100:.1f}% |")
    else:
        lines.append("正式样本无节点观测（或未归属），瓶颈占比不适用。")
    lines.append("")
    lines.append("调用层中耗时最多的 LLM 用途（生成/改写/路由）与检索步骤见 5.1；节点层与调用层是包含关系，不做加总。")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 真实采集
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def sse_request(base_url: str, session_id: str | None, question: str, use_web_search: bool) -> dict:
    """真实 HTTP/SSE 请求：返回事件序列与客户端时间戳（单调时钟）。"""
    payload = json.dumps(
        {
            "question": question,
            **({"session_id": session_id} if session_id else {}),
            "use_web_search": use_web_search,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    failure = None
    wall_start = now_iso()
    t_start = time.perf_counter()
    http_status = 200
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            http_status = response.status
            while True:
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                events.append({"t": time.perf_counter(), "event": event})
    except urllib.error.HTTPError as error:
        http_status = error.code
        failure = {"t": time.perf_counter(), "error": f"HTTPError: {error.code}"}
    except Exception as error:  # noqa: BLE001 — 连接异常也算失败样本，保留
        failure = {"t": time.perf_counter(), "error": f"{type(error).__name__}: {error}"}
    t_end = time.perf_counter()
    wall_end = now_iso()
    return {
        "events": events,
        "failure": failure,
        "t_start": t_start,
        "t_end": t_end,
        "wall_start": wall_start,
        "wall_end": wall_end,
        "http_status": http_status,
    }


def session_id_from_events(events: list) -> str | None:
    """从 SSE 的 session 事件取服务端分配的会话标识（不自己造 id）。"""
    for item in events:
        event = item.get("event") or {}
        if event.get("type") == "session" and event.get("session_id"):
            return str(event["session_id"])
    return None


def run_one_request(
    base_url: str,
    session_id: str | None,
    question: str,
    use_web_search: bool,
    expected_path: dict,
) -> dict:
    result = sse_request(base_url, session_id, question, use_web_search)
    t0 = float(result["t_start"])
    # 原始 SSE 时间线（R1-06）：单调时钟相对偏移 + 事件顺序/类型/是否非空可见文本 + 连接失败
    raw_events = [
        {
            "order": index,
            "type": (item.get("event") or {}).get("type"),
            "t": round(float(item["t"]) - t0, 6),
            "text_nonempty": bool(str((item.get("event") or {}).get("text") or "").strip()),
        }
        for index, item in enumerate(result["events"], start=1)
    ]
    raw_failure = (
        {"t": round(float(result["failure"]["t"]) - t0, 6), "error": result["failure"]["error"]}
        if result.get("failure")
        else None
    )
    metrics = compute_request_metrics(result["events"], result["t_start"], result["t_end"], result["failure"])
    event_types: dict[str, int] = {}
    for item in result["events"]:
        event_type = str((item["event"] or {}).get("type"))
        event_types[event_type] = event_types.get(event_type, 0) + 1
    server_session_id = session_id_from_events(result["events"])
    return {
        "session_id": session_id or server_session_id,
        "server_session_id": server_session_id,
        "question": question,
        "use_web_search": use_web_search,
        "expected_path": expected_path,
        "http_status": result["http_status"],
        "wall_start": result["wall_start"],
        "wall_end": result["wall_end"],
        # 原始证据：t_start=0 为请求起点相对基准，t_end/events.t/failure.t 为单调时钟相对偏移（秒）
        "t_start": 0.0,
        "t_end": round(float(result["t_end"]) - t0, 6),
        "events": raw_events,
        "failure": raw_failure,
        "metrics": metrics,
        "event_types": event_types,
    }


def run_samples(base_url: str, selected: list[dict], warmup_passes: int, formal_reps: int,
                requests_path: Path, log=print) -> int:
    """按固定计划执行预热与正式样本，逐条落盘（不补跑、不重试）。"""
    count = 0
    for phase, passes in (("warmup", warmup_passes), ("formal", formal_reps)):
        for rep in range(1, passes + 1):
            for scenario in selected:
                # 每次重复用新会话：首轮不带 session_id（服务端创建并经 session 事件返回），
                # 场景内后续轮次复用该会话。
                session_id = None
                for turn_index, turn in enumerate(scenario["turns"], start=1):
                    request_id = f"{scenario['id']}-{phase}{rep}-t{turn_index}"
                    log(f"[{now_iso()}] {request_id} ...")
                    record = run_one_request(
                        base_url,
                        session_id,
                        turn["question"],
                        bool(turn.get("use_web_search")),
                        turn["expected_path"],
                    )
                    record.update(
                        {
                            "request_id": request_id,
                            "scenario_id": scenario["id"],
                            "scenario_name": scenario["name"],
                            "phase": phase,
                            "rep": rep,
                            "turn_index": turn_index,
                        }
                    )
                    _append_jsonl(requests_path, record)  # 每条完成后及时落盘
                    count += 1
                    session_id = record.get("server_session_id") or record.get("session_id")
    return count


def _process_table() -> tuple[dict[int, int] | None, str | None]:
    """枚举当前进程表（pid → ppid）。

    返回 (table, reason)：只有返回码为 0、输出可完整解析且非空时才可信；
    命令失败/不可用/空白/含不可信解析结果一律返回 (None, 原因)＝未知，
    **不把诊断失败当作空进程表**（E-102 R1-03）。
    """
    try:
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "$ErrorActionPreference = 'Stop'; "
                "$rows = @(Get-CimInstance -ClassName Win32_Process "
                "-Property ProcessId,ParentProcessId -ErrorAction Stop | "
                "Select-Object ProcessId,ParentProcessId); "
                "ConvertTo-Json -InputObject $rows -Compress",
            ],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as error:  # noqa: BLE001 — 诊断失败保留原因，清理不得记为成功
        return None, f"cim_exception:{type(error).__name__}: {error}"
    if getattr(result, "returncode", 1) != 0:
        stderr = _to_text(getattr(result, "stderr", ""))[:200].replace("\n", " ")
        return None, f"cim_exit_{getattr(result, 'returncode', '?')}: {stderr}"
    output = (getattr(result, "stdout", "") or "").strip()
    if not output:
        return None, "cim_blank_output"
    try:
        rows = json.loads(output)
    except json.JSONDecodeError as error:
        return None, f"cim_untrusted_parse:{error}"
    if not isinstance(rows, list) or not rows:
        return None, "cim_untrusted_parse:expected_nonempty_array"
    table: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None, "cim_untrusted_parse:expected_process_object"
        pid, parent = row.get("ProcessId"), row.get("ParentProcessId")
        # CIM 的 PID 为整数；缺失父关系不能视作 0，部分表不能证明没有后代。
        if type(pid) is not int or type(parent) is not int or pid < 0 or parent < 0:
            return None, "cim_untrusted_parse:invalid_process_ids"
        if pid in table:
            return None, "cim_untrusted_parse:duplicate_process_id"
        table[pid] = parent
    return table, None


def _descendant_pids(pid: int) -> tuple[list[int] | None, str | None]:
    """枚举本次进程的后代 PID（按父子链确认归属，不按端口/进程名猜测）。

    返回 (descendants, reason)：进程表不可信时为 (None, 原因)＝未知，不能读作“无后代”。
    """
    table, reason = _process_table()
    if table is None:
        return None, reason
    descendants: list[int] = []
    frontier = [pid]
    seen = {pid}
    while frontier:
        current = frontier.pop()
        for child, parent in table.items():
            if parent == current and child not in seen:
                seen.add(child)
                descendants.append(child)
                frontier.append(child)
    return descendants, None


def _pid_alive(pid: int) -> tuple[bool | None, str | None]:
    """真实进程状态查询（Windows）。

    返回 (alive, reason)：True=已确认存活；False=已确认退出（真实退出码或
    明确的“无此进程”应答 ERROR_INVALID_PARAMETER）；None=查询失败未知
    （权限拒绝等），不得直接判退出或判存活（E-102 R1-03）。
    """
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            err = ctypes.get_last_error()
            if err == 87:  # ERROR_INVALID_PARAMETER：OS 明确应答“无此进程”
                return False, "process_not_found"
            return None, f"openprocess_error_{err}"
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                err = ctypes.get_last_error()
                return None, f"getexitcode_error_{err}"
            if code.value == 259:  # STILL_ACTIVE
                return True, "still_active"
            return False, f"exit_code_{code.value}"
        finally:
            kernel32.CloseHandle(handle)
    except Exception as error:  # noqa: BLE001
        return None, f"pid_query_exception:{type(error).__name__}: {error}"


def _tcp_listen_table() -> dict[int, int] | None:
    """Windows TCP 监听表（本地端口 → 拥有进程 PID）；查询失败返回 None＝未知。

    这是与连接探测相互独立的监听状态判据（GetExtendedTcpTable）。
    """
    try:
        import ctypes
        from ctypes import wintypes

        iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
        size = wintypes.DWORD(0)
        iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 3, 0)  # AF_INET=2, LISTENER=3
        buf = ctypes.create_string_buffer(size.value)
        ret = iphlpapi.GetExtendedTcpTable(buf, ctypes.byref(size), False, 2, 3, 0)
        if ret != 0:
            return None
        data = buf.raw
        num = int.from_bytes(data[:4], "little")
        table: dict[int, int] = {}
        offset = 4
        for _ in range(num):
            row = data[offset:offset + 24]
            if len(row) < 24:
                return None
            local_port = int.from_bytes(row[8:10], "big")  # dwLocalPort 网络字节序
            pid = int.from_bytes(row[20:24], "little")
            table[local_port] = pid
            offset += 24
        return table
    except Exception:  # noqa: BLE001
        return None


def _port_probe(port: int, timeout: float) -> tuple[str, str | None]:
    """连接探测（原始事实，不做释放推断）。

    返回 (state, reason)：
    - ("connected", None)：连接成功——可证明端口仍可连接；
    - ("not_connected", 原因)：未连接成功——**原因不等于监听消失**，
      是否释放须由 _listen_state 的明确判据决定（E-102 R1-01）。
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return "connected", None
    except Exception as error:  # noqa: BLE001
        return "not_connected", f"{type(error).__name__}: {error}"


def _listen_state(port: int, probe_timeout: float = 1.0) -> tuple[str, str | None, str | None]:
    """端口监听状态的明确判定。

    返回 (state, source, reason)：
    - ("listening", "tcp_table"|"connect_connected", ...)：确认仍在监听；
    - ("released", "tcp_table"|"connect_refused", ...)：确认释放（明确依据）；
    - ("unknown", source, reason)：诊断失败/连接超时/权限错误等——不能判释放。
    """
    table = _tcp_listen_table()
    if table is not None:
        if port in table:
            return "listening", "tcp_table", f"owner_pid={table[port]}"
        return "released", "tcp_table", "no_listen_entry"
    state, reason = _port_probe(port, timeout=probe_timeout)
    if state == "connected":
        return "listening", "connect_connected", None
    if reason and reason.startswith("ConnectionRefusedError"):
        return "released", "connect_refused", reason
    return "unknown", "connect_unclassified", reason


def _to_text(value) -> str:
    """命令输出在边界转为文本（E-102 R2-02）：bytes 按 utf-8(errors=replace) 解码。

    subprocess.capture_output 不带 text=True 时 stdout/stderr 为 bytes；清理证据
    须可 JSON 序列化，因此在命令边界明确解码，不靠 json.dumps(default=str) 掩盖。
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_own_process(process) -> dict:
    """只清理本次创建的进程及其后代（按 PID 归属确认，不按端口/进程名）。

    区分：发起过终止 / 命令报告成功 / 后续确认退出 / 已自行退出 / 仍存活 / 无法确认。
    命令返回 0 不代替后续状态确认（E-102 R1-04）。
    """
    evidence = {
        "pid": getattr(process, "pid", None),
        "method": None,
        "terminated": False,
        "exit_code": None,
        "checked_at_utc": now_iso(),
        "kill_commands": [],
    }
    # 终止前捕获后代（此刻仍挂靠本次进程树，可确认归属）
    descendants, diag = _descendant_pids(evidence["pid"]) if evidence["pid"] else (None, "no_pid")
    evidence["descendants"] = descendants
    evidence["descendants_diagnostic"] = diag
    # 父进程终止
    try:
        process.terminate()
        process.wait(timeout=15)
        evidence["method"] = "terminate"
        evidence["terminated"] = True  # Popen.wait 返回＝真实退出确认
        evidence["exit_code"] = getattr(process, "returncode", None)
    except Exception as first_error:  # noqa: BLE001
        try:
            task = subprocess.run(
                ["taskkill", "/PID", str(evidence["pid"]), "/T", "/F"], capture_output=True, timeout=30
            )
            task_rc = getattr(task, "returncode", None)
            task_err = _to_text(getattr(task, "stderr", ""))[:200]
            evidence["kill_commands"].append(
                {"pid": evidence["pid"], "args": ["taskkill", "/PID", str(evidence["pid"]), "/T", "/F"],
                 "returncode": task_rc, "stderr": task_err}
            )
            process.wait(timeout=15)
            evidence["method"] = "taskkill"
            evidence["terminated"] = task_rc == 0 and getattr(process, "returncode", None) is not None
            evidence["exit_code"] = getattr(process, "returncode", None)
            if task_rc != 0:
                evidence["error"] = f"taskkill_exit_{task_rc}: {task_err}"
        except Exception as second_error:  # noqa: BLE001
            evidence["error"] = f"{type(first_error).__name__}: {first_error} / {type(second_error).__name__}: {second_error}"
    # 后代清理：仅针对已确认属于本次父子链的 PID；尝试与确认分开记账
    results = []
    for desc in descendants or []:
        if desc == evidence["pid"]:
            continue
        alive_before, before_reason = _pid_alive(desc)
        entry = {"pid": desc, "alive_before": alive_before, "before_reason": before_reason,
                 "kill_attempted": False, "command_ok": None, "command_stderr": "", "outcome": None}
        if alive_before is False:
            entry["outcome"] = "exited_on_own"
        elif alive_before is None:
            entry["outcome"] = "unconfirmed_before"
        else:
            entry["kill_attempted"] = True
            try:
                task = subprocess.run(
                    ["taskkill", "/PID", str(desc), "/F"], capture_output=True, timeout=30
                )
                entry["command_ok"] = getattr(task, "returncode", None) == 0
                entry["command_stderr"] = _to_text(getattr(task, "stderr", ""))[:200]
                evidence["kill_commands"].append(
                    {"pid": desc, "args": ["taskkill", "/PID", str(desc), "/F"],
                     "returncode": getattr(task, "returncode", None), "stderr": entry["command_stderr"]}
                )
            except Exception as error:  # noqa: BLE001
                entry["command_ok"] = False
                entry["command_stderr"] = f"{type(error).__name__}: {error}"
            alive_after, after_reason = _pid_alive(desc)
            entry["alive_after"] = alive_after
            entry["after_reason"] = after_reason
            if alive_after is False:
                entry["outcome"] = "confirmed_killed" if entry["command_ok"] else "exited_on_own"
            elif alive_after is True:
                entry["outcome"] = "still_alive"
            else:
                entry["outcome"] = "unconfirmed_after"
        if "alive_after" not in entry:
            entry["alive_after"], entry["after_reason"] = _pid_alive(desc)
        results.append(entry)
    evidence["descendant_results"] = results
    # 已确认成功终止的后代（不含失败/未知/已自行退出）
    evidence["descendants_killed"] = [
        r["pid"] for r in results if r["outcome"] == "confirmed_killed"
    ]
    evidence["descendants_alive_after"] = {str(r["pid"]): r["alive_after"] for r in results}
    return evidence


def start_server(run_dir: Path, port: int) -> tuple[subprocess.Popen, float]:
    """启动本次专用服务进程（隔离数据 + 观测开关），返回进程与启动耗时。

    启动失败（超时/提前退出/检查异常）一律先清理本次已启动进程再抛出；
    RuntimeError 携带 cleanup 证据（.cleanup），绝不泄漏本次进程。
    """
    env = dict(os.environ)
    env["LAWRAG_DB_PATH"] = str(run_dir / "lawrag.sqlite3")
    env["LAWRAG_PERF_OBS_FILE"] = str(run_dir / "observations.jsonl")
    env["LANGFUSE_PUBLIC_KEY"] = ""
    env["LANGFUSE_SECRET_KEY"] = ""
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    log_handle = open(run_dir / "server.log", "w", encoding="utf-8")
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT_PATH), "serve", "--run-dir", str(run_dir), "--port", str(port)],
            cwd=str(ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception as error:  # noqa: BLE001 — 连子进程都没拉起来也要关日志句柄
        log_handle.close()
        raise RuntimeError(f"服务进程启动失败：{type(error).__name__}: {error}") from None
    try:
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.perf_counter() + HEALTHZ_TIMEOUT_SECONDS
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"服务进程提前退出，退出码 {process.returncode}，详见 {run_dir / 'server.log'}")
            try:
                with urllib.request.urlopen(f"{base_url}/healthz", timeout=5) as response:
                    if response.status == 200:
                        log_handle.close()
                        return process, round(time.perf_counter() - started, 3)
            except urllib.error.URLError:
                time.sleep(0.5)
            except Exception:  # noqa: BLE001 — 健康检查异常视为尚未就绪，继续等待
                time.sleep(0.5)
        raise RuntimeError("服务进程未在限时内就绪（/healthz）")
    except BaseException as error:
        cleanup = stop_server(process, port, run_dir)
        log_handle.close()
        wrapped = RuntimeError(f"{error}；本次进程清理：{cleanup}")
        wrapped.cleanup = cleanup  # type: ignore[attr-defined]
        raise wrapped from None


PORT_RELEASE_TIMEOUT_SECONDS = 10.0
PORT_RELEASE_POLL_SECONDS = 0.1


def _port_accepts_connections(port: int, timeout: float = 0.5) -> bool:
    """连接成功→True；其他情况→False（仅用于就绪探测等“是否可连接”场景）。

    注意：False **不表示**端口已释放；释放判定见 _listen_state（E-102 R1-01）。
    """
    return _port_probe(port, timeout=timeout)[0] == "connected"


def stop_server(process: subprocess.Popen, port: int, run_dir: Path) -> dict:
    """只清理本次进程及其后代；返回真实清理证据（父进程事实与整体结果分开）。

    - 首次检查（port_initial_check）如实保留原始结果与诊断原因，不改写为成功；
    - 释放判定必须有明确依据（TCP 监听表 / 明确拒绝连接）；连接超时、权限错误与
      未分类异常一律为未知，不得写 released（E-102 R1-01）；
    - 随后**有上限、每步检查剩余预算、检查真实状态**地等待释放（R1-02）；
    - cleanup_complete 为整体判定：父进程已确认退出 + 后代全部确认不存活（或经
      可信诊断确认无后代）+ 端口确认释放；任一未知/仍存活即不成立（R1-05）。
    """
    info = _terminate_own_process(process)
    start = time.monotonic()  # 独立单调时钟

    def _elapsed_ms() -> float:
        return round((time.monotonic() - start) * 1000, 1)

    def _remaining() -> float:
        return PORT_RELEASE_TIMEOUT_SECONDS - (time.monotonic() - start)

    def _probe_timeout() -> float:
        # 每次探测前按剩余预算截断；不把不足的剩余时间抬高到固定下限（E-102 R2-03）
        return min(1.0, _remaining())

    if _remaining() > 0:
        first_state, first_source, first_reason = _listen_state(port, probe_timeout=_probe_timeout())
    else:
        first_state, first_source, first_reason = "unknown", "budget_exhausted", "remaining<=0，未发起首次探测"
    info["checked_at_utc"] = now_iso()
    info["port_initial_check"] = {"state": first_state, "source": first_source, "reason": first_reason}
    samples = [{"t_ms": _elapsed_ms(), "state": first_state, "source": first_source, "reason": first_reason}]
    state, source, reason = first_state, first_source, first_reason
    while state != "released":
        remaining = _remaining()
        if remaining <= 0:
            break  # 截止后不再发起探测或 sleep
        time.sleep(min(PORT_RELEASE_POLL_SECONDS, remaining))
        remaining = _remaining()
        if remaining <= 0:
            break
        state, source, reason = _listen_state(port, probe_timeout=_probe_timeout())
        samples.append({"t_ms": _elapsed_ms(), "state": state, "source": source, "reason": reason})
    info["port_state"] = state  # released / listening / unknown（含原因）
    info["port_released"] = state == "released"  # 仅明确依据可为 True
    info["port_released_source"] = source if state == "released" else None
    info["port_release_wait_ms"] = _elapsed_ms()
    info["port_check_samples"] = samples
    if state != "released":
        info["port_error"] = f"端口 {port} 状态未确认释放：state={state} source={source} reason={reason}"

    # 整体成功判定（R1-05）：父进程事实 ≠ 整棵树清理完成
    parent_ok = info["terminated"] is True and info["exit_code"] is not None
    results = info.get("descendant_results") or []
    descendants_ok = all(r["outcome"] in ("confirmed_killed", "exited_on_own") for r in results)
    if info.get("descendants") is None:
        descendants_ok = False  # 后代诊断未知＝无法确认整体清理
    info["cleanup_complete"] = bool(parent_ok and descendants_ok and state == "released")
    missing = []
    if not parent_ok:
        missing.append("parent_not_confirmed_exited")
    if not descendants_ok:
        missing.append("descendants_unconfirmed_or_alive")
    if state != "released":
        missing.append(f"port_{state}")
    info["cleanup_incomplete_reasons"] = missing
    return info


def cmd_validate(args) -> int:
    cases = load_cases(Path(args.cases))
    problems = validate_cases(cases)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    print(f"场景校验通过：{len(cases['scenarios'])} 个场景")
    print(f"内容 SHA-256：{cases_sha256(cases)}")
    print(f"文件 SHA-256：{sha256_file(Path(args.cases))}")
    return 0


def cmd_collect(args) -> int:
    cases = load_cases(CASES_PATH)
    problems = validate_cases(cases)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    selected = cases["scenarios"]
    if args.scenarios:
        wanted = {item.strip() for item in args.scenarios.split(",") if item.strip()}
        selected = [scenario for scenario in selected if scenario["id"] in wanted]
        if not selected:
            print(f"[FAIL] 未找到指定场景：{args.scenarios}")
            return 1

    run_dir = PERF_ROOT / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"[FAIL] 运行目录已存在且非空，拒绝覆盖旧结果：{run_dir}")
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)

    before = snapshot_real_dirs()
    if not before["chroma_db"]["exists"]:
        print("[FAIL] 未找到真实 chroma_db，无法建立隔离索引副本")
        return 1

    manifest = {
        "run_id": args.run_id,
        "started_utc": now_iso(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "code_hashes": {name: sha256_file(ROOT / name) for name in APP_FILES_FOR_HASH},
        "cases_sha256": cases_sha256(cases),
        "cases_file_sha256": sha256_file(CASES_PATH),
        "cases_frozen_date": cases.get("frozen_date"),
        "scenarios": [{"id": s["id"], "name": s["name"], "turns": len(s["turns"])} for s in selected],
        "warmup_passes": args.warmup,
        "formal_reps": args.reps,
        "models": {
            "chat": os.getenv("LAWRAG_CHAT_MODEL") or "deepseek-chat",
            "router": os.getenv("LAWRAG_ROUTER_MODEL") or "deepseek-flash",
            "embedding": "BAAI/bge-m3",
            "rerank": "BAAI/bge-reranker-v2-m3",
        },
        "retrieval_strategy": "hybrid_rerank",
        "langfuse": "disabled",
        "pricing": PRICING,
        "isolation": {
            "db_path": str(run_dir / "lawrag.sqlite3"),
            "chroma_source": str(CHROMA_SOURCE_DIR),
            "chroma_copy": str(run_dir / "chroma_copy"),
            # 实际快照证据（不是布尔声明）：采集前后真实目录逐文件 SHA-256
            "snapshot_before": before,
        },
    }

    # 隔离索引副本：复制前后比对源快照，检测并发入库
    index_copy = run_dir / "chroma_copy"
    shutil.copytree(CHROMA_SOURCE_DIR, index_copy)
    source_after_copy = snapshot_dir(CHROMA_SOURCE_DIR)
    if source_after_copy["entries"] != before["chroma_db"]["entries"]:
        print("[FAIL] 复制索引期间源目录变化，疑似并发入库，中止（真实评测未执行）")
        return 1
    manifest["isolation"]["index_copy_consistent"] = True
    manifest["isolation"]["chroma_source_after_copy"] = source_after_copy

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    manifest["port"] = port
    requests_path = run_dir / "requests.jsonl"
    process = None
    startup_error = None
    try:
        process, server_start_seconds = start_server(run_dir, port)
        manifest["server_start_seconds"] = server_start_seconds
    except RuntimeError as error:
        startup_error = error
        manifest["server_start_seconds"] = None
        manifest["cleanup"] = getattr(error, "cleanup", None) or {
            "terminated": False,
            "port_released": False,
            "note": "启动失败且无清理证据",
        }

    if process is not None:
        try:
            run_samples(base_url, selected, args.warmup, args.reps, requests_path)
        finally:
            manifest["cleanup"] = stop_server(process, port, run_dir)

    after = snapshot_real_dirs()
    manifest["isolation"]["snapshot_after"] = after
    manifest["isolation"]["real_dirs_unchanged"] = before == after
    manifest["finished_utc"] = now_iso()

    requests = _read_jsonl(requests_path) if requests_path.exists() else []
    observations = load_observations(run_dir / "observations.jsonl")
    requests, unassigned = annotate_requests(requests, observations)
    manifest["observation_records"] = len(observations)
    manifest["observation_unassigned"] = unassigned
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if requests:
        requests_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in requests),
            encoding="utf-8",
        )
        (run_dir / "report.md").write_text(
            build_report_md(requests, observations, manifest), encoding="utf-8"
        )

    if startup_error is not None:
        print(f"[FAIL] {startup_error}")
        return 1
    ok = all(record["metrics"]["status"] == "ok" for record in requests)
    complete = len(requests) == (args.warmup + args.reps) * sum(len(s["turns"]) for s in selected)
    print(f"采集完成：{run_dir}")
    print(f"请求数：{len(requests)}；全部成功：{ok}；样本完整：{complete}")
    return 0


def cmd_report(args) -> int:
    """离线汇总：只读运行目录，派生结果写入新的输出目录，绝不回写原始文件。"""
    run_dir = PERF_ROOT / args.run_id
    requests_path = run_dir / "requests.jsonl"
    manifest_path = run_dir / "manifest.json"
    if not requests_path.exists() or not manifest_path.exists():
        print(f"[FAIL] 运行目录缺少已保存数据：{run_dir}")
        return 1
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PERF_ROOT / f"{args.run_id}-report-{datetime.now():%Y%m%d-%H%M%S}"
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[FAIL] 输出目录已存在且非空，拒绝覆盖：{output_dir}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = _read_jsonl(requests_path)
    observations = load_observations(run_dir / "observations.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests, _ = annotate_requests(requests, observations)
    report = build_report_md(requests, observations, manifest)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "requests_derived.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in requests),
        encoding="utf-8",
    )
    print(f"已从本地数据离线重建（原始文件未改动）：{output_dir / 'report.md'}")
    return 0


def cmd_serve(args) -> int:
    """内部子命令：隔离数据上启动本次专用 uvicorn（由 collect 拉起）。"""
    run_dir = Path(args.run_dir)
    index_copy = run_dir / "chroma_copy"
    if not index_copy.exists():
        print(f"[FAIL] 缺少隔离索引副本：{index_copy}")
        return 1

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

    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LawRAG 性能与成本基线测量 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="离线校验性能场景并输出冻结哈希")
    p_validate.add_argument("--cases", default=str(CASES_PATH))
    p_validate.set_defaults(func=cmd_validate)

    p_collect = sub.add_parser("collect", help="真实采集（启动隔离服务 + 真实 HTTP/SSE）")
    p_collect.add_argument("--run-id", required=True)
    p_collect.add_argument("--scenarios", default="", help="逗号分隔的场景 id 子集（默认全部）")
    p_collect.add_argument("--warmup", type=int, default=WARMUP_PASSES)
    p_collect.add_argument("--reps", type=int, default=FORMAL_REPS)
    p_collect.set_defaults(func=cmd_collect)

    p_report = sub.add_parser("report", help="离线汇总（不调用服务；写入新的输出目录）")
    p_report.add_argument("--run-id", required=True)
    p_report.add_argument("--output-dir", default="", help="派生输出目录（默认 logs/performance/<run_id>-report-<时间戳>）")
    p_report.set_defaults(func=cmd_report)

    p_serve = sub.add_parser("serve", help="内部子命令：隔离服务进程")
    p_serve.add_argument("--run-dir", required=True)
    p_serve.add_argument("--port", type=int, required=True)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
