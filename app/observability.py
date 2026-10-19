"""Langfuse 可观测性封装（fail-open）。

设计约束（参见 docs/ERRORS.md 的 E-074 纪律与 docs/ENVIRONMENT.md）：
- 未配置密钥、依赖缺失或初始化失败时一律返回 None / 直接调用；
- 可观测性绝不影响问答主流程，出问题只降级、不报错。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables.config import ensure_config

from app import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 性能基线观测（默认关闭，fail-open）
#
# 只在环境变量 LAWRAG_PERF_OBS_FILE 指向的 JSONL 文件中追加测量记录；
# 未设置时全部为空操作。记录只做计时与 usage 抄录，不改变参数、返回值
# 或异常传播，因此不改变实际调用链（第三阶段第一步，P3-01）。
# ---------------------------------------------------------------------------

PERF_ENV_VAR = "LAWRAG_PERF_OBS_FILE"
_perf_lock = threading.Lock()
_perf_seq = 0


def perf_enabled() -> bool:
    return bool(os.getenv(PERF_ENV_VAR))


def perf_write(record: dict) -> None:
    """追加一条观测记录；写入失败只丢弃该条，绝不影响业务（fail-open）。"""
    global _perf_seq
    path = os.getenv(PERF_ENV_VAR)
    if not path:
        return
    try:
        with _perf_lock:
            _perf_seq += 1
            payload = {
                "seq": _perf_seq,
                "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                **record,
            }
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — fail-open：观测写入失败不能影响主流程
        pass


@contextmanager
def perf_span(kind: str, name: str, **fields: Any):
    """计时一段执行并记录；extra 字典可在执行中补充字段，退出时一并写入。

    extra 必须保留调用方传入字典的引用（E-098/F-02）：调用方在执行过程中
    追加的 attempts 等字段要出现在最终记录里，不能因为空字典被换成新字典而丢失。
    """
    if not perf_enabled():
        yield
        return
    extra = fields.pop("extra", None)
    if extra is None:
        extra = {}
    start = time.perf_counter()
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        record = {
            "kind": kind,
            "name": name,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "ok": ok,
            **fields,
            **extra,
        }
        perf_write(record)


def usage_from_response(response: Any) -> dict | None:
    """从模型响应抄录供应商返回的 token 用量；无法取得时返回 None（不是 0）。"""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        details = usage.get("input_token_details") or {}
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cache_read_tokens": details.get("cache_read") if isinstance(details, dict) else None,
        }
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage")
    if isinstance(token_usage, dict):
        return {
            "input_tokens": token_usage.get("prompt_tokens"),
            "output_tokens": token_usage.get("completion_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
            "cache_read_tokens": (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if isinstance(token_usage.get("prompt_tokens_details"), dict)
            else None,
        }
    return None


def _model_name(model: Any) -> str | None:
    name = getattr(model, "model_name", None)
    if name:
        return str(name)
    bound = getattr(model, "bound", None)
    name = getattr(bound, "model_name", None)
    return str(name) if name else None


def get_langfuse_callback() -> Any | None:
    """返回 LangChain 回调 handler；未启用或初始化失败时返回 None。

    每次调用返回新实例：handler 内部按 run_id 维护状态，
    独立实例可避免并发请求之间的状态串扰。
    """
    if not config.langfuse_enabled():
        return None
    try:
        return _new_handler()
    except Exception as error:  # noqa: BLE001 — fail-open：监控初始化失败不能影响主流程
        logger.warning(
            "Langfuse 初始化失败，本轮不采集 trace：%s",
            config._redact_secrets(f"{type(error).__name__}: {error}"),
        )
        return None


def _new_handler() -> Any:
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def traced_invoke(model: Any, prompt: Any, *, session_id: str | None = None, tags: list[str] | None = None) -> Any:
    """带 Langfuse trace 调用 LLM；未启用时等价于直接 model.invoke(prompt)。

    会话与标签通过 LangChain run metadata 传递（langfuse v4 约定：
    langfuse_session_id / langfuse_tags），同一 handler 也能为每次调用
    附上不同的 trace 元数据。回调 handler 内部的异常由 LangChain 的
    回调管理器吞掉（raise_error 默认 False），加上构造失败已在
    get_langfuse_callback 中降级，因此整体 fail-open。

    回调采用合并而非替换（E-075）：显式传入的 callbacks 会覆盖从
    var_child_runnable_config 继承的回调；LangGraph 的 stream_mode="messages"
    依赖继承配置里的流式 handler 转发 token，因此这里用 ensure_config()
    取继承配置副本，把 Langfuse handler 追加到继承 callbacks 之后，
    避免覆盖导致流式 token 丢失。
    """
    callback = get_langfuse_callback()
    purpose = (tags[0] if tags else None) or "generate"
    start = time.perf_counter()
    try:
        if callback is None:
            response = model.invoke(prompt)
        else:
            merged = ensure_config()  # 返回含继承 callbacks/metadata 的副本，不改原对象
            merged["callbacks"] = _merge_callbacks(merged.get("callbacks"), callback)
            metadata = dict(merged.get("metadata") or {})
            metadata["langfuse_session_id"] = session_id
            if tags:
                metadata["langfuse_tags"] = list(tags)
            merged["metadata"] = metadata
            response = model.invoke(prompt, config=merged)
    except BaseException:
        perf_write(
            {
                "kind": "llm",
                "name": purpose,
                "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                "ok": False,
                "session_id": session_id,
                "model": _model_name(model),
                "usage": None,
                "usage_source": "missing",
            }
        )
        raise
    # 单点计量：每次逻辑调用恰好一条记录，不经多回调分发，不会重复计费。
    usage = usage_from_response(response)
    perf_write(
        {
            "kind": "llm",
            "name": purpose,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "ok": True,
            "session_id": session_id,
            "model": _model_name(model),
            "usage": usage,
            "usage_source": "provider" if usage is not None else "missing",
        }
    )
    return response


def _merge_callbacks(existing: Any, callback: Any) -> list[Any]:
    """追加而非替换：None → 新列表；list → 展开后追加；
    回调管理器对象 → 取其 .handlers 展开后追加。绝不原地修改传入对象。"""
    if existing is None:
        return [callback]
    if isinstance(existing, list):
        return [*existing, callback]
    handlers = getattr(existing, "handlers", None)
    if isinstance(handlers, list):
        return [*handlers, callback]
    return [callback]
