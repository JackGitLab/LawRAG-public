from contextlib import asynccontextmanager
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, field_validator

from app.ask import (
    ExternalServiceError,
    IndexNotReadyError,
    ask,
    get_bm25_index,
    load_index,
)
from app.config import get_embedding, get_llm_client, log_current_exception, require_env
from app.db import create_session, delete_session, get_session, list_sessions, touch_session
from app.graph import delete_session_state, get_graph, graph_config, run_turn, stream_turn
from app.mcp_tools import safe_web_source_url


STATIC_DIR = Path(__file__).with_name("static")
INDEX_HTML = STATIC_DIR / "index.html"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"

MAX_QUESTION_LENGTH = 2000


class AskRequest(BaseModel):
    question: str

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("问题不能为空或仅包含空白字符")
            if len(value) > MAX_QUESTION_LENGTH:
                raise ValueError(f"问题长度不能超过 {MAX_QUESTION_LENGTH} 个字符")
        return value


class Citation(BaseModel):
    source: str
    article: str
    score: float


class WebSource(BaseModel):
    title: str
    url: str
    site: str = ""
    date: str = ""


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class ChatRequest(AskRequest):
    session_id: str | None = None
    use_web_search: bool = False


class ChatResponse(AskResponse):
    session_id: str
    web_sources: list[WebSource] = []


def initialize_resources():
    """在启动阶段校验配置并预热进程内单例资源。"""
    require_env("SILICONFLOW_API_KEY")
    require_env("DEEPSEEK_API_KEY")
    get_embedding()
    get_llm_client()
    index = load_index()
    get_bm25_index()
    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index_error = None
    try:
        app.state.index = initialize_resources()
    except IndexNotReadyError as error:
        app.state.index_error = str(error)
    yield


app = FastAPI(title="LawRAG", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    """容器健康检查：不依赖索引、数据库或外部服务，启动期即可响应。"""
    return {"status": "ok"}


@app.get("/", response_class=FileResponse)
def frontend_page() -> FileResponse:
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(
            status_code=404,
            detail="前端尚未构建：请先运行 npm.cmd --prefix frontend run build",
        )
    return FileResponse(FRONTEND_INDEX)


@app.get("/minimal", response_class=FileResponse)
def index_page() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/api/ask", response_model=AskResponse, operation_id="ask_law_question")
def ask_api(payload: AskRequest) -> AskResponse:
    if app.state.index_error:
        raise HTTPException(status_code=503, detail=app.state.index_error)

    try:
        answer, retrieved_nodes = ask(payload.question)
    except IndexNotReadyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except ExternalServiceError:
        log_current_exception("api-ask", "HTTP 问答外部服务调用失败")
        raise HTTPException(
            status_code=502,
            detail="外部服务调用失败，请稍后重试",
        ) from None

    citations = [
        Citation(
            source=item.node.metadata["source"],
            article=item.node.metadata["article"],
            score=float(item.score),
        )
        for item in retrieved_nodes
    ]
    return AskResponse(answer=answer, citations=citations)


@app.post("/api/chat", response_model=ChatResponse, operation_id="chat_law_question")
def chat_api(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id
    is_new_session = session_id is None
    if is_new_session:
        session_id = uuid4().hex
        create_session(session_id, payload.question[:30])
    elif get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在或已失效")
    else:
        touch_session(session_id)

    try:
        answer, retrieved_nodes = run_turn(
            session_id, payload.question, payload.use_web_search
        )
    except IndexNotReadyError as error:
        if is_new_session:
            _cleanup_failed_session(session_id)
        raise HTTPException(status_code=503, detail=str(error)) from None
    except ExternalServiceError:
        if is_new_session:
            _cleanup_failed_session(session_id)
        log_current_exception("api-chat", "HTTP 多轮问答外部服务调用失败")
        raise HTTPException(status_code=502, detail="外部服务调用失败，请稍后重试") from None
    except Exception:
        if is_new_session:
            _cleanup_failed_session(session_id)
        raise

    citations = [
        Citation(
            source=item.node.metadata["source"],
            article=item.node.metadata["article"],
            score=float(item.score),
        )
        for item in retrieved_nodes
    ]
    snapshot = get_graph().get_state(config=graph_config(session_id))
    web_sources = _web_source_payload(snapshot.values.get("web_sources", []))
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        citations=citations,
        web_sources=web_sources,
    )


@app.get("/api/sessions", operation_id="list_sessions")
def sessions_api() -> list[dict]:
    return list_sessions()


@app.get("/api/sessions/{session_id}", operation_id="get_session")
def session_api(session_id: str) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已失效")

    snapshot = get_graph().get_state(config=graph_config(session_id))
    messages = []
    for message in snapshot.values.get("messages", []):
        if message.type not in {"human", "ai"}:
            continue
        item = {
            "role": "user" if message.type == "human" else "assistant",
            "content": str(message.content),
        }
        citations = message.additional_kwargs.get("citations")
        if message.type == "ai" and citations is not None:
            item["citations"] = citations
        messages.append(item)

    assistant_messages = [item for item in messages if item["role"] == "assistant"]
    latest_citations = snapshot.values.get("citations", [])
    if assistant_messages and "citations" not in assistant_messages[-1] and latest_citations:
        assistant_messages[-1]["citations"] = _citation_payload(latest_citations)
    return {**session, "messages": messages}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _citation_fields(item) -> tuple[dict, float]:
    """兼容两种形态：活对象（NodeWithScore + TextNode）与 checkpoint 回放后的 dict（E-032 / E-034）。"""
    node = item["node"] if isinstance(item, dict) else item.node
    metadata = node["metadata"] if isinstance(node, dict) else node.metadata
    score = item["score"] if isinstance(item, dict) else item.score
    return metadata, float(score)


def _citation_payload(nodes: list) -> list[dict]:
    payload = []
    for item in nodes:
        metadata, score = _citation_fields(item)
        payload.append(
            Citation(
                source=metadata["source"],
                article=metadata["article"],
                score=score,
            ).model_dump()
        )
    return payload


def _web_source_payload(sources: list) -> list[dict]:
    return [
        {
            "title": item.get("title") or "",
            "url": safe_web_source_url(item.get("url")),
            "site": item.get("site") or "",
            "date": item.get("date") or "",
        }
        for item in sources
        if isinstance(item, dict)
    ]


def _cleanup_failed_session(session_id: str) -> None:
    """清理本轮新建但未成功完成会话的 Checkpointer 状态与 sessions 元数据行。

    只用于"本次请求新建、且本轮未能成功完成"的会话。先清 Checkpointer 状态、
    再删 sessions 行；两项相互独立，任一侧失败仍尝试另一侧。清理异常仅写入
    日志，不覆盖原始业务失败、不暴露给用户。
    """
    try:
        delete_session_state(session_id)
    except Exception:
        log_current_exception("cleanup-failed-session", "清理失败会话 Checkpointer 状态")
    try:
        delete_session(session_id)
    except Exception:
        log_current_exception("cleanup-failed-session", "清理失败会话 sessions 行")


def chat_event_stream(
    session_id: str,
    question: str,
    use_web_search: bool = False,
    is_new_session: bool = False,
):
    yield _sse({"type": "session", "session_id": session_id})
    try:
        for event in stream_turn(session_id, question, use_web_search):
            event_type = event.get("type")
            if event_type == "token":
                yield _sse(event)
            elif event_type == "done":
                yield _sse(
                    {
                        "type": "done",
                        "citations": _citation_payload(event["citations"]),
                        "web_sources": _web_source_payload(event.get("web_sources", [])),
                    }
                )
                return
            elif event_type == "error":
                if is_new_session:
                    _cleanup_failed_session(session_id)
                yield _sse(
                    {
                        "type": "error",
                        "status": event.get("status", 502),
                        "detail": "外部服务调用失败，请稍后重试",
                    }
                )
                return
        else:
            # 迭代器正常耗尽但未产生明确 done：不视为成功，新会话需清理
            if is_new_session:
                _cleanup_failed_session(session_id)
    except IndexNotReadyError as error:
        if is_new_session:
            _cleanup_failed_session(session_id)
        yield _sse({"type": "error", "status": 503, "detail": str(error)})
    except ExternalServiceError:
        if is_new_session:
            _cleanup_failed_session(session_id)
        log_current_exception("api-chat-stream", "HTTP 流式多轮问答外部服务调用失败")
        yield _sse({"type": "error", "status": 502, "detail": "外部服务调用失败，请稍后重试"})
    except Exception:
        if is_new_session:
            _cleanup_failed_session(session_id)
        raise


@app.post("/api/chat/stream")
def chat_stream_api(payload: ChatRequest) -> StreamingResponse:
    session_id = payload.session_id
    is_new_session = session_id is None
    if is_new_session:
        session_id = uuid4().hex
        create_session(session_id, payload.question[:30])
    elif get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在或已失效")
    else:
        touch_session(session_id)

    return StreamingResponse(
        chat_event_stream(
            session_id,
            payload.question,
            payload.use_web_search,
            is_new_session,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.delete("/api/sessions/{session_id}", operation_id="delete_session")
def delete_session_api(session_id: str) -> dict:
    if get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在或已失效")
    delete_session_state(session_id)
    delete_session(session_id)
    return {"deleted": session_id}


MCP_OPERATION_IDS = [
    "ask_law_question",
    "chat_law_question",
    "list_sessions",
    "get_session",
    "delete_session",
]

mcp_server = FastApiMCP(
    app,
    name="LawRAG",
    description="LawRAG：中国劳动法检索增强问答服务（法条检索 + 工具调用）。",
    include_operations=MCP_OPERATION_IDS,
)
mcp_server.mount_http()


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST), name="frontend")
