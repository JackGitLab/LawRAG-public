import os
import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai_like import OpenAILikeEmbedding


ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "kb"
CHROMA_DIR = ROOT_DIR / "chroma_db"
LOG_DIR = ROOT_DIR / "logs"
DATABASE_PATH = Path(os.getenv("LAWRAG_DB_PATH") or ROOT_DIR / "data" / "lawrag.sqlite3")
COLLECTION_NAME = "lawrag"
EMBEDDING_MODEL = "BAAI/bge-m3"
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_API_BASE = "https://api.siliconflow.cn/v1"
CANDIDATE_K = 10
RETRIEVAL_STRATEGY = "hybrid_rerank"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
# 模型名支持环境变量覆盖；空串视为未配置（compose 的 ${VAR:-} 会传空串，
# 若直接使用 os.getenv 的第二参数默认值，空串会覆盖默认模型名）。
CHAT_MODEL = os.getenv("LAWRAG_CHAT_MODEL") or "deepseek-chat"
ROUTER_MODEL = os.getenv("LAWRAG_ROUTER_MODEL") or "deepseek-flash"
DEEPSEEK_MODEL = CHAT_MODEL

# Langfuse 可观测性（可选）：两个密钥变量齐全才启用，缺失时自动关闭（fail-open）。
# HOST 必须与账号所在区域一致（例如美区为 https://us.cloud.langfuse.com）。
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


def langfuse_enabled() -> bool:
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def configure_utf8_stdio() -> None:
    """避免 Windows 默认 GBK 导致中文日志乱码。"""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"环境变量缺失：{name}")
    return value


def _redact_secrets(text: str) -> str:
    for name, value in os.environ.items():
        if value and (name.endswith("_API_KEY") or name.endswith("_TOKEN") or name.endswith("_SECRET_KEY")):
            text = text.replace(value, "[REDACTED]")
    return text


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _redact_secrets(super().format(record))


def log_current_exception(scene: str, message: str) -> Path:
    """把当前异常的完整堆栈同时写入 UTF-8 文件和控制台。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now():%Y%m%d-%H%M}_{scene}.log"
    formatter = _RedactingFormatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("lawrag.error")
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stderr)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.exception(message)

    for handler in logger.handlers:
        handler.flush()
    return log_path


@lru_cache(maxsize=1)
def get_embedding() -> OpenAILikeEmbedding:
    """入库与查询统一使用同一个模型和服务端点。"""
    return OpenAILikeEmbedding(
        model_name=EMBEDDING_MODEL,
        api_base=SILICONFLOW_API_BASE,
        api_key=require_env("SILICONFLOW_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=require_env("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_API_BASE,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_router_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=ROUTER_MODEL,
        api_key=require_env("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_API_BASE,
        temperature=0,
    )


def get_llm_client() -> ChatOpenAI:
    """保留旧名，避免改动 app.main 的预热入口与既有测试。"""
    return get_chat_model()
