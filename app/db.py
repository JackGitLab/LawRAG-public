import sqlite3
import threading
from datetime import datetime
from functools import lru_cache

from app.config import DATABASE_PATH


WRITE_LOCK = threading.Lock()


def _open_connection() -> sqlite3.Connection:
    """打开指向当前 DATABASE_PATH 的独立连接（WAL、跨线程可用）。"""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


@lru_cache(maxsize=1)
def get_connection() -> sqlite3.Connection:
    """sessions 元数据专用连接（进程内缓存，独立于 Checkpointer 连接）。"""
    connection = _open_connection()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


@lru_cache(maxsize=1)
def get_checkpointer_connection() -> sqlite3.Connection:
    """LangGraph Checkpointer 专用连接（进程内缓存，独立于元数据连接）。"""
    return _open_connection()


def create_session(session_id: str, title: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with WRITE_LOCK:
        connection = get_connection()
        connection.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        connection.commit()


def touch_session(session_id: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with WRITE_LOCK:
        connection = get_connection()
        connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        connection.commit()


def list_sessions() -> list[dict]:
    rows = get_connection().execute(
        "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_session(session_id: str) -> dict | None:
    row = get_connection().execute(
        "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_session(session_id: str) -> bool:
    with WRITE_LOCK:
        connection = get_connection()
        cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        connection.commit()
        return cursor.rowcount > 0
