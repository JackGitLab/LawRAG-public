from pathlib import Path

import app.db as db_module


def reset_connection_caches() -> None:
    """关闭并清除两套连接缓存（Checkpointer 与元数据各自独立）。"""
    if db_module.get_checkpointer_connection.cache_info().currsize:
        db_module.get_checkpointer_connection().close()
    db_module.get_checkpointer_connection.cache_clear()
    if db_module.get_connection.cache_info().currsize:
        db_module.get_connection().close()
    db_module.get_connection.cache_clear()


def use_temp_database(monkeypatch, tmp_path: Path) -> None:
    reset_connection_caches()
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path / "t.sqlite3")


def test_session_crud(monkeypatch, tmp_path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    try:
        assert db_module.list_sessions() == []
        db_module.create_session("session-1", "测试会话")
        created = db_module.get_session("session-1")
        assert created["title"] == "测试会话"
        assert created["created_at"] == created["updated_at"]
        db_module.touch_session("session-1")
        assert db_module.list_sessions()[0]["id"] == "session-1"
        assert db_module.get_session("missing") is None
    finally:
        reset_connection_caches()


def test_delete_session_removes_row(monkeypatch, tmp_path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    try:
        db_module.create_session("session-1", "测试会话")
        assert db_module.delete_session("session-1") is True
        assert db_module.get_session("session-1") is None
        assert db_module.delete_session("session-1") is False
    finally:
        reset_connection_caches()


def test_connection_identity_isolation(monkeypatch, tmp_path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    try:
        metadata = db_module.get_connection()
        metadata_again = db_module.get_connection()
        checkpointer = db_module.get_checkpointer_connection()
        checkpointer_again = db_module.get_checkpointer_connection()

        # 同一入口复用自己的连接
        assert metadata is metadata_again
        assert checkpointer is checkpointer_again
        # 两类连接不是同一对象
        assert metadata is not checkpointer

        for conn in (metadata, checkpointer):
            rows = conn.execute("PRAGMA database_list").fetchall()
            db_file = [row["file"] for row in rows if row["name"] == "main"][0]
            assert Path(db_file).resolve() == (tmp_path / "t.sqlite3").resolve()
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        reset_connection_caches()


def test_metadata_connection_close_does_not_affect_checkpointer(
    monkeypatch, tmp_path
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    try:
        metadata = db_module.get_connection()
        checkpointer = db_module.get_checkpointer_connection()

        db_module.create_session("s-1", "会话")
        assert db_module.get_session("s-1")["title"] == "会话"

        # 关闭并清除元数据连接
        metadata.close()
        db_module.get_connection.cache_clear()

        # 原 Checkpointer 连接仍可执行只读 SQL（底层对象真正隔离）
        row = checkpointer.execute("SELECT 1 AS n").fetchone()
        assert row["n"] == 1
        row = checkpointer.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        assert [r["name"] for r in row] == ["sessions"]
    finally:
        reset_connection_caches()


def test_checkpointer_connection_close_does_not_affect_metadata(
    monkeypatch, tmp_path
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    try:
        metadata = db_module.get_connection()
        checkpointer = db_module.get_checkpointer_connection()

        checkpointer.close()
        db_module.get_checkpointer_connection.cache_clear()

        # 元数据连接仍可完成 sessions CRUD
        db_module.create_session("s-1", "会话")
        assert db_module.get_session("s-1")["title"] == "会话"
        assert db_module.touch_session("s-1") is None
        assert db_module.delete_session("s-1") is True
        assert metadata.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    finally:
        reset_connection_caches()
