from pathlib import Path

import app.config as config


def test_log_current_exception_writes_full_utf8_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)

    try:
        raise RuntimeError("临时测试异常")
    except RuntimeError:
        log_path = config.log_current_exception("test", "测试失败")

    content = log_path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in content
    assert "RuntimeError: 临时测试异常" in content
    assert log_path.name.endswith("_test.log")


def test_log_current_exception_redacts_api_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-token-value")

    try:
        raise RuntimeError("上游返回 secret-token-value")
    except RuntimeError:
        log_path = config.log_current_exception("redaction", "测试敏感信息")

    content = log_path.read_text(encoding="utf-8")
    assert "secret-token-value" not in content
    assert "[REDACTED]" in content
    assert "RuntimeError" in content
