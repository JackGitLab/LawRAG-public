import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

import app.mcp_tools as mcp_tools_module
from app.mcp_tools import run_async, safe_web_source_url, tool_result_text, web_sources_from_text


@pytest.fixture(autouse=True)
def _reset_mcp_load_state(monkeypatch) -> None:
    """每个用例以干净的加载状态开始，防止模块级状态泄漏造成测试顺序依赖。"""
    monkeypatch.setattr(mcp_tools_module, "_loaded", False)
    monkeypatch.setattr(mcp_tools_module, "_load_error", None)
    monkeypatch.setattr(mcp_tools_module, "_mcp_tools", [])


def test_tool_result_text_handles_mcp_list_string_and_unknown() -> None:
    payload = '{"ok": true, "results": []}'
    assert tool_result_text([{"type": "text", "text": payload}]) == payload
    assert tool_result_text(payload) == payload
    assert tool_result_text(123) == "123"


def test_web_sources_from_text_handles_valid_failure_and_invalid_json() -> None:
    valid = (
        '{"ok": true, "results": [{"title": "标题", "url": "https://example.com", '
        '"site": "站点", "date": "2026-09-16", "summary": "摘要"}]}'
    )
    assert web_sources_from_text(valid) == [
        {
            "title": "标题",
            "url": "https://example.com",
            "site": "站点",
            "date": "2026-09-16",
        }
    ]
    assert web_sources_from_text('{"ok": false, "error": "失败"}') == []
    assert web_sources_from_text("not json") == []


def test_run_async_without_running_loop() -> None:
    async def value():
        return 42

    assert run_async(value()) == 42


def test_run_async_with_running_loop() -> None:
    async def value():
        return 42

    async def inside_loop():
        return run_async(value())

    assert asyncio.run(inside_loop()) == 42


def test_failed_load_is_retryable(monkeypatch) -> None:
    attempts = {"n": 0}

    async def flaky_load():
        attempts["n"] += 1
        if attempts["n"] == 1:
            return [], "MCP 工具加载失败：RuntimeError: boom"
        return [SimpleNamespace(name="web_search")], None

    monkeypatch.setattr(mcp_tools_module, "load_mcp_tools", flaky_load)

    first = mcp_tools_module.ensure_mcp_tools()
    assert first == "MCP 工具加载失败：RuntimeError: boom"
    assert mcp_tools_module._loaded is False
    assert attempts["n"] == 1

    # 失败后下一次独立请求重新尝试初始化并成功
    second = mcp_tools_module.ensure_mcp_tools()
    assert second is None
    assert mcp_tools_module._loaded is True
    assert attempts["n"] == 2
    assert [tool.name for tool in mcp_tools_module.get_mcp_tools()] == ["web_search"]


def test_concurrent_first_load_runs_loader_once(monkeypatch) -> None:
    calls = {"n": 0}
    barrier = threading.Barrier(2)
    results = []

    async def slow_load():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return [SimpleNamespace(name="web_search")], None

    monkeypatch.setattr(mcp_tools_module, "load_mcp_tools", slow_load)

    def worker():
        barrier.wait(timeout=5)
        results.append(mcp_tools_module.ensure_mcp_tools())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert calls["n"] == 1  # 底层加载只执行一次
    assert results == [None, None]
    assert mcp_tools_module._loaded is True
    assert [tool.name for tool in mcp_tools_module.get_mcp_tools()] == ["web_search"]


# ---------------------------------------------------------------------------
# P10：MCP 子进程环境变量范围
# ---------------------------------------------------------------------------


_ALLOWED_INPUT = {
    "HTTP_PROXY": "hp", "HTTPS_PROXY": "hs", "ALL_PROXY": "ap", "NO_PROXY": "np",
    "http_proxy": "hp_l", "https_proxy": "hs_l", "all_proxy": "ap_l", "no_proxy": "np_l",
    "SSL_CERT_FILE": "cert_file", "SSL_CERT_DIR": "cert_dir",
}
_FORBIDDEN_KEYS = {
    "DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN",
    "PYTHONPATH", "UNRELATED_SECRET",
}
_FIXED_KEYS = {"BOCHA_API_KEY", "PYTHONIOENCODING", "PYTHONUTF8"}


def test_mcp_subprocess_env_copies_only_explicit_whitelist() -> None:
    source = {**_ALLOWED_INPUT, "RANDOM_VAR": "x", "DEEPSEEK_API_KEY": "sk"}
    result = mcp_tools_module._mcp_subprocess_env(source)

    expected_keys = set(_ALLOWED_INPUT) | _FIXED_KEYS
    assert set(result) == expected_keys
    for name, value in _ALLOWED_INPUT.items():
        assert result[name] == value  # 允许变量按原值复制
    assert source == {**_ALLOWED_INPUT, "RANDOM_VAR": "x", "DEEPSEEK_API_KEY": "sk"}  # 输入未变


def test_mcp_subprocess_env_excludes_unrelated_credentials() -> None:
    source = {key: "v" for key in _FORBIDDEN_KEYS}
    source.update(_ALLOWED_INPUT)
    result = mcp_tools_module._mcp_subprocess_env(source)

    for key in _FORBIDDEN_KEYS:
        assert key not in result


def test_mcp_subprocess_env_bocha_key_boundary() -> None:
    source_with = {**dict.fromkeys(_ALLOWED_INPUT, "v")} | {"BOCHA_API_KEY": "fake-key"}
    assert mcp_tools_module._mcp_subprocess_env(source_with)["BOCHA_API_KEY"] == "fake-key"

    result_missing = mcp_tools_module._mcp_subprocess_env(
        {name: "v" for name in _ALLOWED_INPUT}
    )
    assert result_missing["BOCHA_API_KEY"] == ""


def test_mcp_subprocess_env_utf8_fixed_values() -> None:
    source = {
        "PYTHONIOENCODING": "parent-value",
        "PYTHONUTF8": "0",
    }
    result = mcp_tools_module._mcp_subprocess_env(source)
    assert result["PYTHONIOENCODING"] == "utf-8"
    assert result["PYTHONUTF8"] == "1"


def test_bocha_server_uses_limited_env_dict(monkeypatch) -> None:
    server_env = mcp_tools_module.BOCHA_SERVER["bocha"]["env"]
    assert isinstance(server_env, dict)
    assert server_env is not mcp_tools_module.os.environ
    # 不包含完整父环境（本机模型凭据不存在于应用 env）
    for key in _FORBIDDEN_KEYS:
        assert key not in server_env
    # 配置字段保持
    config = mcp_tools_module.BOCHA_SERVER["bocha"]
    assert config["command"] is mcp_tools_module.sys.executable
    assert str(mcp_tools_module.Path(mcp_tools_module.__file__).with_name("mcp_bocha_server.py")) in config["args"]
    assert config["transport"] == "stdio"


def test_mcp_sdk_default_env_vars_locked() -> None:
    from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS

    expected = {
        "APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT",
        "PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "USERNAME", "USERPROFILE",
    }
    assert set(DEFAULT_INHERITED_ENV_VARS) == expected
    # 代表性禁止变量与 SDK 默认集合无交集
    assert not (set(_FORBIDDEN_KEYS) & expected)


def test_load_mcp_tools_uses_bocha_server(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        def __init__(self, config):
            captured["config"] = config

        async def get_tools(self):
            return ["tool-stub"]

    monkeypatch.setattr(mcp_tools_module, "MultiServerMCPClient", FakeClient)

    tools, error = mcp_tools_module.run_async(mcp_tools_module.load_mcp_tools())
    assert error is None
    assert tools == ["tool-stub"]
    assert captured["config"] is mcp_tools_module.BOCHA_SERVER
    env = captured["config"]["bocha"]["env"]
    for key in _FORBIDDEN_KEYS:
        assert key not in env


def test_persistent_failure_attempts_once_per_call(monkeypatch) -> None:
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        return [], "MCP 工具加载失败：ConnectionError: down"

    monkeypatch.setattr(mcp_tools_module, "load_mcp_tools", always_fail)

    first = mcp_tools_module.ensure_mcp_tools()
    assert first == "MCP 工具加载失败：ConnectionError: down"
    assert calls["n"] == 1
    assert mcp_tools_module._loaded is False

    # 下一次独立调用仍可再尝试一次，不循环、不挂死
    second = mcp_tools_module.ensure_mcp_tools()
    assert second == "MCP 工具加载失败：ConnectionError: down"
    assert calls["n"] == 2
    assert mcp_tools_module._loaded is False


def test_real_production_mcp_server_lists_tools_with_limited_env() -> None:
    """P10-T08：真实生产 server + 有限环境，仅工具发现，不调用工具/不发网络请求。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    config = {name: dict(cfg) for name, cfg in mcp_tools_module.BOCHA_SERVER.items()}
    # 测试用 BOCHA_API_KEY 置空，避免真实密钥进入子进程
    config["bocha"]["env"] = {**config["bocha"]["env"], "BOCHA_API_KEY": ""}
    assert "DEEPSEEK_API_KEY" not in config["bocha"]["env"]

    async def discover():
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
        return [tool.name for tool in tools]

    names = asyncio.run(asyncio.wait_for(discover(), timeout=120))
    assert "web_search" in names
    assert "verify_statute_status" in names


_PROBE_SERVER_CODE = '''\
"""P10 最终环境隔离探针：只返回变量名是否存在，严禁返回任何值。"""
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("env-probe")


@mcp.tool()
def probe_env() -> dict:
    return {
        "bocha_present": "BOCHA_API_KEY" in os.environ,
        "deepseek_present": "DEEPSEEK_API_KEY" in os.environ,
        "siliconflow_present": "SILICONFLOW_API_KEY" in os.environ,
        "github_token_present": "GITHUB_TOKEN" in os.environ,
        "unrelated_secret_present": "UNRELATED_SECRET" in os.environ,
        "pythonpath_present": "PYTHONPATH" in os.environ,
    }


mcp.run()
'''


def test_final_subprocess_env_isolation(tmp_path, monkeypatch) -> None:
    """P10-T09：真实最终子进程环境隔离探针。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    probe_file = tmp_path / "probe_server.py"
    probe_file.write_text(_PROBE_SERVER_CODE, encoding="utf-8")

    # 父测试进程设置虚构哨兵值；结束后由 monkeypatch 自动恢复
    monkeypatch.setenv("BOCHA_API_KEY", "sentinel-bocha")
    for name in (
        "DEEPSEEK_API_KEY",
        "SILICONFLOW_API_KEY",
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "PYTHONPATH",
        "UNRELATED_SECRET",
    ):
        monkeypatch.setenv(name, f"sentinel-{name}")

    app_env = mcp_tools_module._mcp_subprocess_env()
    config = {
        "probe": {
            "command": mcp_tools_module.sys.executable,
            "args": [str(probe_file)],
            "transport": "stdio",
            "cwd": str(
                mcp_tools_module.Path(mcp_tools_module.__file__).resolve().parent.parent
            ),
            "env": app_env,
        }
    }

    async def run_probe():
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
        tool = next(t for t in tools if t.name == "probe_env")
        result = await tool.ainvoke({})
        return result

    result = asyncio.run(asyncio.wait_for(run_probe(), timeout=120))
    payload = json.loads(mcp_tools_module.tool_result_text(result))
    assert payload["bocha_present"] is True
    assert payload["deepseek_present"] is False
    assert payload["siliconflow_present"] is False
    assert payload["github_token_present"] is False
    assert payload["unrelated_secret_present"] is False
    assert payload["pythonpath_present"] is False


def test_successful_load_is_reused(monkeypatch) -> None:
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            return [], "MCP 工具加载失败：RuntimeError: boom"
        return [SimpleNamespace(name="verify_statute_status")], None

    monkeypatch.setattr(mcp_tools_module, "load_mcp_tools", flaky)

    assert mcp_tools_module.ensure_mcp_tools() == "MCP 工具加载失败：RuntimeError: boom"
    assert mcp_tools_module.ensure_mcp_tools() is None
    assert mcp_tools_module._load_error is None  # 成功状态不再携带旧错误
    assert mcp_tools_module._loaded is True

    # 成功后的后续调用不再触发底层加载
    assert mcp_tools_module.ensure_mcp_tools() is None
    assert calls["n"] == 2
    assert [tool.name for tool in mcp_tools_module.get_mcp_tools()] == ["verify_statute_status"]


# ---------------------------------------------------------------------------
# P12：来源链接协议限制
# ---------------------------------------------------------------------------


def test_safe_web_source_url_accepts_absolute_http_https() -> None:
    assert safe_web_source_url("https://example.com/path?x=1#part") == "https://example.com/path?x=1#part"
    assert safe_web_source_url("http://example.com") == "http://example.com"
    assert safe_web_source_url("  https://example.com/  ") == "https://example.com/"
    assert safe_web_source_url("HTTPS://Example.com/A") == "HTTPS://Example.com/A"
    assert safe_web_source_url("http://example.com:8080/x") == "http://example.com:8080/x"


def test_safe_web_source_url_rejects_unsafe_forms() -> None:
    unsafe_values = [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "data:text/html,<script>1</script>",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "//example.com/path",
        "/relative/path",
        "example.com/path",
        "https://",
        "",
        "   ",
        None,
        42,
        ["https://example.com"],
        {"url": "https://example.com"},
        "https://example.com\nx",
        "https://exa\tmple.com",
        "https://example.com\rx",
        "https://example.com\x00",
        "https://example.com\x7f",
        # P12-R1：authority 存在但 hostname 为空，或畸形 IPv6 必须稳定拒绝
        "https://:80",
        "https://user@:443",
        "https://[broken",
        "http://:",
    ]
    for value in unsafe_values:
        assert safe_web_source_url(value) == ""


def test_web_sources_from_text_sanitizes_urls() -> None:
    payload = (
        '{"ok": true, "results": ['
        '{"title": "安全", "url": "https://example.com/a", "site": "站A"},'
        '{"title": "危险", "url": "javascript:alert(1)", "site": "站B", "date": "2026-09-16"},'
        '{"title": "无url", "url": "", "site": "站C"}'
        "]}"
    )
    assert web_sources_from_text(payload) == [
        {"title": "安全", "url": "https://example.com/a", "site": "站A", "date": ""},
        {"title": "危险", "url": "", "site": "站B", "date": "2026-09-16"},
        {"title": "无url", "url": "", "site": "站C", "date": ""},
    ]
