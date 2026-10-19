"""LawRAG 的 MCP 客户端边界与同步桥接。"""

import asyncio
import json
import os
import sys
import urllib.parse
from pathlib import Path
from threading import Lock, Thread

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools import calculate_labor_compensation


MCP_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
MCP_SSL_ENV_VARS = ("SSL_CERT_FILE", "SSL_CERT_DIR")


def _mcp_subprocess_env(source=None) -> dict[str, str]:
    """构造 LawRAG 显式传给 MCP 子进程的有限环境（正向白名单）。

    只复制 source（缺省为 os.environ）中实际存在的代理与 SSL 证书变量，
    显式加入 BOCHA_API_KEY（缺失时为空串），并固定 UTF-8 配置
    （PYTHONIOENCODING=utf-8、PYTHONUTF8=1，父环境不能覆盖）。
    不复制模型凭据、令牌、PYTHONPATH 等无关变量；返回新字典，
    不修改输入映射，也不保留对输入映射的引用。
    """
    env = os.environ if source is None else source
    result: dict[str, str] = {}
    for name in (*MCP_PROXY_ENV_VARS, *MCP_SSL_ENV_VARS):
        value = env.get(name)
        if value is not None:
            result[name] = value
    result["BOCHA_API_KEY"] = env.get("BOCHA_API_KEY", "")
    result["PYTHONIOENCODING"] = "utf-8"
    result["PYTHONUTF8"] = "1"
    return result


BOCHA_SERVER = {
    "bocha": {
        "command": sys.executable,
        "args": [str(Path(__file__).with_name("mcp_bocha_server.py"))],
        "transport": "stdio",
        "cwd": str(Path(__file__).resolve().parent.parent),
        "env": _mcp_subprocess_env(),
    }
}
MCP_TOOL_NAMES = {"web_search", "verify_statute_status"}
LOCAL_TOOLS = [calculate_labor_compensation]

_loaded = False
_load_error = None
_mcp_tools = []
_load_lock = Lock()


async def load_mcp_tools() -> tuple[list, str | None]:
    try:
        client = MultiServerMCPClient(BOCHA_SERVER)
        tools = await client.get_tools()
        return tools, None
    except Exception as error:
        return [], f"MCP 工具加载失败：{type(error).__name__}: {error}"


def run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = []
    failure = []

    def runner():
        try:
            result.append(asyncio.run(coro))
        except BaseException as error:
            failure.append(error)

    thread = Thread(target=runner)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def ensure_mcp_tools() -> str | None:
    """确保 MCP 工具已加载，失败保持可重试，成功后复用。

    - 快速路径：已加载则直接返回当前错误（成功后为 None）。
    - 加锁路径：锁内二次检查，防止并发首次请求重复初始化。
    - 失败时 _loaded 保持 False、_load_error 记录本次错误，
      下一次独立请求可以重新尝试初始化；成功时清除旧错误。
    """
    global _loaded, _load_error, _mcp_tools
    if _loaded:
        return _load_error
    with _load_lock:
        if _loaded:
            return _load_error
        tools, error = run_async(load_mcp_tools())
        if error is None:
            _mcp_tools = list(tools)
            _loaded = True
            _load_error = None
        else:
            _loaded = False
            _load_error = error
        return _load_error


def set_mcp_tools(tools) -> None:
    global _loaded, _load_error, _mcp_tools
    _mcp_tools = list(tools)
    _load_error = None
    _loaded = True


def get_mcp_tools() -> list:
    return list(_mcp_tools)


def get_all_tools() -> list:
    return [*LOCAL_TOOLS, *get_mcp_tools()]


def tool_result_text(result) -> str:
    if isinstance(result, list):
        chunks = []
        for item in result:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text is not None:
                chunks.append(str(text))
        return "".join(chunks)
    if isinstance(result, str):
        return result
    return str(result)


def web_sources_from_text(text: str) -> list[dict]:
    try:
        payload = json.loads(text)
        if payload.get("ok") is not True:
            return []
        results = payload.get("results") or []
        return [
            {
                "title": item.get("title") or "",
                "url": safe_web_source_url(item.get("url")),
                "site": item.get("site") or "",
                "date": item.get("date") or "",
            }
            for item in results
            if isinstance(item, dict)
        ]
    except Exception:
        return []


def safe_web_source_url(value) -> str:
    """只允许绝对 http/https URL 通过，其余一律返回空字符串。

    外部搜索来源属于不可信输入，不能直接用于 <a href> 或 Markdown 链接目标。
    拒绝 javascript:/data:/file:/ftp:、协议相对与相对地址、缺少真实主机
    （https://:80、https://user@:443 等空 hostname）、含控制字符、空白、
    空值与非字符串。畸形 authority（如 https://[broken）返回空字符串而非
    抛出异常。不执行网络请求。
    """
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate:
        return ""
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in candidate):
        return ""
    try:
        parsed = urllib.parse.urlsplit(candidate)
        hostname = parsed.hostname
    except ValueError:
        return ""
    if parsed.scheme.lower() not in ("http", "https"):
        return ""
    if not hostname:
        return ""
    return candidate
