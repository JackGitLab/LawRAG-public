"""博查搜索的薄 MCP stdio server。"""

import json
import os
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("lawrag-bocha")
BOCHA_URL = "https://api.bochaai.com/v1/web-search"
VALID_FRESHNESS = {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}


def _search(query: str, freshness: str, count: int) -> str:
    api_key = os.getenv("BOCHA_API_KEY")
    if not api_key:
        return json.dumps({"ok": False, "error": "未配置 BOCHA_API_KEY"}, ensure_ascii=False)
    if freshness not in VALID_FRESHNESS:
        freshness = "noLimit"
    count = min(max(count, 1), 10)
    payload = json.dumps(
        {"query": query, "freshness": freshness, "summary": True, "count": count},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        BOCHA_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        values = ((body.get("data") or {}).get("webPages") or {}).get("value") or []
        results = [
            {
                "title": item.get("name") or "",
                "url": item.get("url") or "",
                "site": item.get("siteName") or "",
                "date": item.get("datePublished") or "",
                "summary": item.get("summary") or item.get("snippet") or "",
            }
            for item in values
        ]
        return json.dumps({"ok": True, "results": results}, ensure_ascii=False)
    except urllib.error.HTTPError as error:
        message = (
            "错误：联网搜索额度不足（HTTP 403）。请到博查控制台领取免费额度或充值。"
            if error.code == 403
            else f"联网搜索失败（HTTP {error.code}）"
        )
        return json.dumps({"ok": False, "error": message}, ensure_ascii=False)
    except Exception as error:
        return json.dumps(
            {"ok": False, "error": f"联网搜索失败：{type(error).__name__}"},
            ensure_ascii=False,
        )


@mcp.tool()
def web_search(query: str, freshness: str = "noLimit", count: int = 5) -> str:
    """通用联网搜索。"""
    return _search(query, freshness, count)


@mcp.tool()
def verify_statute_status(query: str, freshness: str = "oneYear") -> str:
    """核验法规是否修订、废止及当前有效版本。"""
    return _search(f"{query} 是否已修订或废止 现行有效版本", freshness, 5)


if __name__ == "__main__":
    mcp.run()
