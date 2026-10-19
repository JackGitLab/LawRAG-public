"""app/config.py 的环境变量解析测试（E-083）。

用一次性子进程验证真实 import 行为：不给测试进程做模块 reload，
避免影响 conftest 的 Langfuse 禁用夹具与既有缓存函数。
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import importlib
import os

from app import config

print("EMPTY", config.CHAT_MODEL, config.ROUTER_MODEL)
os.environ["LAWRAG_CHAT_MODEL"] = "custom-chat"
os.environ["LAWRAG_ROUTER_MODEL"] = "custom-router"
importlib.reload(config)
print("OVERRIDE", config.CHAT_MODEL, config.ROUTER_MODEL)
"""


def test_model_name_env_handling_in_fresh_process() -> None:
    """空串（compose 的 ${VAR:-} 语义）回退默认模型名，非空值正常覆盖。"""
    env = {key: value for key, value in __import__("os").environ.items() if not key.startswith("LAWRAG_")}
    env["LAWRAG_CHAT_MODEL"] = ""
    env["LAWRAG_ROUTER_MODEL"] = ""
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    lines = [line.split() for line in proc.stdout.strip().splitlines() if line.strip()]
    assert lines == [
        ["EMPTY", "deepseek-chat", "deepseek-flash"],
        ["OVERRIDE", "custom-chat", "custom-router"],
    ]