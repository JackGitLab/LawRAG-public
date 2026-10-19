"""全局测试夹具。

自动禁用 Langfuse：宿主机环境变量可能携带真实密钥
（Windows 用户级 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY），
若不禁用，测试会向用户的 Langfuse 项目发送真实 trace，
并因网络Flush拖慢测试。遵守 E-074：测试不依赖开发机本地状态。
"""

import pytest

from app import config


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "")
