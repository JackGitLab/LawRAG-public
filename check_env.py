import os
import sys


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

REQUIRED_KEYS = ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY")
missing = [name for name in REQUIRED_KEYS if not os.getenv(name)]

if missing:
    raise SystemExit(f"环境变量缺失：{', '.join(missing)}")

for name in REQUIRED_KEYS:
    print(f"{name}：已读取")
