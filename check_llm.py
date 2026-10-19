import os
import sys

from openai import OpenAI


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("环境变量缺失：DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "只回复：通了"}],
    temperature=0,
)
answer = response.choices[0].message.content or ""
print(f"DeepSeek 回复：{answer}")

if "通了" not in answer:
    raise SystemExit("DeepSeek 自检失败：回复中未包含“通了”")
