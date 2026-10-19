import os
import sys

from llama_index.embeddings.openai_like import OpenAILikeEmbedding


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

api_key = os.getenv("SILICONFLOW_API_KEY")
if not api_key:
    raise SystemExit("环境变量缺失：SILICONFLOW_API_KEY")

embedding = OpenAILikeEmbedding(
    model_name="BAAI/bge-m3",
    api_base="https://api.siliconflow.cn/v1",
    api_key=api_key,
)
vector = embedding.get_text_embedding("劳动合同到期是否有经济补偿？")
print(f"向量长度：{len(vector)}")

if len(vector) != 1024:
    raise SystemExit(f"向量长度异常：期望 1024，实际 {len(vector)}")
