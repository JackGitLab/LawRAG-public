import json
import re
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
import jieba
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi

from app.config import (
    CANDIDATE_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    RERANK_API_BASE,
    RERANK_MODEL,
    RETRIEVAL_STRATEGY,
    configure_utf8_stdio,
    get_embedding,
    get_chat_model,
    log_current_exception,
    require_env,
)
from app.observability import perf_span, traced_invoke


class IndexNotReadyError(RuntimeError):
    """持久化索引尚未创建。"""


class ExternalServiceError(RuntimeError):
    """向量化、重排或生成服务调用失败。"""


LEGAL_RULES = """你是法律顾问问答助手。请严格遵守以下规则：
1. 只能根据“检索资料”回答，不得使用资料外的知识，不得编造。
2. 如果资料不足以回答，只回复“现有资料无法回答”。
3. 回答正文中的每个法律结论句末必须标注对应条号，格式如“……（第四十六条）”；同一结论依据多条时可合并标注。
4. 不要在回答正文中重复文档名，也不要自行生成“引用来源”清单；文档名和条号由程序在回答后统一列出。
5. 如果资料涉及经济补偿，必须明确说明是否应当支付，并说明资料中给出的计算标准。"""

LEGAL_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", f"{LEGAL_RULES}\n\n检索资料：\n{{context}}"),
        ("human", "{question}"),
    ]
)


def format_materials(retrieved_nodes) -> str:
    materials = []
    for item in retrieved_nodes:
        source = item.node.metadata["source"]
        article = item.node.metadata["article"]
        materials.append(f"[来源：《{source}》；{article}]\n{item.node.text}")
    return "\n\n".join(materials)


def build_prompt(question: str, retrieved_nodes) -> str:
    messages = LEGAL_PROMPT_TEMPLATE.format_messages(
        context=format_materials(retrieved_nodes),
        question=question,
    )
    return f"{messages[0].content}\n\n用户问题：{messages[1].content}\n"


def build_prompt_with_tools(question: str, retrieved_nodes: list, tool_text: str = "") -> str:
    if not tool_text:
        return build_prompt(question, retrieved_nodes)

    materials = format_materials(retrieved_nodes)
    context = (
        f"{materials}\n\n"
        f"[联网核验资料]（与上述法条同属“检索资料”）\n{tool_text}\n\n"
        "上面的联网核验资料中，来源标题、站点、日期或摘要在存在时都属于有限可用证据："
        "只能据此陈述这些资料直接支持的事实。若现有联网资料不足以支撑用户要求的具体结论"
        "（例如无法确认是否修订、是否仍然有效），应明确说明该部分无法核实，不得编造。"
        "若联网核验资料明确表明资料不足、联网初始化失败、调用失败、工具不可用或额度不足，"
        "本轮联网核验即视为未完成：对无法证实的联网事实应回答无法核实，不得编造来源、日期、"
        "修订状态或搜索结果；仍可继续回答本地法条资料能够独立支持的部分，"
        "但不得把本地回答冒充成已完成联网核验。"
        "各法条条号与来源站点、日期的引用方式不受“不要自行生成引用来源清单”限制。"
        "不得继续无条件要求给出确定性结论，也不得无条件禁止在资料不足时说明无法核实。"
        "当联网工具失败、额度不足、未返回可用结果，或现有联网证据不足以支持具体结论时，"
        "应明确说明相关事实无法核实，不得编造。"
    )
    messages = LEGAL_PROMPT_TEMPLATE.format_messages(context=context, question=question)
    return f"{messages[0].content}\n\n用户问题：{messages[1].content}\n"


RERANK_MAX_ATTEMPTS = 2
RERANK_RETRY_DELAY_SECONDS = 1
_RERANK_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _is_retryable_rerank_error(error: Exception) -> bool:
    """判断重排请求失败是否属于可重试的暂时性错误。

    HTTPError 是 URLError 的子类，必须先判断 HTTPError，
    再按状态码分类，避免所有 HTTP 状态都被误判为可重试。
    """
    if isinstance(error, HTTPError):
        return error.code in _RERANK_RETRYABLE_HTTP_CODES
    if isinstance(error, URLError):
        return True
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, ConnectionError):
        return True
    if isinstance(error, ssl.SSLError):
        return True
    return False


def rerank_candidates(question: str, candidates: list, top_k: int) -> list:
    """调用重排接口，并把返回的下标映射回原始法条节点。

    暂时性故障（TLS 中断、连接错误、超时、429/5xx）最多重试一次，
    总尝试次数为 2；非暂时性错误立即包装抛出，最终失败不降级。
    """
    extra: dict = {}
    with perf_span("retrieval", "rerank", question=question[:80], extra=extra):
        payload = {
            "model": RERANK_MODEL,
            "query": question,
            "documents": [item.node.text for item in candidates],
            "top_n": top_k,
            "return_documents": False,
        }
        request = Request(
            f"{RERANK_API_BASE}/rerank",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {require_env('SILICONFLOW_API_KEY')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(RERANK_MAX_ATTEMPTS):
            extra["attempts"] = attempt + 1  # 实际可观察的尝试次数
            try:
                with urlopen(request, timeout=30) as response:
                    results = json.loads(response.read().decode("utf-8"))["results"]

                return [
                    NodeWithScore(
                        node=candidates[item["index"]].node,
                        score=float(item["relevance_score"]),
                    )
                    for item in results[:top_k]
                ]
            except Exception as error:
                if attempt == 0 and _is_retryable_rerank_error(error):
                    time.sleep(RERANK_RETRY_DELAY_SECONDS)
                    continue
                raise ExternalServiceError("重排服务调用失败") from error


@lru_cache(maxsize=1)
def load_index() -> VectorStoreIndex:
    """加载现有 Chroma 索引，不触发重新入库。"""
    if not CHROMA_DIR.exists():
        raise IndexNotReadyError("未找到 chroma_db，请先运行入库命令")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as error:
        if error.__class__.__name__ == "NotFoundError":
            raise IndexNotReadyError("未找到 LawRAG 索引，请先运行入库命令") from error
        raise

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=get_embedding(),
    )


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


@lru_cache(maxsize=1)
def get_bm25_index() -> tuple[BM25Okapi, list[TextNode]]:
    """从当前 Chroma 集合加载同一批节点，并构建进程内 BM25 索引。"""
    if not CHROMA_DIR.exists():
        raise IndexNotReadyError("未找到 chroma_db，请先运行入库命令")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as error:
        if error.__class__.__name__ == "NotFoundError":
            raise IndexNotReadyError("未找到 LawRAG 索引，请先运行入库命令") from error
        raise

    stored = collection.get(include=["documents", "metadatas"])
    nodes = [
        TextNode(id_=node_id, text=text, metadata=metadata)
        for node_id, text, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"]
        )
    ]
    if not nodes:
        raise IndexNotReadyError("LawRAG 索引为空，请先运行入库命令")

    tokenized_corpus = [tokenize_for_bm25(node.text) for node in nodes]
    return BM25Okapi(tokenized_corpus), nodes


def bm25_retrieve(query: str, top_k: int) -> list[NodeWithScore]:
    with perf_span("retrieval", "bm25", question=query[:80]):
        bm25, nodes = get_bm25_index()
        scores = bm25.get_scores(tokenize_for_bm25(query))
        ranked_indices = sorted(
            range(len(nodes)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]
        return [
            NodeWithScore(node=nodes[index], score=float(scores[index]))
            for index in ranked_indices
        ]


def vector_retrieve(index: VectorStoreIndex, query: str, top_k: int) -> list:
    with perf_span("retrieval", "vector", question=query[:80]):
        try:
            return index.as_retriever(similarity_top_k=top_k).retrieve(query)
        except Exception as error:
            raise ExternalServiceError("向量化服务调用失败") from error


def generate_short_retrieval_text(
    prompt: str, max_tokens: int, session_id: str | None = None, tags: list[str] | None = None
) -> str:
    """只生成检索辅助短文本，不生成最终法律回答。"""
    try:
        response = traced_invoke(
            get_chat_model().bind(max_tokens=max_tokens),
            prompt,
            session_id=session_id,
            tags=tags,
        )
    except Exception as error:
        raise ExternalServiceError("生成服务调用失败") from error
    content = response.content
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ExternalServiceError("生成服务返回无效结果")
    return content.strip()


def generate_hyde_query(question: str, session_id: str | None = None) -> str:
    result = generate_short_retrieval_text(
        "只生成一段不超过120个汉字的假想法律条文式文本，用于向量检索。"
        "不要回答用户、不要解释、不要引用真实条号。\n"
        f"问题：{question}",
        max_tokens=160,
        session_id=session_id,
        tags=["hyde"],
    )
    if not result:
        raise ExternalServiceError("HyDE 未生成检索文本")
    return result


def parse_query_rewrites(text: str) -> list[str]:
    rewrites = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.、)])\s*", "", raw_line).strip()
        if line and line not in rewrites:
            rewrites.append(line)
    return rewrites[:3]


def generate_query_rewrites(question: str, session_id: str | None = None) -> list[str]:
    result = generate_short_retrieval_text(
        "把下面的问题改写为3个简短法律检索查询，每行一个，不编号，"
        "每行不超过40个汉字。只输出查询，不回答问题。\n"
        f"问题：{question}",
        max_tokens=160,
        session_id=session_id,
        tags=["multi-query"],
    )
    rewrites = parse_query_rewrites(result)
    if len(rewrites) < 2:
        raise ExternalServiceError("Multi-query 未生成至少2个改写查询")
    return rewrites


def reciprocal_rank_fusion(result_sets: list[list], limit: int) -> list:
    """用 RRF 合并多路向量召回结果。"""
    scores = {}
    nodes = {}
    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            node_id = item.node.node_id
            nodes[node_id] = item.node
            scores[node_id] = scores.get(node_id, 0.0) + 1 / (60 + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [NodeWithScore(node=nodes[node_id], score=scores[node_id]) for node_id in ranked_ids]


def retrieve_with_strategy(
    question: str, strategy: str, top_k: int = 3, session_id: str | None = None
) -> list:
    if strategy == "bm25":
        return bm25_retrieve(question, top_k)

    index = load_index()

    if strategy == "vector":
        return vector_retrieve(index, question, top_k)

    if strategy == "vector_rerank":
        candidates = vector_retrieve(index, question, CANDIDATE_K)
        return rerank_candidates(question, candidates, top_k)

    if strategy == "hyde":
        candidates = vector_retrieve(index, generate_hyde_query(question, session_id), CANDIDATE_K)
        return rerank_candidates(question, candidates, top_k)

    if strategy == "multi_query":
        queries = [question, *generate_query_rewrites(question, session_id)]
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            result_sets = list(
                executor.map(
                    lambda query: vector_retrieve(index, query, CANDIDATE_K),
                    queries,
                )
            )
        candidates = reciprocal_rank_fusion(result_sets, CANDIDATE_K)
        return rerank_candidates(question, candidates, top_k)

    if strategy in {"hybrid", "hybrid_rerank"}:
        vector_candidates = vector_retrieve(index, question, CANDIDATE_K)
        bm25_candidates = bm25_retrieve(question, CANDIDATE_K)
        candidate_limit = CANDIDATE_K if strategy == "hybrid_rerank" else top_k
        candidates = reciprocal_rank_fusion(
            [vector_candidates, bm25_candidates],
            candidate_limit,
        )
        if strategy == "hybrid":
            return candidates
        return rerank_candidates(question, candidates, top_k)

    raise ValueError(f"未知检索策略：{strategy}")


def retrieve(question: str, top_k: int = 3, session_id: str | None = None) -> list:
    return retrieve_with_strategy(question, RETRIEVAL_STRATEGY, top_k, session_id=session_id)


def ask(question: str, session_id: str | None = None) -> tuple[str, list]:
    retrieved_nodes = retrieve(question, top_k=3, session_id=session_id)

    try:
        response = traced_invoke(
            get_chat_model(),
            build_prompt(question, retrieved_nodes),
            session_id=session_id,
            tags=["generate"],
        )
    except Exception as error:
        raise ExternalServiceError("生成服务调用失败") from error
    answer = response.content or "现有资料无法回答"
    return answer, retrieved_nodes


def main() -> None:
    configure_utf8_stdio()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入法律问题：").strip()
    if not question:
        raise SystemExit("问题不能为空")

    answer, retrieved_nodes = ask(question)
    print(f"\n回答：\n{answer}")
    print("\n引用来源：")
    for item in retrieved_nodes:
        metadata = item.node.metadata
        print(
            f"- 《{metadata['source']}》{metadata['article']}"
            f"（相关度：{item.score:.4f}）"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        path = log_current_exception("ask", "法律问答失败")
        raise SystemExit(f"法律问答失败，完整日志已保存到：{path}") from None
