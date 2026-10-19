import re
from pathlib import Path
from uuid import uuid4

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.readers.file import DocxReader, FlatReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    KB_DIR,
    configure_utf8_stdio,
    get_embedding,
    log_current_exception,
)


ARTICLE_START = re.compile(r"(?m)^(第[零〇一二三四五六七八九十百千0-9]+条)[ \t]*")
SECTION_HEADING = re.compile(
    r"^第[零〇一二三四五六七八九十百千0-9]+[章节][ \t]*.*$"
)
PAGE_MARKER = re.compile(r"^[—－-]?\s*\d+\s*[—－-]?$|^第\s*\d+\s*页(?:\s*/\s*共\s*\d+\s*页)?$")


def clean_text(text: str) -> str:
    """清理全角空格、重复空行、章节标题和页码残留。"""
    cleaned_lines: list[str] = []
    previous_blank = False

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.replace("\u3000", " ").replace("\xa0", " ").strip()
        if PAGE_MARKER.fullmatch(line) or SECTION_HEADING.fullmatch(line):
            continue
        if not line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def split_articles(text: str) -> list[tuple[str, str]]:
    """每个块恰好从一条法条开始，到下一条法条之前结束。"""
    matches = list(ARTICLE_START.finditer(text))
    articles: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article_text = text[match.start() : end].strip()
        articles.append((match.group(1), article_text))
    return articles


def load_documents(kb_dir: Path = KB_DIR):
    reader = SimpleDirectoryReader(
        input_dir=str(kb_dir),
        recursive=False,
        required_exts=[".docx", ".txt"],
        file_extractor={".docx": DocxReader(), ".txt": FlatReader()},
    )
    return reader.load_data()


def build_article_nodes(kb_dir: Path = KB_DIR) -> list[TextNode]:
    nodes: list[TextNode] = []
    for document in load_documents(kb_dir):
        file_name = document.metadata.get("file_name") or document.metadata.get("filename")
        if not file_name:
            raise ValueError("文档缺少文件名元数据")
        source = Path(file_name).stem
        for article, article_text in split_articles(clean_text(document.text)):
            nodes.append(
                TextNode(
                    text=article_text,
                    metadata={
                        "source": source,
                        "file_name": file_name,
                        "article": article,
                    },
                )
            )
    if not nodes:
        raise RuntimeError("kb/ 中没有解析出任何法条")
    return nodes


def _is_not_found(error: Exception) -> bool:
    """仅把 Chroma 的集合不存在异常当作“不存在”，其他异常继续上抛。"""
    return error.__class__.__name__ == "NotFoundError"


def _get_collection_or_none(client, name: str):
    """获取集合；只有 Chroma 集合不存在时才返回 None，不吞掉其他异常。"""
    try:
        return client.get_collection(name)
    except Exception as error:
        if not _is_not_found(error):
            raise
        return None


def _create_staging_collection(client) -> tuple[str, object]:
    """创建唯一命名的 staging 集合，返回 (staging_name, collection)。

    名称带随机后缀，绝不复用可能残留的同名集合；保持 cosine 空间。
    """
    staging_name = f"{COLLECTION_NAME}__staging__{uuid4()}"
    collection = client.create_collection(
        staging_name,
        metadata={"hnsw:space": "cosine"},
    )
    return staging_name, collection


def _build_index(nodes: list[TextNode], collection) -> None:
    """把本轮法条节点写入指定集合（业务行为与原实现一致）。"""
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=get_embedding(),
        show_progress=True,
    )


def _switch_collections(client, staging_name: str) -> None:
    """把已构建并校验通过的 staging 集合提升为正式集合。

    - 首次入库（不存在正式集合）：staging 直接改名为正式集合，不创建备份。
    - 已有正式集合：旧集合先改名为唯一 backup，staging 再改名为正式集合，
      确认新正式集合可获取后删除 backup。
    - 切换失败：尝试把 backup 恢复为正式集合；恢复也失败时抛出同时包含
      “切换失败、恢复失败、备份集合名”的错误，且不删除 backup。
    - 集合改名切换不是数据库级事务；本函数只保证上述有界流程。
    """
    old = _get_collection_or_none(client, COLLECTION_NAME)
    if old is None:
        client.get_collection(staging_name).modify(name=COLLECTION_NAME)
        return

    backup_name = f"{COLLECTION_NAME}__backup__{uuid4()}"
    old.modify(name=backup_name)
    try:
        client.get_collection(staging_name).modify(name=COLLECTION_NAME)
    except Exception as switch_error:
        try:
            client.get_collection(backup_name).modify(name=COLLECTION_NAME)
        except Exception as restore_error:
            raise RuntimeError(
                f"集合切换失败且恢复失败；切换异常：{switch_error}；"
                f"恢复异常：{restore_error}；备份集合名：{backup_name}"
            ) from switch_error
        # 恢复成功：旧正式集合已还原，仍把原始切换异常作为主错误抛出，
        # 不得进入成功输出。
        raise

    try:
        client.get_collection(COLLECTION_NAME)
    except Exception as confirm_error:
        raise RuntimeError(
            f"新正式集合确认失败；备份集合名：{backup_name}"
        ) from confirm_error
    client.delete_collection(backup_name)


def main() -> None:
    configure_utf8_stdio()
    nodes = build_article_nodes()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    staging_name = None
    try:
        staging_name, staging_collection = _create_staging_collection(client)
        _build_index(nodes, staging_collection)
        if staging_collection.count() != len(nodes):
            raise RuntimeError(
                f"staging 集合完整性检查失败：count={staging_collection.count()}，"
                f"nodes={len(nodes)}，不切换正式集合"
            )
        _switch_collections(client, staging_name)
        staging_name = None
    except Exception as error:
        if staging_name is not None:
            try:
                client.delete_collection(staging_name)
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"入库失败，且 staging 临时集合清理失败（保留：{staging_name}）："
                    f"{cleanup_error}"
                ) from error
        raise

    print(f"入库完成：{len(nodes)} 条法条，持久化目录：{CHROMA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        path = log_current_exception("ingest", "文档入库失败")
        raise SystemExit(f"文档入库失败，完整日志已保存到：{path}") from None
