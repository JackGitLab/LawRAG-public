import chromadb
import pytest
from chromadb.errors import NotFoundError
from llama_index.core.schema import TextNode

import app.ingest as ingest_module
from app.ingest import build_article_nodes, clean_text, split_articles


def test_clean_text_removes_layout_noise() -> None:
    raw = "第一章\u3000总则\n\n\n第一条\u3000正文\n－1－\n"
    assert clean_text(raw) == "第一条 正文"


def test_split_articles_keeps_each_article_whole() -> None:
    text = "标题\n第一条 正文第一段\n正文第二段\n第二条 正文"
    articles = split_articles(text)

    assert articles == [
        ("第一条", "第一条 正文第一段\n正文第二段"),
        ("第二条", "第二条 正文"),
    ]


def test_real_labor_law_keeps_articles_46_and_47_separate() -> None:
    labor_nodes = [
        node
        for node in build_article_nodes()
        if node.metadata["source"] == "中华人民共和国劳动合同法"
    ]
    by_article = {node.metadata["article"]: node.text for node in labor_nodes}

    assert "应当向劳动者支付经济补偿" in by_article["第四十六条"]
    assert "（五）" in by_article["第四十六条"]
    assert "每满一年支付一个月工资" in by_article["第四十七条"]
    assert "第四十七条" not in by_article["第四十六条"]


# ---------------------------------------------------------------------------
# P05 切换流程测试：Fake 客户端
# ---------------------------------------------------------------------------


class FakeCollection:
    def __init__(self, client, name, count=0):
        self.client = client
        self.name = name
        self._count = count
        self.docs = []

    def count(self):
        return self._count

    def modify(self, name=None, metadata=None, configuration=None):
        if name:
            self.client._rename(self, name)

    def add(self, ids=None, embeddings=None, documents=None, metadatas=None):
        self._count = len(ids or [])
        self.docs = list(documents or [])


class FakeClient:
    """严格受控替身：按名称维护集合注册表，模拟 Chroma 改名行为。"""

    def __init__(self, collections=None, on_modify=None):
        self.on_modify_cb = on_modify or (lambda collection, target: None)
        self.staging_count = 0
        self.collections = {}
        for name, count in (collections or {}).items():
            self.collections[name] = FakeCollection(self, name, count=count)

    def _rename(self, collection, new_name):
        self.on_modify_cb(collection, new_name)
        del self.collections[collection.name]
        collection.name = new_name
        self.collections[new_name] = collection

    def create_collection(self, name, metadata=None, **kwargs):
        count = self.staging_count if "__staging__" in name else 0
        collection = FakeCollection(self, name, count=count)
        self.collections[name] = collection
        return collection

    def get_collection(self, name):
        if name not in self.collections:
            raise NotFoundError(f"集合不存在：{name}")
        return self.collections[name]

    def delete_collection(self, name):
        if name not in self.collections:
            raise NotFoundError(f"集合不存在：{name}")
        del self.collections[name]

    def list_collections(self):
        return list(self.collections.values())


def _patch_ingest_env(monkeypatch, client, nodes, index_failure=None) -> None:
    """把 ingest.main 的构建链路替换为受控替身，不触碰真实文件或服务。"""
    monkeypatch.setattr(ingest_module, "build_article_nodes", lambda: nodes)
    monkeypatch.setattr(ingest_module, "get_embedding", lambda: object())
    if index_failure is not None:
        def fail_index(*args, **kwargs):
            raise index_failure
        monkeypatch.setattr(ingest_module, "VectorStoreIndex", fail_index)
    else:
        monkeypatch.setattr(
            ingest_module, "VectorStoreIndex", lambda *args, **kwargs: None
        )
    monkeypatch.setattr(ingest_module.chromadb, "PersistentClient", lambda path: client)


def _node(text: str) -> TextNode:
    return TextNode(text=text)


def test_build_failure_keeps_old_collection(monkeypatch, capsys) -> None:
    client = FakeClient(collections={"lawrag": 2})
    _patch_ingest_env(
        monkeypatch,
        client,
        [_node("第一条 正文")],
        index_failure=RuntimeError("模拟构建失败"),
    )

    with pytest.raises(RuntimeError, match="模拟构建失败"):
        ingest_module.main()

    # 旧 lawrag 仍存在且内容不变
    assert "lawrag" in client.collections
    assert client.collections["lawrag"].count() == 2
    # staging 被清理
    assert not any("__staging__" in name for name in client.collections)
    # 没有成功输出
    assert "入库完成" not in capsys.readouterr().out


def test_completeness_check_failure_does_not_switch(monkeypatch, capsys) -> None:
    nodes = [_node("第一条 正文"), _node("第二条 正文")]
    client = FakeClient(collections={"lawrag": 1})
    _patch_ingest_env(monkeypatch, client, nodes)

    with pytest.raises(RuntimeError, match="完整性检查失败"):
        ingest_module.main()

    assert client.collections["lawrag"].count() == 1  # 正式集合未切换
    assert not any("__staging__" in name for name in client.collections)
    assert "入库完成" not in capsys.readouterr().out


def test_switch_failure_restores_old_collection(monkeypatch, capsys) -> None:
    client = FakeClient(collections={"lawrag": 1})

    def on_modify(collection, target):
        if "__staging__" in collection.name and target == "lawrag":
            raise RuntimeError("模拟 staging 切换失败")

    client.on_modify_cb = on_modify
    _patch_ingest_env(monkeypatch, client, [_node("第一条 正文")])
    client.staging_count = 1

    with pytest.raises(RuntimeError, match="模拟 staging 切换失败"):
        ingest_module.main()

    # 恢复成功：旧集合重新名为 lawrag 且内容不变
    assert "lawrag" in client.collections
    assert client.collections["lawrag"].count() == 1
    # 原切换异常未被吞掉；无成功输出；staging 已清理
    assert not any("__staging__" in name for name in client.collections)
    assert not any("__backup__" in name for name in client.collections)
    assert "入库完成" not in capsys.readouterr().out


def test_switch_and_restore_failure_keeps_backup(monkeypatch, capsys) -> None:
    client = FakeClient(collections={"lawrag": 1})

    def on_modify(collection, target):
        if "__staging__" in collection.name and target == "lawrag":
            raise RuntimeError("模拟 staging 切换失败")
        if "__backup__" in collection.name and target == "lawrag":
            raise RuntimeError("模拟备份恢复失败")

    client.on_modify_cb = on_modify
    _patch_ingest_env(monkeypatch, client, [_node("第一条 正文")])
    client.staging_count = 1

    with pytest.raises(RuntimeError, match="备份集合名") as caught:
        ingest_module.main()

    message = str(caught.value)
    assert "切换失败" in message
    assert "恢复失败" in message
    assert "lawrag__backup__" in message  # 错误信息包含可定位的备份集合名
    # 备份未被删除
    assert any("__backup__" in name for name in client.collections)
    # staging 已清理；无成功输出
    assert not any("__staging__" in name for name in client.collections)
    assert "入库完成" not in capsys.readouterr().out


def test_first_time_run_has_no_backup_and_succeeds(monkeypatch, capsys) -> None:
    client = FakeClient(collections={})
    _patch_ingest_env(monkeypatch, client, [_node("第一条 正文")])
    client.staging_count = 1

    ingest_module.main()

    assert "lawrag" in client.collections
    assert not any("__staging__" in name for name in client.collections)
    assert not any("__backup__" in name for name in client.collections)
    assert "入库完成" in capsys.readouterr().out


def test_old_index_remains_usable_during_build(monkeypatch, capsys) -> None:
    client = FakeClient(collections={"lawrag": 1})
    observed = {}

    def build_index(*args, **kwargs):
        # 构建 staging 过程中检查正式 lawrag 仍为旧集合
        observed["during_build_lawrag_count"] = client.collections["lawrag"].count()
        return None

    monkeypatch.setattr(ingest_module, "build_article_nodes", lambda: [_node("第一条 正文")])
    monkeypatch.setattr(ingest_module, "get_embedding", lambda: object())
    monkeypatch.setattr(ingest_module, "VectorStoreIndex", build_index)
    monkeypatch.setattr(ingest_module.chromadb, "PersistentClient", lambda path: client)
    client.staging_count = 1

    ingest_module.main()

    # 构建期间旧集合一直可读，且切换成功后仍是成功流程
    assert observed["during_build_lawrag_count"] == 1
    assert "入库完成" in capsys.readouterr().out


def test_non_not_found_error_is_not_treated_as_first_run(monkeypatch, capsys) -> None:
    client = FakeClient(collections={})

    def broken_get(name):
        raise RuntimeError("数据库故障")

    client.get_collection = broken_get
    _patch_ingest_env(monkeypatch, client, [_node("第一条 正文")])
    client.staging_count = 1

    with pytest.raises(RuntimeError, match="数据库故障"):
        ingest_module.main()

    # 没有把普通异常当作首次入库：未创建 lawrag，也未创建备份
    assert "lawrag" not in client.collections
    assert not any("__backup__" in name for name in client.collections)
    assert not any("__staging__" in name for name in client.collections)
    assert "入库完成" not in capsys.readouterr().out


def test_cleanup_failure_is_reported_with_original_error(monkeypatch, capsys) -> None:
    client = FakeClient(collections={"lawrag": 1})

    def broken_delete(name):
        if "__staging__" in name:
            raise RuntimeError("模拟清理失败")
        raise NotFoundError(f"集合不存在：{name}")

    client.delete_collection = broken_delete
    _patch_ingest_env(
        monkeypatch,
        client,
        [_node("第一条 正文")],
        index_failure=RuntimeError("模拟构建失败"),
    )

    with pytest.raises(RuntimeError, match="清理失败") as caught:
        ingest_module.main()

    # 原始业务异常通过 __cause__ 保留
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "模拟构建失败" in str(caught.value.__cause__)
    assert "入库完成" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# P05 切换流程测试：tmp_path 下真实 Chroma PersistentClient
# ---------------------------------------------------------------------------


def test_real_chroma_switch_replaces_old_index(tmp_path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    try:
        old = client.create_collection("lawrag", metadata={"hnsw:space": "cosine"})
        old.add(ids=["old-1"], embeddings=[[0.1, 0.2, 0.3]], documents=["旧文档"])
        staging_name, staging = ingest_module._create_staging_collection(client)
        staging.add(ids=["new-1"], embeddings=[[0.9, 0.8, 0.7]], documents=["新文档"])

        ingest_module._switch_collections(client, staging_name)

        current = client.get_collection("lawrag")
        assert current.count() == 1
        assert current.get(ids=["new-1"], include=["documents"])["documents"] == [
            "新文档"
        ]
        names = [c.name for c in client.list_collections()]
        assert "lawrag" in names
        # 本轮 staging 与 backup 均不存在
        assert not any("__staging__" in name or "__backup__" in name for name in names)
    finally:
        client.close()


def test_real_chroma_first_time_promotes_staging(tmp_path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    try:
        assert not any(c.name == "lawrag" for c in client.list_collections())
        staging_name, staging = ingest_module._create_staging_collection(client)
        staging.add(ids=["a-1"], embeddings=[[0.1, 0.2]], documents=["首条"])

        ingest_module._switch_collections(client, staging_name)

        current = client.get_collection("lawrag")
        assert current.count() == 1
        assert current.get(ids=["a-1"], include=["documents"])["documents"] == ["首条"]
        names = [c.name for c in client.list_collections()]
        assert not any("__staging__" in name or "__backup__" in name for name in names)
    finally:
        client.close()
