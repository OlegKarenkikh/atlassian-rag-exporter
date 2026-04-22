"""Tests for vector_store module — mocked adapters, no real DB required."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import (
    SUPPORTED_BACKENDS,
    ChromaAdapter,
    MilvusAdapter,
    OpenSearchAdapter,
    PgVectorAdapter,
    PineconeAdapter,
    QdrantAdapter,
    RAGDocument,
    RedisAdapter,
    VectorStoreAdapter,
    VectorStoreConfig,
    WeaviateAdapter,
    build_adapter,
    load_corpus,
)


def _doc(i: int = 0) -> RAGDocument:
    return RAGDocument(
        doc_id=f"doc{i}",
        text=f"Hello world {i}",
        embedding=[0.1, 0.2, 0.3],
        metadata={"title": f"Doc {i}"},
    )


def _docs(n: int = 3) -> List[RAGDocument]:
    return [_doc(i) for i in range(n)]


class TestRAGDocument:
    def test_from_markdown_file(self, tmp_path):
        p = tmp_path / "doc1_slug"
        p.mkdir()
        (p / "content.md").write_text("---\npage_id: p1\n---\n# Hello")
        (p / "metadata.json").write_text(json.dumps({"page_id": "p1", "title": "Hello"}))
        doc = RAGDocument.from_markdown_file(p / "content.md")
        assert doc.doc_id == "p1"
        assert "Hello" in doc.text

    def test_from_markdown_no_meta(self, tmp_path):
        p = tmp_path / "doc2"
        p.mkdir()
        (p / "content.md").write_text("# Just content")
        doc = RAGDocument.from_markdown_file(p / "content.md")
        assert isinstance(doc.doc_id, str) and len(doc.doc_id) > 0

    def test_defaults(self):
        d = RAGDocument(doc_id="x", text="hello")
        assert d.embedding is None
        assert d.metadata == {}


class TestVectorStoreConfig:
    def test_defaults(self):
        cfg = VectorStoreConfig()
        assert cfg.backend == "chromadb"
        assert cfg.distance == "cosine"

    def test_from_dict(self):
        cfg = VectorStoreConfig.from_dict(
            {"backend": "qdrant", "embedding_dim": 768, "host": "qdrant.internal"}
        )
        assert cfg.backend == "qdrant"
        assert cfg.embedding_dim == 768

    def test_from_dict_extra_fields(self):
        cfg = VectorStoreConfig.from_dict(
            {"backend": "pinecone", "cloud": "gcp", "region": "us-central1"}
        )
        assert cfg.extra["cloud"] == "gcp"

    def test_from_yaml(self, tmp_path):
        (tmp_path / "cfg.yaml").write_text(
            "vector_store:\n  backend: qdrant\n  collection: my-corpus\n"
        )
        cfg = VectorStoreConfig.from_yaml(str(tmp_path / "cfg.yaml"))
        assert cfg.backend == "qdrant"
        assert cfg.collection == "my-corpus"

    def test_from_yaml_missing_section(self, tmp_path):
        (tmp_path / "cfg.yaml").write_text("base_url: https://x.com\n")
        cfg = VectorStoreConfig.from_yaml(str(tmp_path / "cfg.yaml"))
        assert cfg.backend == "chromadb"


class TestBuildAdapter:
    def test_returns_correct_class(self):
        backends = {
            "chromadb": ChromaAdapter,
            "qdrant": QdrantAdapter,
            "weaviate": WeaviateAdapter,
            "pinecone": PineconeAdapter,
            "pgvector": PgVectorAdapter,
            "opensearch": OpenSearchAdapter,
            "milvus": MilvusAdapter,
            "redis": RedisAdapter,
        }
        for backend, cls in backends.items():
            assert isinstance(build_adapter(VectorStoreConfig(backend=backend)), cls)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            build_adapter(VectorStoreConfig(backend="unicorn_db"))

    def test_supported_backends_count(self):
        assert len(SUPPORTED_BACKENDS) == 8


class TestChromaAdapter:
    def _connect(self, tmp_path):
        mock_chroma = MagicMock()
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col
        mock_chroma.PersistentClient.return_value = mock_client
        cfg = VectorStoreConfig(
            backend="chromadb", extra={"persist_path": str(tmp_path / "chroma")}
        )
        adapter = ChromaAdapter(cfg)
        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            adapter.connect()
        adapter._col = mock_col
        return adapter, mock_col

    def test_upsert_with_embeddings(self, tmp_path):
        adapter, mock_col = self._connect(tmp_path)
        assert adapter.upsert(_docs(2)) == 2
        mock_col.upsert.assert_called_once()

    def test_upsert_without_embeddings(self, tmp_path):
        adapter, mock_col = self._connect(tmp_path)
        adapter.upsert([RAGDocument(doc_id="x", text="text")])
        mock_col.upsert.assert_called_once()

    def test_delete(self, tmp_path):
        adapter, mock_col = self._connect(tmp_path)
        assert adapter.delete(["doc0", "doc1"]) == 2
        mock_col.delete.assert_called_once_with(ids=["doc0", "doc1"])

    def test_search(self, tmp_path):
        adapter, mock_col = self._connect(tmp_path)
        mock_col.query.return_value = {
            "ids": [["doc0"]],
            "documents": [["text0"]],
            "metadatas": [[{"title": "T"}]],
            "distances": [[0.1]],
        }
        results = adapter.search([0.1, 0.2, 0.3], k=1)
        assert results[0].score == pytest.approx(0.9)

    def test_connect_http(self, tmp_path):
        mock_chroma = MagicMock()
        mock_client = MagicMock()
        mock_chroma.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()
        cfg = VectorStoreConfig(
            backend="chromadb", url="http://chroma:8000", host="chroma", port=8000
        )
        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            ChromaAdapter(cfg).connect()
        mock_chroma.HttpClient.assert_called_once()


class TestQdrantAdapter:
    def _make(self):
        cfg = VectorStoreConfig(backend="qdrant", collection="test")
        adapter = QdrantAdapter(cfg)
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        adapter._client = mock_client
        return adapter, mock_client

    def test_search(self):
        adapter, mock_client = self._make()
        hit = MagicMock()
        hit.payload = {"_doc_id": "doc0", "_text": "hello"}
        hit.score = 0.95
        hit.id = 12345
        mock_client.search.return_value = [hit]
        results = adapter.search([0.1, 0.2, 0.3], k=1)
        assert results[0].doc_id == "doc0"
        assert results[0].score == pytest.approx(0.95)

    def test_delete(self):
        adapter, mock_client = self._make()
        with patch.dict(
            "sys.modules", {"qdrant_client": MagicMock(), "qdrant_client.models": MagicMock()}
        ):
            adapter.delete(["doc0"])
        mock_client.delete.assert_called_once()

    def test_close(self):
        adapter, mock_client = self._make()
        adapter.close()
        mock_client.close.assert_called_once()

    def test_connect_with_url(self):
        cfg = VectorStoreConfig(
            backend="qdrant", url="https://cloud.qdrant.io", api_key="key123", collection="wiki"
        )
        mock_qdrant = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[MagicMock(name="wiki")])
        mock_qdrant.QdrantClient.return_value = mock_client
        with patch.dict(
            "sys.modules", {"qdrant_client": mock_qdrant, "qdrant_client.models": MagicMock()}
        ):
            QdrantAdapter(cfg).connect()
        mock_qdrant.QdrantClient.assert_called_once_with(
            url="https://cloud.qdrant.io", api_key="key123"
        )


class TestPineconeAdapter:
    def _make(self):
        cfg = VectorStoreConfig(
            backend="pinecone", collection="wiki", api_key="pk-test", embedding_dim=3
        )
        adapter = PineconeAdapter(cfg)
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.list_indexes.return_value = []
        mock_pc.Index.return_value = mock_index
        adapter._pc = mock_pc
        adapter._index = mock_index
        return adapter, mock_pc, mock_index

    def test_upsert_batches(self):
        adapter, _, mock_index = self._make()
        adapter.cfg.extra["batch_size"] = 2
        adapter.upsert(_docs(5))
        assert mock_index.upsert.call_count == 3

    def test_delete(self):
        adapter, _, mock_index = self._make()
        adapter.delete(["id1", "id2"])
        mock_index.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_search(self):
        adapter, _, mock_index = self._make()
        m1 = MagicMock()
        m1.id = "doc0"
        m1.score = 0.88
        m1.metadata = {"_text": "hello"}
        mock_index.query.return_value = MagicMock(matches=[m1])
        results = adapter.search([0.1, 0.2, 0.3], k=1)
        assert results[0].score == pytest.approx(0.88)

    def test_connect_creates_index(self):
        cfg = VectorStoreConfig(
            backend="pinecone", collection="new-idx", api_key="key", embedding_dim=3
        )
        mock_pinecone_mod = MagicMock()
        mock_pc_inst = MagicMock()
        mock_pc_inst.list_indexes.return_value = []
        mock_pinecone_mod.Pinecone.return_value = mock_pc_inst
        mock_pinecone_mod.ServerlessSpec = MagicMock()
        with patch.dict("sys.modules", {"pinecone": mock_pinecone_mod}):
            PineconeAdapter(cfg).connect()
        mock_pc_inst.create_index.assert_called_once()


class TestMilvusAdapter:
    def _make(self):
        cfg = VectorStoreConfig(
            backend="milvus", url="http://localhost:19530", collection="test", embedding_dim=3
        )
        adapter = MilvusAdapter(cfg)
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        adapter._client = mock_client
        return adapter, mock_client

    def test_upsert_skips_no_embedding(self):
        adapter, mock_client = self._make()
        adapter.upsert([RAGDocument(doc_id="no-emb", text="text")])
        mock_client.upsert.assert_not_called()

    def test_upsert_with_embeddings(self):
        adapter, mock_client = self._make()
        adapter.upsert(_docs(2))
        mock_client.upsert.assert_called_once()

    def test_delete(self):
        adapter, mock_client = self._make()
        adapter.delete(["doc0"])
        mock_client.delete.assert_called_once_with(collection_name="test", ids=["doc0"])

    def test_search(self):
        adapter, mock_client = self._make()
        hit = {"id": "doc0", "distance": 0.9, "entity": {"doc_id": "doc0", "text": "hello"}}
        mock_client.search.return_value = [[hit]]
        results = adapter.search([0.1, 0.2, 0.3])
        assert results[0].doc_id == "doc0"

    def test_close(self):
        adapter, mock_client = self._make()
        adapter.close()
        mock_client.close.assert_called_once()


class TestOpenSearchAdapter:
    def _make(self):
        cfg = VectorStoreConfig(
            backend="opensearch", host="localhost", port=9200, collection="rag", embedding_dim=3
        )
        adapter = OpenSearchAdapter(cfg)
        mock_os = MagicMock()
        mock_os.indices.exists.return_value = True
        adapter._os = mock_os
        return adapter, mock_os

    def test_upsert(self):
        adapter, mock_os = self._make()
        assert adapter.upsert(_docs(2)) == 2
        assert mock_os.index.call_count == 2

    def test_delete(self):
        adapter, mock_os = self._make()
        adapter.delete(["doc0", "doc1"])
        assert mock_os.delete.call_count == 2

    def test_search(self):
        adapter, mock_os = self._make()
        mock_os.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "doc0", "_score": 1.5, "_source": {"doc_id": "doc0", "text": "hello"}}
                ]
            }
        }
        results = adapter.search([0.1, 0.2, 0.3], k=1)
        assert results[0].doc_id == "doc0"


class TestPgVectorAdapter:
    def _make(self):
        cfg = VectorStoreConfig(
            backend="pgvector",
            collection="rag",
            embedding_dim=3,
            extra={"dbname": "rag", "user": "pg", "password": "pw"},
        )
        adapter = PgVectorAdapter(cfg)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        adapter._conn = mock_conn
        return adapter, mock_conn, mock_cursor

    def test_upsert_executes_sql(self):
        adapter, mock_conn, mock_cursor = self._make()
        adapter.upsert(_docs(2))
        assert mock_cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()

    def test_upsert_skips_no_embedding(self):
        adapter, mock_conn, mock_cursor = self._make()
        adapter.upsert([RAGDocument(doc_id="x", text="t")])
        mock_cursor.execute.assert_not_called()

    def test_delete(self):
        adapter, mock_conn, mock_cursor = self._make()
        adapter.delete(["doc0", "doc1"])
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    def test_search(self):
        adapter, mock_conn, mock_cursor = self._make()
        mock_cursor.fetchall.return_value = [("doc0", "text0", {"title": "T"}, 0.95)]
        results = adapter.search([0.1, 0.2, 0.3], k=1)
        assert results[0].doc_id == "doc0"
        assert results[0].score == pytest.approx(0.95)

    def test_close(self):
        adapter, mock_conn, _ = self._make()
        adapter.close()
        mock_conn.close.assert_called_once()


class TestRedisAdapter:
    def _make(self):
        cfg = VectorStoreConfig(backend="redis", collection="rag", embedding_dim=3)
        adapter = RedisAdapter(cfg)
        mock_r = MagicMock()
        mock_r.pipeline.return_value = MagicMock()
        adapter._r = mock_r
        return adapter, mock_r

    def test_upsert(self):
        adapter, mock_r = self._make()
        adapter.upsert(_docs(3))
        mock_r.pipeline.return_value.execute.assert_called_once()

    def test_delete(self):
        adapter, mock_r = self._make()
        adapter.delete(["doc0", "doc1"])
        mock_r.pipeline.return_value.execute.assert_called_once()


class TestLoadCorpus:
    def test_loads_files(self, tmp_path):
        for i in range(3):
            d = tmp_path / f"doc{i}"
            d.mkdir()
            (d / "content.md").write_text(f"# Doc {i}")
            (d / "metadata.json").write_text(json.dumps({"page_id": f"p{i}"}))
        mock_adapter = MagicMock(spec=VectorStoreAdapter)
        mock_adapter.upsert.return_value = 3
        result = load_corpus(mock_adapter, str(tmp_path))
        assert result["upserted"] == 3
        assert result["errors"] == 0

    def test_calls_embedder(self, tmp_path):
        d = tmp_path / "doc1"
        d.mkdir()
        (d / "content.md").write_text("# Test")
        (d / "metadata.json").write_text(json.dumps({"page_id": "p1"}))
        mock_adapter = MagicMock(spec=VectorStoreAdapter)
        mock_adapter.upsert.return_value = 1
        embedder = MagicMock(return_value=[0.1, 0.2, 0.3])
        load_corpus(mock_adapter, str(tmp_path), embedder=embedder)
        embedder.assert_called_once()

    def test_handles_error(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "content.md").write_text("bad")
        mock_adapter = MagicMock(spec=VectorStoreAdapter)
        mock_adapter.upsert.return_value = 0
        with patch("vector_store.RAGDocument.from_markdown_file", side_effect=Exception("corrupt")):
            result = load_corpus(mock_adapter, str(tmp_path))
        assert result["errors"] == 1

    def test_batching(self, tmp_path):
        for i in range(5):
            d = tmp_path / f"doc{i}"
            d.mkdir()
            (d / "content.md").write_text(f"# Doc {i}")
        mock_adapter = MagicMock(spec=VectorStoreAdapter)
        mock_adapter.upsert.return_value = 2
        load_corpus(mock_adapter, str(tmp_path), batch_size=2)
        assert mock_adapter.upsert.call_count == 3
