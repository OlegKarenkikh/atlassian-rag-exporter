"""test_new_modules.py — tests for embedder, webhook_listener, rag_tester."""
from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# embedder.py
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedderConfig:
    def test_from_dict(self):
        from embedder import EmbedderConfig
        cfg = EmbedderConfig.from_dict({"backend": "stub", "dim": 384, "batch_size": 16})
        assert cfg.backend == "stub" and cfg.dim == 384

    def test_unknown_fields_go_to_extra(self):
        from embedder import EmbedderConfig
        cfg = EmbedderConfig.from_dict({"backend": "stub", "device": "cuda"})
        assert cfg.extra["device"] == "cuda"

    def test_russian_presets_exist(self):
        from embedder import EmbedderConfig
        assert "sentence_transformers" in EmbedderConfig.RUSSIAN_PRESETS
        assert "model" in EmbedderConfig.RUSSIAN_PRESETS["sentence_transformers"]


class TestRuPreprocess:
    def test_strips_zero_width(self):
        from embedder import _ru_preprocess
        assert "\u200b" not in _ru_preprocess("hello\u200bworld")

    def test_collapses_whitespace(self):
        from embedder import _ru_preprocess
        result = _ru_preprocess("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_truncates_to_max_chars(self):
        from embedder import _ru_preprocess
        assert len(_ru_preprocess("x" * 10000, max_chars=100)) == 100

    def test_unicode_normalize(self):
        from embedder import _ru_preprocess
        result = _ru_preprocess("hello\u00adworld")
        assert "\u00ad" not in result


class TestChunkText:
    def test_single_chunk_short_text(self):
        from embedder import _chunk_text
        chunks = _chunk_text("short text", chunk_size=512)
        assert chunks == ["short text"]

    def test_multiple_chunks(self):
        from embedder import _chunk_text
        long_text = "\n".join([f"sentence {i}" for i in range(100)])
        chunks = _chunk_text(long_text, chunk_size=50, overlap=5)
        assert len(chunks) > 1

    def test_overlap_preserves_context(self):
        from embedder import _chunk_text
        lines = [f"line{i}" for i in range(20)]
        text = "\n".join(lines)
        chunks = _chunk_text(text, chunk_size=30, overlap=3)
        assert len(chunks) >= 2


class TestStubEmbedder:
    def test_returns_zero_vectors(self):
        from embedder import StubEmbedder, EmbedderConfig
        e = StubEmbedder(EmbedderConfig(backend="stub"))
        vecs = e.embed(["hello", "мир"])
        assert len(vecs) == 2
        assert all(v == 0.0 for v in vecs[0])

    def test_embed_one(self):
        from embedder import StubEmbedder, EmbedderConfig
        e = StubEmbedder(EmbedderConfig(backend="stub"))
        vec = e.embed_one("тест")
        assert len(vec) == 384

    def test_dim_property(self):
        from embedder import StubEmbedder, EmbedderConfig
        assert StubEmbedder(EmbedderConfig(backend="stub")).dim == 384

    def test_batch_split(self):
        from embedder import StubEmbedder, EmbedderConfig
        e = StubEmbedder(EmbedderConfig(backend="stub", batch_size=2))
        vecs = e.embed(["a", "b", "c", "d", "e"])
        assert len(vecs) == 5

    def test_preprocess_applied(self):
        from embedder import StubEmbedder, EmbedderConfig
        e = StubEmbedder(EmbedderConfig(backend="stub", ru_normalize=True))
        vecs = e.embed(["Привет\u200b мир!"])
        assert len(vecs) == 1

    def test_preprocess_skip(self):
        from embedder import StubEmbedder, EmbedderConfig
        e = StubEmbedder(EmbedderConfig(backend="stub", ru_normalize=False))
        assert len(e.embed(["raw text"])) == 1


class TestBuildEmbedder:
    def test_build_stub(self):
        from embedder import build_embedder, EmbedderConfig, StubEmbedder
        assert isinstance(build_embedder(EmbedderConfig(backend="stub")), StubEmbedder)

    def test_build_unknown(self):
        from embedder import build_embedder, EmbedderConfig
        with pytest.raises(ValueError, match="Unknown embedder backend"):
            build_embedder(EmbedderConfig(backend="magic"))

    def test_build_openai_compatible(self):
        from embedder import build_embedder, EmbedderConfig, OpenAICompatibleEmbedder
        e = build_embedder(EmbedderConfig(backend="openai_compatible",
                                          api_url="http://localhost:11434/v1"))
        assert isinstance(e, OpenAICompatibleEmbedder)


class TestOpenAICompatibleEmbedder:
    def _embedder(self):
        from embedder import OpenAICompatibleEmbedder, EmbedderConfig
        return OpenAICompatibleEmbedder(EmbedderConfig(
            backend="openai_compatible", api_url="http://localhost:11434/v1", model="test"))

    def test_embed_batch_parses_response(self):
        e = self._embedder()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]},
                     {"index": 1, "embedding": [0.4, 0.5, 0.6]}]}
        with patch.object(e._client, "post", return_value=mock_resp):
            vecs = e._embed_batch(["hello", "world"])
        assert len(vecs) == 2 and vecs[0] == [0.1, 0.2, 0.3]

    def test_embed_batch_orders_by_index(self):
        e = self._embedder()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"index": 1, "embedding": [0.4, 0.5]},
                     {"index": 0, "embedding": [0.1, 0.2]}]}
        with patch.object(e._client, "post", return_value=mock_resp):
            vecs = e._embed_batch(["a", "b"])
        assert vecs[0] == [0.1, 0.2] and vecs[1] == [0.4, 0.5]


class TestChunkDocument:
    def test_basic(self):
        from embedder import chunk_document
        chunks = chunk_document("doc1", "Hello world. " * 50, {})
        assert all(c.doc_id == "doc1" for c in chunks)
        assert chunks[0].chunk_idx == 0

    def test_chunk_ids_unique(self):
        from embedder import chunk_document
        chunks = chunk_document("d1", "\n".join(["x"] * 200), {}, chunk_size=30)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_merged(self):
        from embedder import chunk_document
        chunks = chunk_document("d1", "some text", {"source": "confluence"})
        assert chunks[0].metadata["source"] == "confluence" and "chunk_idx" in chunks[0].metadata

    def test_single_chunk_short(self):
        from embedder import chunk_document
        assert len(chunk_document("d1", "short", {})) == 1


# ─────────────────────────────────────────────────────────────────────────────
# webhook_listener.py
# ─────────────────────────────────────────────────────────────────────────────

class TestSignatureVerification:
    def test_valid_confluence_sig(self):
        from webhook_listener import _verify_confluence_signature
        import hmac as _hmac, hashlib
        secret, body = "mysecret", b'{"event": "page_updated"}'
        sig = "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert _verify_confluence_signature(body, secret, sig)

    def test_invalid_sig(self):
        from webhook_listener import _verify_confluence_signature
        assert not _verify_confluence_signature(b"body", "secret", "sha256=wrong")

    def test_no_secret_always_passes(self):
        from webhook_listener import _verify_confluence_signature
        assert _verify_confluence_signature(b"body", "", "sha256=anything")

    def test_jira_sig_same_scheme(self):
        from webhook_listener import _verify_jira_signature
        import hmac as _hmac, hashlib
        secret, body = "sec", b'{"webhookEvent": "jira:issue_updated"}'
        sig = "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert _verify_jira_signature(body, secret, sig)


class TestEventParsers:
    def test_parse_confluence_page_updated(self):
        from webhook_listener import _parse_confluence_event
        evt = _parse_confluence_event(
            {"webhookEvent": "page_updated", "page": {"id": "123"}, "space": {"key": "ENG"}})
        assert evt is not None and evt.entity_id == "123" and evt.space_key == "ENG"

    def test_parse_confluence_no_page(self):
        from webhook_listener import _parse_confluence_event
        assert _parse_confluence_event({"webhookEvent": "space_created"}) is None

    def test_parse_jira_issue_updated(self):
        from webhook_listener import _parse_jira_event
        evt = _parse_jira_event({"webhookEvent": "jira:issue_updated", "issue": {"key": "PROJ-42"}})
        assert evt is not None and evt.entity_id == "PROJ-42" and evt.source == "jira"

    def test_parse_jira_no_issue(self):
        from webhook_listener import _parse_jira_event
        assert _parse_jira_event({"webhookEvent": "jira:issue_updated"}) is None

    def test_reindex_event_timestamp(self):
        from webhook_listener import ReindexEvent
        evt = ReindexEvent(source="confluence", event_type="page_updated", entity_id="1")
        assert evt.timestamp <= time.time()


class TestCreateApp:
    def test_health_endpoint(self):
        from webhook_listener import create_app
        from fastapi.testclient import TestClient
        q = asyncio.Queue()
        client = TestClient(create_app(q))
        resp = client.get("/health")
        assert resp.status_code == 200 and resp.json()["status"] == "ok"

    def test_confluence_webhook_enqueues(self):
        from webhook_listener import create_app
        from fastapi.testclient import TestClient
        q = asyncio.Queue()
        client = TestClient(create_app(q))
        payload = {"webhookEvent": "page_updated", "page": {"id": "42"}, "space": {"key": "ENG"}}
        resp = client.post("/webhook/confluence", json=payload)
        assert resp.status_code == 200 and resp.json()["queued"] is True and q.qsize() == 1

    def test_confluence_webhook_bad_json(self):
        from webhook_listener import create_app
        from fastapi.testclient import TestClient
        q = asyncio.Queue()
        client = TestClient(create_app(q))
        resp = client.post("/webhook/confluence", content=b"not json",
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 400

    def test_jira_webhook_enqueues(self):
        from webhook_listener import create_app
        from fastapi.testclient import TestClient
        q = asyncio.Queue()
        client = TestClient(create_app(q))
        resp = client.post("/webhook/jira",
                           json={"webhookEvent": "jira:issue_updated", "issue": {"key": "BUG-1"}})
        assert resp.status_code == 200 and q.qsize() == 1

    def test_jira_webhook_bad_signature(self):
        from webhook_listener import create_app
        from fastapi.testclient import TestClient
        q = asyncio.Queue()
        client = TestClient(create_app(q, jira_secret="secret"))
        resp = client.post("/webhook/jira",
                           json={"webhookEvent": "jira:issue_updated", "issue": {"key": "X-1"}},
                           headers={"X-Hub-Signature": "sha256=wrong"})
        assert resp.status_code == 401

    def test_confluence_no_page_not_queued(self):
        from webhook_listener import create_app
        from fastapi.testclient import TestClient
        q = asyncio.Queue()
        client = TestClient(create_app(q))
        resp = client.post("/webhook/confluence", json={"webhookEvent": "space_created"})
        assert resp.json()["queued"] is False and q.qsize() == 0


class TestReindexWorkerBatch:
    def test_delete_calls_indexer_delete(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        exporter, indexer = MagicMock(), MagicMock()
        worker = ReindexWorker(asyncio.Queue(), lambda: exporter, lambda: indexer)
        worker._process_batch(
            [ReindexEvent(source="confluence", event_type="page_trashed", entity_id="p1")])
        indexer.delete.assert_called_once_with(["p1"])

    def test_update_calls_export_upsert(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        exporter, indexer = MagicMock(), MagicMock()
        exporter.export_page_by_id.return_value = {"id": "p1"}
        worker = ReindexWorker(asyncio.Queue(), lambda: exporter, lambda: indexer)
        worker._process_batch(
            [ReindexEvent(source="confluence", event_type="page_updated", entity_id="p1")])
        exporter.export_page_by_id.assert_called_once_with("p1")
        indexer.upsert.assert_called_once()

    def test_jira_deleted(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        indexer = MagicMock()
        worker = ReindexWorker(asyncio.Queue(), lambda: MagicMock(), lambda: indexer)
        worker._process_batch(
            [ReindexEvent(source="jira", event_type="jira:issue_deleted", entity_id="T-1")])
        indexer.delete.assert_called_with(["T-1"])

    def test_jira_updated(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        exporter, indexer = MagicMock(), MagicMock()
        exporter.export_jira_issue_by_key.return_value = {"key": "T-1"}
        worker = ReindexWorker(asyncio.Queue(), lambda: exporter, lambda: indexer)
        worker._process_batch(
            [ReindexEvent(source="jira", event_type="jira:issue_updated", entity_id="T-1")])
        exporter.export_jira_issue_by_key.assert_called_once_with("T-1")
        indexer.upsert.assert_called_once()

    def test_export_error_does_not_crash(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        exporter = MagicMock()
        exporter.export_page_by_id.side_effect = Exception("boom")
        worker = ReindexWorker(asyncio.Queue(), lambda: exporter, lambda: MagicMock())
        worker._process_batch(
            [ReindexEvent(source="confluence", event_type="page_updated", entity_id="p1")])


# ─────────────────────────────────────────────────────────────────────────────
# rag_tester.py
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMClient:
    def _client(self, stream=False):
        from rag_tester import LLMClient
        return LLMClient({"api_url": "http://localhost:11434/v1", "model": "mistral",
                          "stream": stream, "timeout": 5})

    def test_build_messages(self):
        c = self._client()
        msgs = c._build_messages("sys", "ctx", "q", "### Ctx:\n", "### Q:\n")
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        assert "ctx" in msgs[1]["content"] and "q" in msgs[1]["content"]

    def test_complete_non_stream(self):
        c = self._client(stream=False)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        with patch.object(c._client, "post", return_value=mock_resp):
            assert c.complete([{"role": "user", "content": "hi"}], stream=False) == "answer"

    def test_test_connection_success(self):
        c = self._client()
        with patch.object(c._client, "get", return_value=MagicMock(status_code=200)):
            assert c.test_connection() is True

    def test_test_connection_failure(self):
        c = self._client()
        with patch.object(c._client, "get", side_effect=Exception("refused")):
            assert c.test_connection() is False

    def test_test_connection_500(self):
        c = self._client()
        with patch.object(c._client, "get", return_value=MagicMock(status_code=500)):
            assert c.test_connection() is False


class TestEvalMetrics:
    def test_faithfulness_full_overlap(self):
        from rag_tester import _faithfulness_proxy, RetrievedChunk
        chunks = [RetrievedChunk(doc_id="1", chunk_id="1", score=0.9,
                                 text="the answer is forty two and nothing else")]
        assert _faithfulness_proxy("the answer is forty two", chunks) > 0.0

    def test_faithfulness_no_chunks(self):
        from rag_tester import _faithfulness_proxy
        assert _faithfulness_proxy("answer", []) == 0.0

    def test_context_relevance_average_score(self):
        from rag_tester import _context_relevance, RetrievedChunk
        chunks = [RetrievedChunk(doc_id="1", chunk_id="1", score=0.8, text="x"),
                  RetrievedChunk(doc_id="2", chunk_id="2", score=0.6, text="y")]
        assert _context_relevance("q", chunks) == pytest.approx(0.7, abs=0.01)

    def test_context_relevance_empty(self):
        from rag_tester import _context_relevance
        assert _context_relevance("q", []) == 0.0


class TestRAGAnswer:
    def test_to_dict(self):
        from rag_tester import RAGAnswer, RetrievedChunk
        chunk = RetrievedChunk(doc_id="1", chunk_id="c1", score=0.9, text="hi")
        d = RAGAnswer(query="q", answer="a", chunks=[chunk]).to_dict()
        assert d["query"] == "q" and len(d["chunks"]) == 1

    def test_source_label(self):
        from rag_tester import RetrievedChunk
        c = RetrievedChunk(doc_id="p1", chunk_id="p1__0", score=0.9, text="x",
                           metadata={"title": "Page", "space_key": "ENG", "page_id": "1"})
        assert "ENG" in c.source_label and "Page" in c.source_label


class TestRAGTesterCLI:
    def test_print_example_config(self, capsys):
        from rag_tester import main
        assert main(["--print-example-config"]) == 0
        out = capsys.readouterr().out
        assert "embedder" in out

    def test_no_config(self):
        from rag_tester import main
        assert main([]) == 2

    def test_missing_config(self, tmp_path):
        from rag_tester import main
        assert main(["--config", str(tmp_path / "nope.yaml")]) == 1

    def test_run_single_query(self, tmp_path):
        cfg_file = tmp_path / "r.yaml"
        cfg_file.write_text(
            "embedder:\n  backend: stub\nvector_store:\n  backend: qdrant\n"
            "llm:\n  api_url: http://localhost:11434/v1\n  model: mistral\n  stream: false\n"
            "retrieval:\n  top_k: 3\nprompt: {}\nlog_file: " + str(tmp_path / "log.jsonl") + "\n"
        )
        from rag_tester import RetrievedChunk
        chunk = RetrievedChunk(doc_id="1", chunk_id="c1", score=0.9, text="ctx")
        with patch("rag_tester.Retriever") as MockR, patch("rag_tester.LLMClient") as MockL:
            MockR.return_value.retrieve.return_value = ([chunk], 5.0, 3.0)
            MockL.return_value.complete.return_value = "ответ"
            MockL.return_value._build_messages.return_value = [{"role": "user", "content": "x"}]
            MockL.return_value.model = "mistral"
            MockL.return_value.url = "http://x"
            MockL.return_value.stream = False
            from rag_tester import main
            rc = main(["--config", str(cfg_file), "--query", "Как настроить?", "--no-stream"])
        assert rc == 0
