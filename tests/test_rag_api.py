"""test_rag_api.py — tests for rag_api REST endpoints."""
from __future__ import annotations
import json, sys, time, types
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_chunk(text="Контекст SSO", score=0.9, space="ENG", title="SSO Guide"):
    from rag_tester import RetrievedChunk
    return RetrievedChunk(doc_id="p1", chunk_id="c1", score=score, text=text,
                          metadata={"title": title, "space_key": space, "page_id": "1"})


def _make_rag_answer(query="q", answer="Ответ", chunks=None):
    from rag_tester import RAGAnswer
    return RAGAnswer(query=query, answer=answer, chunks=chunks or [_make_chunk()],
                     latency_embed_ms=5.0, latency_retrieve_ms=10.0,
                     latency_llm_ms=200.0, model="tm", timestamp=time.time())


def _app(tmp_path, token="", llm_ok=True):
    """Build test app with mocked RAGTester."""
    import yaml, os
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump({
        "embedder": {"backend": "stub"},
        "vector_store": {"backend": "qdrant"},
        "llm": {"api_url": "http://x", "model": "tm", "stream": False},
        "retrieval": {"top_k": 3},
        "log_file": str(tmp_path / "log.jsonl"),
    }))
    os.environ["RAG_CONFIG"] = str(cfg_file)
    if token:
        os.environ["ATLASSIAN_RAG_API_TOKEN"] = token
    else:
        os.environ.pop("ATLASSIAN_RAG_API_TOKEN", None)

    mock_tester = MagicMock()
    mock_tester._llm.test_connection.return_value = llm_ok
    mock_tester._llm.model = "tm"
    mock_tester._llm.url = "http://x"
    mock_tester._llm.temperature = 0.1
    mock_tester._llm.max_tokens = 512
    mock_tester._llm.stream = False
    mock_tester._llm._build_messages.return_value = [{"role": "user", "content": "x"}]
    mock_tester._retriever._top_k = 3
    mock_tester._retriever.retrieve.return_value = ([_make_chunk()], 5.0, 3.0)
    mock_tester._system = "system"
    mock_tester._ctx_header = "ctx"
    mock_tester._q_header = "q"
    mock_tester._log_path = str(tmp_path / "log.jsonl")
    mock_tester.ask.side_effect = lambda query, **kw: _make_rag_answer(query=query)

    import rag_api
    rag_api._tester = mock_tester

    from rag_api import create_app
    app = create_app(str(cfg_file))
    return TestClient(app), mock_tester


# ─── /health ─────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_llm_reachable_true(self, tmp_path):
        client, _ = _app(tmp_path, llm_ok=True)
        assert client.get("/health").json()["llm_reachable"] is True

    def test_llm_reachable_false(self, tmp_path):
        client, _ = _app(tmp_path, llm_ok=False)
        assert client.get("/health").json()["llm_reachable"] is False

    def test_returns_model(self, tmp_path):
        client, _ = _app(tmp_path)
        assert client.get("/health").json()["llm_model"] == "tm"

    def test_uptime_positive(self, tmp_path):
        client, _ = _app(tmp_path)
        assert client.get("/health").json()["uptime_sec"] >= 0


# ─── /ask ────────────────────────────────────────────────────────────────────

class TestAsk:
    def test_returns_answer(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/ask", json={"query": "Как настроить SSO?"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Ответ"
        assert data["query"] == "Как настроить SSO?"

    def test_returns_sources(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/ask", json={"query": "q"})
        sources = r.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["space_key"] == "ENG"

    def test_returns_latency(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/ask", json={"query": "q"})
        lat = r.json()["latency"]
        assert "embed_ms" in lat and "llm_ms" in lat

    def test_returns_metrics(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/ask", json={"query": "q"})
        m = r.json()["metrics"]
        assert "faithfulness" in m and "context_relevance" in m

    def test_empty_query_rejected(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/ask", json={"query": ""})
        assert r.status_code == 422

    def test_missing_query_rejected(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/ask", json={})
        assert r.status_code == 422

    def test_top_k_override(self, tmp_path):
        client, mock_t = _app(tmp_path)
        client.post("/ask", json={"query": "q", "top_k": 7})
        # top_k should have been restored after request
        assert mock_t._retriever._top_k == 3

    def test_no_chunks_returns_not_found(self, tmp_path):
        client, mock_t = _app(tmp_path)
        from rag_tester import RAGAnswer
        no_chunk_answer = RAGAnswer(
            query="q", answer="Информация не найдена в базе знаний.",
            chunks=[], latency_embed_ms=1, latency_retrieve_ms=1,
            latency_llm_ms=1, model="tm", timestamp=time.time()
        )
        mock_t.ask.side_effect = None  # clear side_effect
        mock_t.ask.return_value = no_chunk_answer
        r = client.post("/ask", json={"query": "q"})
        assert r.status_code == 200
        assert "не найдена" in r.json()["answer"]


# ─── /ask auth ───────────────────────────────────────────────────────────────

class TestAskAuth:
    def test_no_token_no_auth_required(self, tmp_path):
        client, _ = _app(tmp_path, token="")
        r = client.post("/ask", json={"query": "q"})
        assert r.status_code == 200

    def test_valid_token_accepted(self, tmp_path):
        client, _ = _app(tmp_path, token="secret123")
        r = client.post("/ask", json={"query": "q"},
                        headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200

    def test_wrong_token_rejected(self, tmp_path):
        client, _ = _app(tmp_path, token="secret123")
        r = client.post("/ask", json={"query": "q"},
                        headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_missing_token_rejected(self, tmp_path):
        client, _ = _app(tmp_path, token="secret123")
        r = client.post("/ask", json={"query": "q"})
        assert r.status_code == 401


# ─── /eval ────────────────────────────────────────────────────────────────────

class TestEval:
    def test_inline_questions(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/eval", json={"questions": [
            {"question": "Q1"}, {"question": "Q2"}
        ]})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_eval_file(self, tmp_path):
        client, _ = _app(tmp_path)
        ef = tmp_path / "eval.jsonl"
        ef.write_text(json.dumps({"question": "Q1"}) + "\n")
        r = client.post("/eval", json={"eval_file": str(ef)})
        assert r.status_code == 200 and r.json()["total"] == 1

    def test_missing_eval_file_404(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/eval", json={"eval_file": "/no/such/file.jsonl"})
        assert r.status_code == 404

    def test_empty_payload_400(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/eval", json={})
        assert r.status_code == 400

    def test_returns_avg_metrics(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/eval", json={"questions": [{"question": "Q1"}]})
        d = r.json()
        assert "avg_faithfulness" in d and "avg_context_relevance" in d

    def test_duration_positive(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.post("/eval", json={"questions": [{"question": "Q1"}]})
        assert r.json()["duration_sec"] >= 0


# ─── /metrics ────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_empty_log(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.json()["total_queries"] == 0

    def test_counts_entries(self, tmp_path):
        client, mock_t = _app(tmp_path)
        log = tmp_path / "log.jsonl"
        log.write_text(
            json.dumps({"query": "q1", "faithfulness": 0.8, "context_relevance": 0.7,
                        "latency_llm_ms": 300, "latency_retrieve_ms": 20}) + "\n" +
            json.dumps({"query": "q2", "faithfulness": 0.9, "context_relevance": 0.6,
                        "latency_llm_ms": 250, "latency_retrieve_ms": 15}) + "\n"
        )
        r = client.get("/metrics")
        d = r.json()
        assert d["total_queries"] == 2
        assert abs(d["avg_faithfulness"] - 0.85) < 0.01

    def test_missing_log_returns_zeros(self, tmp_path):
        client, _ = _app(tmp_path)
        r = client.get("/metrics")
        assert r.json()["avg_faithfulness"] == 0


# ─── /log DELETE ─────────────────────────────────────────────────────────────

class TestClearLog:
    def test_clears_log(self, tmp_path):
        client, _ = _app(tmp_path)
        log = tmp_path / "log.jsonl"
        log.write_text('{"q":"test"}\n')
        r = client.delete("/log")
        assert r.status_code == 200
        assert r.json()["cleared"] is True
        assert log.read_text() == ""


# ─── /ask/stream (SSE) ───────────────────────────────────────────────────────

class TestAskStream:
    def test_stream_returns_sse(self, tmp_path):
        client, mock_t = _app(tmp_path)
        # Mock LLM stream
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Ответ"}}]}),
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_t._llm._client.stream.return_value = mock_ctx

        r = client.post("/ask/stream", json={"query": "Как SSO?"})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_contains_sources_event(self, tmp_path):
        client, mock_t = _app(tmp_path)
        sse_lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "A"}}]}),
                     "data: [DONE]"]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_t._llm._client.stream.return_value = mock_ctx

        r = client.post("/ask/stream", json={"query": "q"})
        body = r.text
        # First SSE event should be sources
        first_data = [l for l in body.split("\n") if l.startswith("data: ")][0]
        payload = json.loads(first_data[6:])
        assert payload["event"] == "sources"

    def test_stream_no_chunks_done_immediately(self, tmp_path):
        client, mock_t = _app(tmp_path)
        mock_t._retriever.retrieve.return_value = ([], 1.0, 1.0)
        r = client.post("/ask/stream", json={"query": "q"})
        body = r.text
        events = [json.loads(l[6:]) for l in body.split("\n") if l.startswith("data: ")]
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        assert "не найдена" in done_events[0]["answer"]
