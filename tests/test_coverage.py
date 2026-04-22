"""test_coverage.py — coverage boost: streaming LLM, Retriever, RAGTester, ReindexWorker,
VectorStore adapters, embedder pipeline. Target: 80% → ≥90%.
"""
from __future__ import annotations
import asyncio, json, sys, time, types
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sse_lines(tokens, done=True):
    out = ["data: " + json.dumps({"choices": [{"delta": {"content": t}}]}) for t in tokens]
    if done:
        out.append("data: [DONE]")
    return out

def _stream_ctx(lines):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.iter_lines.return_value = iter(lines)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=r)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx

def _llm():
    from rag_tester import LLMClient
    return LLMClient({"api_url": "http://localhost/v1", "model": "m", "stream": True})

def _patch_vs(mock_adapter):
    import vector_store as vs
    orig = vs.VectorStoreConfig, vs.build_adapter
    vs.VectorStoreConfig = MagicMock(from_dict=MagicMock(return_value=MagicMock()))
    vs.build_adapter = MagicMock(return_value=mock_adapter)
    return orig

def _restore_vs(orig):
    import vector_store as vs
    vs.VectorStoreConfig, vs.build_adapter = orig

def _make_retriever(top_k=3, threshold=0.0, score=0.9):
    from rag_tester import Retriever
    res = MagicMock(doc_id="p1", score=score, text="ctx",
                    metadata={"title": "T", "space_key": "ENG", "page_id": "1"})
    adapter = MagicMock()
    adapter.connect = MagicMock()
    adapter.search.return_value = [res]
    emb = MagicMock()
    emb.embed_one.return_value = [0.1]*384
    orig = _patch_vs(adapter)
    try:
        r = Retriever({"backend": "qdrant"}, emb,
                      {"top_k": top_k, "score_threshold": threshold})
    finally:
        _restore_vs(orig)
    r._adapter = adapter
    return r, emb, adapter

def _make_tester(tmp_path):
    import yaml
    from rag_tester import RAGTester, RetrievedChunk
    cfg_file = tmp_path / "cfg.yaml"
    cfg = {"embedder": {"backend": "stub"}, "vector_store": {"backend": "qdrant"},
           "llm": {"api_url": "http://x", "model": "tm", "stream": False},
           "retrieval": {"top_k": 3}, "log_file": str(tmp_path / "log.jsonl")}
    cfg_file.write_text(yaml.dump(cfg))
    chunk = RetrievedChunk(doc_id="1", chunk_id="c1", score=0.9, text="Контекст SSO",
                           metadata={"title": "SSO Guide", "space_key": "ENG", "page_id": "1"})
    emb = MagicMock()
    emb.embed_one.return_value = [0.1]*384
    ret = MagicMock()
    ret.retrieve.return_value = ([chunk], 5.0, 3.0)
    llm = MagicMock()
    llm.complete.return_value = "Ответ про SSO"
    llm.model = "tm"; llm.stream = False; llm.url = "http://x"
    llm.test_connection.return_value = True
    llm._build_messages.return_value = [{"role": "user", "content": "x"}]
    import vector_store as vs; import embedder as emb_mod
    orig_vs = vs.VectorStoreConfig, vs.build_adapter
    orig_emb = emb_mod.EmbedderConfig, emb_mod.build_embedder
    vs.VectorStoreConfig = MagicMock(from_dict=MagicMock(return_value=MagicMock()))
    vs.build_adapter = MagicMock(return_value=MagicMock())
    emb_mod.EmbedderConfig = MagicMock(from_dict=MagicMock(return_value=MagicMock()))
    emb_mod.build_embedder = MagicMock(return_value=emb)
    try:
        t = RAGTester(str(cfg_file))
    finally:
        vs.VectorStoreConfig, vs.build_adapter = orig_vs
        emb_mod.EmbedderConfig, emb_mod.build_embedder = orig_emb
    t._retriever = ret; t._llm = llm
    return t, chunk


# ─── LLMClient streaming ──────────────────────────────────────────────────────

class TestLLMClientStreaming:
    def test_collects_tokens(self, capsys):
        c = _llm()
        with patch.object(c._client, "stream", return_value=_stream_ctx(_sse_lines(["Hi", "!"]))):
            assert c._stream_complete({}) == "Hi!"
    def test_skips_done(self):
        c = _llm()
        with patch.object(c._client, "stream", return_value=_stream_ctx(["data: [DONE]"])):
            assert c._stream_complete({}) == ""
    def test_skips_bad_json(self):
        c = _llm()
        valid = "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]})
        with patch.object(c._client, "stream", return_value=_stream_ctx(["data: bad", valid])):
            assert c._stream_complete({}) == "ok"
    def test_empty_delta_skipped(self):
        c = _llm()
        with patch.object(c._client, "stream", return_value=_stream_ctx(
                ["data: " + json.dumps({"choices": [{"delta": {}}]})])):
            assert c._stream_complete({}) == ""
    def test_complete_calls_stream(self):
        c = _llm()
        with patch.object(c, "_stream_complete", return_value="s") as m:
            assert c.complete([], stream=True) == "s"
        m.assert_called_once()
    def test_complete_sync(self):
        c = _llm()
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"choices": [{"message": {"content": "sync"}}]}
        with patch.object(c._client, "post", return_value=r):
            assert c.complete([], stream=False) == "sync"
    def test_tokens_printed(self, capsys):
        c = _llm()
        with patch.object(c._client, "stream", return_value=_stream_ctx(_sse_lines(["токен"]))):
            c._stream_complete({})
        assert "токен" in capsys.readouterr().out


# ─── Retriever ────────────────────────────────────────────────────────────────

class TestRetriever:
    def test_returns_chunks(self):
        r, _, _ = _make_retriever()
        chunks, e_ms, r_ms = r.retrieve("запрос")
        assert len(chunks) == 1 and chunks[0].score == 0.9

    def test_threshold_filters(self):
        r, _, _ = _make_retriever(threshold=0.95, score=0.9)
        chunks, _, _ = r.retrieve("q")
        assert chunks == []

    def test_top_k(self):
        from rag_tester import Retriever
        import vector_store as vs
        results = [MagicMock(doc_id=str(i), score=0.9, text=f"t{i}", metadata={})
                   for i in range(10)]
        adapter = MagicMock(); adapter.connect = MagicMock(); adapter.search.return_value = results
        emb = MagicMock(); emb.embed_one.return_value = [0.]*384
        orig = _patch_vs(adapter)
        try:
            r = Retriever({}, emb, {"top_k": 3, "score_threshold": 0.0})
        finally:
            _restore_vs(orig)
        r._adapter = adapter
        chunks, _, _ = r.retrieve("q")
        assert len(chunks) == 3

    def test_embed_called_with_query(self):
        r, emb, _ = _make_retriever()
        r.retrieve("мой запрос")
        emb.embed_one.assert_called_once_with("мой запрос")

    def test_search_called(self):
        r, _, adapter = _make_retriever()
        r.retrieve("q")
        assert adapter.search.called


# ─── RAGTester.ask ────────────────────────────────────────────────────────────

class TestRAGTesterAsk:
    def test_returns_answer(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        res = t.ask("Как SSO?")
        assert res.query == "Как SSO?" and res.answer == "Ответ про SSO"

    def test_logs_jsonl(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        t.ask("лог")
        entry = json.loads((tmp_path/"log.jsonl").read_text().strip())
        assert entry["query"] == "лог"

    def test_no_chunks_not_found(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        t._retriever.retrieve.return_value = ([], 1., 1.)
        assert "не найдена" in t.ask("?").answer.lower()

    def test_latency_recorded(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        res = t.ask("q")
        assert res.latency_embed_ms >= 0 and res.latency_llm_ms >= 0

    def test_multiple_log_entries(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        t.ask("q1"); t.ask("q2")
        assert len((tmp_path/"log.jsonl").read_text().strip().split("\n")) == 2

    def test_model_in_result(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        assert t.ask("q").model == "tm"


# ─── print_result ─────────────────────────────────────────────────────────────

class TestPrintResult:
    def test_shows_query_answer(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        t.print_result(t.ask("Как SSO?"))
        out = capsys.readouterr().out
        assert "Как SSO?" in out and "Ответ про SSO" in out

    def test_shows_sources(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        t.print_result(t.ask("q"))
        assert "ENG" in capsys.readouterr().out or "SSO Guide" in capsys.readouterr().out

    def test_shows_metrics(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        t.print_result(t.ask("q"))
        out = capsys.readouterr().out
        assert "faithfulness" in out and "context_relevance" in out


# ─── run_eval ─────────────────────────────────────────────────────────────────

class TestRunEval:
    def test_missing_file(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        t.run_eval(str(tmp_path/"no.jsonl"))
        assert "not found" in capsys.readouterr().out.lower()

    def test_processes_questions(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        ef = tmp_path/"eval.jsonl"
        ef.write_text(json.dumps({"question":"Q1"})+"\n"+json.dumps({"question":"Q2"})+"\n")
        t.run_eval(str(ef))
        out = capsys.readouterr().out
        assert "[1/2]" in out and "[2/2]" in out

    def test_summary_printed(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        ef = tmp_path/"e.jsonl"; ef.write_text(json.dumps({"question":"Q"})+"\n")
        t.run_eval(str(ef))
        assert "faithfulness" in capsys.readouterr().out

    def test_empty_file_no_crash(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        ef = tmp_path/"empty.jsonl"; ef.write_text("")
        t.run_eval(str(ef))

    def test_per_question_score_line(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        ef = tmp_path/"e.jsonl"; ef.write_text(json.dumps({"question":"Q"})+"\n")
        t.run_eval(str(ef))
        assert "faith=" in capsys.readouterr().out


# ─── repl ────────────────────────────────────────────────────────────────────

class TestRepl:
    def test_exits_on_exit(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        with patch("builtins.input", side_effect=["exit"]): t.repl()

    def test_processes_question(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        with patch("builtins.input", side_effect=["Как SSO?", "exit"]): t.repl()
        assert "SSO" in capsys.readouterr().out or "Ответ" in capsys.readouterr().out

    def test_handles_eof(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        with patch("builtins.input", side_effect=EOFError): t.repl()
        assert "Пока" in capsys.readouterr().out

    def test_skips_empty(self, tmp_path):
        t, _ = _make_tester(tmp_path)
        with patch("builtins.input", side_effect=["", "выход"]): t.repl()
        t._retriever.retrieve.assert_not_called()

    def test_eval_command(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        ef = tmp_path/"e.jsonl"; ef.write_text(json.dumps({"question":"Q"})+"\n")
        with patch("builtins.input", side_effect=[f"eval {ef}", "exit"]): t.repl()
        assert "1/1" in capsys.readouterr().out or "faith" in capsys.readouterr().out.lower()

    def test_shows_llm_status(self, tmp_path, capsys):
        t, _ = _make_tester(tmp_path)
        with patch("builtins.input", side_effect=["exit"]): t.repl()
        assert "доступен" in capsys.readouterr().out


# ─── ReindexWorker async ──────────────────────────────────────────────────────

class TestReindexWorker:
    @pytest.mark.asyncio
    async def test_confluence_event(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        q = asyncio.Queue()
        exp = MagicMock(); exp.export_page_by_id.return_value = {}
        w = ReindexWorker(q, lambda: exp, lambda: MagicMock(), debounce_seconds=0.05)
        await q.put(ReindexEvent("confluence", "page_updated", "p1"))
        async def stop():
            await asyncio.sleep(0.3); w.stop()
        await asyncio.gather(asyncio.wait_for(w.run(), 1.), stop(), return_exceptions=True)
        exp.export_page_by_id.assert_called_with("p1")

    @pytest.mark.asyncio
    async def test_deduplicates(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        q = asyncio.Queue()
        exp = MagicMock(); exp.export_page_by_id.return_value = {}
        w = ReindexWorker(q, lambda: exp, lambda: MagicMock(), debounce_seconds=0.05)
        await q.put(ReindexEvent("confluence", "page_updated", "p1"))
        await q.put(ReindexEvent("confluence", "page_updated", "p1"))
        async def stop():
            await asyncio.sleep(0.3); w.stop()
        await asyncio.gather(asyncio.wait_for(w.run(), 1.), stop(), return_exceptions=True)
        assert exp.export_page_by_id.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_queue_no_crash(self):
        from webhook_listener import ReindexWorker
        w = ReindexWorker(asyncio.Queue(), lambda: MagicMock(), lambda: MagicMock(), debounce_seconds=0.05)
        async def stop():
            await asyncio.sleep(0.15); w.stop()
        await asyncio.gather(asyncio.wait_for(w.run(), 1.), stop(), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_jira_event(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        q = asyncio.Queue()
        exp = MagicMock(); exp.export_jira_issue_by_key.return_value = {}
        w = ReindexWorker(q, lambda: exp, lambda: MagicMock(), debounce_seconds=0.05)
        await q.put(ReindexEvent("jira", "jira:issue_updated", "T-1"))
        async def stop():
            await asyncio.sleep(0.3); w.stop()
        await asyncio.gather(asyncio.wait_for(w.run(), 1.), stop(), return_exceptions=True)
        exp.export_jira_issue_by_key.assert_called_with("T-1")

    @pytest.mark.asyncio
    async def test_trash_calls_delete(self):
        from webhook_listener import ReindexWorker, ReindexEvent
        q = asyncio.Queue()
        exp = MagicMock(); idx = MagicMock()
        w = ReindexWorker(q, lambda: exp, lambda: idx, debounce_seconds=0.05)
        await q.put(ReindexEvent("confluence", "page_trashed", "p99"))
        async def stop():
            await asyncio.sleep(0.3); w.stop()
        await asyncio.gather(asyncio.wait_for(w.run(), 1.), stop(), return_exceptions=True)
        idx.delete.assert_called_with(["p99"])
        exp.export_page_by_id.assert_not_called()


# ─── VectorStore config + adapters ────────────────────────────────────────────

class TestVSConfig:
    def test_qdrant(self):
        from vector_store import VectorStoreConfig
        cfg = VectorStoreConfig.from_dict({"backend":"qdrant","collection":"c"})
        assert cfg.backend == "qdrant" and cfg.collection == "c"

    def test_chroma(self):
        from vector_store import VectorStoreConfig
        assert VectorStoreConfig.from_dict({"backend":"chromadb"}).backend == "chromadb"

    def test_rag_document(self):
        from vector_store import RAGDocument
        d = RAGDocument(doc_id="d", text="t", metadata={}, embedding=[0.1])
        assert d.doc_id == "d"

    def test_search_result(self):
        from vector_store import SearchResult
        assert SearchResult(doc_id="d", text="t", score=0.9, metadata={}).score == 0.9

class TestBuildAdapter:
    def test_qdrant(self):
        from vector_store import VectorStoreConfig, build_adapter, QdrantAdapter
        assert isinstance(build_adapter(VectorStoreConfig.from_dict({"backend":"qdrant"})), QdrantAdapter)

    def test_chroma(self):
        from vector_store import VectorStoreConfig, build_adapter, ChromaAdapter
        assert isinstance(build_adapter(VectorStoreConfig.from_dict({"backend":"chromadb"})), ChromaAdapter)

    def test_unknown(self):
        from vector_store import VectorStoreConfig, build_adapter
        with pytest.raises((ValueError, KeyError)):
            build_adapter(VectorStoreConfig.from_dict({"backend":"unicorn"}))

class TestQdrantMocked:
    def _a(self):
        from vector_store import VectorStoreConfig, QdrantAdapter
        return QdrantAdapter(VectorStoreConfig.from_dict(
            {"backend":"qdrant","host":"localhost","port":6333,"collection":"test","embedding_dim":4}))

    def _ensure_qdrant_stub(self):
        """Inject a fake qdrant_client module if the real one isn't installed."""
        if "qdrant_client" not in sys.modules:
            fake = types.ModuleType("qdrant_client")
            fake.QdrantClient = MagicMock()
            models = types.ModuleType("qdrant_client.models")
            models.Distance = MagicMock()
            models.VectorParams = MagicMock()
            fake.models = models
            sys.modules["qdrant_client"] = fake
            sys.modules["qdrant_client.models"] = models
        return sys.modules["qdrant_client"]

    def test_creates_collection(self):
        qc = self._ensure_qdrant_stub()
        a = self._a()
        mc = MagicMock()
        cr = MagicMock(); cr.collections = []
        mc.get_collections.return_value = cr
        orig = qc.QdrantClient
        qc.QdrantClient = MagicMock(return_value=mc)
        try: a.connect()
        finally: qc.QdrantClient = orig
        mc.create_collection.assert_called_once()

    def test_skips_existing(self):
        qc = self._ensure_qdrant_stub()
        a = self._a()
        mc = MagicMock()
        col = MagicMock(); col.name = "test"
        cr = MagicMock(); cr.collections = [col]
        mc.get_collections.return_value = cr
        orig = qc.QdrantClient
        qc.QdrantClient = MagicMock(return_value=mc)
        try: a.connect()
        finally: qc.QdrantClient = orig
        mc.create_collection.assert_not_called()

    def _full_qdrant_stub(self):
        """Set up a qdrant_client stub with all model classes needed."""
        qc = self._ensure_qdrant_stub()
        qm = sys.modules["qdrant_client.models"]
        for attr in ("PointStruct", "FieldCondition", "Filter", "FilterSelector",
                     "MatchAny", "PointIdsList", "Distance", "VectorParams"):
            if not hasattr(qm, attr) or not isinstance(getattr(qm, attr), MagicMock):
                setattr(qm, attr, MagicMock(side_effect=lambda **kw: kw))
        return qc

    def test_upsert(self):
        self._full_qdrant_stub()
        from vector_store import RAGDocument
        a = self._a(); a._client = MagicMock()
        a.upsert([RAGDocument("1", "t", [.1,.2,.3,.4], {})])
        a._client.upsert.assert_called_once()

    def test_search(self):
        a = self._a(); a._client = MagicMock()
        h = MagicMock(); h.id="1"; h.score=0.9; h.payload={"text":"c","title":"T"}
        a._client.search.return_value = [h]
        assert a.search([.1]*4, k=5)[0].score == 0.9

    def test_delete(self):
        self._full_qdrant_stub()
        a = self._a(); a._client = MagicMock()
        a.delete(["i1"]); a._client.delete.assert_called_once()

class TestChromaMocked:
    def _a(self):
        from vector_store import VectorStoreConfig, ChromaAdapter
        return ChromaAdapter(VectorStoreConfig.from_dict({"backend":"chromadb","collection":"t"}))

    def test_upsert(self):
        from vector_store import RAGDocument
        a = self._a(); a._col = MagicMock()
        a.upsert([RAGDocument("1","t",{},[.1])])
        assert a._col.upsert.called or a._col.add.called

    def test_search(self):
        a = self._a(); a._col = MagicMock()
        a._col.query.return_value = {"ids":[["1","2"]],"documents":[["d1","d2"]],
            "metadatas":[[{},{}]],"distances":[[0.1,0.3]]}
        res = a.search([.1], k=2)
        assert len(res)==2 and res[0].score > res[1].score

    def test_delete(self):
        a = self._a(); a._col = MagicMock()
        a.delete(["x"]); a._col.delete.assert_called_once()


# ─── RAGExporter constructor + ADF ────────────────────────────────────────────

def _exp(tmp_path):
    from atlassian_rag_exporter import RAGExporter
    with patch("atlassian_rag_exporter.AtlassianSession"):
        return RAGExporter({"base_url":"https://t.atlassian.net",
            "auth":{"type":"token","email":"a@b.com","token":"t"},
            "output_dir":str(tmp_path/"corpus"),"spaces":["ENG"]})

class TestExporterConstructor:
    def test_output_dir(self, tmp_path):
        assert "corpus" in str(_exp(tmp_path).output_dir)
    def test_has_pages_dir(self, tmp_path):
        assert hasattr(_exp(tmp_path), "pages_dir")
    def test_has_attachments_dir(self, tmp_path):
        assert hasattr(_exp(tmp_path), "attachments_dir")
    def test_has_jira_dir(self, tmp_path):
        assert hasattr(_exp(tmp_path), "jira_dir")

class TestADF:
    def test_paragraph(self):
        from atlassian_rag_exporter import _adf_to_text
        assert "Привет" in _adf_to_text({"type":"doc","content":[
            {"type":"paragraph","content":[{"type":"text","text":"Привет"}]}]})

    def test_heading(self):
        from atlassian_rag_exporter import _adf_to_text
        assert "Заголовок" in _adf_to_text({"type":"doc","content":[
            {"type":"heading","attrs":{"level":1},"content":[{"type":"text","text":"Заголовок"}]}]})

    def test_bullet_list(self):
        from atlassian_rag_exporter import _adf_to_text
        r = _adf_to_text({"type":"doc","content":[{"type":"bulletList","content":[
            {"type":"listItem","content":[{"type":"paragraph","content":[{"type":"text","text":"п1"}]}]},
            {"type":"listItem","content":[{"type":"paragraph","content":[{"type":"text","text":"п2"}]}]},
        ]}]})
        assert "п1" in r and "п2" in r

    def test_empty(self):
        from atlassian_rag_exporter import _adf_to_text
        assert _adf_to_text({}) == ""

    def test_blockquote(self):
        from atlassian_rag_exporter import _adf_to_text
        r = _adf_to_text({"type":"doc","content":[{"type":"blockquote","content":[
            {"type":"paragraph","content":[{"type":"text","text":"цитата"}]}]}]})
        assert "цитата" in r


# ─── Embedder pipeline ────────────────────────────────────────────────────────

class TestEmbedder:
    def test_stub_dim(self):
        from embedder import StubEmbedder, EmbedderConfig
        e = StubEmbedder(EmbedderConfig(backend="stub"))
        assert len(e.embed(["q"])[0]) == 384

    def test_embed_one(self):
        from embedder import StubEmbedder, EmbedderConfig
        assert len(StubEmbedder(EmbedderConfig(backend="stub")).embed_one("q")) == 384

    def test_chunking_metadata(self):
        from embedder import chunk_document
        cs = chunk_document("p1","w "*500,{"space_key":"RU","author":"Иван"})
        for c in cs:
            assert c.metadata["space_key"]=="RU" and c.metadata["chunk_total"]==len(cs)

    def test_russian_preprocessing(self):
        from embedder import StubEmbedder, EmbedderConfig
        e = StubEmbedder(EmbedderConfig(backend="stub", ru_normalize=True))
        assert len(e.embed(["Текст\u200bс\u200cмусором\ufeff"])[0]) == 384

    def test_e5_prefix(self):
        from embedder import SentenceTransformersEmbedder, EmbedderConfig
        if "sentence_transformers" not in sys.modules:
            fake = types.ModuleType("sentence_transformers")
            fake.SentenceTransformer = MagicMock()
            sys.modules["sentence_transformers"] = fake
        cfg = EmbedderConfig(backend="sentence_transformers", model="intfloat/multilingual-e5-large")
        e = object.__new__(SentenceTransformersEmbedder)
        e.cfg = cfg; e._dim = 5; e._e5_prefix = "query: "
        # encode() returns list of objects that have .tolist() (like numpy arrays)
        vec = MagicMock(); vec.tolist.return_value = [.1]*5
        m = MagicMock(); m.encode.return_value = [vec]; e._model = m
        result = e._embed_batch(["тест"])
        assert m.encode.call_args[0][0][0].startswith("query: ")
        assert result == [[.1]*5]
