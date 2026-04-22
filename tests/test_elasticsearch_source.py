"""Tests for elasticsearch_source module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch_source import (
    ElasticsearchImporter,
    ESDocumentConverter,
    ESFieldMapping,
    ESScrollIterator,
    ESSession,
    ESSourceConfig,
)


class TestESSourceConfig:
    def test_defaults(self):
        cfg = ESSourceConfig()
        assert cfg.hosts == ["http://localhost:9200"]
        assert cfg.index == "*"
        assert cfg.auth_type == "none"
        assert cfg.size == 500

    def test_from_dict(self):
        cfg = ESSourceConfig.from_dict(
            {
                "hosts": ["https://es:9200"],
                "index": "wiki-*",
                "auth_type": "api_key",
                "api_key": "key123",
                "fields": {"title": "headline", "body": "content"},
            }
        )
        assert cfg.hosts == ["https://es:9200"]
        assert cfg.index == "wiki-*"
        assert cfg.fields.title == "headline"
        assert cfg.fields.body == "content"

    def test_field_mapping_defaults(self):
        fm = ESFieldMapping()
        assert fm.title == "title"
        assert fm.body_html is None
        assert fm.attachment_urls == []


class TestESSession:
    def test_api_key_auth(self):
        cfg = ESSourceConfig(auth_type="api_key", api_key="encoded==")
        sess = ESSession(cfg)
        assert sess.session.headers.get("Authorization") == "ApiKey encoded=="

    def test_bearer_auth(self):
        cfg = ESSourceConfig(auth_type="bearer", bearer_token="tok123")
        sess = ESSession(cfg)
        assert sess.session.headers.get("Authorization") == "Bearer tok123"

    def test_basic_auth(self):
        cfg = ESSourceConfig(auth_type="basic", username="elastic", password="pw")
        sess = ESSession(cfg)
        assert sess.session.auth == ("elastic", "pw")

    def test_no_auth(self):
        cfg = ESSourceConfig(auth_type="none")
        sess = ESSession(cfg)
        assert "Authorization" not in sess.session.headers

    def test_rate_limit_retry(self):
        cfg = ESSourceConfig()
        sess = ESSession(cfg)
        call_count = {"n": 0}

        def fake_post(url, json, timeout):
            call_count["n"] += 1
            if call_count["n"] == 1:
                r = MagicMock()
                r.status_code = 429
                r.headers = {"Retry-After": "0"}
                r.raise_for_status = MagicMock()
                return r
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"hits": {"hits": []}}
            r.raise_for_status = MagicMock()
            return r

        sess.session.post = fake_post
        with patch("time.sleep"):
            sess.post("/_search", {})
        assert call_count["n"] == 2

    def test_download_success(self, tmp_path):
        cfg = ESSourceConfig(attachment_timeout=5)
        sess = ESSession(cfg)
        dest = tmp_path / "file.bin"
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content.return_value = [b"HELLO", b"WORLD"]
        sess.session.get = MagicMock(return_value=mock_resp)
        result = sess.download("https://example.com/file.bin", dest)
        assert result is True
        assert dest.read_bytes() == b"HELLOWORLD"

    def test_download_failure(self, tmp_path):
        cfg = ESSourceConfig()
        sess = ESSession(cfg)
        sess.session.get = MagicMock(side_effect=Exception("timeout"))
        result = sess.download("https://example.com/fail.bin", tmp_path / "fail.bin")
        assert result is False


class TestESDocumentConverter:
    def test_slugify(self):
        assert ESDocumentConverter._slugify("Hello World! 123") == "hello-world-123"

    def test_slugify_long(self):
        long_text = "a" * 100
        assert len(ESDocumentConverter._slugify(long_text)) <= 60

    def test_get_nested(self):
        cfg = ESSourceConfig(output_dir="/tmp/es_test_get")
        conv = ESDocumentConverter(cfg)
        assert conv._get({"a": {"b": "val"}}, "a.b") == "val"

    def test_get_missing(self):
        cfg = ESSourceConfig(output_dir="/tmp/es_test_get2")
        conv = ESDocumentConverter(cfg)
        assert conv._get({}, "missing.key", "default") == "default"

    def test_get_none_returns_default(self):
        cfg = ESSourceConfig(output_dir="/tmp/es_test_get3")
        conv = ESDocumentConverter(cfg)
        assert conv._get({"k": None}, "k", "fallback") == "fallback"

    def test_save_creates_files(self, tmp_path):
        cfg = ESSourceConfig(output_dir=str(tmp_path / "out"))
        conv = ESDocumentConverter(cfg)
        hit = {
            "_id": "doc1",
            "_index": "wiki",
            "_source": {
                "title": "My Doc",
                "body": "Hello RAG world",
                "author": "Alice",
                "url": "https://example.com/doc1",
                "created_at": "2024-01-01",
                "updated_at": "2024-06-01",
            },
        }
        path = conv.save(hit, MagicMock())
        assert path is not None and path.exists()
        content = path.read_text()
        assert "My Doc" in content
        assert "Hello RAG world" in content
        meta = json.loads((path.parent / "metadata.json").read_text())
        assert meta["doc_id"] == "doc1"
        assert meta["title"] == "My Doc"

    def test_save_empty_source(self, tmp_path):
        cfg = ESSourceConfig(output_dir=str(tmp_path / "out"))
        conv = ESDocumentConverter(cfg)
        hit = {"_id": "empty", "_source": {}}
        path = conv.save(hit, MagicMock())
        assert path is not None and path.exists()
        meta = json.loads((path.parent / "metadata.json").read_text())
        assert meta["doc_id"] == "empty"
        assert meta["title"] == ""

    def test_save_html_body(self, tmp_path):
        cfg = ESSourceConfig(
            output_dir=str(tmp_path / "out"),
            fields=ESFieldMapping(body_html="body_html"),
        )
        conv = ESDocumentConverter(cfg)
        hit = {
            "_id": "doc2",
            "_source": {"title": "HTML Doc", "body_html": "<p>Rich <b>content</b></p>"},
        }
        path = conv.save(hit, MagicMock())
        assert "Rich" in path.read_text()

    def test_html_to_markdown_fallback(self, tmp_path):
        cfg = ESSourceConfig(output_dir=str(tmp_path / "out"))
        conv = ESDocumentConverter(cfg)
        with patch.dict("sys.modules", {"markdownify": None}):
            result = conv._html_to_markdown("<p>Hello <b>world</b></p>")
        assert "Hello" in result

    def test_save_dedup_attachments(self, tmp_path):
        cfg = ESSourceConfig(
            output_dir=str(tmp_path / "out"),
            download_attachments=True,
            fields=ESFieldMapping(attachment_urls=["attach_url"]),
        )
        conv = ESDocumentConverter(cfg)

        def fake_download(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"IMG")
            return True

        mock_session = MagicMock()
        mock_session.download.side_effect = fake_download
        mock_session.cfg = cfg

        hit1 = {"_id": "d1", "_source": {"title": "A", "attach_url": "https://ex.com/img.png"}}
        hit2 = {"_id": "d2", "_source": {"title": "B", "attach_url": "https://ex.com/img.png"}}
        conv.save(hit1, mock_session)
        conv.save(hit2, mock_session)
        assert mock_session.download.call_count == 1

    def test_save_no_attachments_when_disabled(self, tmp_path):
        cfg = ESSourceConfig(
            output_dir=str(tmp_path / "out"),
            download_attachments=False,
            fields=ESFieldMapping(attachment_urls=["attach_url"]),
        )
        conv = ESDocumentConverter(cfg)
        mock_session = MagicMock()
        hit = {"_id": "d1", "_source": {"title": "T", "attach_url": "https://ex.com/f.png"}}
        conv.save(hit, mock_session)
        mock_session.download.assert_not_called()


class TestESScrollIterator:
    def _make_session(self, pages):
        cfg = ESSourceConfig()
        sess = ESSession.__new__(ESSession)
        sess.cfg = cfg
        sess.session = MagicMock()
        sess.host = "http://localhost:9200"
        pages_iter = iter(pages)

        def fake_post(path, body):
            try:
                hits = next(pages_iter)
            except StopIteration:
                hits = []
            return {
                "_scroll_id": "scroll123",
                "hits": {"total": {"value": 10, "relation": "eq"}, "hits": hits},
            }

        sess.post = fake_post
        sess.delete = MagicMock()
        return sess

    def test_scroll_yields_all(self):
        sess = self._make_session([[{"_id": "1"}, {"_id": "2"}], [{"_id": "3"}], []])
        it = ESScrollIterator(sess, ESSourceConfig(size=2), {"match_all": {}})
        results = list(it.scroll())
        assert len(results) == 3

    def test_scroll_cleanup(self):
        sess = self._make_session([[{"_id": "1"}], []])
        it = ESScrollIterator(sess, ESSourceConfig(), {"match_all": {}})
        list(it.scroll())
        assert sess.delete.called

    def test_iter_uses_scroll_by_default(self):
        sess = self._make_session([[{"_id": "x"}], []])
        it = ESScrollIterator(sess, ESSourceConfig(use_pit=False), {"match_all": {}})
        results = list(it)
        assert len(results) == 1


class TestElasticsearchImporter:
    def test_run_saves_documents(self, tmp_path):
        cfg = ESSourceConfig(output_dir=str(tmp_path / "out"), incremental=False)
        importer = ElasticsearchImporter.__new__(ElasticsearchImporter)
        importer.cfg = cfg
        importer.output_dir = tmp_path / "out"
        importer.output_dir.mkdir(parents=True)
        importer._state_file = importer.output_dir / ".es_sync_state.json"
        importer._state = {}
        importer.session = MagicMock()
        importer.converter = ESDocumentConverter(cfg)

        hits = [
            {
                "_id": f"doc{i}",
                "_index": "wiki",
                "_source": {"title": f"Doc {i}", "body": f"Body {i}"},
            }
            for i in range(5)
        ]
        with patch.object(ESScrollIterator, "__iter__", return_value=iter(hits)):
            result = importer.run()
        assert result["saved"] == 5
        assert result["errors"] == 0
        manifest = json.loads((importer.output_dir / "manifest.json").read_text())
        assert manifest["total_saved"] == 5

    def test_run_counts_errors(self, tmp_path):
        cfg = ESSourceConfig(output_dir=str(tmp_path / "out"), incremental=False)
        importer = ElasticsearchImporter.__new__(ElasticsearchImporter)
        importer.cfg = cfg
        importer.output_dir = tmp_path / "out"
        importer.output_dir.mkdir(parents=True)
        importer._state_file = importer.output_dir / ".es_sync_state.json"
        importer._state = {}
        importer.session = MagicMock()
        conv = MagicMock()
        conv.save.side_effect = Exception("disk full")
        importer.converter = conv

        hits = [{"_id": "bad1"}, {"_id": "bad2"}]
        with patch.object(ESScrollIterator, "__iter__", return_value=iter(hits)):
            result = importer.run()
        assert result["errors"] == 2
        assert result["saved"] == 0

    def test_incremental_adds_filter(self, tmp_path):
        cfg = ESSourceConfig(output_dir=str(tmp_path / "out"), incremental=True)
        importer = ElasticsearchImporter.__new__(ElasticsearchImporter)
        importer.cfg = cfg
        importer._state = {"last_sync": "2024-01-01T00:00:00+00:00"}
        query = importer._build_query()
        assert "bool" in query
        assert "filter" in query["bool"]

    def test_no_incremental_returns_base_query(self, tmp_path):
        cfg = ESSourceConfig(
            output_dir=str(tmp_path / "out"),
            incremental=False,
            query={"match_all": {}},
        )
        importer = ElasticsearchImporter.__new__(ElasticsearchImporter)
        importer.cfg = cfg
        importer._state = {}
        assert importer._build_query() == {"match_all": {}}

    def test_load_state_missing_file(self, tmp_path):
        importer = ElasticsearchImporter.__new__(ElasticsearchImporter)
        importer._state_file = tmp_path / "nonexistent.json"
        assert importer._load_state() == {}

    def test_incremental_no_cutoff_returns_base(self, tmp_path):
        cfg = ESSourceConfig(
            output_dir=str(tmp_path / "out"),
            incremental=True,
            query={"term": {"status": "published"}},
        )
        importer = ElasticsearchImporter.__new__(ElasticsearchImporter)
        importer.cfg = cfg
        importer._state = {}
        query = importer._build_query()
        assert query == {"term": {"status": "published"}}
