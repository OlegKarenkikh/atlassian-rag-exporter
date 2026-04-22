"""test_e2e.py — End-to-end tests for RAGExporter pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import atlassian_rag_exporter as m
from tests.conftest import make_png, make_pdf


class TestSaveAttachment:
    def test_saves_png(self, exporter, tmp_out):
        att = {"title": "arch.png", "mediaType": "image/png",
               "_links": {"download": "/wiki/download/arch.png"}}
        exporter.confluence.get_attachment_download_url = MagicMock(
            return_value="https://test.atlassian.net/wiki/download/arch.png")
        exporter.confluence.session.get_binary = MagicMock(return_value=make_png())
        rec = exporter._save_attachment(att)
        assert rec is not None
        assert rec.is_image
        assert Path(tmp_out / rec.local_path).exists()

    def test_deduplication(self, exporter):
        att = {"title": "logo.png", "mediaType": "image/png"}
        png_data = make_png(2, 2, (10, 20, 30))
        exporter.confluence.get_attachment_download_url = MagicMock(
            return_value="https://test.atlassian.net/logo.png")
        exporter.confluence.session.get_binary = MagicMock(return_value=png_data)
        r1 = exporter._save_attachment(att)
        r2 = exporter._save_attachment(att)
        assert r1 is not None and r2 is not None
        assert r1.local_path == r2.local_path

    def test_skips_unsupported_extension(self, exporter):
        att = {"title": "virus.exe", "mediaType": "application/octet-stream"}
        exporter.confluence.get_attachment_download_url = MagicMock(return_value="https://x.com/v.exe")
        rec = exporter._save_attachment(att)
        assert rec is None

    def test_saves_pdf(self, exporter, tmp_out):
        att = {"title": "report.pdf", "mediaType": "application/pdf"}
        exporter.confluence.get_attachment_download_url = MagicMock(
            return_value="https://test.atlassian.net/report.pdf")
        exporter.confluence.session.get_binary = MagicMock(return_value=make_pdf())
        rec = exporter._save_attachment(att)
        assert rec is not None
        assert rec.media_type == "document"

    def test_handles_download_error(self, exporter):
        att = {"title": "broken.png"}
        exporter.confluence.get_attachment_download_url = MagicMock(return_value="https://x.com/b.png")
        exporter.confluence.session.get_binary = MagicMock(side_effect=ConnectionError("timeout"))
        rec = exporter._save_attachment(att)
        assert rec is None

    def test_no_download_url(self, exporter):
        att = {"title": "ghost.png"}
        exporter.confluence.get_attachment_download_url = MagicMock(return_value="")
        rec = exporter._save_attachment(att)
        assert rec is None


class TestHtmlToMarkdown:
    def test_basic_h1(self, exporter):
        md = exporter._html_to_markdown("<h1>Title</h1>", {})
        assert "# Title" in md

    def test_paragraph(self, exporter):
        md = exporter._html_to_markdown("<p>Hello world</p>", {})
        assert "Hello world" in md

    def test_ac_image_resolved(self, exporter):
        html = '<ac:image><ri:attachment ri:filename="flow.png"/></ac:image>'
        md = exporter._html_to_markdown(html, {"flow.png": "attachments/abc_flow.png"})
        assert "attachments/abc_flow.png" in md

    def test_img_tag_resolved(self, exporter):
        html = '<img src="/wiki/download/diag.png" data-linked-resource-default-alias="diag.png">'
        md = exporter._html_to_markdown(html, {"diag.png": "attachments/xyz_diag.png"})
        assert "attachments/xyz_diag.png" in md

    def test_removes_toc(self, exporter):
        html = '<div class="toc"><h2>Contents</h2></div><p>Real content</p>'
        md = exporter._html_to_markdown(html, {})
        assert "Contents" not in md
        assert "Real content" in md

    def test_collapses_extra_newlines(self, exporter):
        html = "<p>A</p>" * 5
        md = exporter._html_to_markdown(html, {})
        assert "\n\n\n" not in md


class TestExportPageE2E:
    def _mock_page(self):
        return {"id": "100", "title": "Architecture Guide",
                "_links": {"webui": "/spaces/ENG/pages/100"}}

    def _mock_body(self):
        return (
            "<h1>Arch</h1>",
            "<h1>Architecture</h1><p>Key decisions here.</p>",
            {
                "version": {"number": 3, "when": "2024-06-01", "by": {"displayName": "Alice"}},
                "history": {"createdDate": "2024-01-01"},
                "ancestors": [{"title": "Home"}, {"title": "ENG"}],
            },
        )

    def test_creates_content_md(self, exporter, tmp_out):
        exporter.confluence.get_page_body = MagicMock(return_value=self._mock_body())
        exporter.confluence.get_attachments = MagicMock(return_value=[])
        exporter.confluence.get_page_labels = MagicMock(return_value=["arch", "backend"])
        exporter.confluence.get_attachment_download_url = MagicMock(return_value="")
        exporter._export_page(self._mock_page(), "ENG", "Engineering")
        md_files = list(tmp_out.rglob("content.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text()
        assert "Architecture Guide" in content or "Architecture" in content

    def test_metadata_json_has_required_keys(self, exporter, tmp_out):
        exporter.confluence.get_page_body = MagicMock(return_value=self._mock_body())
        exporter.confluence.get_attachments = MagicMock(return_value=[])
        exporter.confluence.get_page_labels = MagicMock(return_value=[])
        exporter.confluence.get_attachment_download_url = MagicMock(return_value="")
        exporter._export_page(self._mock_page(), "ENG", "Engineering")
        json_files = list(tmp_out.rglob("metadata.json"))
        assert json_files
        meta = json.loads(json_files[0].read_text())
        for key in ("page_id", "title", "space_key", "word_count", "char_count"):
            assert key in meta, f"Missing key: {key}"

    def test_yaml_front_matter_present(self, exporter, tmp_out):
        exporter.confluence.get_page_body = MagicMock(return_value=self._mock_body())
        exporter.confluence.get_attachments = MagicMock(return_value=[])
        exporter.confluence.get_page_labels = MagicMock(return_value=[])
        exporter.confluence.get_attachment_download_url = MagicMock(return_value="")
        exporter._export_page(self._mock_page(), "ENG", "Engineering")
        md_file = list(tmp_out.rglob("content.md"))[0]
        content = md_file.read_text()
        assert content.startswith("---\n")

    def test_with_png_attachment(self, exporter, tmp_out):
        exporter.confluence.get_page_body = MagicMock(return_value=self._mock_body())
        exporter.confluence.get_attachments = MagicMock(return_value=[
            {"title": "arch.png", "mediaType": "image/png"}
        ])
        exporter.confluence.get_page_labels = MagicMock(return_value=[])
        exporter.confluence.get_attachment_download_url = MagicMock(
            return_value="https://test.atlassian.net/wiki/dl/arch.png")
        exporter.confluence.session.get_binary = MagicMock(return_value=make_png())
        exporter._export_page(self._mock_page(), "ENG", "Engineering")
        md_file = list(tmp_out.rglob("content.md"))[0]
        assert "Attached Images" in md_file.read_text()


class TestJiraExport:
    def _make_issue(self, key="ENG-1"):
        return {
            "key": key,
            "fields": {
                "summary": "Fix the login bug",
                "description": {"type": "doc", "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Details here"}]}
                ]},
                "status": {"name": "In Progress"},
                "priority": {"name": "High"},
                "assignee": {"displayName": "Bob"},
                "reporter": {"displayName": "Alice"},
                "issuetype": {"name": "Bug"},
                "project": {"key": "ENG", "name": "Engineering"},
                "labels": ["login", "auth"],
                "created": "2024-01-15",
                "updated": "2024-06-01",
                "comment": {"comments": [
                    {"author": {"displayName": "Carol"}, "created": "2024-02-01",
                     "body": "Reproduced on prod"}
                ]},
            },
        }

    def test_creates_jira_md(self, exporter, tmp_out):
        exporter._export_jira_issue(self._make_issue())
        md_file = tmp_out / "jira" / "ENG-1.md"
        assert md_file.exists()

    def test_jira_front_matter(self, exporter, tmp_out):
        exporter._export_jira_issue(self._make_issue())
        content = (tmp_out / "jira" / "ENG-1.md").read_text()
        assert "source: jira" in content
        assert "issue_key: ENG-1" in content

    def test_jira_contains_comment(self, exporter, tmp_out):
        exporter._export_jira_issue(self._make_issue())
        content = (tmp_out / "jira" / "ENG-1.md").read_text()
        assert "Carol" in content or "Reproduced" in content

    def test_jira_adf_description(self, exporter, tmp_out):
        exporter._export_jira_issue(self._make_issue())
        content = (tmp_out / "jira" / "ENG-1.md").read_text()
        assert "Details here" in content

    def test_manifest_updated(self, exporter, tmp_out):
        exporter._export_jira_issue(self._make_issue())
        assert any(e["type"] == "jira_issue" for e in exporter._manifest)


class TestManifest:
    def test_write_manifest(self, exporter, tmp_out):
        exporter._manifest = [
            {"type": "confluence_page", "page_id": "1", "title": "T", "space_key": "ENG",
             "md_path": "pages/1_t/content.md", "json_path": "pages/1_t/metadata.json"},
        ]
        exporter._result.confluence_pages = 1
        exporter.write_manifest()
        manifest_path = tmp_out / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == "2.0"
        assert manifest["total_documents"] == 1
        assert len(manifest["documents"]) == 1

    def test_manifest_stats(self, exporter, tmp_out):
        exporter._result.confluence_pages = 5
        exporter._result.jira_issues = 3
        exporter._result.errors = 1
        exporter.write_manifest()
        m_data = json.loads((tmp_out / "manifest.json").read_text())
        assert m_data["confluence_pages"] == 5
        assert m_data["jira_issues"] == 3
        assert m_data["errors"] == 1


class TestIncrementalSync:
    def test_state_saved_after_export_space(self, exporter, tmp_out):
        exporter.confluence.get_pages_in_space = MagicMock(return_value=iter([]))
        exporter.export_space({"id": "10", "key": "ENG", "name": "Engineering"})
        assert "ENG" in exporter._state

    def test_state_file_written(self, exporter, tmp_out):
        exporter.confluence.get_pages_in_space = MagicMock(return_value=iter([]))
        exporter.export_space({"id": "10", "key": "ENG", "name": "Engineering"})
        exporter._save_state()
        assert (tmp_out / ".sync_state.json").exists()

    def test_state_loaded_from_file(self, tmp_out, base_config):
        state_data = {"ENG": "2024-01-01T00:00:00+00:00"}
        (tmp_out / ".sync_state.json").write_text(json.dumps(state_data))
        exp = m.RAGExporter.__new__(m.RAGExporter)
        exp.output_dir = tmp_out
        exp._state = {}
        exp._load_state()
        assert exp._state.get("ENG") == "2024-01-01T00:00:00+00:00"
