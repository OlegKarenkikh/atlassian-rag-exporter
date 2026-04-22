"""test_unit.py — Unit tests for pure functions and data classes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
import atlassian_rag_exporter as m


class TestSlugify:
    def test_basic(self):
        assert m.slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        result = m.slugify("hello: world & test!")
        assert result == "hello-world-test"

    def test_max_length(self):
        long_str = "a" * 200
        assert len(m.slugify(long_str)) <= 80

    def test_empty(self):
        assert m.slugify("") == ""

    def test_numbers(self):
        assert m.slugify("page 42") == "page-42"

    def test_leading_trailing_dashes(self):
        result = m.slugify("---hello---")
        assert not result.startswith("-")
        assert not result.endswith("-")


class TestComputeChecksum:
    def test_length_16(self):
        assert len(m.compute_checksum(b"hello")) == 16

    def test_deterministic(self):
        assert m.compute_checksum(b"data") == m.compute_checksum(b"data")

    def test_different_inputs(self):
        assert m.compute_checksum(b"a") != m.compute_checksum(b"b")


class TestAdfToText:
    def test_plain_text_node(self):
        node = {"type": "text", "text": "Hello"}
        assert m._adf_to_text(node) == "Hello"

    def test_paragraph_adds_newline(self):
        node = {"type": "paragraph", "content": [{"type": "text", "text": "Hi"}]}
        result = m._adf_to_text(node)
        assert "Hi" in result
        assert result.endswith("\n")

    def test_nested(self):
        node = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "A"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "B"}]},
            ],
        }
        result = m._adf_to_text(node)
        assert "A" in result and "B" in result

    def test_string_leaf(self):
        assert m._adf_to_text("plain") == "plain"

    def test_non_dict_non_str(self):
        assert m._adf_to_text(42) == ""


class TestAttachmentRecord:
    def _make(self, media_type="image"):
        return m.AttachmentRecord(
            filename="arch.png",
            local_path="attachments/abc_arch.png",
            checksum="abc123",
            media_type=media_type,
            mime_type="image/png",
            size_bytes=1024,
            created="2024-01-01",
        )

    def test_is_image_true(self):
        assert self._make("image").is_image

    def test_is_image_false(self):
        assert not self._make("document").is_image

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        for key in ("filename", "local_path", "checksum", "media_type", "size_bytes"):
            assert key in d

    def test_getitem(self):
        rec = self._make()
        assert rec["filename"] == "arch.png"

    def test_repr_slots(self):
        rec = self._make()
        assert "arch.png" in repr(rec)


class TestExportResult:
    def test_total_documents(self):
        r = m.ExportResult(confluence_pages=10, jira_issues=5)
        assert r.total_documents == 15

    def test_to_dict_keys(self):
        d = m.ExportResult().to_dict()
        for key in ("confluence_pages", "jira_issues", "errors", "total_documents"):
            assert key in d

    def test_repr(self):
        r = m.ExportResult(confluence_pages=3)
        assert "3" in repr(r)


class TestConfluenceMarkdownConverter:
    def _make_conv(self, att_map=None):
        return m.ConfluenceMarkdownConverter(attachment_map=att_map or {}, heading_style="ATX")

    def test_img_in_map(self):
        html = '<img src="/wiki/dl/arch.png" data-linked-resource-default-alias="arch.png" alt="diagram">'
        soup = BeautifulSoup(html, "html.parser")
        conv = self._make_conv({"arch.png": "attachments/abc_arch.png"})
        result = conv.convert_img(soup.find("img"), "", parent_tags=())
        assert "attachments/abc_arch.png" in result

    def test_img_not_in_map(self):
        html = '<img src="/wiki/dl/x.png" alt="pic">'
        soup = BeautifulSoup(html, "html.parser")
        conv = self._make_conv({})
        result = conv.convert_img(soup.find("img"), "", parent_tags=())
        assert "x.png" in result or "pic" in result or result == ""

    def test_ac_image_with_attachment(self):
        html = '<ac:image><ri:attachment ri:filename="flow.png"/></ac:image>'
        soup = BeautifulSoup(html, "html.parser")
        conv = self._make_conv({"flow.png": "attachments/xyz_flow.png"})
        tag = soup.find("ac:image")
        result = conv.convert_ac_image(tag, "", parent_tags=())
        assert "attachments/xyz_flow.png" in result

    def test_ac_image_with_url(self):
        html = '<ac:image><ri:url ri:value="https://cdn.example.com/img.png"/></ac:image>'
        soup = BeautifulSoup(html, "html.parser")
        conv = self._make_conv({})
        tag = soup.find("ac:image")
        result = conv.convert_ac_image(tag, "", parent_tags=())
        assert "https://cdn.example.com/img.png" in result

    def test_ac_image_no_child(self):
        html = "<ac:image></ac:image>"
        soup = BeautifulSoup(html, "html.parser")
        conv = self._make_conv({})
        result = conv.convert_ac_image(soup.find("ac:image"), "", parent_tags=())
        assert result == ""


class TestAtlassianSession:
    def test_token_auth_sets_basic(self):
        s = m.AtlassianSession("https://x.atlassian.net", "token", email="a@b.com", token="tok")
        assert s.session.auth == ("a@b.com", "tok")

    def test_pat_sets_bearer(self):
        s = m.AtlassianSession("https://x.atlassian.net", "pat", token="pattoken")
        assert "Bearer pattoken" in s.session.headers.get("Authorization", "")

    def test_oauth2_sets_bearer(self):
        s = m.AtlassianSession("https://x.atlassian.net", "oauth2", access_token="oauth_tok")
        assert "Bearer oauth_tok" in s.session.headers.get("Authorization", "")

    def test_empty_base_url_raises(self):
        with pytest.raises((ValueError, KeyError)):
            m.AtlassianSession("", "token", email="a@b.com", token="t")

    def test_unknown_auth_raises(self):
        with pytest.raises(ValueError):
            m.AtlassianSession("https://x.atlassian.net", "magic", token="t")


class TestYamlFrontMatter:
    def test_starts_ends_with_dashes(self, exporter):
        fm = exporter._yaml_front_matter({"title": "Test", "page_id": "1"})
        assert fm.startswith("---\n")
        assert "---\n" in fm[4:]

    def test_contains_keys(self, exporter):
        fm = exporter._yaml_front_matter({"title": "My Page", "space_key": "ENG"})
        assert "title" in fm
        assert "My Page" in fm
