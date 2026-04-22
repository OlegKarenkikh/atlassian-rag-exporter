"""test_integration.py — HTTP-layer integration tests (mocked network)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import responses as resp_lib
from requests.exceptions import RequestException

sys.path.insert(0, str(Path(__file__).parent.parent))
import atlassian_rag_exporter as m
from tests.conftest import make_png


class TestAtlassianSessionHTTP:
    def _session(self):
        return m.AtlassianSession(
            "https://test.atlassian.net", "token", email="a@b.com", token="tok"
        )

    @resp_lib.activate
    def test_get_ok(self):
        resp_lib.add(resp_lib.GET, "https://test.atlassian.net/wiki/api/v2/pages",
                     json={"results": [], "_links": {}}, status=200)
        s = self._session()
        data = s.get_json("https://test.atlassian.net/wiki/api/v2/pages")
        assert "results" in data

    @resp_lib.activate
    def test_relative_url_joined(self):
        resp_lib.add(resp_lib.GET, "https://test.atlassian.net/wiki/api/v2/spaces",
                     json={"results": [], "_links": {}}, status=200)
        s = self._session()
        data = s.get_json("/wiki/api/v2/spaces")
        assert "results" in data

    @resp_lib.activate
    def test_rate_limit_retry(self):
        resp_lib.add(resp_lib.GET, "https://test.atlassian.net/test",
                     json={}, status=429, headers={"Retry-After": "1"})
        resp_lib.add(resp_lib.GET, "https://test.atlassian.net/test",
                     json={"ok": True}, status=200)
        s = self._session()
        with patch("time.sleep") as mock_sleep:
            data = s.get_json("https://test.atlassian.net/test")
        mock_sleep.assert_called_once_with(1)
        assert data.get("ok") is True

    @resp_lib.activate
    def test_404_raises(self):
        resp_lib.add(resp_lib.GET, "https://test.atlassian.net/missing",
                     json={"error": "not found"}, status=404)
        s = self._session()
        with pytest.raises(RequestException):
            s.get_json("https://test.atlassian.net/missing")

    @resp_lib.activate
    def test_get_binary_returns_bytes(self):
        png = make_png()
        resp_lib.add(resp_lib.GET, "https://test.atlassian.net/img.png",
                     body=png, status=200, content_type="image/png")
        s = self._session()
        data = s.get_binary("https://test.atlassian.net/img.png")
        assert isinstance(data, bytes)
        assert len(data) > 0


class TestConfluenceClientPagination:
    def _client(self):
        session = MagicMock(spec=m.AtlassianSession)
        session.base_url = "https://test.atlassian.net"
        return m.ConfluenceClient(session, is_cloud=True)

    def test_paginate_v2_cursor(self):
        client = self._client()
        page1 = {"results": [{"id": "1"}, {"id": "2"}],
                 "_links": {"next": "/wiki/api/v2/pages?cursor=abc"}}
        page2 = {"results": [{"id": "3"}], "_links": {}}
        client.session.get_json = MagicMock(side_effect=[page1, page2])
        results = list(client._paginate_v2("/pages"))
        assert len(results) == 3
        assert results[0]["id"] == "1"
        assert results[2]["id"] == "3"

    def test_paginate_v1_offset(self):
        client = self._client()
        page1 = {"results": [{"id": "a"}, {"id": "b"}],
                 "size": 2, "limit": 2, "start": 0, "totalSize": 3}
        page2 = {"results": [{"id": "c"}],
                 "size": 1, "limit": 2, "start": 2, "totalSize": 3}
        client.session.get_json = MagicMock(side_effect=[page1, page2])
        results = list(client._paginate_v1("/content"))
        assert len(results) == 3

    def test_get_spaces_filtered(self):
        client = self._client()
        all_spaces = [{"key": "ENG", "id": "1"}, {"key": "HR", "id": "2"}]
        client.session.get_json = MagicMock(
            return_value={"results": all_spaces, "_links": {}}
        )
        result = client.get_spaces(space_keys=["ENG"])
        assert len(result) == 1
        assert result[0]["key"] == "ENG"

    def test_get_page_labels(self):
        client = self._client()
        client.session.get_json = MagicMock(
            return_value={"results": [{"name": "backend"}, {"name": "api"}]}
        )
        labels = client.get_page_labels("123")
        assert labels == ["backend", "api"]
