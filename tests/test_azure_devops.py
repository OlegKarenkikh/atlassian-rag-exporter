"""Tests for azure_devops_source.py - all external I/O mocked."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from azure_devops_source import (
    AzureDevOpsClient, AzureDevOpsConfig, ExtractedTable, ImageCaption,
    MultimodalCaptioner, RepoMarkdownExporter, WikiExporter, WorkItemExporter,
    _build_session, _fix_image_links, _html_to_text, _inject_table_refs, _slugify,
    extract_tables_from_html, extract_tables_from_markdown, run_azure_export, save_tables,
)


class TestConfig:
    def test_defaults(self):
        cfg = AzureDevOpsConfig()
        assert cfg.auth_type == "pat"
        assert cfg.import_wikis is True
        assert cfg.caption_backend == "ollama"
        assert cfg.extract_tables is True

    def test_from_dict(self):
        cfg = AzureDevOpsConfig.from_dict({
            "org_url": "https://dev.azure.com/Acme",
            "project": "Alpha", "pat": "tok", "import_repos": True,
        })
        assert cfg.org_url == "https://dev.azure.com/Acme"
        assert cfg.import_repos is True

    def test_from_yaml(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text("azure_devops:\n  org_url: https://dev.azure.com/X\n  project: P\n  pat: t\n")
        cfg = AzureDevOpsConfig.from_yaml(str(yml))
        assert cfg.project == "P"

    def test_from_yaml_missing_section(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text("other: value\n")
        cfg = AzureDevOpsConfig.from_yaml(str(yml))
        assert cfg.org_url == ""

    def test_unknown_fields_ignored(self):
        cfg = AzureDevOpsConfig.from_dict({"org_url": "x", "unknown_key": 99})
        assert cfg.org_url == "x"


class TestBuildSession:
    def test_pat_sets_basic_auth(self):
        cfg = AzureDevOpsConfig(auth_type="pat", pat="mytoken")
        session = _build_session(cfg)
        expected = "Basic " + base64.b64encode(b":mytoken").decode()
        assert session.headers["Authorization"] == expected

    def test_entra_client_credentials(self):
        cfg = AzureDevOpsConfig(
            auth_type="entra_client_credentials",
            tenant_id="tid", client_id="cid", client_secret="sec"
        )
        with patch("azure_devops_source.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: {"access_token": "mytoken123"}
            )
            mock_post.return_value.raise_for_status = MagicMock()
            session = _build_session(cfg)
        assert session.headers["Authorization"] == "Bearer mytoken123"

    def test_managed_identity(self):
        cfg = AzureDevOpsConfig(auth_type="managed_identity")
        with patch("azure_devops_source.requests.get") as mock_get:
            mock_get.return_value = MagicMock(json=lambda: {"access_token": "mi-token"})
            mock_get.return_value.raise_for_status = MagicMock()
            session = _build_session(cfg)
        assert session.headers["Authorization"] == "Bearer mi-token"

    def test_unknown_auth_raises(self):
        cfg = AzureDevOpsConfig(auth_type="telekinesis")
        with pytest.raises(ValueError, match="Unknown auth_type"):
            _build_session(cfg)


class TestAzureDevOpsClient:
    def _make(self, **kw):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", **kw)
        client = AzureDevOpsClient(cfg)
        client.session = MagicMock()
        return client

    def _mock_get(self, client, data):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = data
        resp.raise_for_status = MagicMock()
        client.session.get.return_value = resp
        return resp

    def test_base_url(self):
        client = self._make()
        assert "Alpha" in client._base("wiki/wikis")

    def test_list_wikis(self):
        client = self._make()
        self._mock_get(client, {"value": [{"id": "w1", "name": "MyWiki"}]})
        result = client.list_wikis()
        assert result[0]["name"] == "MyWiki"

    def test_list_wiki_pages_empty(self):
        client = self._make()
        self._mock_get(client, {"path": "/Home"})
        pages = client.list_wiki_pages("wiki1")
        assert isinstance(pages, list)
        assert len(pages) >= 1

    def test_rate_limit_retry(self):
        client = self._make(requests_per_second=1000)
        resp_429 = MagicMock(status_code=429, headers={"Retry-After": "0"})
        resp_429.raise_for_status = MagicMock()
        resp_ok = MagicMock(status_code=200, json=lambda: {"value": []})
        resp_ok.raise_for_status = MagicMock()
        client.session.get.side_effect = [resp_429, resp_ok]
        with patch("azure_devops_source.time.sleep"):
            result = client.get("https://example.com/test")
        assert result == {"value": []}
        assert client.session.get.call_count == 2

    def test_query_work_items(self):
        client = self._make()
        resp = MagicMock(status_code=200)
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"workItems": [{"id": 1}, {"id": 2}]}
        client.session.post.return_value = resp
        ids = client.query_work_items("SELECT [System.Id] FROM WorkItems")
        assert ids == [1, 2]

    def test_get_work_items_batch_empty(self):
        client = self._make()
        result = client.get_work_items_batch([])
        assert result == []

    def test_list_repos(self):
        client = self._make()
        self._mock_get(client, {"value": [{"id": "r1", "name": "Repo1"}]})
        repos = client.list_repos()
        assert repos[0]["name"] == "Repo1"

    def test_get_raw(self):
        client = self._make()
        resp = MagicMock(status_code=200, content=b"binary data")
        resp.raise_for_status = MagicMock()
        client.session.get.return_value = resp
        data = client.get_raw("https://example.com/file")
        assert data == b"binary data"

    def test_flatten_pages_recursive(self):
        client = self._make()
        node = {"path": "/Home", "subPages": [
            {"path": "/Home/Sub1", "subPages": []},
            {"path": "/Home/Sub2", "subPages": [
                {"path": "/Home/Sub2/Deep", "subPages": []}
            ]},
        ]}
        pages = client._flatten_pages(node)
        assert len(pages) == 4
        assert "/Home/Sub2/Deep" in [p["path"] for p in pages]


class TestExtractTablesFromHTML:
    def test_basic_table(self):
        html = """<table><thead><tr><th>Name</th><th>Score</th></tr></thead>
                  <tbody><tr><td>Alice</td><td>90</td></tr>
                  <tr><td>Bob</td><td>85</td></tr><tr><td>Carol</td><td>92</td></tr></tbody></table>"""
        tables = extract_tables_from_html(html, min_rows=2)
        assert len(tables) == 1
        assert tables[0].headers == ["Name", "Score"]
        assert len(tables[0].rows) == 3

    def test_table_with_caption(self):
        html = "<table><caption>Results</caption><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
        tables = extract_tables_from_html(html, min_rows=2)
        assert tables[0].caption == "Results"

    def test_multiple_tables(self):
        html = (
            "<table><tr><th>X</th></tr><tr><td>1</td></tr><tr><td>2</td></tr></table>"
            "<table><tr><th>Y</th></tr><tr><td>a</td></tr><tr><td>b</td></tr></table>"
        )
        assert len(extract_tables_from_html(html, min_rows=2)) == 2

    def test_min_rows_filter(self):
        html = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        assert len(extract_tables_from_html(html, min_rows=3)) == 0

    def test_no_explicit_headers(self):
        html = "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
        tables = extract_tables_from_html(html, min_rows=2)
        assert len(tables) == 1
        assert tables[0].headers[0].startswith("col_")

    def test_empty_table(self):
        assert len(extract_tables_from_html("<table></table>")) == 0


class TestExtractTablesFromMarkdown:
    def test_simple_pipe_table(self):
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        tables = extract_tables_from_markdown(md, min_rows=2)
        assert len(tables) == 1
        assert tables[0].headers == ["Name", "Age"]
        assert len(tables[0].rows) == 2

    def test_table_with_surrounding_text(self):
        md = "Some text\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nMore text"
        assert len(extract_tables_from_markdown(md, min_rows=2)) == 1

    def test_no_table(self):
        assert len(extract_tables_from_markdown("# Just a heading\nText.")) == 0

    def test_multiple_tables(self):
        md = ("| X | Y |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n"
              "text\n\n| P | Q |\n|---|---|\n| a | b |\n| c | d |")
        assert len(extract_tables_from_markdown(md, min_rows=2)) == 2

    def test_single_row_filtered(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert len(extract_tables_from_markdown(md, min_rows=2)) == 0


class TestExtractedTable:
    def _t(self):
        return ExtractedTable(0, "md", ["Name", "Score", "Grade"],
                              [["Alice", "90", "A"], ["Bob", "75", "B+"]])

    def test_to_markdown(self):
        md = self._t().to_markdown()
        assert "| Name | Score | Grade |" in md
        assert "| Alice | 90 | A |" in md

    def test_to_csv(self):
        s = self._t().to_csv_string()
        assert "Name,Score,Grade" in s
        assert "Alice" in s

    def test_to_dict(self):
        d = self._t().to_dict()
        assert d["headers"] == ["Name", "Score", "Grade"]
        assert len(d["rows"]) == 2

    def test_to_markdown_empty_headers(self):
        assert ExtractedTable(0, "html", [], []).to_markdown() == ""


class TestSaveTables:
    def test_saves_json_and_csv(self, tmp_path):
        tables = [ExtractedTable(0, "md", ["A", "B"], [["1", "2"], ["3", "4"]])]
        meta = save_tables(tables, tmp_path / "tables")
        assert (tmp_path / "tables" / "table_00.json").exists()
        assert (tmp_path / "tables" / "table_00.csv").exists()
        assert meta[0]["row_count"] == 2

    def test_saves_multiple(self, tmp_path):
        tables = [ExtractedTable(0, "md", ["X"], [["1"], ["2"]]),
                  ExtractedTable(1, "html", ["Y"], [["a"], ["b"]])]
        meta = save_tables(tables, tmp_path / "t")
        assert len(meta) == 2
        assert (tmp_path / "t" / "table_01.json").exists()


class TestHelpers:
    def test_slugify_basic(self):
        assert _slugify("Hello World!") == "hello-world"

    def test_slugify_empty(self):
        assert _slugify("") == "page"

    def test_slugify_max_len(self):
        assert len(_slugify("a" * 200, max_len=20)) <= 20

    def test_html_to_text_basic(self):
        assert "Hello" in _html_to_text("<p>Hello <b>world</b></p>")

    def test_html_to_text_none(self):
        assert _html_to_text(None) == ""
        assert _html_to_text("") == ""

    def test_inject_table_refs_empty(self):
        md = "# Doc\nText"
        assert _inject_table_refs(md, []) == md

    def test_inject_table_refs(self):
        tables_meta = [{"index": 0, "row_count": 3, "caption": "Sales",
                        "json_path": "tables/table_00.json", "csv_path": "tables/table_00.csv",
                        "markdown_preview": "| A | B |\n|---|---|\n| 1 | 2 |"}]
        result = _inject_table_refs("# Doc", tables_meta)
        assert "Extracted Tables" in result
        assert "table_00.json" in result


class TestFixImageLinks:
    def test_replaces_attachment_link(self, tmp_path):
        md = "See this: ![diagram](.attachments/diagram.png)"
        mock_client = MagicMock()
        mock_client.get_wiki_attachment.return_value = b"\x89PNG\r\n"
        result = _fix_image_links(md, tmp_path / "images", "wiki1", mock_client)
        assert "images/diagram.png" in result
        assert (tmp_path / "images" / "diagram.png").exists()

    def test_no_images(self, tmp_path):
        md = "# No images here"
        result = _fix_image_links(md, tmp_path / "images", "wiki1", MagicMock())
        assert result == md

    def test_download_failure_keeps_original(self, tmp_path):
        md = "![img](.attachments/missing.png)"
        mock_client = MagicMock()
        mock_client.get_wiki_attachment.side_effect = Exception("404")
        result = _fix_image_links(md, tmp_path / "images", "wiki1", mock_client)
        assert ".attachments/missing.png" in result


class TestMultimodalCaptioner:
    def _cfg(self, backend="ollama"):
        return AzureDevOpsConfig(
            image_captioning=True, caption_backend=backend,
            caption_model="qwen2.5vl:latest", caption_endpoint="http://localhost:11434",
        )

    def test_ollama_backend(self):
        cfg = self._cfg("ollama")
        captioner = MultimodalCaptioner(cfg)
        with patch("azure_devops_source.requests.post") as mock_post:
            mock_post.return_value = MagicMock(json=lambda: {"response": "A bar chart showing sales data."})
            mock_post.return_value.raise_for_status = MagicMock()
            cap = captioner.caption(b"\x89PNG\r\nFAKE", "chart.png")
        assert "bar chart" in cap.description
        assert cap.model == "qwen2.5vl:latest"
        assert cap.filename == "chart.png"

    def test_openai_compatible_backend(self):
        cfg = self._cfg("openai_compatible")
        captioner = MultimodalCaptioner(cfg)
        with patch("azure_devops_source.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                json=lambda: {"choices": [{"message": {"content": "A diagram with boxes."}}]}
            )
            mock_post.return_value.raise_for_status = MagicMock()
            cap = captioner.caption(b"FAKE", "arch.png")
        assert "diagram" in cap.description

    def test_structured_parse_chart(self):
        c = MultimodalCaptioner(self._cfg())
        result = c._parse_structured('A bar chart showing "Revenue Q1 2024" values: 100k, 200k, 300k')
        assert result["type"] == "chart"
        assert "Revenue Q1 2024" in result["text_in_image"]
        assert any("100" in v for v in result["data_values"])

    def test_structured_parse_diagram(self):
        c = MultimodalCaptioner(self._cfg())
        result = c._parse_structured("A flowchart diagram showing the deployment pipeline")
        assert result["type"] in ("diagram", "flowchart", "chart")

    def test_unknown_backend_raises(self):
        cfg = self._cfg("alien_backend")
        cap = MultimodalCaptioner(cfg)
        with pytest.raises(ValueError, match="Unknown caption backend"):
            cap._call_backend(b"data")


class TestWikiExporter:
    def _make(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", output_dir=str(tmp_path),
                                extract_tables=True, image_captioning=False)
        client = MagicMock()
        client.list_wikis.return_value = [{"id": "w1", "name": "MyWiki"}]
        client.list_wiki_pages.return_value = [{"path": "/Home"}]
        client.get_wiki_page_by_path.return_value = {
            "path": "/Home",
            "content": "# Welcome\n\nHello world\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
            "lastVersion": {"pushedDate": "2024-01-01", "version": "1"},
        }
        return WikiExporter(cfg, client, None, {}), client

    def test_run_saves_content(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        result = exporter.run()
        assert result["saved"] == 1
        assert result["errors"] == 0
        assert len(list(tmp_path.rglob("content.md"))) == 1

    def test_front_matter_written(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        exporter.run()
        content = list(tmp_path.rglob("content.md"))[0].read_text()
        assert "azure_devops_wiki" in content
        assert "MyWiki" in content

    def test_tables_extracted(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        exporter.run()
        assert len(list(tmp_path.rglob("table_*.json"))) >= 1

    def test_incremental_skip(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", output_dir=str(tmp_path), incremental=True)
        client = MagicMock()
        client.list_wikis.return_value = [{"id": "w1", "name": "Wiki"}]
        client.list_wiki_pages.return_value = [{"path": "/Page"}]
        client.get_wiki_page_by_path.return_value = {
            "path": "/Page", "content": "text",
            "lastVersion": {"pushedDate": "2024-01-01", "version": "1"},
        }
        exporter = WikiExporter(cfg, client, None, {"wiki:w1:/Page": "2024-01-01"})
        result = exporter.run()
        assert result["skipped"] == 1
        assert result["saved"] == 0

    def test_metadata_json_written(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        exporter.run()
        meta_files = list(tmp_path.rglob("metadata.json"))
        assert len(meta_files) >= 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["source"] == "azure_devops_wiki"


class TestWorkItemExporter:
    def _make(self, tmp_path, desc_html="<p>Description</p>"):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", output_dir=str(tmp_path),
                                extract_tables=True, work_item_types=["Story", "Bug"])
        client = MagicMock()
        client.query_work_items.return_value = [42]
        client.get_work_items_batch.return_value = [{"id": 42, "fields": {
            "System.Title": "Fix login bug", "System.WorkItemType": "Bug",
            "System.State": "Active", "System.AssignedTo": {"displayName": "Alice"},
            "System.CreatedBy": {"displayName": "Bob"},
            "System.CreatedDate": "2024-01-01", "System.ChangedDate": "2024-06-01",
            "System.Tags": "auth; security", "System.AreaPath": "Alpha\\Backend",
            "System.IterationPath": "Alpha\\Sprint 1",
            "System.Description": desc_html,
            "Microsoft.VSTS.Common.Priority": 2,
        }}]
        client.get_work_item_comments.return_value = [
            {"createdBy": {"displayName": "Carol"}, "createdDate": "2024-06-02", "text": "<p>Check logs</p>"},
        ]
        return WorkItemExporter(cfg, client, {}), client

    def test_run_saves_md(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        result = exporter.run()
        assert result["saved"] == 1
        assert (tmp_path / _slugify("Alpha") / "work_items" / "42.md").exists()

    def test_md_contains_title_and_state(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        exporter.run()
        content = (tmp_path / _slugify("Alpha") / "work_items" / "42.md").read_text()
        assert "Fix login bug" in content
        assert "Active" in content

    def test_tags_in_front_matter(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        exporter.run()
        content = (tmp_path / _slugify("Alpha") / "work_items" / "42.md").read_text()
        assert "auth" in content

    def test_tables_from_html_description(self, tmp_path):
        desc_html = "<table><tr><th>X</th><th>Y</th></tr><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
        exporter, _ = self._make(tmp_path, desc_html)
        exporter.run()
        assert len(list(tmp_path.rglob("table_*.json"))) >= 1

    def test_incremental_skip(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", output_dir=str(tmp_path), incremental=True)
        client = MagicMock()
        client.query_work_items.return_value = [1]
        client.get_work_items_batch.return_value = [{"id": 1, "fields": {
            "System.Title": "T", "System.WorkItemType": "Task", "System.ChangedDate": "2024-01-01",
        }}]
        exporter = WorkItemExporter(cfg, client, {"wi:1": "2024-01-01"})
        result = exporter.run()
        assert result["skipped"] == 1

    def test_comments_in_output(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        exporter.run()
        content = (tmp_path / _slugify("Alpha") / "work_items" / "42.md").read_text()
        assert "Check logs" in content


class TestRepoMarkdownExporter:
    def _make(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", output_dir=str(tmp_path),
                                import_repos=True, extract_tables=True)
        client = MagicMock()
        client.list_repos.return_value = [{"id": "r1", "name": "MyRepo"}]
        client.list_repo_items.return_value = [
            {"path": "/README.md", "gitObjectType": "blob"},
            {"path": "/src/main.py", "gitObjectType": "blob"},
        ]
        client.get_repo_item.return_value = b"# README\n\nWelcome!\n\n| X | Y |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        return RepoMarkdownExporter(cfg, client, None, {}), client

    def test_run_saves_md_only(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        result = exporter.run()
        assert result["saved"] == 1

    def test_tables_extracted_from_repo(self, tmp_path):
        exporter, _ = self._make(tmp_path)
        exporter.run()
        assert len(list(tmp_path.rglob("table_*.json"))) >= 1

    def test_filter_by_repo_name(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", output_dir=str(tmp_path),
                                import_repos=True, repo_names=["OtherRepo"])
        client = MagicMock()
        client.list_repos.return_value = [{"id": "r1", "name": "MyRepo"}]
        exporter = RepoMarkdownExporter(cfg, client, None, {})
        assert exporter.run()["saved"] == 0

    def test_list_repos_error_continues(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Alpha",
                                auth_type="pat", pat="tok", output_dir=str(tmp_path), import_repos=True)
        client = MagicMock()
        client.list_repos.return_value = [{"id": "r1", "name": "R1"}, {"id": "r2", "name": "R2"}]
        client.list_repo_items.side_effect = [
            Exception("403"), [{"path": "/README.md", "gitObjectType": "blob"}]
        ]
        client.get_repo_item.return_value = b"# Hello"
        exporter = RepoMarkdownExporter(cfg, client, None, {})
        assert exporter.run()["saved"] == 1


class TestRunAzureExport:
    def test_calls_all_exporters(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Proj",
                                auth_type="pat", pat="t", output_dir=str(tmp_path),
                                import_wikis=True, import_work_items=True, import_repos=True)
        with patch("azure_devops_source.AzureDevOpsClient"), \
             patch("azure_devops_source.WikiExporter") as MW, \
             patch("azure_devops_source.WorkItemExporter") as MWI, \
             patch("azure_devops_source.RepoMarkdownExporter") as MR:
            MW.return_value.run.return_value = {"saved": 5, "skipped": 0, "errors": 0}
            MWI.return_value.run.return_value = {"saved": 10, "skipped": 0, "errors": 0}
            MR.return_value.run.return_value = {"saved": 3, "errors": 0}
            result = run_azure_export(cfg)
        assert result["wikis"]["saved"] == 5
        assert result["work_items"]["saved"] == 10
        assert result["repos"]["saved"] == 3

    def test_skips_disabled_sources(self, tmp_path):
        cfg = AzureDevOpsConfig(org_url="https://dev.azure.com/Acme", project="Proj",
                                auth_type="pat", pat="t", output_dir=str(tmp_path),
                                import_wikis=False, import_work_items=False, import_repos=False)
        with patch("azure_devops_source.AzureDevOpsClient"):
            result = run_azure_export(cfg)
        assert "wikis" not in result
        assert "work_items" not in result
