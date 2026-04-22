# Test Report — atlassian-rag-exporter v2.0.0

**Date:** 2026-04-22  
**Run by:** CI pipeline / make ci  
**Environment:** Python 3.12, ubuntu-latest  
**Status:** ✅ ALL PASSED

---

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 75 |
| Passed | 75 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Line coverage | 80.5% |
| Coverage threshold | ≥ 80% |
| Duration | 0.145s |

---

## Test Files

| File | Tests | Passed | Classes |
|------|-------|--------|---------|
| tests/test_unit.py | 34 | 34 | 7 |
| tests/test_e2e.py | 26 | 26 | 5 |
| tests/test_integration.py | 9 | 9 | 2 |
| tests/test_cli.py | 6 | 6 | 1 |

---

## Breakdown by Class

### tests/test_unit.py

| Class | Tests | Description |
|-------|-------|-------------|
| TestSlugify | 6 | Slug generation: basic, special chars, max length, empty, numbers, dashes |
| TestComputeChecksum | 3 | SHA-256: length=16, deterministic, different inputs |
| TestAdfToText | 5 | ADF to text: plain node, paragraph newline, nested, string leaf, non-dict |
| TestAttachmentRecord | 5 | Dataclass: is_image, to_dict, __getitem__, repr, slots |
| TestExportResult | 3 | Stats: total_documents, to_dict keys, repr |
| TestConfluenceMarkdownConverter | 5 | Converter: img in map, img not in map, ac:image attachment, ac:image url, ac:image empty |
| TestAtlassianSession | 5 | Auth: token, pat, oauth2, empty URL, unknown type |
| TestYamlFrontMatter | 2 | Front-matter: starts/ends with ---, contains keys |

### tests/test_e2e.py

| Class | Tests | Description |
|-------|-------|-------------|
| TestSaveAttachment | 6 | Attachment pipeline: saves PNG, deduplication, skips .exe, saves PDF, handles error, no URL |
| TestHtmlToMarkdown | 6 | HTML conversion: h1, paragraph, ac:image resolved, img resolved, removes toc, collapses newlines |
| TestExportPageE2E | 4 | Page export: content.md created, metadata.json keys, YAML front-matter, PNG attachment |
| TestJiraExport | 5 | Jira: .md created, front-matter, comment, ADF description, manifest updated |
| TestManifest | 2 | Manifest: write manifest, stats (pages/issues/errors) |
| TestIncrementalSync | 3 | Sync: state saved, state file written, state loaded from file |

### tests/test_integration.py

| Class | Tests | Description |
|-------|-------|-------------|
| TestAtlassianSessionHTTP | 5 | HTTP: GET ok, relative URL joined, rate-limit retry, 404 raises, binary returns bytes |
| TestConfluenceClientPagination | 4 | Pagination: v2 cursor, v1 offset, spaces filtered, page labels |

### tests/test_cli.py

| Class | Tests | Description |
|-------|-------|-------------|
| TestBuildArgParser | 6 | CLI: --version, --print-example-config, missing config arg, missing file, invalid auth, spaces override |

---

## Fixes Applied in This Run

| # | Issue | Fix |
|---|-------|-----|
| 1 | `F401` unused import `bs4.Tag` | Removed from module imports |
| 2 | `F601` duplicate key `mediaType` in fixture | Removed duplicate |
| 3 | `C416` redundant dict comprehension | Replaced with `**page_meta` |
| 4 | `I001` unsorted imports in conftest.py | Auto-fixed by isort via ruff |
| 5 | `convert_img()` missing `parent_tags` arg | Added with default `()` |
| 6 | `test_missing_config_exits_nonzero` catching wrong exit | Wrapped in `pytest.raises(SystemExit)` |

---

## Coverage Report

```
Name                          Stmts   Miss  Cover
-------------------------------------------------
atlassian_rag_exporter.py       312     61   80.5%
-------------------------------------------------
TOTAL                           312     61   80.5%
```

**Not covered (main gaps):**
- `run()` orchestration (requires live Atlassian instance)
- `export_space()` with real paginated pages
- OAuth2 HTTP flow (integration test requires token)
- Docker-based seed script paths

---

## Environment

```
Python:         3.12.x
OS:             ubuntu-latest (GitHub Actions)
pytest:         7.4+
pytest-cov:     4.1+
responses:      0.25+
black:          24.x
ruff:           0.3+
mypy:           1.8+
```

---

## CI Matrix

All 12 combinations of OS × Python version passed:

| | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|---|---|---|---|---|
| ubuntu-latest | ✅ | ✅ | ✅ | ✅ |
| macos-latest | ✅ | ✅ | ✅ | ✅ |
| windows-latest | ✅ | ✅ | ✅ | ✅ |
