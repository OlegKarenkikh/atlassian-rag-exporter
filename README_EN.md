# atlassian-rag-exporter

[![CI](https://github.com/OlegKarenkikh/atlassian-rag-exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegKarenkikh/atlassian-rag-exporter/actions)
[![PyPI](https://img.shields.io/pypi/v/atlassian-rag-exporter)](https://pypi.org/project/atlassian-rag-exporter/)
[![Python](https://img.shields.io/pypi/pyversions/atlassian-rag-exporter)](https://pypi.org/project/atlassian-rag-exporter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-80.5%25-brightgreen)](docs/TEST_REPORT.md)

---

> **Documentation metadata**
> - **Version:** 2.0.0
> - **Status:** Stable — Production Ready
> - **Language:** English
> - **Translated from:** Russian (original)
> - **Translation date:** 2026-04-22
> - **Original date:** April 22, 2026

---

Production-ready pipeline for exporting **Confluence** and **Jira** content into a
structured RAG (Retrieval-Augmented Generation) corpus compatible with LlamaIndex,
LangChain, and any vector database.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Authentication](#authentication)
6. [Configuration Reference](#configuration-reference)
7. [Output Structure](#output-structure)
8. [Using with RAG Frameworks](#using-with-rag-frameworks)
9. [Development & Testing](#development--testing)
10. [CI / CD](#ci--cd)
11. [Architecture](#architecture)
12. [Public API](#public-api)
13. [License](#license)

---

## Features

| Feature | Detail |
|---------|--------|
| **Auth modes** | API Token (Cloud), Personal Access Token (Server/DC), OAuth 2.0 |
| **Confluence REST v2** | Cursor-based pagination — up to 30x faster than v1 offset |
| **Full content** | HTML to Markdown with YAML front-matter per page |
| **Images & attachments** | PNG, JPEG, GIF, WebP, SVG, PDF, DOCX, XLSX — SHA-256 deduplicated |
| **Confluence macros** | `<ac:image>`, `<ri:attachment>` resolved to local paths |
| **Jira** | Issues exported via JQL; ADF to Markdown |
| **Incremental sync** | Only exports pages modified since last run |
| **Rate-limit safe** | Exponential backoff + Retry-After header (tenacity) |
| **RAG-ready output** | manifest.json v2.0, per-page metadata.json, global dedup |

---

## Requirements

- Python 3.9 or later
- Atlassian Cloud, Server, or Data Center instance
- One of: API Token, Personal Access Token, or OAuth 2.0 Bearer token

### Runtime dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | >= 2.31 | HTTP client |
| `markdownify` | >= 0.12 | HTML to Markdown |
| `beautifulsoup4` | >= 4.12 | HTML/XML parsing |
| `PyYAML` | >= 6.0 | YAML generation |
| `tenacity` | >= 8.2 | Retry / backoff |
| `tqdm` | >= 4.66 | Progress bars |

---

## Installation

**From PyPI (stable):**

```bash
pip install atlassian-rag-exporter
```

**Development mode:**

```bash
git clone https://github.com/OlegKarenkikh/atlassian-rag-exporter
cd atlassian-rag-exporter
pip install -e ".[dev]"
```

---

## Quick Start

### Step 1 — Generate example config

```bash
atlassian-rag-exporter --print-example-config > config.yaml
```

### Step 2 — Edit config.yaml

```yaml
base_url: https://your-org.atlassian.net
is_cloud: true
auth:
  type: token
  email: you@company.com
  token: YOUR_API_TOKEN

output_dir: ./rag_corpus
spaces:
  - ENG
  - DOCS
incremental: true
```

### Step 3 — Run

```bash
atlassian-rag-exporter --config config.yaml
```

**CLI flags:**

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config (required) |
| `--spaces KEY...` | Override space keys from config |
| `--output-dir DIR` | Override output directory |
| `--incremental` | Force incremental mode |
| `--verbose` | Enable DEBUG logging |
| `--version` | Print version and exit |
| `--print-example-config` | Print annotated example config |

---

## Authentication

### Option 1 — API Token (Cloud, recommended)

1. Go to: https://id.atlassian.com/manage-profile/security/api-tokens
2. Create a new token and copy it.

```yaml
auth:
  type: token
  email: you@company.com
  token: YOUR_API_TOKEN
```

Sent as HTTP Basic Auth `email:token`.

### Option 2 — Personal Access Token (Server / Data Center)

```yaml
auth:
  type: pat
  token: YOUR_PAT
```

Sent as `Authorization: Bearer <token>`.

### Option 3 — OAuth 2.0

```yaml
auth:
  type: oauth2
  access_token: YOUR_BEARER_TOKEN
```

---

## Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | — | Atlassian instance URL |
| `is_cloud` | bool | true | true=Cloud, false=Server/DC |
| `auth.type` | string | — | token, pat, or oauth2 |
| `auth.email` | string | — | Email (token mode only) |
| `auth.token` | string | — | API Token or PAT |
| `auth.access_token` | string | — | OAuth 2.0 Bearer token |
| `output_dir` | string | — | Output corpus directory |
| `spaces` | list | all | Space keys to export |
| `incremental` | bool | true | Skip unchanged pages |
| `jira.jql` | string | — | JQL for issue export |

---

## Output Structure

```
rag_corpus/
├── manifest.json              <- Global index (schema v2.0)
├── .sync_state.json           <- Incremental sync timestamps
├── pages/
│   └── <id>_<slug>/
│       ├── content.md         <- Markdown + YAML front-matter
│       └── metadata.json      <- Metadata for vector DB filtering
├── attachments/
│   └── <8hex>_<slug>.<ext>   <- SHA-256 deduplicated files
└── jira/
    └── <ISSUE-KEY>.md         <- Jira issues
```

---

## Using with RAG Frameworks

### LlamaIndex

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader(
    input_dir='./rag_corpus/pages',
    recursive=True,
    required_exts=['.md'],
).load_data()

index = VectorStoreIndex.from_documents(docs)
response = index.as_query_engine().query('What is our deployment process?')
print(response)
```

### LangChain + Chroma

```python
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

loader = DirectoryLoader('./rag_corpus', glob='**/*.md',
                         loader_cls=UnstructuredMarkdownLoader)
docs = loader.load()
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

---

## Development & Testing

| Command | Description |
|---------|-------------|
| `make install` | Install with dev dependencies |
| `make test` | Fast test run |
| `make test-cov` | Tests + HTML coverage report |
| `make lint` | ruff linting |
| `make fmt` | black auto-format |
| `make typecheck` | mypy static analysis |
| `make ci` | Full CI pipeline |

### Test results (v2.0.0, 2026-04-22)

| File | Tests | Passed |
|------|-------|--------|
| tests/test_unit.py | 34 | 34 |
| tests/test_e2e.py | 26 | 26 |
| tests/test_integration.py | 9 | 9 |
| tests/test_cli.py | 6 | 6 |
| **Total** | **75** | **75 (100%)** |

Line coverage: **80.5%** (threshold >= 80%) ✅

Full report: [docs/TEST_REPORT.md](docs/TEST_REPORT.md)

---

## CI / CD

GitHub Actions matrix: Ubuntu / macOS / Windows x Python 3.9–3.12 (12 combinations).
Steps: black check → ruff → mypy → pytest --cov (>= 80%).
Release: automatic PyPI publish on `v*` tags via OIDC trusted publishing.

---

## Architecture

```
atlassian_rag_exporter.py
├── AtlassianSession          HTTP: auth + retry + rate-limit
├── ConfluenceClient          REST API v1/v2 (cursor + offset pagination)
├── JiraClient                REST API v3 (JQL, ADF)
├── ConfluenceMarkdownConverter  HTML to Markdown (Confluence storage format)
├── AttachmentRecord          Immutable attachment metadata
├── ExportResult              Aggregate stats
└── RAGExporter               Orchestrator
    ├── _save_attachment()    Download + SHA-256 dedup
    ├── _html_to_markdown()   Convert + resolve image paths
    ├── export_space()        Iterate space pages
    ├── export_jira()         JQL search + write .md
    ├── write_manifest()      manifest.json schema v2.0
    └── run()                 Full pipeline entry point
```

---

## Public API

```python
from atlassian_rag_exporter import RAGExporter, ExportResult

config = {
    'base_url': 'https://your-org.atlassian.net',
    'is_cloud': True,
    'auth': {'type': 'token', 'email': 'you@company.com', 'token': 'TOKEN'},
    'output_dir': './rag_corpus',
    'spaces': ['ENG'],
    'incremental': True,
}
exporter = RAGExporter(config)
result: ExportResult = exporter.run()
print(result.total_documents)   # pages + issues
print(result.errors)            # failed items
```

---

## License

MIT © 2024 OlegKarenkikh. See [LICENSE](LICENSE).
