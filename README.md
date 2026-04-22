# atlassian-rag-exporter

Production-ready Python script for exporting **Atlassian Confluence** pages and **Jira** issues into a structured corpus ready for Retrieval-Augmented Generation (RAG) pipelines.

## Features

| Capability | Detail |
|---|---|
| Auth modes | API Token (Cloud), PAT (Server/DC), OAuth 2.0 Bearer |
| API version | Confluence REST API **v2** (cursor pagination) with v1 fallback |
| Full content | storage + view format → clean Markdown + YAML front-matter |
| Images | All inline `<ac:image>` and attached images downloaded, deduplicated, local refs resolved |
| All attachments | PDF, DOCX, XLSX, PPTX, SVG, PNG, JPG, GIF… |
| Jira export | Issues + comments via JQL (ADF → plain text) |
| Incremental sync | Only re-exports pages modified since last run |
| Rate-limit safe | Exponential backoff + `Retry-After` header handling |
| RAG-ready output | YAML front-matter Markdown + JSON sidecar + global `manifest.json` |

## Installation

```bash
git clone https://github.com/OlegKarenkikh/atlassian-rag-exporter.git
cd atlassian-rag-exporter
pip install -r requirements.txt
```

## Authentication

### Cloud — API Token
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click **Create API token**, copy the value
3. Set `auth.type: token`, `auth.email`, `auth.token` in your config

### Server / Data Center — PAT
1. In Confluence, go to **Profile → Personal Access Tokens**
2. Create a token, copy the value
3. Set `auth.type: pat`, `auth.token` in your config

## Quick Start

```bash
cp config.yaml.example config.yaml
# Edit config.yaml: set base_url, email, token, spaces

python atlassian_rag_exporter.py --config config.yaml
```

Other options:

```bash
# Only specific spaces
python atlassian_rag_exporter.py --config config.yaml --spaces ENG DOCS

# Verbose logging
python atlassian_rag_exporter.py --config config.yaml --verbose

# Override output directory
python atlassian_rag_exporter.py --config config.yaml --output-dir ./my_corpus
```

## Output Structure

```
rag_corpus/
├── manifest.json              ← Global index of all exported documents
├── .sync_state.json           ← Incremental sync state per space
├── pages/
│   └── <page_id>_<slug>/
│       ├── content.md         ← Markdown + YAML front-matter (primary RAG input)
│       └── metadata.json      ← Structured metadata for vector DB
├── attachments/
│   └── <sha256>_<filename>   ← Deduplicated attachments (images, PDFs…)
└── jira/
    └── <ISSUE-KEY>.md        ← Jira issue (if enabled)
```

### Example `content.md`

```markdown
---
source: confluence
page_id: "123456"
space_key: ENG
title: API Design Guidelines
url: https://yourcompany.atlassian.net/wiki/spaces/ENG/pages/123456
ancestors:
  - Engineering Home
  - Backend Standards
labels:
  - api
  - guidelines
author: Jane Smith
created_at: "2024-03-01T10:00:00.000Z"
updated_at: "2025-11-15T14:22:00.000Z"
version: 7
has_attachments: true
attachment_count: 3
image_count: 2
---

# API Design Guidelines

## Overview

All REST APIs **must** follow...

![diagram.png](attachments/a3f9b012_diagram.png)
```

## Local Testing with Docker

See [`docker/`](docker/) for a complete Docker Compose setup (Confluence + PostgreSQL) and a seed script that populates test pages with images.

```bash
cd docker
docker compose up -d
# Wait ~2 min, then open http://localhost:8090
# Complete setup wizard, create admin account & API token

# Seed test data (3 pages, 6 image attachments)
python seed_test_data.py --base-url http://localhost:8090 \
                          --email admin@example.com \
                          --token YOUR_PAT_TOKEN

# Run exporter against local instance
python atlassian_rag_exporter.py --config config_local.yaml --verbose
```

## Loading into a Vector Store

### LlamaIndex

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader(
    input_dir="./rag_corpus/pages",
    recursive=True,
    required_exts=[".md"],
).load_data()

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("What are the API design standards?")
```

### LangChain

```python
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

loader = DirectoryLoader(
    "./rag_corpus/pages",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
)
docs = loader.load()

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
chunks = splitter.split_text(docs[0].page_content)
```

## License

MIT
