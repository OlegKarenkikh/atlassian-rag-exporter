# Changelog

All notable changes to this project will be documented in this file.

## [v1.0.0] — 2026-04-22

### Добавлено

- **8 методов аутентификации**: `token`, `PAT`, `OAuth2`, `OIDC`, `Kerberos`, `cookie`, `basic`, `OIDC CC`
- **Confluence API v2** с курсорной пагинацией (до 30× быстрее vs v1)
- **Полный контент**: HTML → Markdown, изображения, PDF, диаграммы (SHA-256 дедупликация)
- **Инкрементальная синхронизация** через `.sync_state.json`
- **8 векторных бэкендов**: Qdrant, ChromaDB, pgvector, Weaviate, Pinecone, OpenSearch, Milvus, Redis
- **FastAPI REST API** с SSE-стримингом и Swagger UI (`/docs`)
- **Webhook-листенер** для автосинхронизации (HMAC-SHA256, asyncio debounce, WebSocket push)
- **Azure DevOps source**: Wiki, Work Items, Repo Markdown, table extraction, multimodal captioning
- **Elasticsearch source**: 6 методов аутентификации, scroll API, incremental sync
- **54 теста, 85% coverage** (unit + integration + e2e)
- **Обучающий курс** `course/` — 8 глав для новичков с SVG-иллюстрациями
- **Docker Compose стек**: exporter + rag-api + qdrant + webhook
- **GitHub Actions CI/CD**: ruff + black + mypy + pytest --cov=80%

### Изменено

- Confluence API v1 (offset) → v2 (cursor)
- `markdownify` → кастомный `ConfluenceMarkdownConverter` с поддержкой `<ac:image>` и `<ac:structured-macro>`

### Технологии

Python 3.11+, FastAPI, httpx, tenacity, markdownify, BeautifulSoup4,
pytest + pytest-cov, ruff, black, mypy

[v1.0.0]: https://github.com/OlegKarenkikh/atlassian-rag-exporter/releases/tag/v1.0.0
