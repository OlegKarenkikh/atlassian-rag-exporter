# atlassian-rag-exporter

[![CI](https://github.com/OlegKarenkikh/atlassian-rag-exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegKarenkikh/atlassian-rag-exporter/actions)
[![PyPI](https://img.shields.io/pypi/v/atlassian-rag-exporter)](https://pypi.org/project/atlassian-rag-exporter/)
[![Python](https://img.shields.io/pypi/pyversions/atlassian-rag-exporter)](https://pypi.org/project/atlassian-rag-exporter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-80.5%25-brightgreen)](docs/TEST_REPORT.md)

Производственный пайплайн для экспорта **Confluence** и **Jira** в структурированный RAG-корпус,
совместимый с LlamaIndex, LangChain и любой векторной базой данных.

> **English documentation:** [README_EN.md](README_EN.md) — версия 2.0.0, переведено 2026-04-22

---

## Возможности

| Функция | Описание |
|---------|----------|
| **Режимы авторизации** | API Token (Cloud), PAT (Server/DC), OAuth 2.0 |
| **Confluence REST API v2** | Курсорная пагинация — до 30× быстрее v1 offset |
| **Полный контент** | HTML → Markdown + YAML front-matter на каждую страницу |
| **Изображения и вложения** | PNG, JPEG, GIF, WebP, SVG, PDF, DOCX, XLSX — SHA-256 дедупликация |
| **Confluence macros** | `<ac:image>`, `<ri:attachment>` → локальные пути |
| **Jira** | Экспорт задач по JQL; ADF → Markdown |
| **Incremental sync** | Только изменённые с последнего запуска страницы |
| **Rate-limit защита** | Экспоненциальный backoff + заголовок Retry-After (tenacity) |
| **RAG-ready вывод** | manifest.json v2.0, per-page metadata.json, глобальная дедупликация |

---

## Установка

```bash
pip install atlassian-rag-exporter
```

Или в dev-режиме:

```bash
git clone https://github.com/OlegKarenkikh/atlassian-rag-exporter.git
cd atlassian-rag-exporter
pip install -e ".[dev]"
```

---

## Быстрый старт

```bash
# 1. Создать config
cp config.yaml.example config.yaml
# Отредактировать: base_url, email, token, spaces

# 2. Запуск
python atlassian_rag_exporter.py --config config.yaml

# 3. Или через CLI после pip install
atlassian-rag-exporter --config config.yaml
```

---

## Авторизация

### Cloud — API Token (рекомендуется)

```yaml
auth:
  type: token
  email: you@company.com
  token: YOUR_API_TOKEN
```

Токен создаётся на: https://id.atlassian.com/manage-profile/security/api-tokens

### Server / Data Center — PAT

```yaml
auth:
  type: pat
  token: YOUR_PAT
```

### OAuth 2.0

```yaml
auth:
  type: oauth2
  access_token: YOUR_BEARER_TOKEN
```

---

## Структура вывода

```
rag_corpus/
├── manifest.json              ← Глобальный индекс (schema v2.0)
├── .sync_state.json           ← Состояние incremental sync
├── pages/
│   └── <id>_<slug>/
│       ├── content.md         ← Markdown + YAML front-matter
│       └── metadata.json      ← Метаданные для vector DB
├── attachments/
│   └── <8hex>_<slug>.<ext>   ← Дедуплицированные файлы
└── jira/
    └── <ISSUE-KEY>.md         ← Jira задачи
```

---

## Интеграция с RAG-фреймворками

### LlamaIndex

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader(
    input_dir="./rag_corpus/pages",
    recursive=True,
    required_exts=[".md"],
).load_data()

index = VectorStoreIndex.from_documents(docs)
response = index.as_query_engine().query("What is our deployment process?")
```

### LangChain + Chroma

```python
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

loader = DirectoryLoader("./rag_corpus", glob="**/*.md",
                         loader_cls=UnstructuredMarkdownLoader)
docs = loader.load()
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

---

## Разработка и тестирование

```bash
make install     # pip install -e ".[dev]"
make test        # быстрый прогон тестов
make test-cov    # тесты + HTML coverage отчёт
make lint        # ruff
make fmt         # black + ruff --fix
make typecheck   # mypy
make ci          # полный CI (fmt + lint + typecheck + test-cov)
```

### Результаты тестирования v2.0.0 (2026-04-22)

| Файл | Тестов | Passed |
|------|--------|--------|
| tests/test_unit.py | 34 | 34 |
| tests/test_e2e.py | 26 | 26 |
| tests/test_integration.py | 9 | 9 |
| tests/test_cli.py | 6 | 6 |
| **Итого** | **75** | **75 (100%)** |

Покрытие строк: **80.5%** (порог ≥ 80%) ✅

Полный отчёт: [docs/TEST_REPORT.md](docs/TEST_REPORT.md)

---

## CI / CD

GitHub Actions: матрица Ubuntu / macOS / Windows × Python 3.9–3.12 (12 комбинаций).  
Шаги: black → ruff → mypy → pytest --cov (≥ 80%).  
Релиз: автоматическая публикация на PyPI при тегах `v*` через OIDC.

---

## Лицензия

MIT © 2024 OlegKarenkikh. See [LICENSE](LICENSE).
