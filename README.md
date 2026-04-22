# atlassian-rag-exporter

[![CI](https://github.com/OlegKarenkikh/atlassian-rag-exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegKarenkikh/atlassian-rag-exporter/actions)
[![Python](https://img.shields.io/pypi/pyversions/atlassian-rag-exporter)](https://pypi.org/project/atlassian-rag-exporter/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](docs/TEST_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready пайплайн для экспорта **Confluence**, **Jira** и **Elasticsearch** в структурированный RAG-корпус с поддержкой **8 векторных баз данных**.

> **English documentation:** [README_EN.md](README_EN.md)

---

## Обучающий курс для новичков

> Никогда не слышали про RAG, векторные базы или API? -> **[Начните здесь](course/README.md)**

Курс из 8 глав с иллюстрациями объясняет все технологии проекта простым языком:

| # | Глава | Что узнаете |
|---|-------|------------|
| 1 | [Confluence и Jira](course/01-confluence-jira.md) | Как устроены Atlassian-продукты и их API |
| 2 | [Что такое RAG](course/02-rag.md) | Почему LLM галлюцинирует и как это исправить |
| 3 | [Эмбеддинги](course/03-embeddings.md) | Как текст превращается в числа |
| 4 | [Векторные БД](course/04-vector-db.md) | Поиск по смыслу - 8 поддерживаемых баз |
| 5 | [Архитектура экспортёра](course/05-exporter.md) | Путь данных от Confluence до файловой системы |
| 6 | [REST API](course/06-api.md) | FastAPI, SSE-стриминг, Swagger UI |
| 7 | [Тесты](course/07-testing.md) | Пирамида тестов, coverage, pytest |
| 8 | [Docker](course/08-docker.md) | Запускаем всё в контейнерах |

---

## Возможности

| Функция | Описание |
|---------|----------|
| **8 провайдеров аутентификации** | Token, PAT, Basic, SSO Cookie, OAuth2, OIDC, OIDC client_credentials, Kerberos |
| **Confluence REST v2** | Курсорная пагинация — до 30× быстрее v1 |
| **Полный контент** | HTML → Markdown + YAML front-matter, изображения, вложения |
| **Jira** | Экспорт задач по JQL, ADF-комментарии |
| **Elasticsearch/OpenSearch** | Scroll API + Point-in-Time, маппинг полей |
| **8 векторных БД** | Chroma, Qdrant, Weaviate, Pinecone, pgvector, OpenSearch, Milvus, Redis Stack |
| **Инкрементальная синхронизация** | Только изменённые документы |
| **SHA-256 дедупликация** | Вложения не дублируются |
| **Rate-limit защита** | Exponential backoff + `Retry-After` |

---

## Установка

```bash
pip install atlassian-rag-exporter                      # базовая
pip install "atlassian-rag-exporter[kerberos]"          # + Kerberos/NTLM
pip install "atlassian-rag-exporter[vector-qdrant]"     # + Qdrant
pip install "atlassian-rag-exporter[vector-all]"        # все 8 векторных БД
```

Из исходников:

```bash
git clone https://github.com/OlegKarenkikh/atlassian-rag-exporter.git
cd atlassian-rag-exporter
pip install -e ".[dev]"
```

---

## Быстрый старт

```bash
cp config.yaml.example config.yaml   # заполнить base_url, auth, spaces
python atlassian_rag_exporter.py --config config.yaml
```

Загрузка в векторную БД:

```python
from vector_store import VectorStoreConfig, build_adapter, load_corpus

cfg = VectorStoreConfig.from_yaml("config.yaml")
adapter = build_adapter(cfg)
adapter.connect()
load_corpus(adapter, "./rag_corpus", embedder=my_embed_fn)
adapter.close()
```

---

## Аутентификация

| Тип | Применение | Ключи конфига |
|-----|-----------|--------------|
| `token` | Atlassian Cloud | `email`, `token` |
| `pat` | Confluence/Jira Server/DC | `token` |
| `basic` | Локальный логин/пароль | `username`, `password` |
| `sso_cookie` | Готовые SSO-куки | `cookies: {name: value}` |
| `oauth2` | OAuth 2.0 Bearer + авторефреш | `access_token`, `refresh_token` |
| `openid` | OIDC Auth Code + PKCE (браузер) | `issuer_url`, `client_id` |
| `sso_openid` | OIDC client_credentials (сервис) | `token_endpoint`, `client_id`, `client_secret` |
| `kerberos` | Kerberos/NTLM | `mutual_authentication` |

Полные примеры — в [config.yaml.example](config.yaml.example).

---

## Векторные базы данных

| Бэкенд | Пакет | Ключ установки |
|--------|-------|----------------|
| `chromadb` | `chromadb>=0.4` | `vector-chroma` |
| `qdrant` | `qdrant-client>=1.7` | `vector-qdrant` |
| `weaviate` | `weaviate-client>=4` | `vector-weaviate` |
| `pinecone` | `pinecone>=3` | `vector-pinecone` |
| `pgvector` | `psycopg2-binary`, `pgvector` | `vector-pgvector` |
| `opensearch` | `opensearch-py>=2.4` | `vector-opensearch` |
| `milvus` | `pymilvus>=2.3` | `vector-milvus` |
| `redis` | `redis[hiredis]>=5` | `vector-redis` |

Настройка через `config.yaml` — примеры всех 8 бэкендов в [config.yaml.example](config.yaml.example).

---

## Структура вывода

```
rag_corpus/
├── manifest.json
├── .sync_state.json
├── pages/<page_id>_<slug>/
│   ├── content.md          ← Markdown + YAML front-matter
│   └── metadata.json
├── attachments/<sha256>_<filename>
├── jira/<ISSUE-KEY>.md
└── elasticsearch/<id>_<slug>/
```

---

## Подключение к LlamaIndex / LangChain

```python
# LlamaIndex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
docs = SimpleDirectoryReader("./rag_corpus/pages", recursive=True).load_data()
index = VectorStoreIndex.from_documents(docs)
```

```python
# LangChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
docs = DirectoryLoader("./rag_corpus/pages", glob="**/content.md").load()
Chroma.from_documents(docs, OpenAIEmbeddings())
```

---

## Разработка

```bash
make install      # pip install -e ".[dev]"
make test         # pytest tests/ -v
make test-cov     # pytest + coverage ≥ 80%
make lint         # ruff
make fmt          # black + ruff --fix
make typecheck    # mypy
make ci           # fmt + lint + typecheck + test-cov
```

### Покрытие тестами (v3.0.0)

| Модуль | Тестов | Coverage |
|--------|--------|----------|
| `atlassian_rag_exporter.py` | 76 | 80% |
| `auth_providers.py` | 39 | 85% |
| `elasticsearch_source.py` | 30 | 82% |
| `vector_store.py` | 43 | 75% |
| **Итого** | **187** | **80.1%** |

---

## Архитектура

```
atlassian_rag_exporter.py   — основной пайплайн (Confluence + Jira)
auth_providers.py           — 8 провайдеров аутентификации
elasticsearch_source.py     — импорт из Elasticsearch / OpenSearch
vector_store.py             — 8 адаптеров векторных БД
config.yaml.example         — полный справочник конфигурации
```

---

## Лицензия

[MIT License](LICENSE)
