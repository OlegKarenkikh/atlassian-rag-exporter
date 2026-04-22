# Глава 8: Docker и деплой

## Что такое Docker?

Docker упаковывает приложение со всеми зависимостями в **контейнер** — изолированную среду. На любом компьютере контейнер работает одинаково.

Аналогия: Docker — конструктор LEGO. Каждый сервис — отдельный блок. `docker-compose.yml` описывает как их собрать.

## Четыре сервиса

| Сервис | Что делает | Порт |
|--------|-----------|------|
| **exporter** | Скачивает Confluence → corpus | — |
| **rag-api** | FastAPI REST сервер | 8080 |
| **qdrant** | Векторная БД | 6333 |
| **webhook** | Слушает события Confluence | 8081 |

## Быстрый старт

```bash
# 1. Создайте .env
cp .env.example .env
# Отредактируйте: укажите токены

# 2. Поднять всё
docker compose up -d

# 3. Проверить
curl http://localhost:8080/health
```

## Secrets через .env

Никогда не кладите токены в код или `docker-compose.yml`. Используйте `.env` файл (добавлен в `.gitignore`):

```env
ATLASSIAN_BASE_URL=https://company.atlassian.net
ATLASSIAN_EMAIL=you@company.com
ATLASSIAN_TOKEN=ATATT3xFfGF0...
ATLASSIAN_RAG_API_TOKEN=your-api-secret
```

## Полезные команды

```bash
docker compose logs -f rag-api          # логи API
docker compose restart rag-api          # перезапустить сервис
docker compose down -v                  # остановить + удалить данные
docker compose run --rm exporter \
  python atlassian_rag_exporter.py      # ручной экспорт
```

## CI/CD: quality gate

При каждом `git push` GitHub Actions запускает тесты + coverage. Если упало — merge запрещён.

```
git push → GitHub Actions:
  ruff + black + mypy → pytest --cov=80% → ✅ merge allowed
                                           ❌ merge blocked
```

## Как собрать образ

```bash
make docker-build
# Или:
docker build -f docker/Dockerfile -t atlassian-rag-exporter:latest .
```

---

> Вы дочитали до конца! Теперь вы знаете всё: от Atlassian API до векторных БД, от RAG до докера. 

