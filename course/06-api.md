# Глава 6: REST API и FastAPI

## Что такое REST API?

REST API — как меню в ресторане. Отправляете HTTP-запрос: адрес (`/ask`), метод (`POST`), данные (вопрос в JSON). В ответ — JSON с ответом.

## FastAPI: почему он?

- Автоматически генерирует **Swagger UI** (`/docs`) — тестируйте прямо в браузере
- Валидирует запросы через **Pydantic** — понятные ошибки вместо загадочных крашей
- Работает **асинхронно** — обслуживает много запросов одновременно

## Задать вопрос

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret" \
  -d '{"query": "Как настроить SSO через SAML?"}'
```

Ответ:
```json
{
  "answer": "Для настройки SSO через SAML...",
  "sources": [{"title": "SSO Guide", "score": 0.93, "space_key": "ENG"}],
  "metrics": {"faithfulness": 0.87, "context_relevance": 0.94},
  "latency": {"embed_ms": 12, "retrieve_ms": 8, "llm_ms": 1240}
}
```

## SSE-стриминг — как ChatGPT

Вместо ожидания полного ответа — токены приходят по одному:

```javascript
const resp = await fetch("/ask/stream", {
  method: "POST",
  body: JSON.stringify({ query: "Что такое Jira?" })
});
// Токены: "Jira", " —", " это", " трекер", ...
```

Три типа событий: `sources` (сначала), `token` (по одному), `done` (финал с метриками).

## Безопасность

```bash
export ATLASSIAN_RAG_API_TOKEN=my-secret
uvicorn rag_api:app --port 8080
```

Если переменная задана — запросы без токена получают `401 Unauthorized`. Не задана — API открыт (для локальной разработки).

## Все эндпоинты

| Метод | Путь | Что делает |
|-------|------|------------|
| `POST` | `/ask` | Вопрос → ответ с источниками |
| `POST` | `/ask/stream` | Вопрос → SSE-стриминг токенов |
| `POST` | `/eval` | Пакетная оценка по JSONL-файлу |
| `GET` | `/health` | Статус LLM и uptime |
| `GET` | `/metrics` | Агрегированные метрики |
| `DELETE` | `/log` | Очистить лог Q&A |

## Swagger UI

Откройте [http://localhost:8080/docs](http://localhost:8080/docs) → нажмите **Try it out** → введите вопрос → Execute.
