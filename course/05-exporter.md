# Глава 5: Как устроён экспортёр

![Путь данных от Confluence до векторной БД](img/05-exporter.svg)

## Три больших шага

### Шаг 1: Скачать данные

`ConfluenceClient` использует REST API v2 с курсорной пагинацией. Для каждой страницы скачивается: полный HTML, вложения, метаданные (автор, дата, версия, иерархия).

### Шаг 2: HTML → Markdown

Confluence хранит страницы в **Storage Format** — HTML с тегами `<ac:image>`, `<ac:structured-macro>`. `ConfluenceMarkdownConverter` обрабатывает их:

- `<ac:image><ri:attachment filename="diagram.png"/>` → `![diagram](../attachments/abc123_diagram.png)`
- `<ac:structured-macro name="code">` → ` ```python ... ``` `

### Шаг 3: Сохранить структурированно

Каждая страница — `.md` файл с **YAML front-matter**:

```markdown
---
page_id: "12345"
title: "SSO Configuration Guide"
space_key: "ENG"
url: "https://company.atlassian.net/wiki/..."
labels: ["sso", "saml"]
author: "john.doe@company.com"
updated_at: "2026-03-15T10:23:00Z"
---

# SSO Configuration Guide
...
```

Front-matter попадает в векторную БД как метаданные — для фильтрации по space, дате, лейблам.

## Дедупликация вложений

Одно изображение может встречаться на 10 страницах. Каждый файл проверяется по **SHA-256 хэшу** — если уже есть в `manifest.json`, не скачивается повторно.

## Rate-limit защита

Если сервер отвечает `429` с `Retry-After: 30` — `AtlassianSession` автоматически ждёт 30 секунд и повторяет через **tenacity** с экспоненциальным backoff.

---
[Следующая глава: REST API →](06-api.md)
