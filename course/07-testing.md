# Глава 7: Тесты и качество кода

## Зачем тесты?

Исправили баг в `auth_providers.py` — не знаете, не сломали ли `atlassian_rag_exporter.py`. С тестами: запустили `make test` и за 3 секунды знаете что всё OK. Тесты — это ваша страховая сеть.

## Пирамида тестирования

```
        /\
       /  \   E2E (10 шт) — полный пайплайн
      /----\
     /      \  Integration (80 шт) — несколько компонент
    /--------\
   /          \ Unit (320 шт) — одна функция
  /____________\
```

### Unit-тесты (320 шт) — самые быстрые

Тестируют одну функцию в изоляции. HTTP-запросы заменяются **моками** — реальный сервер не нужен:

```python
def test_slugify():
    assert slugify("Hello World!") == "hello-world"

def test_rate_limit(monkeypatch):
    # вместо реального time.sleep — меряем время
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    # ...429 ответ → проверяем slept == [1]
```

### Интеграционные (80 шт)

Несколько компонентов вместе. FastAPI тестируется через `TestClient` (не нужен запущенный сервер):

```python
def test_ask(tmp_path):
    client, _ = _app(tmp_path)
    r = client.post("/ask", json={"query": "Как SSO?"})
    assert r.status_code == 200
    assert "answer" in r.json()
```

### E2E-тесты (10 шт)

Полный пайплайн: скачать → конвертировать → сохранить → проверить файлы (`content.md`, `metadata.json`, изображения).

## Coverage

```bash
make test-cov-full
# TOTAL: 85% (--cov-fail-under=80)
```

85% — хороший результат. 100% обычно избыточно — вы тратите время на тестирование очевидных вещей.

## Полезные команды

```bash
make test                           # все тесты
make test-api                       # только API
python -m pytest tests/ -v -k sso  # по ключевому слову
make lint                           # ruff проверяет стиль
make fmt                            # black форматирует автоматически
```
