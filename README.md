# OpenAI CLI Chat

Interactive terminal chat with OpenAI using the **Responses API**, streaming, and multi-turn dialogue.

**Requirements:** Python 3.11+

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Перед каждым запуском чата **активируйте venv** (`source .venv/bin/activate`) или вызывайте `./.venv/bin/python3 main.py`.

## Run

Ключ берётся из переменной окружения **или из файла `.env`** в корне проекта (через `python-dotenv`). Создайте `.env`:

```bash
# В корне проекта создайте файл .env с одной строкой:
echo "OPENAI_API_KEY=sk-ваш-ключ" > .env
```

Либо задайте переменную вручную:

```bash
export OPENAI_API_KEY=sk-...
python3 main.py
```

С промптами (ключ можно по-прежнему держать в `.env`). Несколько строк — передайте аргумент несколько раз:

```bash
python3 main.py --system "You are helpful." --system "Answer briefly."
python3 main.py --assistant "Line one." --assistant "Line two."
```

Override the model (default is `gpt-5.2`):

```bash
python3 main.py --model gpt-4o
```

## In-chat commands

| Command     | Description |
|------------|-------------|
| `/system`  | Show current system prompt; then enter new text in several lines, finish with an empty line (or single `.` to clear). |
| `/assistant` | Same for assistant prompt: multi-line input, end with empty line or `.` to clear. |
| `/reset`   | Clear dialogue history; next message starts a new conversation (system/assistant prompts are re-applied). |
| `/exit`    | Quit the chat. |

### Examples

**View and change system prompt (multi-line):**

```
You: /system
Current system prompt: 'You are helpful.'
New system prompt (end with empty line, or single '.' to clear):
You are a concise coding assistant.
Use bullet points when listing.
[empty line]
System prompt updated.
```

**View and change assistant prompt (multi-line):**

```
You: /assistant
Current assistant prompt: 'I answer briefly.'
New assistant prompt (end with empty line, or single '.' to clear):
Hi! I'll keep answers short.
[empty line]
Assistant prompt updated.
```

**Clear history and start over:**

```
You: /reset
Dialog history cleared. Next message will start a new conversation.
You: What is 2+2?
AI: 4
```

**Exit:**

```
You: /exit
```

## Web UI: сравнение нескольких чатов с разными настройками

Веб-интерфейс для параллельного запуска **нескольких чатов** с разными конфигами: один общий запрос пользователя отправляется во все чаты, ответы отображаются в колонках.

**Возможности:**

- **Несколько чатов** — по умолчанию один, можно добавлять и удалять. У каждого чата свой набор настроек и своя история диалога.
- **Настройки на чат:** System prompt, Stop sequences (теги, до 4), Max output tokens, модель, опциональные параметры сэмплирования (temperature, top_p, frequency_penalty, presence_penalty, seed) с чекбоксом «Включить» — в API уходят только включённые и заполненные.
- **Имена чатов** — можно задать кастомное название (иначе «Чат 1», «Чат 2»).
- **Продолжить диалог** — чекбокс под полем ввода: если выключен, запрос уходит без истории (один тур); если включен — с полной историей чата.
- **Оценить** — модуль Evaluator (LLM-as-Judge): ранжирование ответов по критериям, баллы, сильные/слабые стороны, рекомендации, сравнение топ-2. Пресеты весов: Общий, Код, Факты, Текст. Вывод на русском.
- **История запросов** — боковая панель (drawer), последние запросы в localStorage, восстановление по клику.
- **Diff** — после получения ответов можно включить сравнение двух колонок (unified/split).
- **Тема** — переключатель светлая/тёмная, настройки сохраняются в localStorage.

**Запуск:**

```bash
source .venv/bin/activate
flask --app web_app run
```

Если порт 5000 занят (например, «Приёмник AirPlay» на macOS):

```bash
flask --app web_app run --port 5001
```

Откройте в браузере: **http://127.0.0.1:5000** (или 5001 при `--port 5001`).

Введите сообщение, при необходимости настройте каждый чат в своей карточке и нажмите **Run**. Ответы появятся в колонках; затем можно нажать **Оценить** для ранжирования. Веб-интерфейс использует **Chat Completions API** и поддерживает множество моделей (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, o1, o1-mini и др.). CLI-чат использует Responses API.

## Project layout

- `main.py` — Точка входа CLI: argparse, `load_dotenv()`, проверка `OPENAI_API_KEY`, вызов `run_chat`.
- `chat_cli.py` — Цикл чата: команды `/system`, `/assistant`, `/reset`, `/exit`, retry при вызовах API.
- `openai_client.py` — Клиент OpenAI: `stream_message()` (Responses API, CLI), `create_message_chat()` (Chat Completions, веб).
- `web_app.py` — Flask: маршруты `/`, `POST /api/compare`, `POST /api/compare-many`, `POST /api/evaluate`, `GET /api/evaluate/last`.
- `templates/index.html` — Веб-форма: общее поле ввода, список чатов с настройками, колонки результатов, Evaluator, история, diff, тема.
- `evaluator/` — Модуль оценки ответов: классификация вопроса, LLM-as-Judge, эвристический fallback, схемы (Pydantic), промпты, защита от prompt injection.
- `tests/` — Тесты (в т.ч. `test_evaluator.py`: парсинг JSON, injection, fallback).
- `requirements.txt` — `openai`, `python-dotenv`, `flask`, `pydantic`, `pytest`.
