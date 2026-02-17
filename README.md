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

## Web UI: сравнение с конфигом и без

Веб-интерфейс для отправки одного запроса **дважды** (параллельно): с настройками и без. Удобно сравнивать влияние System prompt, Stop sequence и Max output tokens.

После установки зависимостей запустите:

```bash
source .venv/bin/activate
flask --app web_app run
```

Откройте в браузере: **http://127.0.0.1:5000**

В форме укажите (опционально): System prompt, Stop sequence (по одной на строку, до 4), Max output tokens. Введите сообщение пользователя и нажмите «Отправить оба запроса». Слева отобразится ответ **с конфигом**, справа — **без конфига** (только текст пользователя, без system/stop/max_tokens). Веб-сравнение использует **Chat Completions API** и по умолчанию модели с поддержкой stop: **gpt-4o**, gpt-4o-mini, gpt-4-turbo, gpt-4 (выбор в форме). CLI-чат по-прежнему использует Responses API.

## Project layout

- `main.py` — Entry point: argparse, `load_dotenv()`, `OPENAI_API_KEY` check, calls `run_chat`.
- `chat_cli.py` — Chat loop, `/system` / `/assistant` / `/reset` / `/exit`, retry around API calls.
- `openai_client.py` — OpenAI client: `stream_message()` (Responses API, CLI), `create_message_chat()` (Chat Completions, web compare).
- `web_app.py` — Flask app: маршруты `/` и `POST /api/compare`, параллельный вызов двух запросов.
- `templates/index.html` — форма (system, stop, max_tokens, user message) и два блока для ответов.
- `requirements.txt` — `openai`, `python-dotenv`, `flask`.
