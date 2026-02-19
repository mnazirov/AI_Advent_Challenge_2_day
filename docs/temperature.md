# Temperature и параметры сэмплирования в проекте

В веб-интерфейсе сравнения чатов для каждого чата можно включить параметры сэмплирования: **temperature**, **top_p**, **frequency_penalty**, **presence_penalty**, **seed**. Они отправляются в API только для моделей, которые их поддерживают.

## Как это устроено

- Список моделей без поддержки сэмплирования задаётся в `openai_client.MODELS_WITHOUT_SAMPLING`.
- Для таких моделей в UI блок сэмплирования отключается (подсказка «Для выбранной модели недоступно»). При переключении на такую модель чекбоксы сбрасываются.
- В исходящем JSON запроса (и в блоке «Запрос и ответ») поля temperature/top_p/… **не включаются** для этих моделей, чтобы payload совпадал с тем, что реально принимает API.
- Бэкенд при вызове Chat Completions API добавляет sampling-параметры в запрос только если модель **не** в `MODELS_WITHOUT_SAMPLING`.

## Матрица совместимости (ModelCapabilities)

| Модель | Провайдер | temperature / top_p / penalties / seed |
|--------|-----------|----------------------------------------|
| gpt-5.2, gpt-5.2-instant, gpt-5.2-pro, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo | OpenAI | Да. Параметр `temperature` (0–2), `top_p` (0–1) и др. в Chat Completions. |
| gpt-5.2-thinking, o1, o1-mini, o3-mini | OpenAI | **Нет.** Reasoning-модели; параметры сэмплирования не поддерживаются. Не передавать. |
| deepseek-chat | DeepSeek | Да. API совместим с OpenAI; дефолт temperature 1.0, см. [Parameter settings](https://api-docs.deepseek.com/quick_start/parameter_settings). |
| deepseek-reasoner | DeepSeek | **Нет.** Temperature, top_p, presence_penalty, frequency_penalty не поддерживаются (игнорируются API). В проекте не отправляем. См. [Reasoning model](https://api-docs.deepseek.com/guides/reasoning_model). |

## Официальная документация

- **OpenAI**  
  - [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) — параметр `temperature`.  
  - [Reasoning models](https://platform.openai.com/docs/guides/reasoning) — ограничения для o1/o3 и reasoning-моделей.
- **DeepSeek**  
  - [The Temperature Parameter](https://api-docs.deepseek.com/quick_start/parameter_settings) — для deepseek-chat.  
  - [Reasoning Model (deepseek-reasoner)](https://api-docs.deepseek.com/guides/reasoning_model) — неподдерживаемые параметры.

## Модели, где была проблема / ограничение

| Модель | Причина |
|--------|--------|
| deepseek-reasoner | В коде не была в `MODELS_WITHOUT_SAMPLING`; параметры уходили в запросе и игнорировались API. Добавлена в список, UI и payload приведены в соответствие. |
| o1, o1-mini, o3-mini, gpt-5.2-thinking | Ограничение API (reasoning): temperature/sampling не поддерживаются. Уже были в `MODELS_WITHOUT_SAMPLING`; исправлена только фильтрация payload на фронте и сброс чекбоксов при смене модели. |

## Диапазоны и дефолты в проекте

- **temperature:** 0–2 (clamp в коде); в UI placeholder 0.7.  
- **top_p:** 0–1; в UI placeholder 1.  
- **frequency_penalty / presence_penalty:** -2–2.  
- **seed:** целое (опционально).

Значения приводятся к числу и ограничиваются диапазоном на бэкенде (`_parse_float`) и во фронте (`getRunConfig`).
