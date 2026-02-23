"""OpenAI Responses API client: stream_message, create_message and input building."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.2"

# Модели OpenAI для веб-сравнения (Chat Completions API). Источник: https://developers.openai.com/api/docs/models
OPENAI_CHAT_MODELS = (
    "gpt-5.2",
    "gpt-5.2-chat-latest",
    "gpt-5.2-pro",
    "gpt-5.1",
    "gpt-5.1-chat-latest",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-5-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o3",
    "o3-pro",
    "o4-mini",
    "o3-mini",
    "o1",
    "o1-pro",
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
)

# Модели DeepSeek (API совместим с OpenAI; ключ DEEPSEEK_API_KEY в .env)
DEEPSEEK_CHAT_MODELS = (
    "deepseek-chat",
    "deepseek-reasoner",
)

# Модели Hugging Face (OpenAI-совместимый API router.huggingface.co; ключ HUGGINGFACE_API_KEY в .env)
HUGGINGFACE_CHAT_MODELS = (
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen2.5-7B-Instruct-1M",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-4B-Thinking-2507",
    "zai-org/GLM-4.5",
)

# Все модели для выбора в веб-интерфейсе (включая HF для валидации)
CHAT_MODELS = OPENAI_CHAT_MODELS + DEEPSEEK_CHAT_MODELS + HUGGINGFACE_CHAT_MODELS
# Модели только для dropdown «Модель» (без HF; HF — отдельный поиск)
MODELS_FOR_DROPDOWN = OPENAI_CHAT_MODELS + DEEPSEEK_CHAT_MODELS
DEFAULT_CHAT_MODEL = "gpt-4o"

# Модели без поддержки параметров сэмплирования (temperature, top_p, penalties, seed).
# Reasoning-модели OpenAI: o1, o1-mini, o1-pro, o3, o3-mini, o3-pro, o4-mini. DeepSeek: deepseek-reasoner.
MODELS_WITHOUT_SAMPLING = frozenset(("o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o3-pro", "o4-mini", "deepseek-reasoner"))

# Цены USD за 1M токенов (input, output) для расчёта стоимости. Источник: https://platform.openai.com/docs/pricing
MODELS_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.2": (2.5, 12.5),
    "gpt-5.2-chat-latest": (2.5, 12.5),
    "gpt-5.2-pro": (2.5, 12.5),
    "gpt-5.1": (2.5, 12.5),
    "gpt-5.1-chat-latest": (2.5, 12.5),
    "gpt-5": (2.5, 12.5),
    "gpt-5-chat-latest": (2.5, 12.5),
    "gpt-5-pro": (2.5, 12.5),
    "gpt-5-mini": (0.4, 1.6),
    "gpt-5-nano": (0.2, 0.8),
    "gpt-4.1": (2.5, 10.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.15, 0.6),
    "o3": (15.0, 60.0),
    "o3-pro": (20.0, 80.0),
    "o4-mini": (3.0, 12.0),
    "o3-mini": (4.0, 16.0),
    "o1": (15.0, 60.0),
    "o1-pro": (20.0, 80.0),
    "o1-mini": (3.0, 12.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.2),
}
# Hugging Face: цены зависят от модели и провайдера; для известных — приблизительно.
for _m in HUGGINGFACE_CHAT_MODELS:
    if _m not in MODELS_PRICING:
        MODELS_PRICING[_m] = (0.2, 0.4)  # примерный диапазон


def compute_cost(
    model: str,
    usage: dict[str, int] | None,
    price_input_per_1m: float | None = None,
    price_output_per_1m: float | None = None,
) -> float | None:
    """Return estimated cost in USD. Uses custom prices if both are set, else MODELS_PRICING. Prices in $/1M tokens."""
    if not usage:
        return None
    if price_input_per_1m is not None and price_output_per_1m is not None:
        input_per_1m, output_per_1m = price_input_per_1m, price_output_per_1m
    else:
        prices = MODELS_PRICING.get(model)
        if not prices:
            return None
        input_per_1m, output_per_1m = prices
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return (prompt_tokens / 1_000_000.0) * input_per_1m + (completion_tokens / 1_000_000.0) * output_per_1m


def is_huggingface_model(model: str) -> bool:
    """True if model is from Hugging Face (fixed list or org/name from search)."""
    return model in HUGGINGFACE_CHAT_MODELS or ("/" in model and model not in OPENAI_CHAT_MODELS and model not in DEEPSEEK_CHAT_MODELS)


def get_chat_client(model: str) -> OpenAI:
    """Return OpenAI client for the given model (OpenAI, DeepSeek, or Hugging Face)."""
    if model in DEEPSEEK_CHAT_MODELS:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set; add it to .env for DeepSeek models")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    if is_huggingface_model(model):
        api_key = (os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN") or "").strip()
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY is not set; add it to .env for Hugging Face models")
        return OpenAI(api_key=api_key, base_url="https://router.huggingface.co/v1")
    return OpenAI()


def _build_first_input(
    user_text: str,
    system_prompt: str | None,
    assistant_prompt: str | None,
) -> list[dict[str, Any]]:
    """Build input list for the first request: [system?, assistant?, user]."""
    items: list[dict[str, Any]] = []
    if system_prompt:
        items.append({"type": "message", "role": "system", "content": system_prompt})
    if assistant_prompt:
        items.append({"type": "message", "role": "assistant", "content": assistant_prompt})
    items.append({"type": "message", "role": "user", "content": user_text})
    return items


def stream_message(
    client: OpenAI,
    user_text: str,
    *,
    system_prompt: str | None,
    assistant_prompt: str | None,
    model: str,
    previous_response_id: str | None,
) -> str | None:
    """
    Send user message and stream the assistant reply. Print tokens as they arrive.
    Returns the new response id for use as previous_response_id in the next turn.
    """
    if previous_response_id is not None:
        api_input: str | list[dict[str, Any]] = user_text
        prev_id: str | None = previous_response_id
    else:
        api_input = _build_first_input(user_text, system_prompt, assistant_prompt)
        prev_id = None

    kwargs: dict[str, Any] = {
        "model": model,
        "input": api_input,
        "stream": True,
    }
    if prev_id is not None:
        kwargs["previous_response_id"] = prev_id
    # Явно передаём system prompt через instructions (рекомендуемый способ в Responses API)
    if system_prompt:
        kwargs["instructions"] = system_prompt

    stream = client.responses.create(**kwargs)
    response_id: str | None = None

    with stream:
        for event in stream:
            event_type = getattr(event, "type", None) or ""
            if event_type == "response.created":
                resp = getattr(event, "response", None)
                if resp is not None:
                    response_id = getattr(resp, "id", None)
            elif event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None) or getattr(event, "output", None)
                if delta:
                    print(delta, end="", flush=True)
            elif "refusal" in event_type:
                delta = getattr(event, "delta", None) or getattr(event, "output", None)
                if delta:
                    print(f"[Refusal: {delta}]", end="", flush=True)
            elif "error" in event_type.lower():
                err_msg = getattr(event, "message", None) or getattr(event, "error", str(event))
                print(f"\n[Error: {err_msg}]", flush=True)
            elif event_type == "response.completed":
                resp = getattr(event, "response", None)
                if resp is not None and response_id is None:
                    response_id = getattr(resp, "id", None)

    print(end="\n")
    return response_id


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key."""
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _extract_output_text(response: Any) -> str:
    """Extract full text from a non-streaming Response object."""
    parts: list[str] = []
    output = _get_attr(response, "output") or []
    for item in output:
        if _get_attr(item, "type") == "message":
            content = _get_attr(item, "content") or []
            for part in content:
                if _get_attr(part, "type") == "output_text":
                    text = _get_attr(part, "text")
                    if text:
                        parts.append(str(text))
    return "".join(parts)


def create_message(
    client: OpenAI,
    user_text: str,
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    max_output_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    """
    Send user message without streaming (Responses API); return full assistant text.
    Used for CLI/legacy. For web UI comparison use create_message_chat (Chat Completions).
    """
    api_input: str | list[dict[str, Any]] = user_text
    if system_prompt:
        api_input = [
            {"type": "message", "role": "system", "content": system_prompt},
            {"type": "message", "role": "user", "content": user_text},
        ]

    kwargs: dict[str, Any] = {
        "model": model,
        "input": api_input,
        "stream": False,
    }
    if system_prompt:
        kwargs["instructions"] = system_prompt
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**kwargs)
    return _extract_output_text(response)


def create_message_chat(
    client: OpenAI,
    user_text: str = "",
    *,
    model: str = DEFAULT_CHAT_MODEL,
    system_prompt: str | None = None,
    max_output_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    messages: list[dict[str, str]] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Chat Completions API: one shot, no streaming. Supports system, stop, max_tokens, sampling params.
    Used for web UI comparison (with and without config).
    If messages is provided and non-empty, uses them (with optional system_prompt prepended).
    Otherwise builds from user_text (single turn).
    Sampling params (temperature, top_p, frequency_penalty, presence_penalty, seed) are only sent
    for models not in MODELS_WITHOUT_SAMPLING.
    Returns dict with text, usage (completion_tokens), finish_reason.
    """
    if messages:
        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)
    else:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.append({"role": "user", "content": user_text})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "stream": False,
    }
    if max_output_tokens is not None:
        kwargs["max_completion_tokens"] = max_output_tokens
    if stop_sequences:
        kwargs["stop"] = stop_sequences[:4]  # API allows up to 4

    # Add sampling params only when set and not equal to API defaults (don't send defaults)
    if model not in MODELS_WITHOUT_SAMPLING:
        if temperature is not None:
            t = max(0.0, min(2.0, float(temperature)))
            if t != 1.0:
                kwargs["temperature"] = t
        if top_p is not None:
            p = max(0.0, min(1.0, float(top_p)))
            if p != 1.0:
                kwargs["top_p"] = p
        if frequency_penalty is not None:
            f = max(-2.0, min(2.0, float(frequency_penalty)))
            if f != 0.0:
                kwargs["frequency_penalty"] = f
        if presence_penalty is not None:
            p = max(-2.0, min(2.0, float(presence_penalty)))
            if p != 0.0:
                kwargs["presence_penalty"] = p
        if seed is not None:
            kwargs["seed"] = int(seed)

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        err_msg = str(e).lower()
        if stop_sequences and "stop" in err_msg and ("not supported" in err_msg or "unsupported" in err_msg):
            del kwargs["stop"]
            response = client.chat.completions.create(**kwargs)
        elif "temperature" in err_msg and ("unsupported" in err_msg or "only the default" in err_msg or "unsupported_value" in str(e)):
            # Model supports only default temperature (e.g. 1); retry without sampling params
            for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"):
                kwargs.pop(key, None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    choice = (response.choices or [None])[0]
    content = ""
    finish_reason: str | None = None
    if choice is not None:
        raw = getattr(choice.message, "content", None)
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, dict):
            content = str(raw.get("text") or raw.get("content") or raw.get("refusal") or "")
        elif isinstance(raw, list):
            # Newer models return content as array of parts: [{ "type": "text", "text": "..." }, ...]
            parts: list[str] = []
            for part in raw:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if isinstance(part, dict):
                    d = part
                else:
                    try:
                        d = part.model_dump() if hasattr(part, "model_dump") else {}
                    except Exception:
                        d = {}
                    if not isinstance(d, dict):
                        d = {}
                t = d.get("text") or d.get("content") or d.get("refusal")
                if not t and not isinstance(part, dict):
                    t = getattr(part, "text", None) or getattr(part, "content", None) or getattr(part, "refusal", None)
                if t:
                    parts.append(str(t))
            content = "".join(parts)
        elif raw is not None:
            content = str(raw)
        if not content:
            refusal = getattr(choice.message, "refusal", None)
            if refusal:
                content = str(refusal)
        finish_reason = getattr(choice, "finish_reason", None)

    usage: dict[str, int] | None = None
    if getattr(response, "usage", None) is not None:
        u = response.usage
        prompt_tokens = getattr(u, "prompt_tokens", None)
        completion_tokens = getattr(u, "completion_tokens", None)
        if prompt_tokens is not None or completion_tokens is not None:
            usage = {
                "prompt_tokens": prompt_tokens if prompt_tokens is not None else 0,
                "completion_tokens": completion_tokens if completion_tokens is not None else 0,
            }

    return {
        "text": content,
        "usage": usage,
        "finish_reason": finish_reason,
    }
