"""OpenAI Responses API client: stream_message, create_message and input building."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.2"

# Модели с поддержкой stop для веб-сравнения (Chat Completions)
CHAT_MODELS_WITH_STOP = ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4")
DEFAULT_CHAT_MODEL = "gpt-4o"


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
) -> dict[str, Any]:
    """
    Chat Completions API: one shot, no streaming. Supports system, stop, max_tokens.
    Used for web UI comparison (with and without config).
    If messages is provided and non-empty, uses them (with optional system_prompt prepended).
    Otherwise builds from user_text (single turn).
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

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        err_msg = str(e).lower()
        if stop_sequences and "stop" in err_msg and ("not supported" in err_msg or "unsupported" in err_msg):
            del kwargs["stop"]
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    choice = (response.choices or [None])[0]
    content = ""
    finish_reason: str | None = None
    if choice is not None:
        content = getattr(choice.message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", None)

    usage: dict[str, int] | None = None
    if getattr(response, "usage", None) is not None:
        u = response.usage
        completion_tokens = getattr(u, "completion_tokens", None)
        if completion_tokens is not None:
            usage = {"completion_tokens": completion_tokens}

    return {
        "text": content,
        "usage": usage,
        "finish_reason": finish_reason,
    }
