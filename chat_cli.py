"""Interactive chat loop: commands /system, /assistant, /reset, /exit and API calls with retry."""

from __future__ import annotations

import time

from openai import OpenAI

from openai_client import stream_message

MAX_RETRIES = 3
BASE_DELAY = 1.0


def _retry_stream(
    client: OpenAI,
    user_text: str,
    *,
    system_prompt: str | None,
    assistant_prompt: str | None,
    model: str,
    previous_response_id: str | None,
) -> str | None:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return stream_message(
                client,
                user_text,
                system_prompt=system_prompt,
                assistant_prompt=assistant_prompt,
                model=model,
                previous_response_id=previous_response_id,
            )
        except Exception as e:  # noqa: BLE001
            last_error = e
            err_str = str(e).lower()
            if "rate" in err_str or "limit" in err_str or "timeout" in err_str:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2**attempt)
                    time.sleep(delay)
                    continue
            raise
    if last_error is not None:
        raise last_error
    return None


def run_chat(
    system: str | None,
    assistant: str | None,
    model: str,
) -> None:
    """Run the interactive chat loop. Uses OPENAI_API_KEY from environment."""
    client = OpenAI()
    current_system: str | None = system
    current_assistant: str | None = assistant
    last_response_id: str | None = None

    print(f"Model: {model}")
    print("System:", current_system if current_system else "(не задан)")
    print("Assistant:", current_assistant if current_assistant else "(не задан)")
    print("Commands: /system, /assistant, /reset, /exit")
    print()

    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/reset":
            last_response_id = None
            print("Dialog history cleared. Next message will start a new conversation.")
            continue
        if line == "/system":
            print("Current system prompt:", repr(current_system or ""))
            print("New system prompt (end with empty line, or single '.' to clear):")
            try:
                lines: list[str] = []
                while True:
                    ln = input()
                    if ln == "" or ln.strip() == ".":
                        break
                    lines.append(ln)
                current_system = "\n".join(lines).strip() if lines else None
            except EOFError:
                break
            print("System prompt updated.")
            continue
        if line == "/assistant":
            print("Current assistant prompt:", repr(current_assistant or ""))
            print("New assistant prompt (end with empty line, or single '.' to clear):")
            try:
                lines = []
                while True:
                    ln = input()
                    if ln == "" or ln.strip() == ".":
                        break
                    lines.append(ln)
                current_assistant = "\n".join(lines).strip() if lines else None
            except EOFError:
                break
            print("Assistant prompt updated.")
            continue

        print("AI: ", end="", flush=True)
        try:
            new_id = _retry_stream(
                client,
                line,
                system_prompt=current_system,
                assistant_prompt=current_assistant,
                model=model,
                previous_response_id=last_response_id,
            )
            if new_id is not None:
                last_response_id = new_id
        except Exception as e:
            print(f"\nError: {e}")
