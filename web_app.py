"""Flask web app: compare one user message with config vs without config."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> bool:
        return False


from flask import Flask, jsonify, render_template, request

from openai import OpenAI
from openai_client import create_message_chat, DEFAULT_CHAT_MODEL, CHAT_MODELS_WITH_STOP

load_dotenv()

app = Flask(__name__)


def _normalize_messages(raw: list | None) -> list[dict[str, str]]:
    """Extract list of {role, content} from API payload."""
    if not raw or not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def _run_with_config(
    messages: list[dict[str, str]],
    system_prompt: str | None,
    stop_sequences: list[str],
    max_output_tokens: int | None,
    model: str,
) -> tuple[dict, str | None]:
    """Call create_message_chat (Chat Completions) with config; return (result dict, error)."""
    try:
        client = OpenAI()
        result = create_message_chat(
            client,
            model=model,
            system_prompt=system_prompt or None,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences if stop_sequences else None,
            messages=messages if messages else None,
        )
        return (result, None)
    except Exception as e:
        return ({"text": "", "usage": None, "finish_reason": None}, str(e))


def _run_without_config(messages: list[dict[str, str]], model: str) -> tuple[dict, str | None]:
    """Call create_message_chat without system/stop/max_tokens; return (result dict, error)."""
    try:
        client = OpenAI()
        result = create_message_chat(
            client,
            model=model,
            system_prompt=None,
            max_output_tokens=None,
            stop_sequences=None,
            messages=messages if messages else None,
        )
        return (result, None)
    except Exception as e:
        return ({"text": "", "usage": None, "finish_reason": None}, str(e))


@app.route("/")
def index():
    return render_template("index.html", models=CHAT_MODELS_WITH_STOP, default_model=DEFAULT_CHAT_MODEL)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json() or {}
    user_message = (data.get("user_message") or "").strip()
    baseline_messages = _normalize_messages(data.get("baseline_messages"))
    with_config_messages = _normalize_messages(data.get("with_config_messages"))

    if baseline_messages or with_config_messages:
        if not baseline_messages or not with_config_messages:
            return jsonify({"error": "baseline_messages and with_config_messages required when continuing dialog"}), 400
    else:
        if not user_message:
            return jsonify({"error": "user_message is required"}), 400
        baseline_messages = [{"role": "user", "content": user_message}]
        with_config_messages = [{"role": "user", "content": user_message}]

    system_prompt = (data.get("system_prompt") or "").strip() or None
    stop_raw = data.get("stop_sequences")
    if isinstance(stop_raw, list):
        stop_sequences = [s for s in stop_raw if isinstance(s, str) and s.strip()]
    elif isinstance(stop_raw, str):
        stop_sequences = [s.strip() for s in stop_raw.split("\n") if s.strip()]
    else:
        stop_sequences = []

    max_output_tokens = data.get("max_output_tokens")
    if max_output_tokens is not None:
        try:
            max_output_tokens = int(max_output_tokens)
            if max_output_tokens < 1:
                max_output_tokens = None
        except (TypeError, ValueError):
            max_output_tokens = None

    model = (data.get("model") or "").strip() or DEFAULT_CHAT_MODEL
    if model not in CHAT_MODELS_WITH_STOP:
        model = DEFAULT_CHAT_MODEL

    with_config_result: dict = {"text": "", "usage": None, "finish_reason": None}
    with_config_error: str | None = None
    without_config_result: dict = {"text": "", "usage": None, "finish_reason": None}
    without_config_error: str | None = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_with = executor.submit(
            _run_with_config,
            with_config_messages,
            system_prompt,
            stop_sequences,
            max_output_tokens,
            model,
        )
        f_without = executor.submit(_run_without_config, baseline_messages, model)

        with_config_result, with_config_error = f_with.result()
        without_config_result, without_config_error = f_without.result()

    return jsonify({
        "with_config": {
            "text": with_config_result.get("text", ""),
            "error": with_config_error,
            "usage": with_config_result.get("usage"),
            "finish_reason": with_config_result.get("finish_reason"),
        },
        "without_config": {
            "text": without_config_result.get("text", ""),
            "error": without_config_error,
            "usage": without_config_result.get("usage"),
            "finish_reason": without_config_result.get("finish_reason"),
        },
    })


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it in the environment or .env file.")
    app.run(host="127.0.0.1", port=5000, debug=False)
