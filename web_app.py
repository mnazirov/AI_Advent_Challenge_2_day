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
from openai_client import (
    create_message_chat,
    DEFAULT_CHAT_MODEL,
    CHAT_MODELS,
    MODELS_WITHOUT_SAMPLING,
)
from evaluator.pipeline import run_evaluation
from evaluator.schemas import RunInput

load_dotenv()

app = Flask(__name__)

MAX_RUNS = 10  # max chats per request in compare-many


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
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
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
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
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


def _parse_float(val, lo: float, hi: float) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if lo <= f <= hi else None
    except (TypeError, ValueError):
        return None


def _parse_run_config(c: dict | None) -> tuple:
    """Parse a single run's config dict into args for _run_with_config. Returns (system_prompt, stop_sequences, max_output_tokens, model, temperature, top_p, frequency_penalty, presence_penalty, seed)."""
    if not c or not isinstance(c, dict):
        return (None, [], None, DEFAULT_CHAT_MODEL, None, None, None, None, None)
    system_prompt = (c.get("system_prompt") or "").strip() or None
    stop_raw = c.get("stop_sequences")
    if isinstance(stop_raw, list):
        stop_sequences = [s for s in stop_raw if isinstance(s, str) and s.strip()]
    elif isinstance(stop_raw, str):
        stop_sequences = [s.strip() for s in stop_raw.split("\n") if s.strip()]
    else:
        stop_sequences = []
    max_output_tokens = c.get("max_output_tokens")
    if max_output_tokens is not None:
        try:
            max_output_tokens = int(max_output_tokens)
            if max_output_tokens < 1:
                max_output_tokens = None
        except (TypeError, ValueError):
            max_output_tokens = None
    model = (c.get("model") or "").strip() or DEFAULT_CHAT_MODEL
    if model not in CHAT_MODELS:
        model = DEFAULT_CHAT_MODEL
    temperature = _parse_float(c.get("temperature"), 0.0, 2.0)
    top_p = _parse_float(c.get("top_p"), 0.0, 1.0)
    frequency_penalty = _parse_float(c.get("frequency_penalty"), -2.0, 2.0)
    presence_penalty = _parse_float(c.get("presence_penalty"), -2.0, 2.0)
    seed = c.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
    return (system_prompt, stop_sequences, max_output_tokens, model, temperature, top_p, frequency_penalty, presence_penalty, seed)


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=CHAT_MODELS,
        default_model=DEFAULT_CHAT_MODEL,
        models_without_sampling=list(MODELS_WITHOUT_SAMPLING),
    )


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
    if model not in CHAT_MODELS:
        model = DEFAULT_CHAT_MODEL

    temperature = _parse_float(data.get("temperature"), 0.0, 2.0)
    top_p = _parse_float(data.get("top_p"), 0.0, 1.0)
    frequency_penalty = _parse_float(data.get("frequency_penalty"), -2.0, 2.0)
    presence_penalty = _parse_float(data.get("presence_penalty"), -2.0, 2.0)
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

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
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
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


@app.route("/api/compare-many", methods=["POST"])
def api_compare_many():
    """Accept user_message and runs: [{ messages?, config }, ...]. Return results: [{ text, error?, usage?, finish_reason? }, ...]."""
    data = request.get_json() or {}
    user_message = (data.get("user_message") or "").strip()
    runs_raw = data.get("runs")
    if not runs_raw or not isinstance(runs_raw, list):
        return jsonify({"error": "runs array is required"}), 400
    if len(runs_raw) > MAX_RUNS:
        return jsonify({"error": f"at most {MAX_RUNS} runs allowed"}), 400

    runs: list[tuple[list[dict[str, str]], tuple]] = []
    for run in runs_raw:
        if not isinstance(run, dict):
            return jsonify({"error": "each run must be an object"}), 400
        messages = _normalize_messages(run.get("messages"))
        if not messages:
            if not user_message:
                return jsonify({"error": "user_message is required when run has no messages"}), 400
            messages = [{"role": "user", "content": user_message}]
        config_args = _parse_run_config(run.get("config"))
        runs.append((messages, config_args))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(runs)) as executor:
        futures = []
        for messages, config_args in runs:
            (system_prompt, stop_sequences, max_output_tokens, model, temperature, top_p, frequency_penalty, presence_penalty, seed) = config_args
            fut = executor.submit(
                _run_with_config,
                messages,
                system_prompt,
                stop_sequences,
                max_output_tokens,
                model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
            )
            futures.append(fut)
        for fut in futures:
            result, err = fut.result()
            results.append({
                "text": result.get("text", ""),
                "error": err,
                "usage": result.get("usage"),
                "finish_reason": result.get("finish_reason"),
            })

    return jsonify({"results": results})


# Last evaluation report (for UI / optional get)
_last_eval_report: dict | None = None


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    Evaluate runs: question + list of { run_id, answer_text, ... }.
    Returns EvalReport (winner, ranking, per_run scores, confidence, is_fallback).
    """
    global _last_eval_report
    data = request.get_json() or {}
    question = (data.get("user_message") or data.get("user_question") or "").strip()
    runs_raw = data.get("runs") or data.get("results")
    if not isinstance(runs_raw, list) or not runs_raw:
        return jsonify({"error": "runs array with run_id and answer_text is required"}), 400

    run_inputs: list[RunInput] = []
    for i, r in enumerate(runs_raw):
        if not isinstance(r, dict):
            return jsonify({"error": "each run must be an object"}), 400
        run_id = r.get("run_id")
        if run_id is None:
            run_id = str(i)
        answer_text = r.get("answer_text") or r.get("text") or ""
        run_inputs.append(RunInput(
            run_id=str(run_id),
            answer_text=answer_text,
            model_id=r.get("model_id"),
            config_id=r.get("config_id"),
            latency_ms=r.get("latency_ms"),
            tokens_in=r.get("tokens_in"),
            tokens_out=r.get("tokens_out"),
            cost=r.get("cost"),
        ))

    weights_preset = (data.get("weights_preset") or "").strip() or None
    use_heuristic_only = bool(data.get("use_heuristic_only"))

    try:
        report = run_evaluation(
            question=question or "No question",
            runs=run_inputs,
            weights_preset=weights_preset,
            use_heuristic_only=use_heuristic_only,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    out = report.model_dump(mode="json")
    _last_eval_report = out
    return jsonify(out)


@app.route("/api/evaluate/last", methods=["GET"])
def api_evaluate_last():
    """Return last evaluation report if any."""
    if _last_eval_report is None:
        return jsonify({"report": None}), 200
    return jsonify({"report": _last_eval_report}), 200


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it in the environment or .env file.")
    app.run(host="127.0.0.1", port=5000, debug=False)
