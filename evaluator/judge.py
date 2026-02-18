"""LLM-as-Judge: call judge model, parse JSON, map labels back to run_ids."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from openai import OpenAI

from evaluator.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from evaluator.schemas import (
    EvalReport,
    EVAL_CRITERIA,
    PerRunEval,
    Top2Comparison,
    RunInput,
)

# Default judge model (fast, cheap)
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
JUDGE_VERSION = "1.0"


def _labels_for(n: int) -> list[str]:
    """A, B, C, ... Z, AA, AB, ..."""
    if n <= 0:
        return []
    out = []
    for i in range(n):
        if i < 26:
            out.append(chr(ord("A") + i))
        else:
            out.append(chr(ord("A") + i // 26 - 1) + chr(ord("A") + i % 26))
    return out


def _extract_json_from_response(text: str) -> dict[str, Any]:
    """Strip markdown code fence if present and parse JSON."""
    s = (text or "").strip()
    # Remove ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    return json.loads(s)


def _parse_and_validate_judge_output(
    raw: dict[str, Any],
    label_to_run_id: dict[str, str],
) -> tuple[list[str], dict[str, PerRunEval], Top2Comparison | None, float]:
    """
    Parse judge JSON into ranking (run_ids), per_run (run_id -> PerRunEval), top2, confidence.
    Raises ValueError if invalid.
    """
    winner_label = str(raw.get("winner_run_label") or raw.get("winner_run_id") or "").strip().upper()
    ranking_labels = raw.get("ranking") or []
    if not isinstance(ranking_labels, list):
        ranking_labels = [ranking_labels]
    ranking_labels = [str(x).strip().upper() for x in ranking_labels if x]
    if winner_label and winner_label not in ranking_labels:
        ranking_labels.insert(0, winner_label)
    if not ranking_labels and label_to_run_id:
        ranking_labels = list(label_to_run_id.keys())

    ranking = []
    for lab in ranking_labels:
        if lab in label_to_run_id:
            ranking.append(label_to_run_id[lab])

    # Ensure all run_ids appear in ranking (judge may return only top 3)
    seen = set(ranking)
    for lab in sorted(label_to_run_id.keys(), key=lambda l: (len(l), l)):
        rid = label_to_run_id[lab]
        if rid not in seen:
            ranking.append(rid)
            seen.add(rid)

    per_run_raw = raw.get("per_run") or raw.get("per_run_scores") or {}
    per_run: dict[str, PerRunEval] = {}
    for label, run_id in label_to_run_id.items():
        pr = per_run_raw.get(label) or per_run_raw.get(run_id)
        if not isinstance(pr, dict):
            pr = {}
        overall = float(pr.get("overall_score", 0))
        overall = max(0, min(100, overall))
        per_criterion = pr.get("per_criterion_scores") or pr.get("scores") or {}
        if isinstance(per_criterion, dict):
            per_criterion = {k: max(0, min(10, float(v))) for k, v in per_criterion.items() if k in EVAL_CRITERIA}
        strengths = pr.get("strengths") or []
        weaknesses = pr.get("weaknesses") or []
        fix_suggestions = pr.get("fix_suggestions") or pr.get("suggestions") or []
        flags = pr.get("flags") or {}
        if not isinstance(strengths, list):
            strengths = [str(strengths)]
        if not isinstance(weaknesses, list):
            weaknesses = [str(weaknesses)]
        if not isinstance(fix_suggestions, list):
            fix_suggestions = [str(fix_suggestions)]
        per_run[run_id] = PerRunEval(
            run_id=run_id,
            overall_score=overall,
            per_criterion_scores=per_criterion,
            strengths=strengths[:6],
            weaknesses=weaknesses[:6],
            fix_suggestions=fix_suggestions[:10],
            flags={k: v for k, v in flags.items() if isinstance(v, (bool, str, int, float))},
        )

    top2 = None
    top2_raw = raw.get("top2") or raw.get("top2_comparison")
    if isinstance(top2_raw, dict) and len(ranking) >= 2:
        top2 = Top2Comparison(
            why_winner_better=str(top2_raw.get("why_winner_better") or "")[:500],
            where_runner_up_better=str(top2_raw.get("where_runner_up_better") or "")[:500],
        )

    confidence = float(raw.get("confidence", 0.8))
    confidence = max(0, min(1, confidence))

    return ranking, per_run, top2, confidence


def run_llm_judge(
    question: str,
    runs: list[RunInput],
    question_type: str | None = None,
    weights_preset: str = "general",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    openai_client: OpenAI | None = None,
) -> tuple[EvalReport, str]:
    """
    Run LLM judge on answers. Returns (EvalReport, raw_judge_response_text).
    Answers are shuffled and labeled A/B/C so judge is blind.
    """
    if not runs:
        raise ValueError("No runs to evaluate")
    if not question or not question.strip():
        raise ValueError("Question is required")

    # Shuffle order so judge does not see any ordering bias
    indices = list(range(len(runs)))
    random.shuffle(indices)
    labeled: list[tuple[str, str]] = []
    label_to_run_id: dict[str, str] = {}
    labels = _labels_for(len(runs))
    for i, idx in enumerate(indices):
        r = runs[idx]
        lab = labels[i]
        label_to_run_id[lab] = r.run_id
        text = (r.answer_text or "").strip()[:50000]
        labeled.append((lab, text))

    user_prompt = build_judge_user_prompt(question, question_type, labeled, weights_preset)
    client = openai_client or OpenAI()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0.2,
        max_tokens=4096,
    )
    choice = (response.choices or [None])[0]
    raw_text = (getattr(choice.message, "content", None) or "").strip()

    try:
        raw_json = _extract_json_from_response(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge returned invalid JSON: {e}") from e

    ranking, per_run, top2, confidence = _parse_and_validate_judge_output(raw_json, label_to_run_id)

    winner = ranking[0] if ranking else runs[0].run_id
    # Ensure all run_ids have a per_run entry
    for r in runs:
        if r.run_id not in per_run:
            per_run[r.run_id] = PerRunEval(run_id=r.run_id, overall_score=0)

    report = EvalReport(
        winner_run_id=winner,
        ranking=ranking,
        per_run=per_run,
        top2_comparison=top2,
        confidence=confidence,
        is_fallback=False,
        metadata={
            "judge_model": judge_model,
            "version": JUDGE_VERSION,
            "timestamp": time.time(),
        },
    )
    return report, raw_text
