"""Heuristic fallback when LLM judge is unavailable. Low confidence, marked is_fallback."""

from __future__ import annotations

import re
import time

from evaluator.schemas import (
    EvalReport,
    PerRunEval,
    RunInput,
    EVAL_CRITERIA,
    WEIGHTS_PRESETS,
)


def _score_length(text: str) -> float:
    """0-10: too short -> low, reasonable length -> high, very long -> cap."""
    n = len((text or "").strip())
    if n < 50:
        return max(0, n / 10.0)
    if n < 500:
        return 5 + min(5, (n - 50) / 90)
    if n < 3000:
        return 8 + min(2, (n - 500) / 1250)
    return 10.0


def _score_structure(text: str) -> float:
    """0-10: has headings/lists/numbers -> higher."""
    t = (text or "").strip()
    if not t:
        return 0
    score = 3
    if re.search(r"\n\s*[-*•]\s", t) or re.search(r"\n\s*\d+[.)]\s", t):
        score += 3
    if re.search(r"\n#+\s|\n\*{2}[^*]|\n_{2}[^_]", t):
        score += 2
    if len(t.split("\n")) >= 3:
        score += 1
    return min(10, score + 1)


def _score_uncertainty_handling(text: str) -> float:
    """0-10: phrases like 'might', 'could', 'not sure' -> higher for factual context."""
    t = (text or "").lower()
    cues = ["might", "may", "could", "perhaps", "uncertain", "not sure", "возможно", "не уверен"]
    count = sum(1 for c in cues if c in t)
    return min(10, 3 + count * 2)


def _score_safety(text: str) -> float:
    """0-10: no obvious dangerous keywords -> 8-10."""
    t = (text or "").lower()
    bad = ["kill yourself", "hack into", "взломай", "убий", "оружие"]
    if any(b in t for b in bad):
        return 2
    return 8


def _score_relevance(text: str, question: str) -> float:
    """0-10: simple word overlap."""
    qw = set(re.findall(r"\w+", (question or "").lower()))
    tw = set(re.findall(r"\w+", (text or "").lower()))
    if not qw:
        return 5
    overlap = len(qw & tw) / len(qw)
    return min(10, 3 + overlap * 7)


def run_heuristic_evaluation(question: str, runs: list[RunInput]) -> EvalReport:
    """Compute simple heuristic scores. confidence=0.3, is_fallback=True."""
    if not runs:
        raise ValueError("No runs to evaluate")

    per_run: dict[str, PerRunEval] = {}
    for r in runs:
        text = (r.answer_text or "").strip()
        c1 = _score_length(text)
        c2 = _score_structure(text)
        c3 = _score_uncertainty_handling(text)
        c4 = _score_safety(text)
        c5 = _score_relevance(text, question)
        # Equal-ish weights for fallback
        scores = {
            "correctness": 5.0,
            "completeness": c1,
            "relevance": c5,
            "clarity": c2,
            "actionability": c2,
            "constraint_following": 5.0,
            "uncertainty_handling": c3,
            "safety": c4,
            "efficiency": min(10, c1 + 1),
        }
        w = WEIGHTS_PRESETS.get("general", {})
        total = sum(scores.get(k, 5) * w.get(k, 1/9) for k in EVAL_CRITERIA)
        total = max(0, min(100, total * 10))
        per_run[r.run_id] = PerRunEval(
            run_id=r.run_id,
            overall_score=round(total, 1),
            per_criterion_scores=scores,
            strengths=[],
            weaknesses=[],
            fix_suggestions=[],
            flags={"heuristic_fallback": True},
        )

    sorted_run_ids = sorted(per_run.keys(), key=lambda rid: per_run[rid].overall_score, reverse=True)
    winner = sorted_run_ids[0] if sorted_run_ids else runs[0].run_id

    return EvalReport(
        winner_run_id=winner,
        ranking=sorted_run_ids,
        per_run=per_run,
        top2_comparison=None,
        confidence=0.3,
        is_fallback=True,
        metadata={"judge_model": "heuristic", "version": "1.0", "timestamp": time.time()},
    )
