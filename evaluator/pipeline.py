"""Orchestrate: classify question -> LLM judge or heuristic fallback -> EvalReport."""

from __future__ import annotations

from evaluator.schemas import EvalReport, RunInput, WEIGHTS_PRESETS
from evaluator.question_classifier import classify_question
from evaluator.judge import run_llm_judge
from evaluator.heuristic_fallback import run_heuristic_evaluation


def run_evaluation(
    question: str,
    runs: list[RunInput],
    *,
    weights_preset: str | None = None,
    judge_model: str = "gpt-4o-mini",
    use_heuristic_only: bool = False,
) -> EvalReport:
    """
    Run full evaluation: classify question, pick weights, run LLM judge or heuristic fallback.
    If weights_preset is None, it is inferred from question type.
    """
    if not question or not runs:
        if not runs:
            raise ValueError("No runs to evaluate")
        question = ""

    preset = weights_preset
    if not preset or preset not in WEIGHTS_PRESETS:
        qtype = classify_question(question)
        preset = qtype if qtype in WEIGHTS_PRESETS else "general"

    if use_heuristic_only:
        return run_heuristic_evaluation(question, runs)

    try:
        report, _ = run_llm_judge(
            question=question,
            runs=runs,
            question_type=preset,
            weights_preset=preset,
            judge_model=judge_model,
        )
        return report
    except Exception:
        return run_heuristic_evaluation(question, runs)
