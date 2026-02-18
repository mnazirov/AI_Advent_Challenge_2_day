"""Pydantic schemas for evaluation input/output and config."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Criteria (0–10 each); overall = weighted sum → 0–100 ---
EVAL_CRITERIA = [
    "correctness",
    "completeness",
    "relevance",
    "clarity",
    "actionability",
    "constraint_following",
    "uncertainty_handling",
    "safety",
    "efficiency",
]

# --- Flags (boolean / level) ---
EVAL_FLAGS = [
    "hallucination_risk",  # low | med | high
    "missing_critical_info",
    "contradicts_user_constraints",
    "too_verbose",
    "too_short",
    "unsafe_or_policy_violation",
    "questionable_facts_no_sources",
]


class RunInput(BaseModel):
    """Single run for evaluation (anonymous: no model name to judge)."""

    run_id: str = Field(description="Stable id for this run (e.g. index or uuid)")
    answer_text: str = Field(description="Raw answer from the model")
    model_id: str | None = Field(default=None, description="Not sent to judge; for UI only")
    config_id: str | None = Field(default=None, description="Optional config label")
    latency_ms: int | None = Field(default=None, description="Response time in ms")
    tokens_in: int | None = Field(default=None)
    tokens_out: int | None = Field(default=None)
    cost: float | None = Field(default=None)
    tool_usage_metadata: dict[str, Any] | None = Field(default=None)


class PerRunEval(BaseModel):
    """Per-run evaluation result."""

    run_id: str
    overall_score: float = Field(ge=0, le=100, description="0–100")
    per_criterion_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Criterion name → 0–10",
    )
    strengths: list[str] = Field(default_factory=list, max_length=6)
    weaknesses: list[str] = Field(default_factory=list, max_length=6)
    fix_suggestions: list[str] = Field(default_factory=list)
    flags: dict[str, str | bool] = Field(
        default_factory=dict,
        description="e.g. hallucination_risk: low|med|high, too_verbose: true",
    )


class Top2Comparison(BaseModel):
    """Comparison of winner vs runner-up."""

    why_winner_better: str = Field(description="Short reason winner is better")
    where_runner_up_better: str = Field(description="Short where runner-up is better (if any)")


class EvalReport(BaseModel):
    """Full evaluation report (machine-readable)."""

    winner_run_id: str
    ranking: list[str] = Field(description="Ordered run_ids, best first")
    per_run: dict[str, PerRunEval] = Field(description="run_id → PerRunEval")
    top2_comparison: Top2Comparison | None = Field(default=None, description="When ≥2 runs")
    confidence: float = Field(ge=0, le=1, description="0–1")
    is_fallback: bool = Field(default=False, description="True if heuristic fallback was used")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="judge_model, version, timestamp",
    )


# --- Question type for weight presets ---
QuestionType = str  # factual | code | plan | creative | writing | analysis | advice | general

# --- Preset names for UI ---
CriterionWeightsPreset = str  # "code" | "factual" | "writing" | "general" | ...

# Default weights (sum = 1.0 for 0–10 criteria → 0–100)
DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.15,
    "completeness": 0.12,
    "relevance": 0.12,
    "clarity": 0.12,
    "actionability": 0.12,
    "constraint_following": 0.10,
    "uncertainty_handling": 0.08,
    "safety": 0.09,
    "efficiency": 0.10,
}

WEIGHTS_PRESETS: dict[str, dict[str, float]] = {
    "general": DEFAULT_WEIGHTS,
    "code": {
        "correctness": 0.22,
        "completeness": 0.18,
        "relevance": 0.10,
        "clarity": 0.10,
        "actionability": 0.18,
        "constraint_following": 0.08,
        "uncertainty_handling": 0.04,
        "safety": 0.04,
        "efficiency": 0.06,
    },
    "factual": {
        "correctness": 0.25,
        "completeness": 0.20,
        "relevance": 0.15,
        "clarity": 0.10,
        "actionability": 0.05,
        "constraint_following": 0.05,
        "uncertainty_handling": 0.12,
        "safety": 0.04,
        "efficiency": 0.04,
    },
    "writing": {
        "correctness": 0.10,
        "completeness": 0.10,
        "relevance": 0.12,
        "clarity": 0.20,
        "actionability": 0.08,
        "constraint_following": 0.18,
        "uncertainty_handling": 0.04,
        "safety": 0.04,
        "efficiency": 0.14,
    },
}
