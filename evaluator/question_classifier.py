"""Classify user question type for criterion weight presets."""

from __future__ import annotations

import re

# Keywords per type (lowercased); first match wins unless we score
QUESTION_TYPE_KEYWORDS = {
    "code": [
        "code", "function", "debug", "bug", "implement", "script", "program",
        "algorithm", "syntax", "error", "exception", "api", "класс", "функци",
        "код", "отлад", "реализуй", "напиши программу",
    ],
    "factual": [
        "what is", "who was", "when did", "define", "definition", "meaning of",
        "how many", "how much", "сколько", "когда", "что такое", "определение",
    ],
    "plan": [
        "plan", "strategy", "roadmap", "steps to", "how to approach",
        "план", "стратеги", "как подойти", "с чего начать",
    ],
    "creative": [
        "creative", "story", "idea", "imagine", "write a", "придумай",
        "рассказ", "историю", "креатив",
    ],
    "writing": [
        "write", "letter", "email", "edit", "rewrite", "tone", "style",
        "напиши письмо", "редакт", "тон", "стиль", "сочинение",
    ],
    "analysis": [
        "analyze", "analysis", "compare", "pros and cons", "разбор", "сравни",
        "анализ", "плюсы и минусы",
    ],
    "advice": [
        "should i", "recommend", "advice", "совет", "рекомендуй", "какой лучше",
    ],
}


def classify_question(question: str) -> str:
    """
    Classify question type from keywords. Returns preset key for WEIGHTS_PRESETS
    (code, factual, writing, general) or 'general'.
    """
    if not question or not isinstance(question, str):
        return "general"
    q = question.lower().strip()
    if len(q) < 2:
        return "general"

    for qtype, keywords in QUESTION_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                # Map internal types to preset names
                if qtype in ("code", "factual", "writing"):
                    return qtype
                return "general"
    return "general"
