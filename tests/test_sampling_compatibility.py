"""Tests for temperature/sampling compatibility: MODELS_WITHOUT_SAMPLING and create_message_chat kwargs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openai_client import (
    CHAT_MODELS,
    MODELS_WITHOUT_SAMPLING,
    create_message_chat,
)


def _make_mock_response(text: str = "ok", completion_tokens: int = 1, finish_reason: str = "stop"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_models_without_sampling_subset_of_chat_models():
    """MODELS_WITHOUT_SAMPLING must only contain model IDs that exist in CHAT_MODELS."""
    chat_models_set = set(CHAT_MODELS)
    for model in MODELS_WITHOUT_SAMPLING:
        assert model in chat_models_set, f"{model} must be in CHAT_MODELS"


def test_deepseek_reasoner_in_models_without_sampling():
    """deepseek-reasoner does not support temperature/top_p per DeepSeek docs; must be in the list."""
    assert "deepseek-reasoner" in MODELS_WITHOUT_SAMPLING


def test_sampling_not_sent_for_models_without_sampling():
    """For models in MODELS_WITHOUT_SAMPLING, temperature/top_p/etc. must not be in API kwargs."""
    captured = {}

    def capture_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _make_mock_response()

    mock_client = MagicMock()
    mock_client.chat.completions.create = capture_create

    for model in ("o1", "o1-mini", "o3-mini", "gpt-5.2-thinking", "deepseek-reasoner"):
        captured.clear()
        create_message_chat(
            mock_client,
            "Hi",
            model=model,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.1,
            seed=42,
        )
        assert "temperature" not in captured, f"{model} must not receive temperature"
        assert "top_p" not in captured, f"{model} must not receive top_p"
        assert "frequency_penalty" not in captured
        assert "presence_penalty" not in captured
        assert "seed" not in captured


def test_sampling_sent_for_supported_model():
    """For a model with sampling support, temperature/top_p are included and clamped."""
    captured = {}

    def capture_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _make_mock_response()

    mock_client = MagicMock()
    mock_client.chat.completions.create = capture_create

    create_message_chat(
        mock_client,
        "Hi",
        model="gpt-4o",
        temperature=0.7,
        top_p=0.9,
    )
    assert captured.get("temperature") == 0.7
    assert captured.get("top_p") == 0.9


def test_temperature_clamped_to_valid_range():
    """Temperature above 2 is clamped to 2, below 0 to 0."""
    captured = {}

    def capture_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _make_mock_response()

    mock_client = MagicMock()
    mock_client.chat.completions.create = capture_create

    create_message_chat(mock_client, "Hi", model="gpt-4o", temperature=3.0)
    assert captured.get("temperature") == 2.0

    create_message_chat(mock_client, "Hi", model="gpt-4o", temperature=-0.5)
    assert captured.get("temperature") == 0.0


def test_top_p_clamped():
    """top_p above 1 clamped to 1, below 0 to 0."""
    captured = {}

    def capture_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _make_mock_response()

    mock_client = MagicMock()
    mock_client.chat.completions.create = capture_create

    create_message_chat(mock_client, "Hi", model="gpt-4o", top_p=1.5)
    assert captured.get("top_p") == 1.0

    create_message_chat(mock_client, "Hi", model="gpt-4o", top_p=-0.1)
    assert captured.get("top_p") == 0.0


def test_none_sampling_params_omitted():
    """When temperature/top_p are None, they are not added to kwargs."""
    captured = {}

    def capture_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _make_mock_response()

    mock_client = MagicMock()
    mock_client.chat.completions.create = capture_create

    create_message_chat(mock_client, "Hi", model="gpt-4o", temperature=None, top_p=None)
    assert "temperature" not in captured
    assert "top_p" not in captured


def test_string_temperature_converted_and_clamped():
    """String numeric temperature is converted to float and clamped."""
    captured = {}

    def capture_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _make_mock_response()

    mock_client = MagicMock()
    mock_client.chat.completions.create = capture_create

    create_message_chat(mock_client, "Hi", model="gpt-4o", temperature=1.5)  # float
    assert captured.get("temperature") == 1.5
