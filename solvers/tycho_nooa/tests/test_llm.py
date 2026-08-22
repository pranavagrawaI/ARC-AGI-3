from __future__ import annotations

from typing import Any

import pytest

from tycho_nooa.llm import build_llm, generation_max_tokens
from tycho_nooa.llm_trace import (
    CAPTURE_RESPONSE_TEXT_ENV,
    REQUEST_REASONING_ENV,
    capture_response_text,
    request_reasoning,
)


def test_live_llm_requires_a_model() -> None:
    with pytest.raises(RuntimeError, match="ARC_LLM_MODEL is required"):
        build_llm({})


def test_generation_max_tokens_is_runtime_configurable() -> None:
    assert generation_max_tokens({}) == 16_000
    assert generation_max_tokens({"ARC_LLM_MAX_TOKENS": "8000"}) == 8_000


@pytest.mark.parametrize("value", ["0", "-1", "eight thousand"])
def test_generation_max_tokens_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_MAX_TOKENS"):
        generation_max_tokens({"ARC_LLM_MAX_TOKENS": value})


@pytest.mark.parametrize(
    "environment",
    [
        {"ARC_LLM_MODEL": "model", "ARC_LLM_BASE_URL": "https://gateway"},
        {"ARC_LLM_MODEL": "model", "ARC_LLM_API_KEY": "secret"},
    ],
)
def test_live_llm_requires_a_complete_gateway_pair(
    environment: dict[str, str],
) -> None:
    with pytest.raises(RuntimeError, match="must be set together"):
        build_llm(environment)


def test_live_llm_normalizes_gateway_and_optional_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_client(model: str, **kwargs: Any) -> object:
        captured.update(model=model, **kwargs)
        return sentinel

    monkeypatch.setattr("tycho_nooa.llm.get_llm_client", fake_client)
    client, kind = build_llm(
        {
            "ARC_LLM_MODEL": "gpt-5-test",
            "ARC_LLM_BASE_URL": "https://gateway",
            "ARC_LLM_API_KEY": "secret",
            "ARC_LLM_CONTEXT_WINDOW": "131072",
            "ARC_LLM_PROVIDER_ENDPOINT": "open-inference/bf16",
            REQUEST_REASONING_ENV: "1",
        }
    )

    assert client is sentinel
    assert kind == "live"
    assert captured == {
        "model": "openai/gpt-5-test",
        "client_type": "responses",
        "timeout": 180.0,
        "api_base": "https://gateway",
        "api_key": "secret",
        "context_window": 131072,
        "reasoning": {"enabled": True},
        "extra_body": {
            "provider": {
                "order": ["open-inference/bf16"],
                "allow_fallbacks": False,
            }
        },
    }


def test_live_llm_always_requests_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_client(model: str, **kwargs: Any) -> object:
        captured.update(model=model, **kwargs)
        return object()

    monkeypatch.setattr("tycho_nooa.llm.get_llm_client", fake_client)
    build_llm({"ARC_LLM_MODEL": "openai/custom"})

    assert captured["reasoning"] == {"enabled": True}


def test_reasoning_and_response_capture_cannot_be_disabled() -> None:
    assert request_reasoning({}) is True
    assert capture_response_text({}) is True
    assert request_reasoning({REQUEST_REASONING_ENV: "1"}) is True
    assert capture_response_text({CAPTURE_RESPONSE_TEXT_ENV: "1"}) is True

    with pytest.raises(RuntimeError, match="reasoning must remain enabled"):
        request_reasoning({REQUEST_REASONING_ENV: "0"})
    with pytest.raises(RuntimeError, match="response capture must remain enabled"):
        capture_response_text({CAPTURE_RESPONSE_TEXT_ENV: "0"})


def test_live_llm_keeps_direct_completion_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_client(model: str, **kwargs: Any) -> object:
        captured.update(model=model, **kwargs)
        return sentinel

    monkeypatch.setattr("tycho_nooa.llm.get_llm_client", fake_client)
    client, kind = build_llm(
        {"ARC_LLM_MODEL": "openai/custom", "ARC_LLM_USE_RESPONSES": "0"}
    )

    assert (client, kind) == (sentinel, "live")
    assert captured == {
        "model": "openai/custom",
        "client_type": "completion",
        "timeout": 180.0,
        "reasoning": {"enabled": True},
    }
