from __future__ import annotations

from typing import Any

import pytest

from tycho_nooa.llm import (
    build_llm,
    generation_action_demand_disable_thinking,
    generation_action_demand_max_tokens,
    generation_force_action_tool,
    generation_max_actor_calls,
    generation_max_actor_calls_after_initial,
    generation_max_tokens,
    generation_max_tokens_after_initial,
    generation_retries,
    generation_timeout_seconds,
)
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
    assert generation_max_tokens_after_initial({}) == 16_000
    assert generation_max_tokens_after_initial(
        {"ARC_LLM_MAX_TOKENS": "8000"}
    ) == 8_000
    assert generation_max_tokens_after_initial(
        {
            "ARC_LLM_MAX_TOKENS": "16000",
            "ARC_LLM_MAX_TOKENS_AFTER_INITIAL": "4096",
        }
    ) == 4_096
    assert generation_action_demand_max_tokens({}) is None
    assert generation_action_demand_max_tokens(
        {"ARC_LLM_MAX_TOKENS_AFTER_INITIAL": "4096"}
    ) is None
    assert generation_action_demand_max_tokens(
        {
            "ARC_LLM_MAX_TOKENS_AFTER_INITIAL": "4096",
            "ARC_LLM_ACTION_DEMAND_MAX_TOKENS": "2048",
        }
    ) == 2_048


def test_generation_decision_economy_is_runtime_configurable() -> None:
    assert generation_max_actor_calls({}) == 26
    assert generation_max_actor_calls({"ARC_LLM_MAX_ACTOR_CALLS": "4"}) == 4
    assert generation_max_actor_calls_after_initial({}) == 26
    assert generation_max_actor_calls_after_initial(
        {"ARC_LLM_MAX_ACTOR_CALLS": "6"}
    ) == 6
    assert generation_max_actor_calls_after_initial(
        {
            "ARC_LLM_MAX_ACTOR_CALLS": "6",
            "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL": "1",
        }
    ) == 1
    assert generation_timeout_seconds({}) == 180.0
    assert generation_timeout_seconds({"ARC_LLM_TIMEOUT_SECONDS": "60"}) == 60.0
    assert generation_retries({}) is None
    assert generation_retries({"ARC_LLM_RETRIES": "0"}) == 0
    assert generation_force_action_tool({}) is False
    assert generation_force_action_tool({"ARC_LLM_FORCE_ACTION_TOOL": "1"}) is True
    assert generation_action_demand_disable_thinking({}) is False
    assert generation_action_demand_disable_thinking(
        {"ARC_LLM_ACTION_DEMAND_DISABLE_THINKING": "1"}
    ) is True


@pytest.mark.parametrize("value", ["0", "-1", "eight thousand"])
def test_generation_max_tokens_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_MAX_TOKENS"):
        generation_max_tokens({"ARC_LLM_MAX_TOKENS": value})


@pytest.mark.parametrize("value", ["0", "-1", "eight thousand"])
def test_generation_max_tokens_after_initial_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_MAX_TOKENS_AFTER_INITIAL"):
        generation_max_tokens_after_initial(
            {"ARC_LLM_MAX_TOKENS_AFTER_INITIAL": value}
        )


@pytest.mark.parametrize("value", ["0", "-1", "two thousand"])
def test_generation_action_demand_max_tokens_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_ACTION_DEMAND_MAX_TOKENS"):
        generation_action_demand_max_tokens(
            {"ARC_LLM_ACTION_DEMAND_MAX_TOKENS": value}
        )


@pytest.mark.parametrize("value", ["0", "-1", "four"])
def test_generation_max_actor_calls_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_MAX_ACTOR_CALLS"):
        generation_max_actor_calls({"ARC_LLM_MAX_ACTOR_CALLS": value})


@pytest.mark.parametrize("value", ["0", "-1", "four"])
def test_generation_max_actor_calls_after_initial_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL"):
        generation_max_actor_calls_after_initial(
            {"ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL": value}
        )


@pytest.mark.parametrize("value", ["0", "-1", "one minute"])
def test_generation_timeout_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_TIMEOUT_SECONDS"):
        generation_timeout_seconds({"ARC_LLM_TIMEOUT_SECONDS": value})


@pytest.mark.parametrize("value", ["-1", "once"])
def test_generation_retries_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_RETRIES"):
        generation_retries({"ARC_LLM_RETRIES": value})


@pytest.mark.parametrize("value", ["true", "required", "2", ""])
def test_generation_force_action_tool_rejects_ambiguous_values(value: str) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_FORCE_ACTION_TOOL"):
        generation_force_action_tool({"ARC_LLM_FORCE_ACTION_TOOL": value})


@pytest.mark.parametrize("value", ["true", "disabled", "2", ""])
def test_generation_action_demand_thinking_rejects_ambiguous_values(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="ARC_LLM_ACTION_DEMAND_DISABLE_THINKING"):
        generation_action_demand_disable_thinking(
            {"ARC_LLM_ACTION_DEMAND_DISABLE_THINKING": value}
        )


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


def test_live_llm_disables_both_nooa_and_litellm_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_client(model: str, **kwargs: Any) -> object:
        captured.update(model=model, **kwargs)
        return object()

    monkeypatch.setattr("tycho_nooa.llm.get_llm_client", fake_client)
    build_llm(
        {
            "ARC_LLM_MODEL": "openai/custom",
            "ARC_LLM_TIMEOUT_SECONDS": "60",
            "ARC_LLM_RETRIES": "0",
        }
    )

    assert captured["timeout"] == 60.0
    assert captured["num_retries"] == 0
    retry_config = captured["retry_config"]
    assert retry_config.max_retries == 0
    assert retry_config.rate_limit_extra_retries == 0


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


def test_live_llm_accepts_single_segment_provider_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_client(model: str, **kwargs: Any) -> object:
        captured.update(model=model, **kwargs)
        return object()

    monkeypatch.setattr("tycho_nooa.llm.get_llm_client", fake_client)
    build_llm(
        {
            "ARC_LLM_MODEL": "openai/gpt-5-test",
            "ARC_LLM_PROVIDER_ENDPOINT": "openai",
        }
    )

    assert captured["extra_body"] == {
        "provider": {"order": ["openai"], "allow_fallbacks": False}
    }


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
