# Modified for the Tycho/dfranzen candidate; see canonical-candidate.patch.
"""Construct the single NOOA model client used by a game-long worker."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from nooa.unifiedllm import (
    FakeLLMClient,
    LLMResponse,
    RetryConfig,
    ToolCall,
    UnifiedLLM,
)
from nooa.unifiedllm.registry import get_llm_client

from tycho_nooa.llm_trace import request_reasoning

SCRIPTED_TOOL_CALLS_ENV = "TYCHO_NOOA_SCRIPTED_TOOL_CALLS"
MAX_TOKENS_ENV = "ARC_LLM_MAX_TOKENS"
MAX_TOKENS_AFTER_INITIAL_ENV = "ARC_LLM_MAX_TOKENS_AFTER_INITIAL"
ACTION_DEMAND_MAX_TOKENS_ENV = "ARC_LLM_ACTION_DEMAND_MAX_TOKENS"
ACTION_DEMAND_DISABLE_THINKING_ENV = "ARC_LLM_ACTION_DEMAND_DISABLE_THINKING"
DEFAULT_MAX_TOKENS = 16_000
MAX_ACTOR_CALLS_ENV = "ARC_LLM_MAX_ACTOR_CALLS"
MAX_ACTOR_CALLS_AFTER_INITIAL_ENV = "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL"
DEFAULT_MAX_ACTOR_CALLS = 26
TIMEOUT_SECONDS_ENV = "ARC_LLM_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_SECONDS = 180.0
RETRIES_ENV = "ARC_LLM_RETRIES"
FORCE_ACTION_TOOL_ENV = "ARC_LLM_FORCE_ACTION_TOOL"
PROVIDER_ENDPOINT_ENV = "ARC_LLM_PROVIDER_ENDPOINT"


def generation_max_tokens(
    environment: Mapping[str, str] | None = None,
) -> int:
    """Return one validated output ceiling for every model-using lane."""
    env = os.environ if environment is None else environment
    raw = env.get(MAX_TOKENS_ENV)
    if raw is None:
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{MAX_TOKENS_ENV} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{MAX_TOKENS_ENV} must be a positive integer")
    return value


def generation_max_tokens_after_initial(
    environment: Mapping[str, str] | None = None,
) -> int:
    """Return the later-turn output ceiling, defaulting to the main ceiling."""
    env = os.environ if environment is None else environment
    return _positive_integer(
        env,
        MAX_TOKENS_AFTER_INITIAL_ENV,
        generation_max_tokens(env),
    )


def generation_action_demand_max_tokens(
    environment: Mapping[str, str] | None = None,
) -> int | None:
    """Return an explicit final action-only response ceiling, when configured."""
    env = os.environ if environment is None else environment
    if ACTION_DEMAND_MAX_TOKENS_ENV not in env:
        return None
    return _positive_integer(env, ACTION_DEMAND_MAX_TOKENS_ENV, 1)


def generation_action_demand_disable_thinking(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Whether the final action-only request suppresses model reasoning."""
    env = os.environ if environment is None else environment
    return _binary_flag(env, ACTION_DEMAND_DISABLE_THINKING_ENV, default=False)


def generation_max_actor_calls(
    environment: Mapping[str, str] | None = None,
) -> int:
    """Return the exact Actor request ceiling, including the final demand."""
    env = os.environ if environment is None else environment
    return _positive_integer(env, MAX_ACTOR_CALLS_ENV, DEFAULT_MAX_ACTOR_CALLS)


def generation_max_actor_calls_after_initial(
    environment: Mapping[str, str] | None = None,
) -> int:
    """Return the later-turn Actor ceiling, including its final demand."""
    env = os.environ if environment is None else environment
    return _positive_integer(
        env,
        MAX_ACTOR_CALLS_AFTER_INITIAL_ENV,
        generation_max_actor_calls(env),
    )


def generation_timeout_seconds(
    environment: Mapping[str, str] | None = None,
) -> float:
    """Return the timeout applied to each model request."""
    env = os.environ if environment is None else environment
    raw = env.get(TIMEOUT_SECONDS_ENV)
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{TIMEOUT_SECONDS_ENV} must be a positive number") from error
    if value <= 0:
        raise ValueError(f"{TIMEOUT_SECONDS_ENV} must be a positive number")
    return value


def generation_retries(
    environment: Mapping[str, str] | None = None,
) -> int | None:
    """Return an explicit retry ceiling, or None to preserve NOOA defaults."""
    env = os.environ if environment is None else environment
    raw = env.get(RETRIES_ENV)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{RETRIES_ENV} must be a non-negative integer") from error
    if value < 0:
        raise ValueError(f"{RETRIES_ENV} must be a non-negative integer")
    return value


def generation_force_action_tool(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Whether the final action-only request must select its sole tool."""
    env = os.environ if environment is None else environment
    return _binary_flag(env, FORCE_ACTION_TOOL_ENV, default=False)


def _binary_flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{name} must be exactly 0 or 1")


def _positive_integer(
    env: Mapping[str, str], name: str, default: int
) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_llm(environment: Mapping[str, str] | None = None) -> tuple[UnifiedLLM, str]:
    """Build a live client, or an explicit scripted client for hermetic tests."""
    env = os.environ if environment is None else environment
    if script := env.get(SCRIPTED_TOOL_CALLS_ENV):
        return _scripted_llm(script), "scripted"
    model = env.get("ARC_LLM_MODEL")
    if not model:
        raise RuntimeError(
            "ARC_LLM_MODEL is required unless TYCHO_NOOA_SCRIPTED_TOOL_CALLS is set"
        )
    overrides: dict[str, Any] = {"timeout": generation_timeout_seconds(env)}
    if (retries := generation_retries(env)) is not None:
        # NOOA owns an outer retry loop and LiteLLM owns an inner one. Both
        # must share the same explicit ceiling or one logical generation can
        # silently multiply endpoint attempts and wall time.
        overrides["retry_config"] = RetryConfig(
            max_retries=retries,
            rate_limit_extra_retries=0,
        )
        overrides["num_retries"] = retries
    model = _apply_gateway(env, model, overrides)
    _apply_optional_overrides(env, overrides)
    use_responses = env.get(
        "ARC_LLM_USE_RESPONSES", "1" if "gpt-5" in model.lower() else "0"
    )
    overrides['extra_body'] = {**overrides.get('extra_body', {}), 'chat_template_kwargs': {'enable_thinking': True}, 'top_k': 20}
    overrides['top_p'] = 0.95
    client_type = "responses" if use_responses == "1" else "completion"
    return get_llm_client(model, client_type=client_type, **overrides), "live"


def _apply_gateway(
    env: Mapping[str, str], model: str, overrides: dict[str, Any]
) -> str:
    base_url = env.get("ARC_LLM_BASE_URL")
    api_key = env.get("ARC_LLM_API_KEY")
    if bool(base_url) != bool(api_key):
        raise RuntimeError("ARC_LLM_BASE_URL and ARC_LLM_API_KEY must be set together")
    if not base_url or not api_key:
        return model
    overrides.update(api_base=base_url, api_key=api_key)
    return model if model.startswith("openai/") else f"openai/{model}"


def _apply_optional_overrides(
    env: Mapping[str, str], overrides: dict[str, Any]
) -> None:
    # Without this, `client.context_window` is None for any model litellm does
    # not know, and NOOA's whole context-limit defence is gated off it: the
    # proactive block budget in `_build_messages`, `_archive_on_context_error`
    # (`if not stats or not ctx_window: return`) and the reduced-max_tokens
    # retry all no-op, so an overflowing request kills the run instead of
    # shedding history. Env-set rather than hardcoded so it cannot drift out of
    # step with ARC_LLM_MODEL.
    if window := env.get("ARC_LLM_CONTEXT_WINDOW"):
        overrides["context_window"] = int(window)
    # Not needed to *see* reasoning: muse-glimmer returns it unasked, ~200
    # characters a call, and the flag raises that to ~340 (six trials, A/B, same
    # prompt). So this asks for more thinking, it does not switch capture on --
    # and it changes the run, because the reasoning that comes back is replayed
    # on the next tool call.
    if request_reasoning(env):
        overrides["reasoning"] = {"enabled": True}
    if endpoint := env.get(PROVIDER_ENDPOINT_ENV):
        if endpoint != endpoint.strip():
            raise ValueError(
                f"{PROVIDER_ENDPOINT_ENV} must be one exact provider endpoint"
            )
        overrides["extra_body"] = {
            "provider": {"order": [endpoint], "allow_fallbacks": False}
        }


def _scripted_llm(script: str) -> FakeLLMClient:
    value = json.loads(script)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{SCRIPTED_TOOL_CALLS_ENV} must be a non-empty JSON list")
    responses: list[LLMResponse] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"tool", "arguments"}:
            raise ValueError(
                f"{SCRIPTED_TOOL_CALLS_ENV}[{index}] must contain tool and arguments"
            )
        tool = item["tool"]
        arguments = item["arguments"]
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise TypeError(
                "scripted tool must be text and arguments must be an object"
            )
        call = ToolCall(
            id=f"scripted-{index}",
            name=tool,
            arguments=json.dumps(arguments, separators=(",", ":")),
        )
        responses.append(
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[call],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": None},
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
        )
    return FakeLLMClient(scripted_responses=responses)
