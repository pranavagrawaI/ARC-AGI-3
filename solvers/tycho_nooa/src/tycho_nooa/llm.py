"""Construct the single NOOA model client used by a game-long worker."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall, UnifiedLLM
from nooa.unifiedllm.registry import get_llm_client

from tycho_nooa.llm_trace import request_reasoning

SCRIPTED_TOOL_CALLS_ENV = "TYCHO_NOOA_SCRIPTED_TOOL_CALLS"
MAX_TOKENS_ENV = "ARC_LLM_MAX_TOKENS"
DEFAULT_MAX_TOKENS = 16_000
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
    overrides: dict[str, Any] = {"timeout": 180.0}
    model = _apply_gateway(env, model, overrides)
    _apply_optional_overrides(env, overrides)
    use_responses = env.get(
        "ARC_LLM_USE_RESPONSES", "1" if "gpt-5" in model.lower() else "0"
    )
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
