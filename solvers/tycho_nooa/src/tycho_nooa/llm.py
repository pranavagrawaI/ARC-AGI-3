"""Construct the single NOOA model client used by a game-long worker."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall, UnifiedLLM
from nooa.unifiedllm.registry import get_llm_client

SCRIPTED_TOOL_CALLS_ENV = "TYCHO_NOOA_SCRIPTED_TOOL_CALLS"


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
    base_url = env.get("ARC_LLM_BASE_URL")
    api_key = env.get("ARC_LLM_API_KEY")
    if bool(base_url) != bool(api_key):
        raise RuntimeError("ARC_LLM_BASE_URL and ARC_LLM_API_KEY must be set together")
    if base_url and api_key:
        overrides.update(api_base=base_url, api_key=api_key)
        if not model.startswith("openai/"):
            model = f"openai/{model}"
    use_responses = env.get(
        "ARC_LLM_USE_RESPONSES", "1" if "gpt-5" in model.lower() else "0"
    )
    client_type = "responses" if use_responses == "1" else "completion"
    return get_llm_client(model, client_type=client_type, **overrides), "live"


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
