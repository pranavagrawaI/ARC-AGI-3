"""Verbatim from thinharness v0.5.4 ``thinharness/providers.py``: chat-completions helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Json


@dataclass
class ModelToolCall:
    """A normalized model tool call."""

    id: str
    name: str
    arguments: str

@dataclass(frozen=True)
class TokenUsage:
    """Normalized provider token usage; fields are None when unreported."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None

@dataclass
class ReasoningPart:
    """Provider-neutral carrier for one native reasoning block.

    The opaque blob is re-emitted natively only when ``provider_name`` matches the
    resuming provider; otherwise ``text`` is replayed as a leading ``<thinking>`` block.
    """

    text: str = ""                       # plain reasoning text — always kept; cross-provider fallback
    signature: str | None = None         # opaque blob: Anthropic signature / redacted data,
    #                                      OpenAI encrypted_content, OpenRouter signature|data
    id: str | None = None                # provider reasoning-item id (OpenAI rs_…; "redacted_thinking" marker)
    provider_name: str | None = None     # origin provider prefix; native re-emit only when this matches
    provider_details: Json | None = None  # spillover: OpenAI summary raw_content; OpenRouter raw reasoning_details entry

@dataclass
class ModelTurn:
    """A normalized model response turn."""

    text: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    raw: Json = field(default_factory=dict)
    reasoning: list[ReasoningPart] = field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    response_model: str | None = None

def _responses_tool_to_chat(tool: Json) -> Json:
    """Convert a Responses API function tool to Chat Completions format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }

def _extract_chat_tool_calls(message: Json) -> list[ModelToolCall]:
    """Extract normalized tool calls from a chat completion message."""
    calls = []
    for call in message.get("tool_calls", []) or []:
        function = call.get("function") or {}
        calls.append(ModelToolCall(id=str(call.get("id")), name=str(function.get("name")), arguments=function.get("arguments") or "{}"))
    return calls

def extract_token_usage(raw: Json) -> TokenUsage | None:
    """Best-effort normalized token usage from a raw provider response.

    Handles both key styles (input_tokens/output_tokens and
    prompt_tokens/completion_tokens), plus provider cache-read breakdowns;
    missing keys yield None fields.
    """
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    cached_tokens = usage.get("cache_read_input_tokens")
    if not isinstance(cached_tokens, int):
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict):
            cached_tokens = input_details.get("cached_tokens")
    if not isinstance(cached_tokens, int):
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        cached_tokens=cached_tokens if isinstance(cached_tokens, int) else None,
    )

def extract_finish_reason(raw: Json) -> str | None:
    """Best-effort normalized finish reason from a raw provider response.

    Precedence: stop_reason, then top-level finish_reason, then
    choices[0].finish_reason; only the first choice is normalized.
    """
    if isinstance(raw.get("stop_reason"), str):
        return raw["stop_reason"]
    if isinstance(raw.get("finish_reason"), str):
        return raw["finish_reason"]
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    return None
