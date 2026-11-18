from __future__ import annotations

import sys
from pathlib import Path

LIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIVE))
from harness import ACTOR_SUMMARIZER_CAP, HTTP_POST_SECONDS, MAX_CALLS  # noqa: E402
from request_wire import CRITIC_MAX_TOKENS, MODEL, conservative_reserve_usd, project_outbound, request_purpose, usage_receipt  # noqa: E402


def test_strips_unsupported_and_pins_coreweave() -> None:
    body = project_outbound({
        "model": "other",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "execute_python", "arguments": "{}"}}],
        "max_tokens": 16384,
        "parallel_tool_calls": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "reasoning": {"enabled": False}},
        "reasoning": {"enabled": False},
    })
    assert "parallel_tool_calls" not in body
    assert "chat_template_kwargs" not in body
    assert "chat_template_kwargs" not in (body.get("extra_body") or {})
    assert body["model"] == MODEL
    assert body["provider"]["only"] == ["coreweave/fp8"]
    assert body["provider"]["allow_fallbacks"] is False
    assert body["provider"]["require_parameters"] is True
    assert body["max_tokens"] == 16384


def test_preserves_critic_192() -> None:
    body = project_outbound({"messages": [{"role": "user", "content": "{}"}], "max_tokens": 192, "reasoning": {"enabled": False}})
    assert body["max_tokens"] == CRITIC_MAX_TOKENS
    assert request_purpose({"max_tokens": 192}) == "critic"
    assert request_purpose({"max_tokens": 16384, "tools": [{"function": {"name": "execute_python"}}]}) == "actor"
    assert request_purpose({"tools": [{"function": {"name": "submit_candidate"}}]}) == "constructor"


def test_rejects_output_over_16384() -> None:
    try:
        project_outbound({"max_tokens": 16385, "reasoning": {"enabled": False}})
    except ValueError as error:
        assert "16384" in str(error)
    else:
        raise AssertionError("expected max_tokens rejection")


def test_caps_and_http_timeout_constants() -> None:
    assert MAX_CALLS == 24
    assert ACTOR_SUMMARIZER_CAP == 20
    assert HTTP_POST_SECONDS == 30
    reserve = conservative_reserve_usd({"max_tokens": 16384, "messages": []})
    assert 0 < reserve < 2
    assert usage_receipt({"choices": [{}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}) == {"prompt_tokens": 1, "completion_tokens": 2}
    assert usage_receipt({"choices": [{}]}) is None

