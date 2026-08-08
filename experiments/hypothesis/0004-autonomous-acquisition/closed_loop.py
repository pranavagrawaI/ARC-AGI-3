"""Closed one-action contract for feedback-conditioned planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = """Choose one next legal action in an unfamiliar grid game. You receive the public successor frame after every action. Use the supplied action model, objective, and replay-supported causal affordance; adapt after blocked moves or unexpected transitions. Reply with exactly one digit: 1, 2, 3, or 4. Do not emit a route, label, JSON, or narration."""


@dataclass(frozen=True)
class ActionDecision:
    action: int


def parse_response(payload: dict[str, Any]) -> ActionDecision:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must have exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("response did not finish completely")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or content.strip() not in {"1", "2", "3", "4"}:
        raise ValueError("response must contain exactly one legal action digit")
    return ActionDecision(int(content.strip()))
