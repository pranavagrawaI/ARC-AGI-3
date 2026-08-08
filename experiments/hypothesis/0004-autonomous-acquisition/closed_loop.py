"""Closed one-action contract for feedback-conditioned planning."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


SYSTEM_PROMPT = """Choose one next legal action in an unfamiliar grid game. You receive the public successor frame after every action. Use the supplied action model, objective, and replay-supported causal affordance; adapt after blocked moves or unexpected transitions. Reply with exactly one digit: 1, 2, 3, or 4. Do not emit a route, label, JSON, or narration."""

DELIBERATIVE_SYSTEM_PROMPT = """Choose one next legal action in an unfamiliar grid game. You receive the public successor frame after every action. Think through the path-dependent plan as fully as useful. Use the supplied action model, objective, replay-supported causal affordance, prior replies, and observed outcomes; explicitly adapt after blocked moves or unexpected transitions. In your visible reply, explain your current subgoal, route logic, and predicted successor. End with exactly one separate final line `ACTION: N`, where N is 1, 2, 3, or 4."""


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


def parse_deliberative_response(payload: dict[str, Any]) -> ActionDecision:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must have exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("response did not finish completely")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("response content must be text")
    matches = re.findall(r"(?m)^ACTION:\s*([1-4])\s*$", content.strip())
    if len(matches) != 1 or not content.strip().endswith(f"ACTION: {matches[0]}"):
        raise ValueError("response must end with exactly one ACTION: N line")
    return ActionDecision(int(matches[0]))
