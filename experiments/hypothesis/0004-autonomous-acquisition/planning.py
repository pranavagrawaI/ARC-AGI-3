"""Closed one-shot planning contract for an acquired objective."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


SYSTEM_PROMPT = """Plan in an unfamiliar grid game from chronological public frames and a machinery-acquired objective. Actions move the controlled object according to the supplied action model. Produce your own route: no candidate plan, historical suffix, or protected successor is supplied. Submit a plan only if you expect it to complete one level; otherwise abstain. Use exactly one tool and never invent an action or evidence reference."""
EVIDENCE_REFS = ("chronological_frames", "action_model", "acquired_objective", "tracked_positions")


@dataclass(frozen=True)
class PlanningDecision:
    tool: str
    arguments: dict[str, Any]

    @property
    def signature(self) -> tuple[Any, ...]:
        if self.tool == "SUBMIT_PLAN":
            return self.tool, tuple(self.arguments["actions"])
        return self.tool, self.arguments["reason_code"]


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "SUBMIT_PLAN",
                "description": "Submit a self-generated route expected to complete one level.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array", "items": {"type": "integer", "enum": [1, 2, 3, 4]},
                            "minItems": 1, "maxItems": 64,
                        },
                        "predicted_level_gain": {"type": "integer", "enum": [1]},
                        "evidence_refs": {
                            "type": "array", "items": {"type": "string", "enum": list(EVIDENCE_REFS)},
                            "minItems": 1,
                        },
                    },
                    "required": ["actions", "predicted_level_gain", "evidence_refs"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ABSTAIN",
                "description": "Decline when the supplied evidence cannot support a route.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason_code": {"type": "string"},
                        "missing_evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    },
                    "required": ["reason_code", "missing_evidence"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def parse_response(payload: dict[str, Any]) -> PlanningDecision:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must have exactly one choice")
    calls = choices[0].get("message", {}).get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("response must have exactly one tool call")
    function = calls[0].get("function", {})
    tool = function.get("name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if tool == "SUBMIT_PLAN":
        required = {"actions", "predicted_level_gain", "evidence_refs"}
        if not isinstance(arguments, dict) or set(arguments) != required:
            raise ValueError("invalid SUBMIT_PLAN fields")
        actions = arguments["actions"]
        refs = arguments["evidence_refs"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 64 or any(type(a) is not int or a not in {1, 2, 3, 4} for a in actions):
            raise ValueError("actions must contain 1..64 legal integer actions")
        if arguments["predicted_level_gain"] != 1:
            raise ValueError("predicted_level_gain must be 1")
        if not isinstance(refs, list) or not refs or any(ref not in EVIDENCE_REFS for ref in refs):
            raise ValueError("unknown or empty evidence_refs")
    elif tool == "ABSTAIN":
        required = {"reason_code", "missing_evidence"}
        if not isinstance(arguments, dict) or set(arguments) != required:
            raise ValueError("invalid ABSTAIN fields")
        if not isinstance(arguments["reason_code"], str) or not arguments["reason_code"]:
            raise ValueError("reason_code must be non-empty")
        missing = arguments["missing_evidence"]
        if not isinstance(missing, list) or not missing or any(not isinstance(x, str) or not x for x in missing):
            raise ValueError("missing_evidence must contain non-empty strings")
    else:
        raise ValueError("undeclared planning tool")
    return PlanningDecision(tool, arguments)
