"""Closed acquisition contract, request assembly, and held-out scoring."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any


SYSTEM_PROMPT = """You acquire one objective relation from prefix-only neutral measurements. Every candidate is mechanically executable and fits the stated scale checks; several may be incidental markers. Select a relation only when the supplied structural, temporal, and causal evidence supports adopting it as the game objective. Otherwise abstain and name the missing evidence. Use exactly one tool. Never invent a pair, program, factor, observation, or outcome."""

EVIDENCE_REFS = (
    "invariant.area", "invariant.extent", "invariant.colors", "invariant.patch",
    "invariant.observed_frames", "mutable.initial_area", "mutable.initial_extent",
    "mutable.initial_colors", "mutable.initial_patch", "mutable.current_patch",
    "mutable.observed_frames", "mutable.appearance_changes",
    "mutable.contact_change_rows", "mutable.background_change_rate",
    "prefix_replay.checks", "representation_deduplication.evidence",
)


@dataclass(frozen=True)
class AcquisitionDecision:
    tool: str
    arguments: dict[str, Any]

    @property
    def signature(self) -> tuple[Any, ...]:
        if self.tool == "SELECT_RELATION":
            return self.tool, self.arguments["pair_id"], self.arguments["program_kind"], self.arguments["factor"]
        return self.tool, self.arguments["reason_code"]


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "SELECT_RELATION",
                "description": "Adopt one supplied compiled objective relation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair_id": {"type": "integer"},
                        "program_kind": {"type": "string", "enum": ["integer_scale_objective"]},
                        "factor": {"type": "integer", "enum": [2, 3, 4, 5]},
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(EVIDENCE_REFS)},
                            "minItems": 1,
                        },
                    },
                    "required": ["pair_id", "program_kind", "factor", "evidence_refs"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ABSTAIN",
                "description": "Decline when the prefix cannot distinguish surviving hypotheses.",
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


def assemble_request(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(case, sort_keys=True, indent=2)},
        ],
        "tools": tool_definitions(),
        "tool_choice": "required",
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "max_tokens": config["max_completion_tokens"],
        "reasoning": deepcopy(config["reasoning"]),
        "seed": config["seed"],
    }


def parse_response(payload: dict[str, Any]) -> AcquisitionDecision:
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
    if tool == "SELECT_RELATION":
        required = {"pair_id", "program_kind", "factor", "evidence_refs"}
        if not isinstance(arguments, dict) or set(arguments) != required:
            raise ValueError("invalid SELECT_RELATION fields")
        if type(arguments["pair_id"]) is not int or type(arguments["factor"]) is not int:
            raise ValueError("pair_id and factor must be integers")
        if arguments["program_kind"] != "integer_scale_objective":
            raise ValueError("unknown acquisition program")
        refs = arguments["evidence_refs"]
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("evidence_refs must be non-empty strings")
    elif tool == "ABSTAIN":
        required = {"reason_code", "missing_evidence"}
        if not isinstance(arguments, dict) or set(arguments) != required:
            raise ValueError("invalid ABSTAIN fields")
        missing = arguments["missing_evidence"]
        if not isinstance(arguments["reason_code"], str) or not arguments["reason_code"]:
            raise ValueError("reason_code must be non-empty")
        if not isinstance(missing, list) or not missing or any(not isinstance(item, str) or not item for item in missing):
            raise ValueError("missing_evidence must be non-empty strings")
    else:
        raise ValueError("undeclared acquisition tool")
    return AcquisitionDecision(tool, arguments)


def score(case: dict[str, Any], label: dict[str, Any], decision: AcquisitionDecision) -> dict[str, Any]:
    if decision.tool == "ABSTAIN":
        return {"contract": True, "prefix_semantics": True, "held_out_support": None, "acquired": False, "safe": True}
    candidate = next(
        (row for row in case["candidate_hypotheses"] if row["pair_id"] == decision.arguments["pair_id"]),
        None,
    )
    allowed_evidence_refs = set(EVIDENCE_REFS)
    prefix_semantics = bool(
        candidate
        and candidate["program"]["kind"] == decision.arguments["program_kind"]
        and candidate["program"]["factor"] == decision.arguments["factor"]
        and candidate["prefix_replay"]["status"] == "working"
        and set(decision.arguments["evidence_refs"]) <= allowed_evidence_refs
    )
    held_out = bool(
        prefix_semantics
        and decision.arguments["pair_id"] == label["valid_pair_id"]
        and label["terminal_scale_normalised_match"]
    )
    return {
        "contract": True,
        "prefix_semantics": prefix_semantics,
        "held_out_support": held_out,
        "acquired": held_out,
        "safe": held_out,
    }
