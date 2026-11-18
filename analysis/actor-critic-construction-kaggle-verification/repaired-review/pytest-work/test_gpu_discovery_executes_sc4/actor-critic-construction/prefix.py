"""Project worker-executed transitions.jsonl onto the constructor-visible case."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from observation_abi import observation_from_state, validate_constructor_surface


class PrefixError(ValueError):
    """Executed prefix cannot be projected without inventing a transition."""


def _grid(payload: Any, side: str) -> list[list[int]]:
    if not isinstance(payload, dict) or "grid" not in payload:
        raise PrefixError(f"{side} must contain integer grid")
    grid = payload["grid"]
    if not isinstance(grid, list) or not grid:
        raise PrefixError(f"{side}.grid must be a nonempty matrix")
    return observation_from_state({"grid": grid})["grid"]


def load_transitions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise PrefixError("transition row must be an object")
        rows.append(row)
    if not rows:
        raise PrefixError("transitions.jsonl is empty")
    return rows


@dataclass(frozen=True)
class ActorPrefixCase:
    visible: tuple[dict[str, Any], ...]
    frontier: int

    def constructor_view(self) -> dict[str, Any]:
        payload = {
            "schema": "actor-executed-prefix-v1",
            "objective": "Construct the smallest model that supports the next useful decision, while preserving what is known and exposing what remains unresolved.",
            "frontier": self.frontier,
            "initial_observation": observation_from_state(self.visible[0]["before"]),
            "current_observation": observation_from_state(self.visible[-1]["after"]),
        }
        validate_constructor_surface(payload)
        return payload

    def compact_view(self) -> dict[str, Any]:
        payload = self.constructor_view()
        payload["visible_transition_summaries"] = [
            {"step": row["step"], "action": row["action"]} for row in self.visible
        ]
        payload["full_evidence"] = {
            "path": "/evidence/case.json",
            "scope": "chronological visible transitions only; no future observations",
        }
        validate_constructor_surface(payload)
        return payload

    def inspection_view(self) -> dict[str, Any]:
        payload = self.constructor_view()
        payload["visible_transitions"] = [
            {
                "step": row["step"],
                "action": row["action"],
                "before": observation_from_state(row["before"]),
                "after": observation_from_state(row["after"]),
            }
            for row in self.visible
        ]
        validate_constructor_surface(payload)
        return payload

    def evaluator_payload(self, include_suffix: bool = False) -> dict[str, Any]:
        if include_suffix:
            raise PrefixError("prospective suffix is not constructor-visible and is not assessed here")
        return {
            "frontier": self.frontier,
            "visible": list(self.visible),
            "suffix": [],
            "branch_cap": 32,
            "required_targets": [],
        }


def project_transitions(path: Path) -> ActorPrefixCase:
    visible: list[dict[str, Any]] = []
    for row in load_transitions(path):
        action = row.get("action")
        step = row.get("step")
        if not isinstance(action, str) or not action:
            raise PrefixError("transition action must be a nonempty string")
        if not isinstance(step, int) or step < 1:
            raise PrefixError("transition step must be a positive integer")
        visible.append(
            {
                "step": step,
                "turn": row.get("turn"),
                "action": action,
                "before": {"grid": _grid(row.get("before"), "before")},
                "after": {"grid": _grid(row.get("after"), "after")},
            }
        )
    steps = [row["step"] for row in visible]
    if steps != list(range(1, len(visible) + 1)):
        raise PrefixError("transition steps must be chronological 1..N")
    return ActorPrefixCase(visible=tuple(visible), frontier=len(visible))
