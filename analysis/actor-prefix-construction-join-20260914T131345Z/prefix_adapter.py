"""Project an Actor-executed IPC prefix onto the constructor-visible case ABI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from observation_abi import observation_from_state, validate_constructor_surface

HEX = "0123456789abcdef"


class PrefixError(ValueError):
    """Actor prefix cannot be projected without inventing a transition."""


def decode_grid_rows(rows: Any) -> list[list[int]]:
    if not isinstance(rows, list) or not rows:
        raise PrefixError("grid_rows must be a nonempty list of hex strings")
    grid: list[list[int]] = []
    width = None
    for row in rows:
        if not isinstance(row, str) or not row:
            raise PrefixError("each grid_rows entry must be a nonempty hex string")
        cells: list[int] = []
        for char in row:
            index = HEX.find(char)
            if index < 0:
                raise PrefixError("grid_rows contains a non-hex cell")
            cells.append(index)
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise PrefixError("grid_rows is not rectangular")
        grid.append(cells)
    return grid


def _load_jsonl(path: Path) -> list[Any]:
    text = path.read_text()
    rows: list[Any] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class ActorPrefixCase:
    visible: tuple[dict[str, Any], ...]
    frontier: int

    def constructor_view(self) -> dict[str, Any]:
        payload = {
            "schema": "actor-prefix-construction-case-v1",
            "objective": "Construct the smallest model that supports the next useful decision, while preserving what is known and exposing what remains unresolved.",
            "frontier": self.frontier,
            "initial_observation": observation_from_state(self.visible[0]["before"]),
            "current_observation": observation_from_state(self.visible[-1]["after"]),
        }
        validate_constructor_surface(payload)
        return payload

    def compact_view(self) -> dict[str, Any]:
        payload = self.constructor_view()
        payload.pop("initial_observation")
        payload.pop("current_observation")
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


def project_actor_prefix(states_path: Path, actions_path: Path) -> ActorPrefixCase:
    states = _load_jsonl(Path(states_path))
    actions = _load_jsonl(Path(actions_path))
    if len(states) != len(actions) + 1:
        raise PrefixError("need N+1 states and N executed actions")
    visible: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, str) or not action:
            raise PrefixError("executed action must be a nonempty primitive string")
        before = states[index]
        after = states[index + 1]
        visible.append(
            {
                "step": index + 1,
                "action": action,
                "before": {"grid": decode_grid_rows(before.get("grid_rows"))},
                "after": {"grid": decode_grid_rows(after.get("grid_rows"))},
            }
        )
    return ActorPrefixCase(visible=tuple(visible), frontier=len(visible))
