"""Synthetic traces with a single constructor-visible observation encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from observation_abi import observation_from_state, validate_constructor_surface

BLOCK = 12
BAR = 11
FLOOR = 4
ROWS = 64
COLS = 64
BAR_ROW = 61


def make_grid(block: list[int], bar_left: int) -> list[list[int]]:
    grid = [[FLOOR] * COLS for _ in range(ROWS)]
    grid[block[0]][block[1]] = BLOCK
    grid[BAR_ROW][bar_left] = BAR
    return grid


def state(block: list[int], bar_left: int) -> dict[str, Any]:
    return {"grid": make_grid(block, bar_left)}


def measure(grid: list[list[int]]) -> dict[str, Any]:
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == BLOCK]
    if not cells:
        raise ValueError("observer found no block")
    bar = next((c for c, v in enumerate(grid[BAR_ROW]) if v == BAR), None)
    if bar is None:
        raise ValueError("observer found no meter")
    return {"block": [min(r for r, _c in cells), min(c for _r, c in cells)], "bar_left": bar}


@dataclass(frozen=True)
class SyntheticCase:
    visible: tuple[dict[str, Any], ...]
    sealed_suffix: tuple[dict[str, Any], ...]
    frontier: int

    def constructor_view(self) -> dict[str, Any]:
        payload = {
            "schema": "offline-construction-case-v1",
            "objective": "Construct the smallest model that supports the next useful decision, while preserving what is known and exposing what remains unresolved.",
            "frontier": self.frontier,
            "visible_transition_summaries": [
                {
                    "step": row["step"],
                    "action": row["action"],
                    "before": measure(row["before"]["grid"]),
                    "after": measure(row["after"]["grid"]),
                }
                for row in self.visible
            ],
            "initial_observation": observation_from_state(self.visible[0]["before"]),
            "current_observation": observation_from_state(self.visible[-1]["after"]),
            "host_observer": {
                "owned_targets": ["block", "bar_left"],
                "authority": "factual measurement only; no transition rule or semantic diagnosis",
            },
        }
        validate_constructor_surface(payload)
        return payload

    def compact_view(self) -> dict[str, Any]:
        payload = self.constructor_view()
        payload.pop("initial_observation")
        payload.pop("current_observation")
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
        return {
            "frontier": self.frontier,
            "visible": list(self.visible),
            "suffix": list(self.sealed_suffix) if include_suffix else [],
            "branch_cap": 32,
            "required_targets": ["block", "bar_left"],
        }


def translated_prefix_case(
    origin_block: list[int],
    origin_bar: int,
    delta_block: list[int],
    delta_bar: int,
) -> SyntheticCase:
    start_block = [origin_block[0] + delta_block[0], origin_block[1] + delta_block[1]]
    start_bar = origin_bar + delta_bar
    after_block = [start_block[0], start_block[1] + 5]
    after_bar = start_bar + 1
    visible = (
        {
            "step": 1,
            "action": "RIGHT",
            "before": state(start_block, start_bar),
            "after": state(after_block, after_bar),
        },
    )
    suffix = (
        {
            "step": 6,
            "action": "RIGHT",
            "before": state(after_block, after_bar),
            "after": state([after_block[0], after_block[1] + 5], after_bar + 1),
        },
    )
    return SyntheticCase(visible=visible, sealed_suffix=suffix, frontier=1)
