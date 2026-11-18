from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = (
    REPO
    / "analysis/deployment-probes/20260912-ls20-sparse-reasoning-public-v1/raw/qualification/solver-workspaces"
    / "ls20-9607627b-jz1v9k2m"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def observe(state: dict[str, Any]) -> dict[str, Any]:
    grid = state["grid"]
    block_cells = [
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value == 12
    ]
    if not block_cells:
        raise ValueError("host observer found no color-12 block")
    bar_left = next(
        (column for column, value in enumerate(grid[61]) if value == 11), None
    )
    if bar_left is None:
        raise ValueError("host observer found no color-11 meter")
    return {
        "block": [
            min(row for row, _ in block_cells),
            min(column for _, column in block_cells),
        ],
        "bar_left": bar_left,
    }


def _grid_rows(state: dict[str, Any]) -> list[str]:
    return ["".join(format(int(value), "x") for value in row) for row in state["grid"]]


@dataclass(frozen=True)
class RealTraceCase:
    visible: tuple[dict[str, Any], ...]
    sealed_suffix: tuple[dict[str, Any], ...]
    planned_actions: tuple[str, ...]
    incumbent_source: str
    incumbent_sha256: str
    frontier: int = 5

    @property
    def sealed_suffix_sha256(self) -> str:
        body = json.dumps(
            self.sealed_suffix, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(body).hexdigest()

    def constructor_view(self) -> dict[str, Any]:
        initial = self.visible[0]["before"]
        current = self.visible[-1]["after"]
        summaries = [
            {
                "step": row["step"],
                "action": row["action"],
                "before": observe(row["before"]),
                "after": observe(row["after"]),
            }
            for row in self.visible
        ]
        return {
            "schema": "ls20-construction-case-v1",
            "objective": (
                "Construct the smallest model that supports the next useful decision, "
                "while preserving what is known and exposing what remains unresolved."
            ),
            "frontier": self.frontier,
            "visible_transition_summaries": summaries,
            "initial_observation": {"grid_rows": _grid_rows(initial)},
            "current_observation": {"grid_rows": _grid_rows(current)},
            "planned_fixed_actions": list(self.planned_actions),
            "incumbent": {
                "source": self.incumbent_source,
                "sha256": self.incumbent_sha256,
                "visible_status": "five informative predictions matched",
            },
            "host_observer": {
                "owned_targets": ["block", "bar_left"],
                "block": "top-left row and column of observed color-12 cells",
                "bar_left": "leftmost color-11 cell in observed row 61",
                "authority": (
                    "factual measurement only; the host supplies no transition rule or "
                    "semantic diagnosis"
                ),
            },
        }

    def inspection_view(self) -> dict[str, Any]:
        packet = self.constructor_view()
        packet['visible_transitions'] = [
            {'step': row['step'], 'action': row['action'],
             'before': {'grid_rows': _grid_rows(row['before'])},
             'after': {'grid_rows': _grid_rows(row['after'])}}
            for row in self.visible
        ]
        return packet

    def compact_view(self) -> dict[str, Any]:
        packet = self.constructor_view()
        del packet['initial_observation'], packet['current_observation']
        packet['full_evidence'] = {
            'path': '/evidence/case.json',
            'format': 'initial_observation/current_observation.grid_rows and visible_transitions[{step,action,before,after}]; all grid_rows are hexadecimal cell rows',
            'scope': 'Every chronological visible before/after grid; no later observations. Inspect only as needed; do not print entire grids gratuitously.'}
        return packet

    def evaluator_payload(self, *, include_suffix: bool) -> dict[str, Any]:
        def detached(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "step": row["step"],
                "action": row["action"],
                "before": row["before"],
                "after": row["after"],
            }

        return {
            "frontier": self.frontier,
            "visible": [detached(row) for row in self.visible],
            "suffix": [detached(row) for row in self.sealed_suffix]
            if include_suffix
            else [],
            "branch_cap": 32,
            "required_targets": ["block", "bar_left"],
        }

    def legacy_incumbent_source(self) -> str:
        return '''"""Pure-stdlib adapter for the exact incumbent's latent dynamics."""

def _observed(grid):
    cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 12]
    block = [min(r for r, _ in cells), min(c for _, c in cells)]
    bar_left = next(c for c, value in enumerate(grid[61]) if value == 11)
    return block, bar_left


def initialize(observation):
    block, bar_left = _observed(observation["grid"])
    return {"block": block, "bar_left": bar_left}


def predict(state, action):
    dr, dc = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}.get(action, (0, 0))
    successor = dict(state)
    successor["block"] = [state["block"][0] + dr, state["block"][1] + dc]
    successor["bar_left"] = state["bar_left"] + 1
    return [{"state": successor, "commitments": {"block": successor["block"], "bar_left": successor["bar_left"]}, "declared_dependencies": ["block", "bar_left"]}]


def assimilate(predicted_state, observation, verdict):
    block, bar_left = _observed(observation["grid"])
    revised = dict(predicted_state)
    revised["block"] = block
    revised["bar_left"] = bar_left
    return revised
'''


def load_case() -> RealTraceCase:
    transitions = _jsonl(WORKSPACE / "ipc/transitions.jsonl")
    predictions = _jsonl(WORKSPACE / "ipc/model_predictions.jsonl")
    actions = _jsonl(WORKSPACE / "ipc/actions.jsonl")
    visible = tuple(row for row in transitions if row["step"] <= 5)
    suffix = tuple(row for row in transitions if 5 < row["step"] <= 8)
    incumbent = next(row for row in predictions if row.get("turn") == 5)
    decision = next(row for row in actions if row.get("turn") == 5)
    case = RealTraceCase(
        visible=visible,
        sealed_suffix=suffix,
        planned_actions=tuple(decision["actions"]),
        incumbent_source=incumbent["source"],
        incumbent_sha256=incumbent["model_sha256"],
    )
    if [row["step"] for row in visible] != [1, 2, 3, 4, 5]:
        raise ValueError("visible transition frontier is not contiguous")
    if [row["step"] for row in suffix] != [6, 7, 8]:
        raise ValueError("sealed suffix is not the expected chronological prefix")
    if tuple(row["action"] for row in suffix) != case.planned_actions[: len(suffix)]:
        raise ValueError("sealed suffix does not match the decision-time action prefix")
    if hashlib.sha256(case.incumbent_source.encode()).hexdigest() != case.incumbent_sha256:
        raise ValueError("incumbent source digest does not reconcile")
    return case
