"""Host-owned action choice over a frozen initialize/predict/assimilate candidate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

PASSIVE = frozenset({"RESET", "UNDO"})


@dataclass(frozen=True)
class Decision:
    action: str
    row: int | None = None
    col: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"action": self.action, "row": self.row, "col": self.col}


def _commitments(state: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(state[key]) for key in ("block", "bar_left") if key in state}


def _predict_commitments(module: Any, state: dict[str, Any], action: str) -> dict[str, Any]:
    produced = module.predict(copy.deepcopy(state), action)
    if not isinstance(produced, list) or not produced:
        raise ValueError("predict must return a nonempty list")
    item = produced[0]
    if not isinstance(item, dict) or "commitments" not in item or "state" not in item:
        raise ValueError("predict branch must contain state and commitments")
    commitments = item["commitments"]
    if not isinstance(commitments, dict):
        raise ValueError("commitments must be an object")
    return commitments


def select_action(module: Any, state: dict[str, Any], available: list[str] | tuple[str, ...]) -> Decision:
    if not available:
        raise ValueError("no available actions")
    moving = [action for action in available if action not in PASSIVE and action != "CLICK"]
    pool = moving or [action for action in available if action not in PASSIVE] or list(available)
    current = _commitments(state)
    changing: list[str] = []
    static: list[str] = []
    for action in pool:
        predicted = _predict_commitments(module, state, action)
        if current and any(predicted.get(key) != current.get(key) for key in current):
            changing.append(action)
        else:
            static.append(action)
    chosen = (changing or static)[0]
    if chosen == "CLICK":
        return Decision("CLICK", row=0, col=0)
    return Decision(chosen)


def predict_successor(module: Any, state: dict[str, Any], action: str) -> dict[str, Any]:
    produced = module.predict(copy.deepcopy(state), action)
    return copy.deepcopy(produced[0]["state"])
