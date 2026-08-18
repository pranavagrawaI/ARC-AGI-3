"""Translate human control names only at the ARC SDK boundary."""

from __future__ import annotations

from typing import Any

from arcengine import GameAction

from arc_host.types import ActionDecision

ACTION_ID_TO_NAME = {
    0: "RESET",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "SPACEBAR",
    6: "CLICK",
    7: "UNDO",
}
ACTION_NAME_TO_ID = {name: action_id for action_id, name in ACTION_ID_TO_NAME.items()}


def action_name(action: Any) -> str:
    """Return the human-facing label for one public ARC action."""
    action_id = int(action)
    try:
        return ACTION_ID_TO_NAME[action_id]
    except KeyError as error:
        raise ValueError(f"unsupported ARC action id {action_id}") from error


def game_action(decision: ActionDecision) -> tuple[GameAction, dict[str, int] | None]:
    """Convert a validated decision into the SDK action and coordinate payload."""
    try:
        action = GameAction.from_id(ACTION_NAME_TO_ID[decision.action])
    except KeyError as error:
        raise ValueError(f"unsupported ARC action label {decision.action!r}") from error
    data = (
        {"x": decision.col, "y": decision.row} if decision.action == "CLICK" else None
    )
    return action, data
