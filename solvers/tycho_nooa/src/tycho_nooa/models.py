"""The small public interface between an ARC host and the solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

TYCHO_WORKSPACE_TURN_SCHEMA = "tycho-workspace-turn-v1"
_TYCHO_WORKSPACE_AVAILABLE_ACTIONS_FIELD = "available"


class GameTurn(BaseModel):
    """One public observation delivered to the persistent solver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid: list[list[int]]
    available_actions: list[str]
    sequence: int | None = Field(default=None, ge=0)
    level: int
    turn_in_level: int
    levels_completed: int = 0
    state: str = "NOT_FINISHED"
    previous_action: str = ""
    previous_row: int | None = None
    previous_col: int | None = None


def tycho_workspace_turn_record(turn: GameTurn) -> dict[str, Any]:
    """Project the live turn onto Tycho's durable workspace-v1 record call."""
    return {
        "level": turn.level,
        "turn_in_level": turn.turn_in_level,
        "action": turn.previous_action,
        "row": turn.previous_row,
        "col": turn.previous_col,
        "state": turn.state,
        _TYCHO_WORKSPACE_AVAILABLE_ACTIONS_FIELD: list(turn.available_actions),
    }


def tycho_workspace_available_actions(metadata: Mapping[str, Any]) -> list[str]:
    """Read live `available_actions` from Tycho's workspace-v1 field name."""
    value = metadata.get(_TYCHO_WORKSPACE_AVAILABLE_ACTIONS_FIELD)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"{TYCHO_WORKSPACE_TURN_SCHEMA} has invalid "
            f"{_TYCHO_WORKSPACE_AVAILABLE_ACTIONS_FIELD!r}"
        )
    return list(value)


class ActionDecision(BaseModel):
    """The sole environment commitment returned by a solver turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    row: int | None = Field(default=None, ge=0, le=63)
    col: int | None = Field(default=None, ge=0, le=63)

    @model_validator(mode="after")
    def coordinates_match_action(self) -> ActionDecision:
        if self.action == "CLICK":
            if self.row is None or self.col is None:
                raise ValueError("CLICK requires row and col")
        elif self.row is not None or self.col is not None:
            raise ValueError(f"{self.action} does not accept row or col")
        return self
