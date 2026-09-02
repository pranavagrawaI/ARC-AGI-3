"""The small public interface between an ARC host and the solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

TYCHO_WORKSPACE_TURN_SCHEMA = "tycho-workspace-turn-v1"
_TYCHO_WORKSPACE_AVAILABLE_ACTIONS_FIELD = "available"


def _exclude_when_none(value: object) -> bool:
    return value is None


def _validate_action_horizon(
    *,
    sequence: int | None,
    action_limit: int | None,
    actions_used: int | None,
    actions_remaining: int | None,
    final_permitted_action: bool | None,
) -> None:
    values = (
        action_limit,
        actions_used,
        actions_remaining,
        final_permitted_action,
    )
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("action horizon fields must be supplied together")
    if sequence is None or actions_used != sequence:
        raise ValueError("actions_used must equal the turn sequence")
    if actions_used >= action_limit:
        raise ValueError("actions_used must be less than action_limit")
    if actions_remaining != action_limit - actions_used:
        raise ValueError("actions_remaining must equal action_limit - actions_used")
    if final_permitted_action is not (actions_remaining == 1):
        raise ValueError("final_permitted_action must match actions_remaining == 1")


class GameTurn(BaseModel):
    """One public observation delivered to the persistent solver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid: list[list[int]]
    available_actions: list[str]
    sequence: int | None = Field(default=None, ge=0)
    action_limit: int | None = Field(
        default=None, strict=True, gt=0, exclude_if=_exclude_when_none
    )
    actions_used: int | None = Field(
        default=None, strict=True, ge=0, exclude_if=_exclude_when_none
    )
    actions_remaining: int | None = Field(
        default=None, strict=True, gt=0, exclude_if=_exclude_when_none
    )
    final_permitted_action: bool | None = Field(
        default=None, strict=True, exclude_if=_exclude_when_none
    )
    level: int
    turn_in_level: int
    levels_completed: int = 0
    state: str = "NOT_FINISHED"
    previous_action: str = ""
    previous_row: int | None = None
    previous_col: int | None = None

    @model_validator(mode="after")
    def action_horizon_is_exact(self) -> GameTurn:
        _validate_action_horizon(
            sequence=self.sequence,
            action_limit=self.action_limit,
            actions_used=self.actions_used,
            actions_remaining=self.actions_remaining,
            final_permitted_action=self.final_permitted_action,
        )
        return self


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
