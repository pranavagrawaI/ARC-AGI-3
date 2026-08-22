"""The small public interface between an ARC host and the solver."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
