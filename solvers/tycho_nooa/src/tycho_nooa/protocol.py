"""Strict Python 3.12 mirror of the ARC-host JSON protocol."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from tycho_nooa.models import ActionDecision

PROTOCOL_VERSION = 1
HOST_PREFIX = "ARC_HOST\t"
WORKER_PREFIX = "ARC_AGENT\t"


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[1]


class ObservationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grid: list[list[int]]
    available_actions: list[str]
    levels_completed: int = Field(ge=0)
    state: str
    full_reset: bool

    @field_validator("grid")
    @classmethod
    def validate_grid(cls, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            raise ValueError("grid must be non-empty")
        width = len(grid[0])
        if len(grid) > 64 or width > 64:
            raise ValueError("grid must fit within 64x64")
        if any(len(row) != width for row in grid):
            raise ValueError("grid must be rectangular")
        if any(
            type(cell) is not int or not 0 <= cell <= 15 for row in grid for cell in row
        ):
            raise ValueError("grid cells must be integers in [0, 15]")
        return grid


class StartGameMessage(ProtocolModel):
    type: Literal["start_game"]
    game_id: str
    seed: int
    workspace_root: str
    initial_observation: ObservationPayload


class TurnMessage(ProtocolModel):
    type: Literal["turn"]
    sequence: int = Field(ge=0)
    level: int = Field(ge=0)
    turn_in_level: int = Field(ge=0)
    observation: ObservationPayload
    previous_decision: ActionDecision | None


class ActionResultMessage(ProtocolModel):
    type: Literal["action_result"]
    sequence: int = Field(ge=0)
    level: int = Field(ge=0)
    turn_in_level: int = Field(ge=0)
    decision: ActionDecision
    before: ObservationPayload
    after: ObservationPayload


class FinishMessage(ProtocolModel):
    type: Literal["finish"]
    stop_reason: str
    actions: int = Field(ge=0)
    levels_completed: int = Field(ge=0)
    final_observation: ObservationPayload


class CloseMessage(ProtocolModel):
    type: Literal["close"]


HostMessage = Annotated[
    StartGameMessage | TurnMessage | ActionResultMessage | FinishMessage | CloseMessage,
    Field(discriminator="type"),
]
HOST_MESSAGE_ADAPTER = TypeAdapter(HostMessage)


def parse_host_message(payload: str) -> HostMessage:
    return HOST_MESSAGE_ADAPTER.validate_json(payload)
