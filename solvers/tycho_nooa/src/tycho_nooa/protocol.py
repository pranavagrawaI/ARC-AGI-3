"""Strict Python 3.12 mirror of the ARC-host JSON protocol."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from tycho_nooa.models import ActionDecision

PROTOCOL_VERSION = 3
HOST_PREFIX = "ARC_HOST\t"
WORKER_PREFIX = "ARC_AGENT\t"


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[3]


class EvidenceFramePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["animation"]
    source_index: int = Field(ge=0)
    frame: list[list[int]]

    @field_validator("frame", mode="before")
    @classmethod
    def validate_frame(cls, frame: list[list[int]]) -> list[list[int]]:
        return _validated_grid(frame)


class FrameSelectionProvenancePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_frame_count: int = Field(ge=1)
    decision_frame_index: int = Field(ge=0)
    terminal_frame_index: int | None = Field(default=None, ge=0)
    event_frame_indices: list[int]
    terminal_reason: Literal["level_advanced", "terminal_state"] | None


class EvidenceEnvelopePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_frame: list[list[int]]
    terminal_frame: list[list[int]] | None
    event_frames: list[EvidenceFramePayload]
    provenance: FrameSelectionProvenancePayload
    available_actions: list[str]
    levels_completed: int = Field(ge=0)
    state: str
    full_reset: bool

    @property
    def grid(self) -> list[list[int]]:
        """The settled frame used for action decisions."""
        return self.decision_frame

    @field_validator("decision_frame", "terminal_frame", mode="before")
    @classmethod
    def validate_grid(
        cls, grid: list[list[int]] | None
    ) -> list[list[int]] | None:
        return None if grid is None else _validated_grid(grid)

    @model_validator(mode="after")
    def evidence_indices_match_frames(self) -> EvidenceEnvelopePayload:
        provenance = self.provenance
        _validate_provenance_indices(provenance)
        _validate_terminal_provenance(self.terminal_frame, provenance)
        event_indices = [event.source_index for event in self.event_frames]
        if event_indices != provenance.event_frame_indices:
            raise ValueError("event frames and provenance indices must agree")
        return self


def _validate_provenance_indices(provenance: FrameSelectionProvenancePayload) -> None:
    event_roles = [provenance.decision_frame_index, *provenance.event_frame_indices]
    indices = [*event_roles]
    if provenance.terminal_frame_index is not None:
        indices.append(provenance.terminal_frame_index)
    if any(index >= provenance.raw_frame_count for index in indices):
        raise ValueError("frame provenance index exceeds raw_frame_count")
    if len(event_roles) != len(set(event_roles)):
        raise ValueError("event frames must not overlap the decision frame")
    if provenance.terminal_frame_index in provenance.event_frame_indices:
        raise ValueError("event frames must not overlap the terminal frame")


def _validate_terminal_provenance(
    terminal_frame: list[list[int]] | None,
    provenance: FrameSelectionProvenancePayload,
) -> None:
    absent = terminal_frame is None
    if absent != (provenance.terminal_frame_index is None):
        raise ValueError("terminal frame and provenance must agree")
    if absent != (provenance.terminal_reason is None):
        raise ValueError("terminal frame and terminal reason must agree")


# Compatibility name for existing worker internals; the wire schema is the
# EvidenceEnvelopePayload above and carries no legacy `grid` field.
ObservationPayload = EvidenceEnvelopePayload


def _validated_grid(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if len(grid) > 64 or width > 64:
        raise ValueError("grid must fit within 64x64")
    if any(len(row) != width for row in grid):
        raise ValueError("grid must be rectangular")
    if not _has_valid_cells(grid):
        raise ValueError("grid cells must be integers in [0, 15]")
    return grid


def _has_valid_cells(grid: list[list[int]]) -> bool:
    return all(type(cell) is int and 0 <= cell <= 15 for row in grid for cell in row)


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
    decision_timeout_seconds: float = Field(gt=0)
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
