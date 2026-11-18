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

from tycho_nooa.models import (
    ActionDecision,
    _exclude_when_none,
    _validate_action_horizon,
)

PROTOCOL_VERSION = 3
HOST_PREFIX = "ARC_HOST\t"
WORKER_PREFIX = "ARC_AGENT\t"

ARC_ACTION_ID_TO_NAME = {
    0: "RESET",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "SPACEBAR",
    6: "CLICK",
    7: "UNDO",
}
ARC_ACTION_NAME_TO_ID = {
    name: action_id for action_id, name in ARC_ACTION_ID_TO_NAME.items()
}


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


def project_arc_observation(
    *,
    frames: list[list[list[int]]] | list[list[int]],
    available_action_ids: list[int],
    state: str,
    levels_completed: int,
    full_reset: bool,
    previous_levels_completed: int | None = None,
) -> ObservationPayload:
    """Project one public ARC frame sequence onto the Solver's evidence contract."""
    sequence = _frame_sequence(frames)
    grids = [_validated_grid(grid) for grid in sequence]
    level_advanced = (
        previous_levels_completed is not None
        and levels_completed > previous_levels_completed
    )
    normalized_state = state.upper().replace("GAMESTATE.", "")
    terminal_state = normalized_state in {"WIN", "GAME_OVER"}
    terminal_index = 0 if level_advanced or terminal_state else None
    terminal_reason = (
        "level_advanced"
        if level_advanced
        else "terminal_state"
        if terminal_state
        else None
    )
    decision_index = len(grids) - 1
    event_indices = [
        index
        for index in range(len(grids))
        if index not in {decision_index, terminal_index}
    ]
    available_actions = [
        "RESET",
        *(
            action
            for action in (_arc_action_name(action_id) for action_id in available_action_ids)
            if action != "RESET"
        ),
    ]
    return ObservationPayload(
        decision_frame=grids[decision_index],
        terminal_frame=(grids[terminal_index] if terminal_index is not None else None),
        event_frames=[
            EvidenceFramePayload(
                kind="animation",
                source_index=index,
                frame=grids[index],
            )
            for index in event_indices
        ],
        provenance=FrameSelectionProvenancePayload(
            raw_frame_count=len(grids),
            decision_frame_index=decision_index,
            terminal_frame_index=terminal_index,
            event_frame_indices=event_indices,
            terminal_reason=terminal_reason,
        ),
        available_actions=available_actions,
        levels_completed=levels_completed,
        state=normalized_state,
        full_reset=full_reset,
    )


def arc_action_payload(decision: ActionDecision) -> tuple[int, dict[str, int] | None]:
    """Translate one validated Solver decision to ARC's action id and data."""
    try:
        action_id = ARC_ACTION_NAME_TO_ID[decision.action]
    except KeyError as error:
        raise ValueError(f"unsupported ARC action label {decision.action!r}") from error
    if decision.action == "CLICK":
        assert decision.row is not None and decision.col is not None
        data = {"x": decision.col, "y": decision.row}
    else:
        data = None
    return action_id, data


def _arc_action_name(action_id: int) -> str:
    try:
        return ARC_ACTION_ID_TO_NAME[action_id]
    except KeyError as error:
        raise ValueError(f"unsupported ARC action id {action_id}") from error


def _frame_sequence(
    frames: list[list[list[int]]] | list[list[int]],
) -> list[list[list[int]]]:
    if not frames:
        raise ValueError("public frame sequence must be non-empty")
    first = frames[0]
    if not first:
        raise ValueError("public frame must be a non-empty grid")
    if type(first[0]) is int:
        return [frames]  # type: ignore[list-item]
    if isinstance(first[0], list):
        return frames  # type: ignore[return-value]
    raise ValueError("public frame must be a grid or frame sequence")


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
    level: int = Field(ge=0)
    turn_in_level: int = Field(ge=0)
    decision_timeout_seconds: float = Field(gt=0)
    observation: ObservationPayload
    previous_decision: ActionDecision | None

    @model_validator(mode="after")
    def action_horizon_is_exact(self) -> TurnMessage:
        _validate_action_horizon(
            sequence=self.sequence,
            action_limit=self.action_limit,
            actions_used=self.actions_used,
            actions_remaining=self.actions_remaining,
            final_permitted_action=self.final_permitted_action,
        )
        return self


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
