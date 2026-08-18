"""Typed values crossing the ARC-host/solver-process seam."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
type Grid = tuple[tuple[int, ...], ...]


def _object(value: JsonValue, *, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    value: dict[str, JsonValue],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


@dataclass(frozen=True)
class ActionDecision:
    """One human-facing action chosen by a solver."""

    action: str
    row: int | None = None
    col: int | None = None

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("action must be non-empty")
        if self.action == "CLICK":
            if type(self.row) is not int or type(self.col) is not int:
                raise ValueError("CLICK requires integer row and col")
            if not 0 <= self.row <= 63 or not 0 <= self.col <= 63:
                raise ValueError("CLICK row and col must be in [0, 63]")
        elif self.row is not None or self.col is not None:
            raise ValueError(f"{self.action} does not accept row or col")

    def to_json(self) -> dict[str, JsonValue]:
        return {"action": self.action, "row": self.row, "col": self.col}

    @classmethod
    def from_json(cls, value: JsonValue) -> Self:
        data = _object(value, context="decision")
        _exact_keys(
            data,
            required={"action", "row", "col"},
            context="decision",
        )
        action = data["action"]
        row = data["row"]
        col = data["col"]
        if not isinstance(action, str):
            raise TypeError("decision.action must be text")
        if row is not None and type(row) is not int:
            raise TypeError("decision.row must be an integer or null")
        if col is not None and type(col) is not int:
            raise TypeError("decision.col must be an integer or null")
        return cls(action=action, row=row, col=col)


@dataclass(frozen=True)
class Observation:
    """The complete public information made visible to a solver."""

    grid: Grid
    available_actions: tuple[str, ...]
    levels_completed: int
    state: str
    full_reset: bool

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "grid": [list(row) for row in self.grid],
            "available_actions": list(self.available_actions),
            "levels_completed": self.levels_completed,
            "state": self.state,
            "full_reset": self.full_reset,
        }


@dataclass(frozen=True)
class GameStart:
    game_id: str
    seed: int
    workspace_root: Path
    initial_observation: Observation

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "workspace_root": str(self.workspace_root),
            "initial_observation": self.initial_observation.to_json(),
        }


@dataclass(frozen=True)
class WorkerReady:
    worker_id: str
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class Turn:
    sequence: int
    level: int
    turn_in_level: int
    observation: Observation
    previous_decision: ActionDecision | None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "sequence": self.sequence,
            "level": self.level,
            "turn_in_level": self.turn_in_level,
            "observation": self.observation.to_json(),
            "previous_decision": (
                self.previous_decision.to_json() if self.previous_decision else None
            ),
        }


@dataclass(frozen=True)
class Transition:
    turn: Turn
    decision: ActionDecision
    after: Observation

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "sequence": self.turn.sequence,
            "level": self.turn.level,
            "turn_in_level": self.turn.turn_in_level,
            "decision": self.decision.to_json(),
            "before": self.turn.observation.to_json(),
            "after": self.after.to_json(),
        }


@dataclass(frozen=True)
class EpisodeResult:
    initial: Observation
    transitions: tuple[Transition, ...]
    final: Observation
    stop_reason: str

    @property
    def actions(self) -> int:
        return len(self.transitions)

    @property
    def levels_completed(self) -> int:
        return self.final.levels_completed


class SolverPort(Protocol):
    """The entire interface the ARC host requires from a persistent solver."""

    def start_game(self, start: GameStart) -> WorkerReady: ...

    def decide(self, turn: Turn) -> ActionDecision: ...

    def observe(self, transition: Transition) -> None: ...

    def finish(self, result: EpisodeResult) -> None: ...

    def close(self) -> None: ...


class Environment(Protocol):
    def reset(self) -> Any: ...

    def step(
        self,
        action: Any,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> Any: ...
