"""Types crossing the boundary between an environment and a contestant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, TypeAlias

import numpy as np


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class Observation:
    """The deliberately narrow subset of an ARC frame visible to an agent.

    ``pixels`` preserves all rendered layers with shape ``(layers, height, width)``.
    The array is copied from the environment and marked read-only by the runner.
    """

    pixels: np.ndarray
    available_actions: tuple[int, ...]
    levels_completed: int
    state: str
    full_reset: bool


@dataclass(frozen=True)
class Decision:
    """An action and optional JSON-serialisable agent explanation."""

    action: int
    data: Mapping[str, JsonValue] | None = None
    reasoning: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True)
class Transition:
    before: Observation
    decision: Decision
    after: Observation


@dataclass(frozen=True)
class RunResult:
    initial: Observation
    transitions: tuple[Transition, ...]
    final: Observation
    stop_reason: str

    @property
    def actions(self) -> int:
        return len(self.transitions)


class Agent(Protocol):
    """Minimum stateful contestant interface."""

    def act(self, observation: Observation) -> Decision:
        """Choose one of ``observation.available_actions``."""

    def observe(self, transition: Transition) -> None:
        """Update memory after the environment executes the decision."""


class TraceSink(Protocol):
    def start(self, observation: Observation) -> None: ...

    def append(self, step: int, transition: Transition) -> None: ...

    def finish(self, result: RunResult) -> None: ...
