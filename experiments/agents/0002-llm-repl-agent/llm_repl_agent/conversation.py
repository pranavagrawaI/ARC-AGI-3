"""Rolling per-episode conversation with explicit, logged eviction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ConversationTurn:
    step: int
    messages: tuple[dict[str, Any], ...]


class RollingConversation:
    """Retain the system prompt and the newest complete action turns."""

    def __init__(self, *, system_message: Mapping[str, Any], max_turns: int = 6) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.system_message = dict(system_message)
        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []

    def append(self, *, step: int, messages: list[dict[str, Any]]) -> tuple[int, ...]:
        self._turns.append(ConversationTurn(step=step, messages=tuple(messages)))
        evicted: list[int] = []
        while len(self._turns) > self.max_turns:
            evicted.append(self._turns.pop(0).step)
        return tuple(evicted)

    def messages(self) -> list[dict[str, Any]]:
        output = [dict(self.system_message)]
        for turn in self._turns:
            output.extend(dict(message) for message in turn.messages)
        return output

    @property
    def retained_steps(self) -> tuple[int, ...]:
        return tuple(turn.step for turn in self._turns)
