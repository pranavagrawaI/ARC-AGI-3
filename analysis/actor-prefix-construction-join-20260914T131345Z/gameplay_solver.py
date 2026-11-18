"""Persistent solver: frozen candidate plus host-owned action selection."""

from __future__ import annotations

import copy
from typing import Any

from action_selector import Decision, predict_successor, select_action
from observation_abi import observation_from_host


class GameplaySolver:
    def __init__(self, module: Any, *, worker_id: str = "construction-gameplay"):
        self.module = module
        self.worker_id = worker_id
        self.state: dict[str, Any] | None = None
        self.pending_state: dict[str, Any] | None = None
        self.actions: list[str] = []

    def start_game(self, observation: Any) -> dict[str, Any]:
        canonical = observation_from_host(observation)
        self.state = self.module.initialize(copy.deepcopy(canonical))
        if not isinstance(self.state, dict):
            raise ValueError("initialize must return an object")
        self.pending_state = None
        self.actions = []
        return {
            "worker_id": self.worker_id,
            "metadata": {"solver": "GameplaySolver", "candidate": getattr(self.module, "__name__", "candidate")},
        }

    def decide(self, observation: Any, available_actions: list[str] | tuple[str, ...]) -> Decision:
        if self.state is None:
            raise RuntimeError("decide before start_game")
        observation_from_host(observation)
        decision = select_action(self.module, self.state, available_actions)
        self.pending_state = predict_successor(self.module, self.state, decision.action)
        self.actions.append(decision.action)
        return decision

    def observe(self, observation: Any) -> None:
        if self.state is None or self.pending_state is None:
            raise RuntimeError("observe before decide")
        canonical = observation_from_host(observation)
        self.state = self.module.assimilate(
            copy.deepcopy(self.pending_state),
            canonical,
            {"status": "compatible"},
        )
        if not isinstance(self.state, dict):
            raise ValueError("assimilate must return an object")
        self.pending_state = None
