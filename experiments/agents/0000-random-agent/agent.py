"""Seeded uniform-random ARC contestant."""

from __future__ import annotations

import random

from arcengine import GameAction

from experiments.harness import Decision, Observation, Transition


class RandomAgent:
    """Choose uniformly from the actions exposed by the current observation."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._draws = 0

    def act(self, observation: Observation) -> Decision:
        action = self._rng.choice(observation.available_actions)
        self._draws += 1
        data = None
        if GameAction.from_id(action).is_complex():
            height, width = observation.pixels.shape[-2:]
            data = {
                "x": self._rng.randrange(width),
                "y": self._rng.randrange(height),
            }
        return Decision(
            action=action,
            data=data,
            reasoning={"policy": "uniform_random", "draw": self._draws},
        )

    def observe(self, transition: Transition) -> None:
        pass
