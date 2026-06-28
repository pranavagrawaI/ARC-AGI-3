from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.harness import Observation


AGENT_PATH = Path(__file__).resolve().parents[1] / "0000-random-agent/agent.py"
SPEC = importlib.util.spec_from_file_location("experiment_0000_random_agent", AGENT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
RandomAgent = MODULE.RandomAgent


def observation(actions=(1, 2, 3, 4), *, height=8, width=10) -> Observation:
    pixels = np.zeros((1, height, width), dtype=np.int8)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=tuple(actions),
        levels_completed=0,
        state="NOT_FINISHED",
        full_reset=False,
    )


class RandomAgentTests(unittest.TestCase):
    def test_same_seed_reproduces_the_action_sequence(self) -> None:
        left = RandomAgent(seed=17)
        right = RandomAgent(seed=17)
        obs = observation()

        left_actions = [left.act(obs).action for _ in range(20)]
        right_actions = [right.act(obs).action for _ in range(20)]

        self.assertEqual(left_actions, right_actions)

    def test_every_decision_uses_an_available_action(self) -> None:
        agent = RandomAgent(seed=3)
        obs = observation(actions=(2, 4))

        decisions = [agent.act(obs) for _ in range(30)]

        self.assertTrue(all(decision.action in obs.available_actions for decision in decisions))
        self.assertTrue(all(decision.data is None for decision in decisions))

    def test_complex_action_coordinates_stay_inside_the_frame(self) -> None:
        agent = RandomAgent(seed=5)
        obs = observation(actions=(6,), height=7, width=11)

        decisions = [agent.act(obs) for _ in range(20)]

        self.assertTrue(all(0 <= decision.data["x"] < 11 for decision in decisions))
        self.assertTrue(all(0 <= decision.data["y"] < 7 for decision in decisions))


if __name__ == "__main__":
    unittest.main()
