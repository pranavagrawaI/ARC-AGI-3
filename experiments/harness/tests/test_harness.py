from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from experiments.harness import Decision, JsonlTraceWriter
from experiments.harness.runner import IllegalActionError, public_observation, run_episode


class State(str, Enum):
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"


@dataclass
class FakeFrame:
    frame: list[np.ndarray]
    available_actions: list[int]
    levels_completed: int = 0
    state: State = State.NOT_FINISHED
    full_reset: bool = False


class FakeEnvironment:
    def __init__(self, frames: list[FakeFrame]) -> None:
        self.frames = frames
        self.index = 0
        self.actions: list[int] = []
        self.action_data = []
        self.reasoning = []
        self.private_value = "must not cross the boundary"

    def reset(self) -> FakeFrame:
        self.index = 0
        return self.frames[0]

    def step(self, action, data=None, reasoning=None) -> FakeFrame:
        self.actions.append(action.value)
        self.action_data.append(data)
        self.reasoning.append(reasoning)
        self.index += 1
        return self.frames[self.index]


class SequenceAgent:
    def __init__(self, actions: list[int]) -> None:
        self.actions = iter(actions)
        self.observations = []
        self.transitions = []

    def act(self, observation):
        self.observations.append(observation)
        return Decision(next(self.actions), reasoning={"source": "test"})

    def observe(self, transition) -> None:
        self.transitions.append(transition)


def frame(value: int, *, state: State = State.NOT_FINISHED, actions=(1, 2)) -> FakeFrame:
    return FakeFrame(
        frame=[np.full((2, 2), value, dtype=np.int8)],
        available_actions=list(actions),
        levels_completed=int(state is State.WIN),
        state=state,
    )


class HarnessTests(unittest.TestCase):
    def test_public_observation_is_an_isolated_read_only_copy(self) -> None:
        raw = frame(3)
        observation = public_observation(raw)

        raw.frame[0][0, 0] = 9

        self.assertEqual(observation.pixels.shape, (1, 2, 2))
        self.assertEqual(int(observation.pixels[0, 0, 0]), 3)
        self.assertFalse(observation.pixels.flags.writeable)
        with self.assertRaises(ValueError):
            observation.pixels[0, 0, 0] = 7

    def test_runner_delivers_transitions_and_stops_at_terminal(self) -> None:
        env = FakeEnvironment([frame(0), frame(1), frame(2, state=State.WIN)])
        agent = SequenceAgent([1, 2, 1])

        result = run_episode(env, agent, max_actions=2)

        self.assertEqual(env.actions, [1, 2])
        self.assertEqual(result.actions, 2)
        self.assertEqual(result.stop_reason, "terminal")
        self.assertEqual(result.final.state, "WIN")
        self.assertEqual(len(agent.transitions), 2)
        self.assertIs(agent.transitions[0].after, agent.transitions[1].before)

    def test_runner_rejects_an_action_before_stepping(self) -> None:
        env = FakeEnvironment([frame(0), frame(1)])
        agent = SequenceAgent([4])

        with self.assertRaisesRegex(IllegalActionError, "available actions are"):
            run_episode(env, agent, max_actions=1)

        self.assertEqual(env.actions, [])

    def test_runner_forwards_parameterized_action_data(self) -> None:
        env = FakeEnvironment([frame(0, actions=(6,)), frame(1)])

        class PointAgent(SequenceAgent):
            def act(self, observation):
                return Decision(6, data={"x": 12, "y": 34}, reasoning={"probe": "point"})

        run_episode(env, PointAgent([]), max_actions=1)

        self.assertEqual(env.actions, [6])
        self.assertEqual(env.action_data, [{"x": 12, "y": 34}])
        self.assertEqual(env.reasoning, [{"probe": "point"}])

    def test_runner_accepts_reset_and_action7_when_publicly_available(self) -> None:
        env = FakeEnvironment([
            frame(0, actions=(0,)),
            frame(1, actions=(7,)),
            frame(2, state=State.WIN),
        ])

        result = run_episode(env, SequenceAgent([0, 7]), max_actions=2)

        self.assertEqual(env.actions, [0, 7])
        self.assertEqual(result.stop_reason, "terminal")

    def test_runner_rejects_unknown_public_action_id(self) -> None:
        env = FakeEnvironment([frame(0, actions=(99,)), frame(1)])

        with self.assertRaisesRegex(IllegalActionError, "unknown ARC action id 99"):
            run_episode(env, SequenceAgent([99]), max_actions=1)

        self.assertEqual(env.actions, [])

    def test_runner_rejects_missing_parameterized_action_data(self) -> None:
        env = FakeEnvironment([frame(0, actions=(6,)), frame(1)])

        with self.assertRaisesRegex(IllegalActionError, "requires action data"):
            run_episode(env, SequenceAgent([6]), max_actions=1)

        self.assertEqual(env.actions, [])

    def test_action_budget_and_trace_are_reconstructable(self) -> None:
        env = FakeEnvironment([frame(0), frame(1), frame(2)])
        agent = SequenceAgent([1, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with JsonlTraceWriter(path) as trace:
                result = run_episode(env, agent, max_actions=2, trace=trace)
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(result.stop_reason, "action_budget")
        self.assertEqual([record["type"] for record in records], [
            "observation",
            "transition",
            "transition",
            "end",
        ])
        self.assertEqual(records[1]["action"], 1)
        self.assertEqual(records[2]["observation"]["pixels"], [[[2, 2], [2, 2]]])
        self.assertEqual(records[-1]["actions"], 2)

    def test_empty_action_set_stops_without_calling_the_agent(self) -> None:
        env = FakeEnvironment([frame(0, actions=())])
        agent = SequenceAgent([1])

        result = run_episode(env, agent, max_actions=3)

        self.assertEqual(result.stop_reason, "no_available_actions")
        self.assertEqual(agent.observations, [])


if __name__ == "__main__":
    unittest.main()
