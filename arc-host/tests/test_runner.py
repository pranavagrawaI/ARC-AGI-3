from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arc_host.frames import public_observation
from arc_host.runner import IllegalActionError, run_episode
from arc_host.types import (
    ActionDecision,
    EpisodeResult,
    GameStart,
    Transition,
    Turn,
    WorkerReady,
)


def frame(
    value: int,
    *,
    actions: tuple[int, ...] = (1, 6),
    levels: int = 0,
    state: str = "NOT_FINISHED",
    full_reset: bool = False,
):
    return SimpleNamespace(
        frame=np.asarray([[[value, value + 1], [value + 2, value + 3]]]),
        available_actions=actions,
        levels_completed=levels,
        state=state,
        full_reset=full_reset,
    )


class FakeEnvironment:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.steps = []

    def reset(self):
        return next(self.frames)

    def step(self, action, data=None, reasoning=None):
        self.steps.append((action.value, data))
        return next(self.frames)


@dataclass
class FakeSolver:
    decisions: list[ActionDecision]

    def __post_init__(self):
        self.starts: list[GameStart] = []
        self.turns: list[Turn] = []
        self.transitions: list[Transition] = []
        self.results: list[EpisodeResult] = []

    def start_game(self, start):
        self.starts.append(start)
        return WorkerReady("fake", {})

    def decide(self, turn):
        self.turns.append(turn)
        return self.decisions[len(self.turns) - 1]

    def observe(self, transition):
        self.transitions.append(transition)

    def finish(self, result):
        self.results.append(result)

    def close(self):
        pass


def test_host_translates_click_and_preserves_one_solver(tmp_path: Path):
    environment = FakeEnvironment([frame(0), frame(4, state="WIN")])
    solver = FakeSolver([ActionDecision("CLICK", row=1, col=0)])

    result = run_episode(
        environment,
        solver,
        game_id="fake-game",
        seed=7,
        workspace_root=tmp_path,
        max_actions=3,
    )

    assert result.stop_reason == "terminal"
    assert result.actions == 1
    assert environment.steps == [(6, {"x": 0, "y": 1})]
    assert solver.starts[0].initial_observation.available_actions == (
        "RESET",
        "UP",
        "CLICK",
    )
    assert solver.turns[0].observation.grid == ((0, 1), (2, 3))
    assert solver.transitions[0].after.grid == ((4, 5), (6, 7))
    assert solver.results == [result]


def test_level_boundary_resets_level_turn_and_previous_action(tmp_path: Path):
    environment = FakeEnvironment(
        [
            frame(0, actions=(0, 1)),
            frame(4, actions=(2,), levels=1),
            frame(8, actions=(), levels=1, state="WIN"),
        ]
    )
    solver = FakeSolver([ActionDecision("UP"), ActionDecision("DOWN")])

    result = run_episode(
        environment,
        solver,
        game_id="fake-game",
        seed=0,
        workspace_root=tmp_path,
        max_actions=3,
    )

    assert result.stop_reason == "terminal"
    assert [(turn.level, turn.turn_in_level) for turn in solver.turns] == [
        (0, 0),
        (1, 0),
    ]
    assert solver.turns[1].previous_decision is None


def test_level_limit_wins_when_successor_has_no_actions(tmp_path: Path):
    environment = FakeEnvironment(
        [frame(0, actions=(1,)), frame(4, actions=(), levels=1)]
    )
    solver = FakeSolver([ActionDecision("UP")])

    result = run_episode(
        environment,
        solver,
        game_id="fake-game",
        seed=0,
        workspace_root=tmp_path,
        max_actions=3,
        stop_after_levels=1,
    )

    assert result.stop_reason == "level_limit"
    assert result.levels_completed == 1


def test_illegal_model_action_never_reaches_environment(tmp_path: Path):
    environment = FakeEnvironment([frame(0, actions=(1,))])
    solver = FakeSolver([ActionDecision("DOWN")])

    with pytest.raises(IllegalActionError, match="available actions"):
        run_episode(
            environment,
            solver,
            game_id="fake-game",
            seed=0,
            workspace_root=tmp_path,
            max_actions=1,
        )

    assert environment.steps == []


def test_public_observation_rejects_out_of_palette_cells():
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        public_observation(frame(15))


def test_completing_frame_sequence_preserves_terminal_transient_and_settled_frames():
    completing = SimpleNamespace(
        frame=np.asarray([[[1]], [[2]], [[3]]]),
        available_actions=(1,),
        levels_completed=1,
        state="NOT_FINISHED",
        full_reset=False,
    )

    evidence = public_observation(completing, previous_levels_completed=0)

    assert evidence.decision_frame == ((3,),)
    assert evidence.terminal_frame == ((1,),)
    assert [event.kind for event in evidence.event_frames] == ["animation"]
    assert [event.source_index for event in evidence.event_frames] == [1]
    assert [event.frame for event in evidence.event_frames] == [((2,),)]
    assert evidence.provenance.raw_frame_count == 3
    assert evidence.provenance.decision_frame_index == 2
    assert evidence.provenance.terminal_frame_index == 0
    assert evidence.provenance.terminal_reason == "level_advanced"


def test_reset_is_always_exposed_and_starts_a_fresh_attempt_turn(tmp_path: Path):
    environment = FakeEnvironment(
        [
            frame(0, actions=(1,)),
            frame(4, actions=(1,), full_reset=True),
            frame(8, actions=(1,)),
        ]
    )
    solver = FakeSolver([ActionDecision("RESET"), ActionDecision("UP")])

    result = run_episode(
        environment,
        solver,
        game_id="fake-game",
        seed=0,
        workspace_root=tmp_path,
        max_actions=2,
    )

    assert result.stop_reason == "action_budget"
    assert result.actions == 2
    assert environment.steps == [(0, None), (1, None)]
    assert solver.starts[0].initial_observation.available_actions == ("RESET", "UP")
    assert [(turn.level, turn.turn_in_level) for turn in solver.turns] == [
        (0, 0),
        (0, 0),
    ]
    assert solver.turns[1].previous_decision is None
