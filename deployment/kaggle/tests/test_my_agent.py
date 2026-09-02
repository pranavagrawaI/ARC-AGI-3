from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from arcengine import FrameData, GameAction, GameState

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import (
    ActionResultMessage,
    ObservationPayload,
    project_arc_observation,
)

AGENT_PATH = Path(__file__).parents[1] / "agent" / "my_agent.py"


class StubAgent:
    MAX_ACTIONS = 80

    def __init__(
        self,
        *,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        arc_env: object,
        tags: list[str] | None = None,
    ) -> None:
        self.card_id = card_id
        self.game_id = game_id
        self.agent_name = agent_name
        self.ROOT_URL = ROOT_URL
        self.arc_env = arc_env
        self.tags = tags or []
        self.frames = [FrameData(levels_completed=0, win_levels=0)]
        self.action_counter = 0
        self.base_cleanup_calls = 0

    @property
    def name(self) -> str:
        return f"{self.game_id}.{type(self).__name__.lower()}"

    def cleanup(self, scorecard: object | None = None) -> None:
        self.base_cleanup_calls += 1


class FakeSolver:
    def __init__(self, decisions: list[ActionDecision]) -> None:
        self.decisions = decisions
        self.events: list[tuple[str, GameTurn | ActionResultMessage]] = []
        self.finished: list[tuple[ObservationPayload, int]] = []
        self.close_calls = 0

    def decide(self, turn: GameTurn) -> ActionDecision:
        self.events.append(("decide", turn))
        return self.decisions.pop(0)

    def observe(self, result: ActionResultMessage) -> None:
        self.events.append(("observe", result))

    def finish(self, final_observation: ObservationPayload, actions: int) -> None:
        self.finished.append((final_observation, actions))

    def close(self) -> None:
        self.close_calls += 1


class FinishFailingSolver(FakeSolver):
    def finish(self, final_observation: ObservationPayload, actions: int) -> None:
        raise RuntimeError("finish failed")


class RecordingFactory:
    def __init__(self, solver: FakeSolver) -> None:
        self.solver = solver
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        game_id: str,
        workspace_root: Path,
        initial_observation: ObservationPayload,
    ) -> FakeSolver:
        self.calls.append(
            {
                "game_id": game_id,
                "workspace_root": workspace_root,
                "initial_observation": initial_observation,
            }
        )
        return self.solver


def load_agent_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    agents = ModuleType("agents")
    agent_module = ModuleType("agents.agent")
    setattr(agent_module, "Agent", StubAgent)
    setattr(agents, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.agent", agent_module)

    spec = importlib.util.spec_from_file_location("tested_kaggle_agent", AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(
    *grids: list[list[int]],
    state: GameState = GameState.NOT_FINISHED,
    levels_completed: int = 0,
    actions: tuple[GameAction, ...] = (GameAction.ACTION1, GameAction.ACTION6),
    full_reset: bool = False,
) -> FrameData:
    return FrameData(
        game_id="game",
        frame=list(grids),
        state=state,
        levels_completed=levels_completed,
        win_levels=2,
        available_actions=[action.value for action in actions],
        full_reset=full_reset,
    )


def new_agent(module: ModuleType, factory: RecordingFactory, tmp_path: Path) -> Any:
    return module.MyAgent(
        card_id="card",
        game_id="game",
        agent_name="myagent",
        ROOT_URL="http://gateway",
        record=False,
        arc_env=object(),
        solver_factory=factory,
        workspace_root=tmp_path,
    )


def test_launcher_environment_owns_the_default_workspace_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "launcher-owned-solver-workspaces"
    monkeypatch.setenv("TYCHO_KAGGLE_WORKSPACE_ROOT", str(workspace_root))
    module = load_agent_module(monkeypatch)
    factory = RecordingFactory(FakeSolver([ActionDecision(action="UP")]))
    agent = module.MyAgent(
        card_id="card",
        game_id="game",
        agent_name="myagent",
        ROOT_URL="http://gateway",
        record=False,
        arc_env=object(),
        solver_factory=factory,
    )

    assert agent.choose_action(agent.frames, frame([[1]])) is GameAction.ACTION1
    assert factory.calls[0]["workspace_root"] == workspace_root


def test_action_cap_is_explicitly_runtime_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "1")
    module = load_agent_module(monkeypatch)
    agent = new_agent(module, RecordingFactory(FakeSolver([])), tmp_path)

    assert agent._solver_action_limit == 1
    assert agent.MAX_ACTIONS == 3
    assert agent.name.endswith(".1")


def test_action_cap_counts_solver_decisions_not_official_loop_iterations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "1")
    module = load_agent_module(monkeypatch)
    solver = FakeSolver(
        [ActionDecision(action="UP"), ActionDecision(action="CLICK", row=7, col=9)]
    )
    agent = new_agent(module, RecordingFactory(solver), tmp_path)

    assert agent.choose_action(agent.frames, frame([[1]])) is GameAction.ACTION1
    assert agent.is_done(agent.frames, frame([[2]])) is True
    assert [kind for kind, _ in solver.events] == ["decide", "observe"]


@pytest.mark.parametrize("action_limit", (32, 80))
def test_each_solver_turn_carries_the_exact_remaining_action_horizon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action_limit: int,
) -> None:
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", str(action_limit))
    module = load_agent_module(monkeypatch)
    solver = FakeSolver(
        [ActionDecision(action="UP") for _ in range(action_limit)]
    )
    agent = new_agent(module, RecordingFactory(solver), tmp_path)

    for action_index in range(action_limit):
        assert (
            agent.choose_action(agent.frames, frame([[action_index % 16]]))
            is GameAction.ACTION1
        )

    turns = [event for kind, event in solver.events if kind == "decide"]
    assert len(turns) == action_limit
    for actions_used, actions_remaining, final in (
        (action_limit - 2, 2, False),
        (action_limit - 1, 1, True),
    ):
        turn = turns[actions_used]
        assert isinstance(turn, GameTurn)
        assert turn.sequence == actions_used
        assert turn.action_limit == action_limit
        assert turn.actions_used == actions_used
        assert turn.actions_remaining == actions_remaining
        assert turn.final_permitted_action is final


def test_reset_bootstrap_does_not_consume_the_solver_action_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "1")
    module = load_agent_module(monkeypatch)
    solver = FakeSolver([ActionDecision(action="UP")])
    agent = new_agent(module, RecordingFactory(solver), tmp_path)

    reset = agent.choose_action(
        agent.frames,
        frame([[0]], state=GameState.NOT_PLAYED),
    )
    decision = agent.choose_action(agent.frames, frame([[1]]))

    assert reset is GameAction.RESET
    assert decision is GameAction.ACTION1
    assert agent.is_done(agent.frames, frame([[2]])) is True
    assert [kind for kind, _ in solver.events] == ["decide", "observe"]


def test_game_over_reset_does_not_let_the_official_backstop_truncate_solver_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "2")
    module = load_agent_module(monkeypatch)
    solver = FakeSolver(
        [ActionDecision(action="UP"), ActionDecision(action="UP")]
    )
    agent = new_agent(module, RecordingFactory(solver), tmp_path)

    assert agent.MAX_ACTIONS == 5
    assert agent.choose_action(agent.frames, frame([[1]])) is GameAction.ACTION1
    game_over = frame([[2]], state=GameState.GAME_OVER)
    assert agent.choose_action(agent.frames, game_over) is GameAction.RESET
    assert agent.choose_action(agent.frames, frame([[3]])) is GameAction.ACTION1
    assert agent.is_done(agent.frames, frame([[4]])) is True
    assert [kind for kind, _ in solver.events] == [
        "decide",
        "observe",
        "decide",
        "observe",
    ]


def test_projection_preserves_terminal_event_and_decision_roles() -> None:
    observation = project_arc_observation(
        frames=[[[1]], [[2]], [[3]]],
        available_action_ids=[1, 6],
        state="NOT_FINISHED",
        levels_completed=1,
        full_reset=False,
        previous_levels_completed=0,
    )

    assert observation.decision_frame == [[3]]
    assert observation.terminal_frame == [[1]]
    assert [event.frame for event in observation.event_frames] == [[[2]]]
    assert observation.provenance.model_dump() == {
        "raw_frame_count": 3,
        "decision_frame_index": 2,
        "terminal_frame_index": 0,
        "event_frame_indices": [1],
        "terminal_reason": "level_advanced",
    }
    assert observation.available_actions == ["RESET", "UP", "CLICK"]


def test_one_solver_closes_each_action_before_next_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_agent_module(monkeypatch)
    solver = FakeSolver(
        [ActionDecision(action="UP"), ActionDecision(action="CLICK", row=7, col=9)]
    )
    factory = RecordingFactory(solver)
    agent = new_agent(module, factory, tmp_path)

    first_action = agent.choose_action(agent.frames, frame([[1]]))
    second_action = agent.choose_action(agent.frames, frame([[4]], [[5]]))

    assert first_action is GameAction.ACTION1
    assert second_action is GameAction.ACTION6
    action_data = second_action.action_data.model_dump()
    assert {key: action_data[key] for key in ("x", "y")} == {"x": 9, "y": 7}
    assert len(factory.calls) == 1
    assert factory.calls[0]["initial_observation"].decision_frame == [[1]]
    assert [kind for kind, _ in solver.events] == ["decide", "observe", "decide"]

    first_turn = solver.events[0][1]
    first_result = solver.events[1][1]
    second_turn = solver.events[2][1]
    assert isinstance(first_turn, GameTurn)
    assert isinstance(first_result, ActionResultMessage)
    assert isinstance(second_turn, GameTurn)
    assert (first_turn.sequence, first_turn.level, first_turn.turn_in_level) == (0, 0, 0)
    assert first_result.decision == ActionDecision(action="UP")
    assert first_result.before.decision_frame == [[1]]
    assert first_result.after.decision_frame == [[5]]
    assert [event.frame for event in first_result.after.event_frames] == [[[4]]]
    assert (second_turn.sequence, second_turn.level, second_turn.turn_in_level) == (
        1,
        0,
        1,
    )
    assert (second_turn.previous_action, second_turn.previous_row, second_turn.previous_col) == (
        "UP",
        None,
        None,
    )

    winning = frame([[8]], state=GameState.WIN, levels_completed=1)
    assert agent.is_done(agent.frames, winning) is True
    assert [kind for kind, _ in solver.events] == [
        "decide",
        "observe",
        "decide",
        "observe",
    ]
    final_result = solver.events[-1][1]
    assert isinstance(final_result, ActionResultMessage)
    assert final_result.after.state == "WIN"
    assert final_result.after.terminal_frame == [[8]]

    agent.frames.append(winning)
    agent.action_counter = 2
    agent.cleanup()
    assert solver.finished == [(final_result.after, 2)]
    assert solver.close_calls == 1
    assert agent.base_cleanup_calls == 1


def test_cleanup_closes_solver_when_finish_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_agent_module(monkeypatch)
    solver = FinishFailingSolver([ActionDecision(action="UP")])
    factory = RecordingFactory(solver)
    agent = new_agent(module, factory, tmp_path)

    agent.choose_action(agent.frames, frame([[1]]))
    successor = frame([[2]])
    agent.is_done(agent.frames, successor)
    agent.frames.append(successor)
    agent.action_counter = 1

    with pytest.raises(RuntimeError, match="finish failed"):
        agent.cleanup()

    assert solver.close_calls == 1
    assert agent.base_cleanup_calls == 1


def test_illegal_solver_action_fails_before_environment_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_agent_module(monkeypatch)
    factory = RecordingFactory(FakeSolver([ActionDecision(action="LEFT")]))
    agent = new_agent(module, factory, tmp_path)

    with pytest.raises(ValueError, match="not available"):
        agent.choose_action(
            agent.frames,
            frame([[1]], actions=(GameAction.ACTION1,)),
        )


@pytest.mark.parametrize("state", [GameState.NOT_PLAYED, GameState.GAME_OVER])
def test_non_playing_state_resets_before_constructing_solver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: GameState
) -> None:
    module = load_agent_module(monkeypatch)
    solver = FakeSolver([ActionDecision(action="UP")])
    factory = RecordingFactory(solver)
    agent = new_agent(module, factory, tmp_path)

    action = agent.choose_action(
        agent.frames,
        FrameData(state=state, levels_completed=0, win_levels=2),
    )

    assert action is GameAction.RESET
    assert factory.calls == []
    assert solver.events == []


def test_missing_local_solver_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_agent_module(monkeypatch)
    agent = module.MyAgent(
        card_id="card",
        game_id="game",
        agent_name="myagent",
        ROOT_URL="http://gateway",
        record=False,
        arc_env=object(),
        workspace_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="local Solver"):
        agent.choose_action(agent.frames, frame([[1]]))
