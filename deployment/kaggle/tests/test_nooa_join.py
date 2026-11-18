from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from arcengine import GameAction, GameState
from tycho_nooa.models import GameTurn
from tycho_nooa.protocol import (
    PROTOCOL_VERSION,
    ActionResultMessage,
    project_arc_observation,
)

from test_my_agent import frame, load_agent_module

CANDIDATE = Path(__file__).resolve().parents[1] / "s0-nooa-join-candidate"
STUB = CANDIDATE / "stub_agent.py"


def load_create_solver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "s0_nooa_join_local_solver", CANDIDATE / "local_solver.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(*, grid: list[list[int]] | None = None):
    return project_arc_observation(
        frames=grid or [[0, 1], [0, 0]],
        available_action_ids=[1, 2, 3, 4],
        state="NOT_FINISHED",
        levels_completed=0,
        full_reset=False,
    )


def turn_for(obs, sequence: int = 0) -> GameTurn:
    return GameTurn(
        grid=obs.grid,
        available_actions=obs.available_actions,
        sequence=sequence,
        level=0,
        turn_in_level=sequence,
        levels_completed=0,
        state=obs.state,
    )


@pytest.fixture
def stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "NOOA_AGENT_COMMAND", json.dumps([sys.executable, str(STUB)])
    )
    monkeypatch.setenv("NOOA_AGENT_READY_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("TYCHO_NOOA_WORKER_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def test_stub_agent_returns_typed_up_through_protocol_v3(
    tmp_path: Path, stub_env: None
) -> None:
    module = load_create_solver()
    before = observation()
    after = observation(grid=[[0, 0], [0, 1]])
    solver = module.create_solver(
        game_id="nooa-join",
        workspace_root=tmp_path / "workspaces",
        initial_observation=before,
    )
    try:
        decision = solver.decide(turn_for(before))
        assert decision.action == "UP"
        assert decision.row is None and decision.col is None
        assert solver.worker_metadata["chassis"] == "q3c-mdfiles-after"
        assert solver.worker_metadata["join"] == "s0-nooa-join"
        solver.observe(
            ActionResultMessage(
                protocol_version=PROTOCOL_VERSION,
                type="action_result",
                sequence=0,
                level=0,
                turn_in_level=0,
                decision=decision,
                before=before,
                after=after,
            )
        )
        solver.finish(after, actions=1)
        state_lines = (
            (solver.session_root / "ipc" / "states.jsonl").read_text().strip().splitlines()
        )
        actions = (solver.session_root / "ipc" / "actions.jsonl").read_text().strip()
        assert json.loads(actions)["actions"] == ["UP"]
        # Two lines: the turn-0 observation, and the result state published by
        # `observe` the moment the batch finished. The second line is what ends the
        # agent's WAIT. Publishing it only at the next `decide` left the agent
        # polling a frozen file, once per generation, for the whole time the host
        # spent inside other games.
        assert len(state_lines) == 2
        first, result = (json.loads(line) for line in state_lines)
        assert first["grid_rows"] == ["01", "00"]
        assert first["action_results"] == []
        assert result["turn"] == 1
        assert result["step"] == 1
        assert result["grid_rows"] == ["00", "01"]
        assert [r["action"] for r in result["action_results"]] == ["UP"]
    finally:
        solver.close()


def test_batch_is_drained_without_a_second_state_line(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOOA_STUB_BATCH", "UP,DOWN")
    module = load_create_solver()
    first = observation()
    mid = observation(grid=[[1, 1], [0, 0]])
    last = observation(grid=[[1, 1], [1, 0]])
    solver = module.create_solver(
        game_id="nooa-join-batch",
        workspace_root=tmp_path / "workspaces",
        initial_observation=first,
    )
    try:
        up = solver.decide(turn_for(first, 0))
        assert up.action == "UP"
        solver.observe(
            ActionResultMessage(
                protocol_version=PROTOCOL_VERSION,
                type="action_result",
                sequence=0,
                level=0,
                turn_in_level=0,
                decision=up,
                before=first,
                after=mid,
            )
        )
        down = solver.decide(
            GameTurn(
                grid=mid.grid,
                available_actions=mid.available_actions,
                sequence=1,
                level=0,
                turn_in_level=1,
                levels_completed=0,
                state=mid.state,
                previous_action="UP",
            )
        )
        assert down.action == "DOWN"
        states_after_second_decide = (
            solver.session_root / "ipc" / "states.jsonl"
        ).read_text().strip().splitlines()
        assert len(states_after_second_decide) == 1
        solver.observe(
            ActionResultMessage(
                protocol_version=PROTOCOL_VERSION,
                type="action_result",
                sequence=1,
                level=0,
                turn_in_level=1,
                decision=down,
                before=mid,
                after=last,
            )
        )
        solver.finish(last, actions=2)
        actions = [
            json.loads(line)
            for line in (solver.session_root / "ipc" / "actions.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        assert len(actions) == 1
        assert actions[0]["actions"] == ["UP", "DOWN"]
    finally:
        solver.close()


def test_my_agent_commits_one_official_action(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_agent_module(monkeypatch)
    factory_module = load_create_solver()
    moves = (
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
    )
    first = frame([[0, 1], [0, 0]], actions=moves)
    successor = frame([[0, 0], [0, 1]], actions=moves)
    agent = module.MyAgent(
        card_id="card",
        game_id="game",
        agent_name="myagent",
        ROOT_URL="http://example.test",
        record=False,
        arc_env=object(),
        solver_factory=factory_module.create_solver,
        workspace_root=tmp_path / "workspaces",
    )
    agent.frames = [first]
    try:
        action = agent.choose_action(agent.frames, first)
        assert action is GameAction.ACTION1
        agent.is_done(agent.frames, successor)
        agent.frames.append(successor)
        agent.action_counter = 1
    finally:
        agent.cleanup(scorecard=None)


def test_game_over_reset_does_not_raise_turn_continuity(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth v1 died here: GAME_OVER then NOT_FINISHED vs LocalSolver's last grid."""
    module = load_agent_module(monkeypatch)
    factory_module = load_create_solver()
    moves = (
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.RESET,
    )
    first = frame([[0, 1], [0, 0]], actions=moves)
    game_over = frame([[1, 1], [1, 0]], state=GameState.GAME_OVER, actions=moves)
    after_reset = frame([[2, 2], [0, 0]], actions=moves, full_reset=True)
    agent = module.MyAgent(
        card_id="card",
        game_id="ls20",
        agent_name="myagent",
        ROOT_URL="http://example.test",
        record=False,
        arc_env=object(),
        solver_factory=factory_module.create_solver,
        workspace_root=tmp_path / "workspaces",
    )
    agent.frames = [first]
    agent.action_counter = 1
    try:
        assert agent.choose_action(agent.frames, first) is GameAction.ACTION1
        assert agent.choose_action(agent.frames, game_over) is GameAction.RESET
        resumed = agent.choose_action(agent.frames, after_reset)
        assert resumed is GameAction.ACTION1
        sessions = [
            path
            for path in (tmp_path / "workspaces").iterdir()
            if path.is_dir()
        ]
        assert len(sessions) == 2
    finally:
        agent.cleanup(scorecard=None)


def test_wrapper_assigns_hardened_host_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1 died here: HARDENED_HOST=1 and no GAME_UID. The wrapper must assign both."""
    captured: dict = {}

    class FakeSolver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = load_create_solver()
    monkeypatch.setattr(module, "LocalSolver", FakeSolver)
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setenv("TYCHO_KAGGLE_HARDENED_HOST", "1")
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    module.create_solver(
        game_id="ls20",
        workspace_root=tmp_path / "workspaces",
        initial_observation=observation(),
    )
    environment = captured["environment"]
    assert environment["TYCHO_KAGGLE_GAME_UID"] == environment["TYCHO_KAGGLE_GAME_GID"]
    assert int(environment["TYCHO_KAGGLE_GAME_UID"]) >= 60_001
    assert str(captured["command"][-1]).endswith("nooa_worker.py")
