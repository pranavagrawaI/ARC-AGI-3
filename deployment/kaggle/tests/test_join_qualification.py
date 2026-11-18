"""Control-structure qualification for the official join.

Drives scripted lifecycle sequences through the real
``MyAgent -> LocalSolver -> nooa_worker -> stub agent`` path and requires zero
continuity failures and exact action accounting.

Every test here uses the real ``create_solver``. The defect this suite exists for
-- GAME_OVER RESET desynchronising ``LocalSolver`` and killing the official loop
after 131 frames on ``public3-depth`` -- was invisible to the existing adapter
test because that test used a ``FakeSolver`` which does not check continuity. A
double whose process model is weaker than the real component cannot qualify it.

Prerequisite for any Kaggle q6 run: until this passes, a kernel failure cannot be
read as evidence about the solver.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from arcengine import FrameData, GameAction, GameState
from tycho_nooa.protocol import PROTOCOL_VERSION, ActionResultMessage

from test_my_agent import frame, load_agent_module
from test_nooa_join import CANDIDATE, load_create_solver, observation, turn_for

STUB = CANDIDATE / "stub_agent.py"
MOVES = (
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.RESET,
)


@pytest.fixture
def stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOOA_AGENT_COMMAND", json.dumps([sys.executable, str(STUB)]))
    monkeypatch.setenv("NOOA_AGENT_READY_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("TYCHO_NOOA_WORKER_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def build_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: FrameData) -> Any:
    module = load_agent_module(monkeypatch)
    factory = load_create_solver()
    agent = module.MyAgent(
        card_id="card",
        game_id="ls20",
        agent_name="myagent",
        ROOT_URL="http://example.test",
        record=False,
        arc_env=object(),
        solver_factory=factory.create_solver,
        workspace_root=tmp_path / "workspaces",
    )
    agent.frames = [first]
    agent.action_counter = 1
    return agent


def drive(agent: Any, script: list[FrameData]) -> list[GameAction]:
    """Walk the official loop shape: is_done, then choose_action, per frame.

    Any ``LocalSolverError`` escaping here is a qualification failure, not a
    caught condition -- the official loop has no handler for it either.
    """
    emitted: list[GameAction] = []
    for latest in script:
        # The official loop accumulates frames; `cleanup` reconciles the pending
        # decision against `frames[-1]`, so a driver that never appends is not
        # exercising the same control structure.
        if agent.frames and agent.frames[-1] is not latest:
            agent.frames.append(latest)
        if agent.is_done(agent.frames, latest):
            break
        action = agent.choose_action(agent.frames, latest)
        agent.action_counter += 1
        emitted.append(action)
    return emitted


def sessions(tmp_path: Path) -> list[Path]:
    root = tmp_path / "workspaces"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def test_ordinary_play_holds_continuity_across_many_turns(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    script = [first] + [
        frame([[i % 3, 1], [0, i % 2]], actions=MOVES) for i in range(1, 9)
    ]
    agent = build_agent(monkeypatch, tmp_path, first)
    try:
        emitted = drive(agent, script)
        assert len(emitted) == len(script)
        assert all(a is not GameAction.RESET for a in emitted)
        assert agent._solver_decisions_emitted == len(script)
    finally:
        agent.cleanup(scorecard=None)


def test_level_completion_boundary_does_not_break_continuity(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """levels_completed increments; the accepted-history check must reset with it."""
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    script = [
        first,
        frame([[0, 1], [1, 0]], actions=MOVES),
        frame([[3, 3], [3, 3]], levels_completed=1, actions=MOVES),
        frame([[4, 0], [0, 4]], levels_completed=1, actions=MOVES),
        frame([[4, 1], [0, 4]], levels_completed=1, actions=MOVES),
    ]
    agent = build_agent(monkeypatch, tmp_path, first)
    try:
        emitted = drive(agent, script)
        assert len(emitted) == len(script)
        assert agent._level == 1
        # One episode: a level boundary is not an episode boundary.
        assert len(sessions(tmp_path)) == 1
    finally:
        agent.cleanup(scorecard=None)


def test_two_consecutive_game_over_episodes_each_get_a_fresh_worker(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    over = frame([[1, 1], [1, 0]], state=GameState.GAME_OVER, actions=MOVES)
    resumed = frame([[2, 2], [0, 0]], actions=MOVES, full_reset=True)
    over2 = frame([[5, 5], [5, 0]], state=GameState.GAME_OVER, actions=MOVES)
    resumed2 = frame([[6, 6], [0, 0]], actions=MOVES, full_reset=True)
    agent = build_agent(monkeypatch, tmp_path, first)
    try:
        emitted = drive(agent, [first, over, resumed, over2, resumed2, resumed2])
        # Each GAME_OVER costs one RESET that is not a Solver decision.
        assert emitted.count(GameAction.RESET) == 2
        assert agent._solver_decisions_emitted == 4
        assert len(sessions(tmp_path)) == 3
    finally:
        agent.cleanup(scorecard=None)


def test_budget_survives_an_episode_restart_and_the_horizon_stays_truthful(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GAME_OVER restart must not refund actions, and the Solver must be told.

    ``GameTurn`` requires ``actions_used == sequence``, and sequence restarts with
    each worker, so the horizon is session-relative. The advertised limit therefore
    has to shrink by what earlier sessions spent, or the second episode is told it
    has a full budget and ``final_permitted_action`` never fires.
    """
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "2")
    seen: list[Any] = []
    factory = load_create_solver()
    real_create = factory.create_solver

    def recording_create(**kwargs: Any) -> Any:
        solver = real_create(**kwargs)
        real_decide = solver.decide

        def decide(turn: Any) -> Any:
            seen.append(turn)
            return real_decide(turn)

        solver.decide = decide  # type: ignore[method-assign]
        return solver

    module = load_agent_module(monkeypatch)
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    over = frame([[1, 1], [1, 0]], state=GameState.GAME_OVER, actions=MOVES)
    resumed = frame([[2, 2], [0, 0]], actions=MOVES, full_reset=True)
    agent = module.MyAgent(
        card_id="card",
        game_id="ls20",
        agent_name="myagent",
        ROOT_URL="http://example.test",
        record=False,
        arc_env=object(),
        solver_factory=recording_create,
        workspace_root=tmp_path / "workspaces",
    )
    agent.frames = [first]
    agent.action_counter = 1
    try:
        drive(agent, [first, over, resumed, resumed])
        assert agent._solver_decisions_emitted == 2
        assert agent.is_done(agent.frames, resumed) is True
        # Episode 1 turn 0 of 2; episode 2 turn 0 with only one action left.
        assert [(t.action_limit, t.actions_used, t.final_permitted_action) for t in seen] == [
            (2, 0, False),
            (1, 0, True),
        ]
    finally:
        agent.cleanup(scorecard=None)


def test_result_state_is_published_on_observe_not_at_the_next_decide(
    tmp_path: Path, stub_env: None
) -> None:
    """The agent's WAIT must end when the batch executes, not when the host returns.

    ``submit_actions`` sleeps until a new ``states.jsonl`` line appears. If that line
    is only written at the start of the next ``decide``, the agent waits out whatever
    the host does in between -- under the single-generation admission semaphore, the
    other games' entire thinking pass -- and polls a file that cannot change, burning
    one full generation per poll. Asserting a final line count does not catch this;
    the invariant is about *when* the line exists.
    """
    factory = load_create_solver()
    before = observation()
    after = observation(grid=[[0, 0], [0, 1]])
    solver = factory.create_solver(
        game_id="nooa-join-timing",
        workspace_root=tmp_path / "workspaces",
        initial_observation=before,
    )
    states = solver.session_root / "ipc" / "states.jsonl"

    def lines() -> list[str]:
        return [line for line in states.read_text().splitlines() if line.strip()]

    try:
        decision = solver.decide(turn_for(before))
        assert len(lines()) == 1, "turn 0 observation should be published to decide"
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
        published = lines()
        assert len(published) == 2, "observe must publish the result state itself"
        result = json.loads(published[1])
        assert [r["action"] for r in result["action_results"]] == ["UP"]
        assert result["grid_rows"] == ["00", "01"]
    finally:
        solver.close()


def test_win_stops_the_loop(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    win = frame([[9, 9], [9, 9]], state=GameState.WIN, levels_completed=2, actions=MOVES)
    agent = build_agent(monkeypatch, tmp_path, first)
    try:
        drive(agent, [first, win, win])
        assert agent.is_done(agent.frames, win) is True
    finally:
        agent.cleanup(scorecard=None)


def test_action_budget_is_exact_and_stops_the_loop(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Solver cap counts Solver decisions, not official actions."""
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "3")
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    script = [first] + [frame([[i, 1], [0, 0]], actions=MOVES) for i in range(1, 8)]
    agent = build_agent(monkeypatch, tmp_path, first)
    try:
        assert agent._solver_action_limit == 3
        assert agent.MAX_ACTIONS == 7  # 2 * limit + 1 official-counter backstop
        emitted = drive(agent, script)
        assert agent._solver_decisions_emitted == 3
        assert len(emitted) == 3
    finally:
        agent.cleanup(scorecard=None)


def test_batch_is_interrupted_by_a_level_boundary(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued tail must not be released against a grid from the next level."""
    monkeypatch.setenv("NOOA_STUB_BATCH", "UP,DOWN,LEFT")
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    script = [
        first,
        frame([[0, 1], [1, 0]], actions=MOVES),
        frame([[3, 3], [3, 3]], levels_completed=1, actions=MOVES),
        frame([[4, 0], [0, 4]], levels_completed=1, actions=MOVES),
    ]
    agent = build_agent(monkeypatch, tmp_path, first)
    try:
        emitted = drive(agent, script)
        assert len(emitted) == len(script)
        assert agent._level == 1
    finally:
        agent.cleanup(scorecard=None)


def test_cleanup_is_idempotent_after_an_episode_ended(
    tmp_path: Path, stub_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = frame([[0, 1], [0, 0]], actions=MOVES)
    over = frame([[1, 1], [1, 0]], state=GameState.GAME_OVER, actions=MOVES)
    agent = build_agent(monkeypatch, tmp_path, first)
    drive(agent, [first, over])
    agent.cleanup(scorecard=None)
    agent.cleanup(scorecard=None)
