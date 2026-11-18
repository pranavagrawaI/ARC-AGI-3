"""Native q3c parity plus the real official join, without model calls."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import test_join_qualification as qualification
import test_nooa_join as join
from arcengine import GameAction

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "deployment/kaggle/q6-nooa-candidate"
SOURCE = (
    ROOT
    / "analysis/local-probes/20260905T-stall-authority-q6/patched-source/before/harness.py"
)


def load_host():
    sys.path.insert(0, str(CANDIDATE / "chassis"))
    spec = importlib.util.spec_from_file_location(
        "q6_host", CANDIDATE / "chassis/host_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repaired(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "deployment/kaggle"))
    monkeypatch.setattr(join, "CANDIDATE", CANDIDATE)
    monkeypatch.setenv(
        "NOOA_AGENT_COMMAND",
        json.dumps([sys.executable, str(CANDIDATE / "stub_agent.py")]),
    )
    monkeypatch.setenv("NOOA_AGENT_READY_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("TYCHO_NOOA_WORKER_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def test_observation_and_prediction_semantics_match_frozen_native_harness():
    tree = ast.parse(SOURCE.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("ObservationRecord", "score_predicted_path")
    ]
    namespace = {"RETRODICTION_MIN": 0.6}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    host = load_host()
    native, candidate = namespace["ObservationRecord"](), host.ObservationRecord()
    before = [[0, 0], [0, 0]]
    outcomes = []
    for step in range(8):
        after = [[step % 2, 0], [0, 0]]
        state = "WIN" if step == 7 else "NOT_FINISHED"
        for record in (native, candidate):
            record.observe(before, after)
            label = record.classify(before, after)
            outcome = record.outcome(label, state, step == 7)
            record.note(before, "UP", outcome)
        assert candidate.to_dict(after, ["UP", "DOWN"], {}) == native.to_dict(
            after, ["UP", "DOWN"], {}
        )
        outcomes.append({"outcome": outcome})
        before = after
    assert outcomes[-1]["outcome"] == "terminal"
    assert any(r["outcome"] == "no_impact" for r in outcomes)
    for path in (None, [], [{"expect": "move"}] * 8, [{"expect": "terminal"}] * 8):
        assert host.score_predicted_path(path, outcomes) == namespace[
            "score_predicted_path"
        ](path, outcomes)


def test_level_boundary_publishes_evidence_and_discards_tail(
    tmp_path, repaired, monkeypatch
):
    monkeypatch.setenv("NOOA_STUB_BATCH", "UP,DOWN,LEFT")
    first = qualification.frame([[0, 1], [0, 0]], actions=qualification.MOVES)
    mid = qualification.frame([[0, 1], [1, 0]], actions=qualification.MOVES)
    boundary = qualification.frame(
        [[3, 3], [3, 3]], levels_completed=1, actions=qualification.MOVES
    )
    agent = qualification.build_agent(monkeypatch, tmp_path, first)
    try:
        assert qualification.drive(agent, [first, mid]) == [
            GameAction.ACTION1,
            GameAction.ACTION2,
        ]
        agent.frames.append(boundary)
        assert not agent.is_done(agent.frames, boundary)
        ipc = agent._solver.session_root / "ipc"
        states = [
            json.loads(line) for line in (ipc / "states.jsonl").read_text().splitlines()
        ]
        assert [s["step"] for s in states] == [0, 2]
        assert states[-1]["levels_completed"] == 1
        evidence = json.loads((ipc / "level_0_terminal.json").read_text())
        assert evidence["winning_action"] == "DOWN"
        assert evidence["pre_win_grid_rows"] == ["01", "10"]
        assert evidence["terminal_grid_rows"] == ["33", "33"]
        assert evidence["actions_spent_on_level"] == 2
        assert agent.choose_action(agent.frames, boundary) is GameAction.ACTION1
    finally:
        agent.cleanup(scorecard=None)


def test_no_impact_interrupts_tail_and_records_bad_action(
    tmp_path, repaired, monkeypatch
):
    monkeypatch.setenv("NOOA_STUB_BATCH", "UP,DOWN")
    first = qualification.frame([[0, 1], [0, 0]], actions=qualification.MOVES)
    agent = qualification.build_agent(monkeypatch, tmp_path, first)
    try:
        qualification.drive(agent, [first])
        assert not agent.is_done(agent.frames, first)
        ipc = agent._solver.session_root / "ipc"
        observations = json.loads((ipc / "observations.json").read_text())
        assert observations["known_from_here"]["UP"] == "no_impact"
        assert observations["totals"]["truncations"] == 1
        assert len((ipc / "states.jsonl").read_text().splitlines()) == 2
        assert agent.choose_action(agent.frames, first) is GameAction.ACTION1
    finally:
        agent.cleanup(scorecard=None)


def test_failed_prediction_is_published_before_state(tmp_path):
    host = load_host()
    evidence = host.HostEvidence(tmp_path)
    before = {
        "grid": [[0]],
        "levels_completed": 0,
        "state": "NOT_FINISHED",
        "available_actions": ["UP", "DOWN"],
    }
    evidence.begin(
        {
            "turn": 0,
            "actions": ["UP", "DOWN"],
            "predicted_path": [{"expect": "move"}, {"expect": "move"}],
        }
    )
    assert evidence.observe(before, before, "UP", 1)
    evidence.publish(before, 1)
    published = json.loads((tmp_path / "observations.json").read_text())
    assert published["last_retrodiction"] == {
        "turn": 0,
        "checked": 1,
        "matched": 0,
        "accuracy": 0.0,
        "ok": False,
    }
    assert published["known_from_here"] == {"UP": "no_impact"}


def test_game_over_and_budget_lifecycle(tmp_path, repaired, monkeypatch):
    qualification.test_budget_survives_an_episode_restart_and_the_horizon_stays_truthful(
        tmp_path, None, monkeypatch
    )


def test_dead_worker_fails_closed_without_replaying_actions(tmp_path, repaired):
    module = join.load_create_solver()
    obs = join.observation()
    solver = module.create_solver(
        game_id="worker-death", workspace_root=tmp_path, initial_observation=obs
    )
    try:
        child_pid = solver.worker_metadata["agent_pid"]
        solver._process.kill()
        solver._process.wait(timeout=5)
        import time

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{child_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("Actor survived worker SIGKILL")
        with pytest.raises((RuntimeError, BrokenPipeError)):
            solver.decide(join.turn_for(obs))
        assert not (solver.session_root / "ipc/actions.jsonl").read_text().strip()
    finally:
        solver.close()


def test_rejected_codeact_submission_can_be_repaired_through_real_join(
    tmp_path, repaired, monkeypatch
):
    script = tmp_path / "scripted_actor.py"
    script.write_text("""import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from solver_agent import MdArcSolverAgent
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

def response(code, identity):
    return LLMResponse(raw_response=None, content='',
        tool_calls=[ToolCall(id=identity, name='execute_python', arguments=json.dumps({'code':code}))],
        finish_reason='tool_calls', assistant_message={'role':'assistant','content':''})

async def main():
    fake = FakeLLMClient(scripted_responses=[
        response("self.submit_actions(['NOT_LEGAL'], rationale='invalid attempt')", 'rejected'),
        response("self.submit_actions(['UP'], rationale='repaired')", 'accepted'),
    ])
    agent = MdArcSolverAgent(llm=fake, run_dir=Path.cwd())
    while not (Path('ipc/states.jsonl').stat().st_size):
        await asyncio.sleep(.01)
    await agent.handle({'user_messages':['Make one legal move.']})
asyncio.run(main())
""")
    monkeypatch.setenv(
        "NOOA_AGENT_COMMAND",
        json.dumps([sys.executable, str(script), str(CANDIDATE / "chassis")]),
    )
    monkeypatch.setenv("ARC_STALL_REVIEW", "1")
    solver = join.load_create_solver().create_solver(
        game_id="repair",
        workspace_root=tmp_path / "sessions",
        initial_observation=join.observation(),
    )
    try:
        decision = solver.decide(join.turn_for(join.observation()))
        assert decision.action == "UP"
        ipc = solver.session_root / "ipc"
        assert len((ipc / "actions.jsonl").read_text().splitlines()) == 1
        assert len((ipc / "model_calls.jsonl").read_text().splitlines()) == 2
        assert (
            json.loads((solver.session_root / "launch.json").read_text())["arm"]
            == "treatment"
        )
    finally:
        solver.close()
