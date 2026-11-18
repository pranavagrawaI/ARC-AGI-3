import json
import ast
import hashlib
import shutil
import sys
from pathlib import Path

import pytest
from arcengine import FrameData, FrameDataRaw
import test_join_qualification as qualification
import test_nooa_join as join

OUT = Path(__file__).resolve().parent
PAYLOAD = OUT / "extracted"


def setup(monkeypatch):
    monkeypatch.setenv(
        "NOOA_AGENT_COMMAND",
        json.dumps([sys.executable, str(PAYLOAD / "stub_agent.py")]),
    )
    monkeypatch.setenv("NOOA_AGENT_READY_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("TYCHO_NOOA_WORKER_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "200")


def test_real_ls20_full_budget(tmp_path, monkeypatch):
    from arc_agi import Arcade, OperationMode

    setup(monkeypatch)
    monkeypatch.setenv("NOOA_STUB_BATCH", "UP,DOWN,LEFT,RIGHT")
    root = OUT.parents[2]
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(root / "environment_files"),
        recordings_dir=str(tmp_path / "recordings"),
    )
    env = arcade.make("ls20")
    assert env is not None
    source = (
        root
        / "analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/ARC-AGI-3-Agents/agents/agent.py"
    )
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_convert_raw_frame_data"
    )
    namespace = {"FrameData": FrameData, "FrameDataRaw": FrameDataRaw}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    def convert(raw):
        return namespace["_convert_raw_frame_data"](None, raw)
    first = convert(env.reset())
    agent = qualification.build_agent(monkeypatch, tmp_path, first)
    emitted = []
    try:
        current = first
        for _ in range(402):
            if agent.is_done(agent.frames, current):
                break
            action = agent.choose_action(agent.frames, current)
            agent.action_counter += 1
            emitted.append(action.name)
            current = convert(env.step(action))
            assert current is not None
            agent.frames.append(current)
        else:
            pytest.fail("adapter exceeded its full reset-inclusive horizon")
        assert agent._solver_decisions_emitted == 200
    finally:
        agent.cleanup(scorecard=None)
    sessions = qualification.sessions(tmp_path)
    transitions = [
        json.loads(line)
        for session in sessions
        for line in (session / "ipc/transitions.jsonl").read_text().splitlines()
    ]
    assert len(transitions) == 200
    assert (
        sum(
            len((p / "ipc/transitions.jsonl").read_text().splitlines())
            for p in sessions
        )
        == agent._solver_decisions_emitted
    )
    launches = [json.loads((p / "launch.json").read_text()) for p in sessions]
    assert len({p["run_id"] for p in launches}) == 1
    assert len({p["session_id"] for p in launches}) == len(sessions)
    assert all((p / "stop.json").exists() for p in sessions)
    report = {
        "decisions": 200,
        "environment_calls": len(emitted),
        "resets": emitted.count("RESET"),
        "sessions": len(sessions),
        "levels_completed": current.levels_completed,
        "session_paths": [str(p) for p in sessions],
        "policy": "deterministic stub; no model",
    }
    (OUT / "LS20.json").write_text(json.dumps(report, indent=2) + "\n")


def test_corrupt_extracted_worker_payload_refuses_startup(tmp_path, monkeypatch):
    setup(monkeypatch)
    corrupted = tmp_path / "corrupted"
    shutil.copytree(PAYLOAD, corrupted)
    with (corrupted / "chassis/stall_contract.py").open("a") as handle:
        handle.write("\n# injected byte drift\n")
    monkeypatch.setattr(join, "CANDIDATE", corrupted)
    with pytest.raises(RuntimeError, match="candidate drift"):
        join.load_create_solver().create_solver(
            game_id="ls20",
            workspace_root=tmp_path / "sessions",
            initial_observation=join.observation(),
        )
