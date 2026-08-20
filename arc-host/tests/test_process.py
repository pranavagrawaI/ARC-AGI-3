from __future__ import annotations

import json
import sys
from pathlib import Path

from test_runner import FakeEnvironment, frame

from arc_host.process import JsonlWorker, worker_environment
from arc_host.runner import run_episode


def test_persistent_prefixed_jsonl_process_runs_whole_episode(tmp_path: Path):
    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        """
import json
import os
import sys

HOST = "ARC_HOST\\t"
AGENT = "ARC_AGENT\\t"

def send(kind, **fields):
    payload = {"protocol_version": 1, "type": kind, "pid": os.getpid(), **fields}
    print(AGENT + json.dumps(payload), flush=True)

for line in sys.stdin:
    if not line.startswith(HOST):
        continue
    message = json.loads(line[len(HOST):])
    kind = message["type"]
    if kind == "start_game":
        print("incidental worker output", flush=True)
        send("ready", worker_id=str(os.getpid()), metadata={"kind": "fake"})
    elif kind == "turn":
        send(
            "decision",
            sequence=message["sequence"],
            decision={"action": "UP", "row": None, "col": None},
        )
    elif kind == "action_result":
        send("observed", sequence=message["sequence"])
    elif kind == "finish":
        send("finished")
    elif kind == "close":
        break
""".lstrip(),
        encoding="utf-8",
    )
    environment = FakeEnvironment(
        [frame(0, actions=(1,)), frame(4, actions=(), levels=1, state="WIN")]
    )
    stdout_log = tmp_path / "stdout.log"
    with JsonlWorker(
        [sys.executable, str(worker_script)],
        stdout_log=stdout_log,
        stderr_log=tmp_path / "stderr.log",
        timeout_seconds=2,
    ) as worker:
        result = run_episode(
            environment,
            worker,
            game_id="fake-game",
            seed=0,
            workspace_root=tmp_path / "workspace",
            max_actions=2,
        )

    assert result.actions == 1
    assert result.levels_completed == 1
    assert stdout_log.read_text(encoding="utf-8") == "incidental worker output\n"


def test_recording_solver_writes_plain_structured_evidence(tmp_path: Path):
    from test_runner import FakeSolver

    from arc_host.trace import RecordingSolver
    from arc_host.types import ActionDecision

    trace_path = tmp_path / "trace.jsonl"
    solver = RecordingSolver(FakeSolver([ActionDecision("UP")]), trace_path)
    result = run_episode(
        FakeEnvironment([frame(0, actions=(1,)), frame(4, actions=(), state="WIN")]),
        solver,
        game_id="fake-game",
        seed=0,
        workspace_root=tmp_path / "workspace",
        max_actions=2,
    )

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["type"] for record in records] == [
        "start_game",
        "ready",
        "turn",
        "decision",
        "action_result",
        "finish",
    ]
    assert records[-1]["stop_reason"] == result.stop_reason


def test_worker_environment_does_not_receive_arc_environment_key():
    result = worker_environment(
        {
            "ARC_API_KEY": "environment-secret",
            "ARC_LLM_API_KEY": "model-secret",
            "PATH": "/bin",
            "UNRELATED_SECRET": "no",
        }
    )

    assert result == {"ARC_LLM_API_KEY": "model-secret", "PATH": "/bin"}


def test_worker_environment_forwards_solver_context_and_capture_settings():
    """An allowlist silently drops what it does not name.

    Both of these change what a run does, not just what it reports:
    ARC_LLM_CONTEXT_WINDOW arms the model client's context-limit defence, and
    TYCHO_NOOA_CAPTURE_REASONING makes the run ask the provider for reasoning.
    TYCHO_MODE selects which orchestration topology the solver runs; dropped, a
    comparison arm silently becomes the default `trigger` control and the whole
    run measures nothing. Dropped, they leave a run that looks valid and answers
    nothing.
    """
    result = worker_environment(
        {
            "ARC_LLM_CONTEXT_WINDOW": "131072",
            "ARC_LLM_MAX_TOKENS": "8000",
            "TYCHO_MODE": "no_world_model",
            "TYCHO_NOOA_CAPTURE_REASONING": "1",
        }
    )

    assert result == {
        "ARC_LLM_CONTEXT_WINDOW": "131072",
        "ARC_LLM_MAX_TOKENS": "8000",
        "TYCHO_MODE": "no_world_model",
        "TYCHO_NOOA_CAPTURE_REASONING": "1",
    }
