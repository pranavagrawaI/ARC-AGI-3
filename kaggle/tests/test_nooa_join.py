#!/usr/bin/env python3
"""Exercise the NOOA-upstream join without a GPU, a model, or a Kaggle kernel.

Two checks:

1. ``stub`` — the worker's harness role: a scripted agent writing one
   ``submit_actions`` batch to ``ipc/actions.jsonl`` becomes one protocol-v3
   decision, and a multi-action batch is drained one official turn at a time
   without a second state line.
2. ``boot`` — the pristine upstream ``MdArcSolverAgent`` actually starts under the
   offline wheelhouse: the two ``arc_llm.py`` deployment deltas hold, the pristine
   51-line skill loads, the agent writes its ``agent_ready`` marker, and it issues
   a first request to the configured gateway.

Run with the wheelhouse venv:

    refvenv/bin/python kaggle/tests/test_nooa_join.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = HERE.parent / "nooa-upstream"
WORKER = ARM / "worker"
sys.path.insert(0, str(WORKER))

from tycho_nooa.protocol import (  # noqa: E402
    HOST_PREFIX,
    PROTOCOL_VERSION,
    WORKER_PREFIX,
    project_arc_observation,
)

GAME = "ls20"
GRID = [[(row + col) % 16 for col in range(64)] for row in range(64)]
SETTLED = [[(row + col + 1) % 16 for col in range(64)] for row in range(64)]


def observation(frames, *, state="NOT_FINISHED", levels=0):
    return project_arc_observation(
        frames=frames,
        available_action_ids=[1, 2, 3, 4, 5, 6],
        state=state,
        levels_completed=levels,
        full_reset=False,
    )


def send(process, payload: dict) -> None:
    message = {"protocol_version": PROTOCOL_VERSION, **payload}
    process.stdin.write(HOST_PREFIX + json.dumps(message) + "\n")
    process.stdin.flush()


def receive(process, expected: str) -> dict:
    while True:
        line = process.stdout.readline()
        if not line:
            raise AssertionError(f"worker exited before {expected}")
        if not line.startswith(WORKER_PREFIX):
            print("  worker stdout:", line.rstrip())
            continue
        payload = json.loads(line[len(WORKER_PREFIX) :])
        if payload.get("type") == "error":
            raise AssertionError(f"worker error waiting for {expected}: {payload}")
        assert payload["type"] == expected, payload
        return payload


def turn_message(sequence: int, payload, *, used: int, previous=None) -> dict:
    return {
        "type": "turn",
        "sequence": sequence,
        "action_limit": 16,
        "actions_used": used,
        "actions_remaining": 16 - used,
        "final_permitted_action": False,
        "level": 0,
        "turn_in_level": used,
        "decision_timeout_seconds": 60.0,
        "observation": payload.model_dump(mode="json"),
        "previous_decision": previous,
    }


# --------------------------------------------------------------------------
# 1. The worker's harness role, with a scripted agent
# --------------------------------------------------------------------------

SCRIPTED_AGENT = '''#!/usr/bin/env python3
"""Two fixed batches over NOOA's file IPC: one action, then three."""
import json, time
from pathlib import Path

ipc = Path("ipc")
ipc.mkdir(parents=True, exist_ok=True)
states, actions = ipc / "states.jsonl", ipc / "actions.jsonl"
states.touch(); actions.touch()
(ipc / "agent_ready").write_text(str(time.time()))
batches = [["UP"], ["LEFT", "LEFT", "CLICK 7 9"]]
position = 0
index = 0
while True:
    with states.open() as handle:
        handle.seek(position)
        line = handle.readline()
        if line.endswith("\\n"):
            position = handle.tell()
            if line.strip():
                state = json.loads(line)
                batch = batches[min(index, len(batches) - 1)]
                index += 1
                with actions.open("a") as sink:
                    sink.write(json.dumps({
                        "turn": state.get("turn"),
                        "actions": batch,
                        "rationale": "predict: scripted",
                    }) + "\\n")
    time.sleep(0.02)
'''



def test_stub(tmp: Path) -> None:
    session = tmp / "stub-session"
    workspace = session / "workspace"
    workspace.mkdir(parents=True)
    # A scripted agent owned by this test: it speaks the same file IPC upstream
    # NOOA speaks, and emits two batches so the drain path is exercised.
    scripted = tmp / "scripted_agent.py"
    scripted.write_text(SCRIPTED_AGENT)
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "NOOA_AGENT_COMMAND": json.dumps([sys.executable, str(scripted)]),
    }
    stderr_path = tmp / "stub-stderr.log"
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(WORKER / "nooa_worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            cwd=session,
            env=environment,
        )
        try:
            initial = observation([GRID])
            send(
                process,
                {
                    "type": "start_game",
                    "game_id": GAME,
                    "seed": 0,
                    "workspace_root": str(workspace),
                    "initial_observation": initial.model_dump(mode="json"),
                },
            )
            metadata = receive(process, "ready")["metadata"]
            assert metadata["join"] == "s0-nooa-upstream-join", metadata
            print("stub: ready ->", metadata["agent"], "|", metadata["chassis"])

            states = session / "ipc" / "states.jsonl"
            decisions = []
            current = initial
            for sequence in range(4):
                send(
                    process,
                    turn_message(
                        sequence,
                        current,
                        used=sequence,
                        previous=decisions[-1] if decisions else None,
                    ),
                )
                decision = receive(process, "decision")["decision"]
                decisions.append(decision)
                after = observation([SETTLED if sequence % 2 == 0 else GRID])
                send(
                    process,
                    {
                        "type": "action_result",
                        "sequence": sequence,
                        "level": 0,
                        "turn_in_level": sequence,
                        "decision": decision,
                        "before": current.model_dump(mode="json"),
                        "after": after.model_dump(mode="json"),
                    },
                )
                receive(process, "observed")
                current = after

            names = [d["action"] for d in decisions]
            assert names == ["UP", "LEFT", "LEFT", "CLICK"], names
            assert decisions[3]["row"] == 9 and decisions[3]["col"] == 7, decisions[3]
            state_lines = [l for l in states.read_text().splitlines() if l.strip()]
            # Two batches means two state lines: the batch is drained without
            # asking the agent again.
            assert len(state_lines) == 2, f"{len(state_lines)} state lines: {state_lines}"
            print("stub: decisions", names, "from", len(state_lines), "state lines")

            send(process, {"type": "close"})
            process.wait(timeout=20)
            print("stub: ok")
        finally:
            if process.poll() is None:
                process.kill()


# --------------------------------------------------------------------------
# 2. The pristine upstream agent boots under the offline wheelhouse
# --------------------------------------------------------------------------


class RecordingGateway(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        RecordingGateway.requests.append(body)
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "test"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "print('hello')"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


def test_boot(tmp: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingGateway)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    run_dir = tmp / "boot-run"
    (run_dir / "ipc").mkdir(parents=True)
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "agg",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "ARC_LLM_API_KEY": "local-model-no-key",
        "ARC_LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "ARC_LLM_MODEL": "qwen3.8-27b-fp8",
        "ARC_LLM_USE_RESPONSES": "0",
        "ARC_LLM_EXTRA_BODY": '{"chat_template_kwargs":{"enable_thinking":true}}',
    }
    stderr_path = tmp / "boot-stderr.log"
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                str(WORKER / "nooa_headless.py"),
                "--run-dir",
                str(run_dir),
                "--visual",
                "additive",
            ],
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            text=True,
            cwd=run_dir,
            env=environment,
        )
        try:
            marker = run_dir / "ipc" / "agent_ready"
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if marker.exists():
                    break
                if process.poll() is not None:
                    raise AssertionError(
                        "headless agent exited before agent_ready:\n"
                        + stderr_path.read_text()[-4000:]
                    )
                time.sleep(0.1)
            else:
                raise AssertionError("timed out waiting for agent_ready")
            print("boot: pristine MdArcSolverAgent reached agent_ready")

            # Hand it a first state so it makes a model call.
            state = {
                "turn": 0,
                "step": 0,
                "level": 0,
                "state": "NOT_FINISHED",
                "levels_completed": 0,
                "available_actions": ["RESET", "UP", "DOWN", "LEFT", "RIGHT", "SPACEBAR", "CLICK"],
                "action_results": [],
                "grid_rows": ["".join("0123456789abcdef"[v] for v in row) for row in GRID],
                "diff_summary": "first state",
            }
            with (run_dir / "ipc" / "states.jsonl").open("a") as handle:
                handle.write(json.dumps(state) + "\n")

            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and not RecordingGateway.requests:
                if process.poll() is not None:
                    raise AssertionError(
                        "headless agent exited before a model call:\n"
                        + stderr_path.read_text()[-4000:]
                    )
                time.sleep(0.1)
            assert RecordingGateway.requests, "upstream NOOA made no model call"
            body = RecordingGateway.requests[0]
            print("boot: first model call ->", body.get("model"))
            print("boot: request keys:", sorted(body))
            print("boot: extra_body thinking present:", "chat_template_kwargs" in json.dumps(body))
            messages = body.get("messages") or []
            print("boot: message roles:", [m.get("role") for m in messages][:6])
            has_image = any(
                isinstance(m.get("content"), list)
                and any(
                    isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"]
                )
                for m in messages
            )
            print("boot: sent an image part:", has_image)
            workspace = run_dir / "team_nemo" / "shared"
            print("boot: workspace dirs:", sorted(p.name for p in workspace.iterdir()))
            print("boot: ok")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            server.shutdown()
    tail = stderr_path.read_text()[-1500:]
    if tail.strip():
        print("--- headless stderr tail ---")
        print(tail)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        test_stub(root)
        test_boot(root)
