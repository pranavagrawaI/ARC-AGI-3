#!/usr/bin/env python3
"""Exercise the Tycho-upstream join without a GPU, a model, or a Kaggle kernel.

Two checks:

1. ``projection`` — a host observation built by the seam's own
   ``project_arc_observation`` survives the round trip into the ``FrameData``
   upstream Tycho reads, frames and action ids included.
2. ``join`` — the real worker subprocess, running the real ``TychoAgent`` through
   upstream's own ``run_env``, answers a protocol-v3 turn with a decision. The
   model is a local fake OpenAI-compatible gateway that returns one
   ``take_action`` tool call, so this covers the transport configuration, the
   tool surface, the thread bridge, and the action translation.

Run with the wheelhouse venv:

    refvenv/bin/python kaggle/tests/test_tycho_join.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE.parent / "tycho-upstream" / "worker"
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


def observation(frames, *, state="NOT_FINISHED", levels=0, previous=None):
    return project_arc_observation(
        frames=frames,
        available_action_ids=[1, 2, 3, 4, 5, 6],
        state=state,
        levels_completed=levels,
        full_reset=False,
        previous_levels_completed=previous,
    )


# --------------------------------------------------------------------------
# 1. Projection
# --------------------------------------------------------------------------


def test_projection() -> None:
    import tycho_worker

    payload = observation([GRID, SETTLED])
    frame = tycho_worker.frame_data(payload, GAME)
    assert frame.game_id == GAME, frame.game_id
    # Both raw frames survive: Tycho reads animation evidence, not just the settled grid.
    assert frame.frame == [GRID, SETTLED], "raw frame sequence was not reconstructed"
    assert frame.levels_completed == 0
    assert frame.available_actions == [0, 1, 2, 3, 4, 5, 6], frame.available_actions
    assert str(frame.state).endswith("NOT_FINISHED"), frame.state

    # A level-completing observation carries a terminal frame at index 0.
    advanced = observation([GRID, SETTLED], levels=1, previous=0)
    assert advanced.provenance.terminal_frame_index == 0
    assert tycho_worker.frame_data(advanced, GAME).frame == [GRID, SETTLED]

    from arcengine import GameAction

    assert tycho_worker.decision_payload(GameAction.ACTION1, None) == {
        "action": "UP",
        "row": None,
        "col": None,
    }
    assert tycho_worker.decision_payload(GameAction.ACTION6, {"x": 7, "y": 9}) == {
        "action": "CLICK",
        "row": 9,
        "col": 7,
    }
    assert tycho_worker.decision_payload(GameAction.RESET, None)["action"] == "RESET"
    print("projection: ok")


# --------------------------------------------------------------------------
# 2. Full worker join against a fake gateway
# --------------------------------------------------------------------------


class FakeGateway(BaseHTTPRequestHandler):
    """Answer every chat completion with one take_action tool call."""

    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeGateway.requests.append(body)
        tool_names = [t["function"]["name"] for t in body.get("tools", [])]
        action = "ACTION1" if "take_action" in tool_names else "RESET"
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "moving up to test the transition",
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "take_action",
                                    "arguments": json.dumps({"action": action}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


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


def test_join(tmp: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    session = tmp / "session"
    workspace = session / "workspace"
    workspace.mkdir(parents=True)
    environment = {
        **os.environ,
        "PATH": os.environ["PATH"],
        "PYTHONUNBUFFERED": "1",
        "LLM_BACKEND": "openai",
        "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "LLM_MODEL": "qwen3.8-27b-fp8",
        "LLM_API_KEY": "EMPTY",
        "LLM_EFFORT": "high",
        "TYCHO_MODE": "single",
        "TYCHO_VISION": "1",
        "TYCHO_MAX_LLM_CALLS": "8",
        "TYCHO_MAX_TOOL_STEPS": "2",
        "TYCHO_MAX_TOKENS": "512",
        "TYCHO_MAX_INFERENCE_COST_PER_GAME": "0",
        "TYCHO_MAX_INFERENCE_COST_PER_LEVEL": "0",
        "TYCHO_SANDBOX_RUNTIME": "host",
        "TYCHO_SYNTHETIC_BASELINE": "64",
        "TYCHO_SYNTHETIC_LEVELS": "4",
    }
    stderr_path = tmp / "worker-stderr.log"
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(WORKER / "tycho_worker.py")],
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
            ready = receive(process, "ready")
            metadata = ready["metadata"]
            assert metadata["join"] == "s0-tycho-upstream-join", metadata
            assert metadata["agent"].startswith("tycho."), metadata
            assert (workspace / GAME).is_dir(), "per-game session marker was not created"
            print("join: ready ->", metadata["agent"], "mode", metadata["mode"])

            send(
                process,
                {
                    "type": "turn",
                    "sequence": 0,
                    "action_limit": 16,
                    "actions_used": 0,
                    "actions_remaining": 16,
                    "final_permitted_action": False,
                    "level": 0,
                    "turn_in_level": 0,
                    "decision_timeout_seconds": 120.0,
                    "observation": initial.model_dump(mode="json"),
                    "previous_decision": None,
                },
            )
            decision = receive(process, "decision")["decision"]
            assert decision["action"] in {
                "UP",
                "DOWN",
                "LEFT",
                "RIGHT",
                "SPACEBAR",
                "CLICK",
                "RESET",
                "UNDO",
            }, decision
            print("join: decision ->", decision)
            assert FakeGateway.requests, "Tycho made no model call"
            body = FakeGateway.requests[0]
            names = [t["function"]["name"] for t in body.get("tools", [])]
            assert "take_action" in names, names
            assert body["model"] == "qwen3.8-27b-fp8", body["model"]
            has_image = any(
                isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"])
                for m in body["messages"]
            )
            print(f"join: tools {names}")
            print(f"join: sent an image part: {has_image}")

            after = observation([SETTLED])
            send(
                process,
                {
                    "type": "action_result",
                    "sequence": 0,
                    "level": 0,
                    "turn_in_level": 0,
                    "decision": decision,
                    "before": initial.model_dump(mode="json"),
                    "after": after.model_dump(mode="json"),
                },
            )
            receive(process, "observed")
            print("join: observed")

            send(
                process,
                {
                    "type": "finish",
                    "stop_reason": "test",
                    "actions": 1,
                    "levels_completed": 0,
                    "final_observation": after.model_dump(mode="json"),
                },
            )
            receive(process, "finished")
            send(process, {"type": "close"})
            process.wait(timeout=30)
            print("join: ok")
        finally:
            if process.poll() is None:
                process.kill()
            server.shutdown()
    tail = stderr_path.read_text()[-2000:]
    if tail.strip():
        print("--- worker stderr tail ---")
        print(tail)


if __name__ == "__main__":
    import tempfile

    test_projection()
    with tempfile.TemporaryDirectory() as raw:
        test_join(Path(raw))
