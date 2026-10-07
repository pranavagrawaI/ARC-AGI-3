"""Time-aware Actor behavior through the real worker and a loopback provider.

No provider/GPU access. A held HTTP response and a sleeping Python subprocess
exercise the two waits that formerly consumed the host's whole decision budget.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "runtime"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import arcengine
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame
from hybrid.solver import HybridSolver
from hybrid.decision_time import decision_runtime


def body_text(body):
    parts = []
    for message in body["messages"]:
        content = message.get("content", "")
        parts.append(content if isinstance(content, str) else "".join(p.get("text", "") for p in content))
    return "\n".join(parts)


def run_game(mode, *, enabled=True):
    requests = []
    decision_seconds = 6.0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"healthy")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            final = [tool["function"]["name"] for tool in body["tools"]] == ["take_actions"]
            requests.append({"final": final, "body": body, "at": time.monotonic()})
            if not final and len(requests) == 1 and mode in {"blocked", "refuse"}:
                time.sleep(decision_seconds * 2)
            python = (mode == "python" and len(requests) == 1) or (mode == "refuse" and final)
            name = "run_python" if python else "take_actions"
            arguments = ({"code": "import time; time.sleep(30)", "timeout_s": 300} if python
                         else {"actions": [{"action": "UP"}], "rationale": "fixture action"})
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call-{len(requests)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}]}
            wire = json.dumps({"id": "clock", "object": "chat.completion", "created": 0,
                "model": "flashnext", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 100,
                "completion_tokens": 10, "total_tokens": 110}}).encode()
            try:
                trickle = not final and len(requests) == 1 and mode in {"trickle", "error_trickle"}
                self.send_response(503 if trickle and mode == "error_trickle" else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire))); self.end_headers()
                if trickle:
                    for byte in wire:
                        self.wfile.write(bytes([byte])); self.wfile.flush(); time.sleep(0.1)
                else:
                    self.wfile.write(wire)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The deliberate slow response outlives the client deadline.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    environment = {key: value for key, value in os.environ.items() if not key.startswith(
        ("ARC_LLM_", "OPENAI_", "OPENROUTER_", "ANTHROPIC_", "NOOA_LLM_", "TYCHO_"))}
    environment.update(LITELLM_LOCAL_MODEL_COST_MAP="True", TYCHO_SANDBOX_RUNTIME="host",
        TYCHO_MODE="no_world_model", TYCHO_NOOA_TREATMENT="lean_prune",
        TYCHO_TIME_AWARE="1" if enabled else "0", MULTIMODAL_CONTEXT="current_grid",
        MULTIMODAL_UPSCALE="10", ARC3_MAX_ACTIVE_STREAMS="0", ARC3_PRIORITY_TAIL_FADE="0",
        ARC_LLM_MODEL="flashnext", ARC_LLM_API_KEY="loopback-fixture",
        ARC_LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
        ARC_LLM_CONTEXT_WINDOW=str(128 * 1024), ARC_LLM_RETRIES="0", ARC_LLM_USE_RESPONSES="0",
        ARC_LLM_TIMEOUT_SECONDS="60", ARC3_HTTP_RETRY_INITIAL_SECONDS="10")
    try:
        with patch.dict(os.environ, environment, clear=True), tempfile.TemporaryDirectory() as directory:
            game = ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]], label="clock-fixture")
            benchmark = Benchmark(games=[game], job_dir=Path(directory), solver=HybridSolver(
                worker_python=sys.executable, runtime_root=str(HERE / "runtime"),
                max_runtime_s_per_game=30, max_actions_per_game=1, analyzer_timeout=decision_seconds))
            asyncio.run(benchmark.run())
            root = next((Path(directory) / "hybrid-games").iterdir())
            def records(name):
                path = root / name
                return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            return {"requests": requests, "receipt": json.loads((root / "execution-receipt.json").read_text()),
                    "lifecycle": records("actor-lifecycle.jsonl"), "actions": records("host-actions.jsonl"),
                    "tools": records("tool-trace.jsonl"), "completed": records("llm-trace.jsonl")}
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


class DecisionTimeRegressions(unittest.TestCase):
    def test_disabled_runtime_requires_no_deadline(self):
        sentinel = object()
        with patch.dict(os.environ, {"TYCHO_TIME_AWARE": "0"}):
            self.assertIs(decision_runtime(sentinel), sentinel)

    def test_disabled_worker_request_has_no_clock(self):
        result = run_game("fast", enabled=False)
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(len(result["requests"]), 1)
        self.assertNotIn("HOST DECISION CLOCK", body_text(result["requests"][0]["body"]))
        self.assertEqual(result["lifecycle"], [])

    def test_fast_voluntary_action_sees_both_clocks(self):
        result = run_game("fast")
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(len(result["requests"]), 1)
        prompt = body_text(result["requests"][0]["body"])
        self.assertIn("HOST DECISION CLOCK", prompt)
        self.assertIn("Whole-game time remaining:", prompt)
        self.assertFalse(result["receipt"]["unresolved_provider_usage"])

    def test_blocked_http_leaves_time_for_real_action_and_unknown_usage(self):
        result = run_game("blocked")
        self.assertEqual([r["final"] for r in result["requests"]], [False, True])
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["receipt"]["stop_reason"], "won")
        self.assertTrue(result["receipt"]["unresolved_provider_usage"])
        self.assertEqual(len(result["completed"]), 1)  # Only the returned final action has known usage.
        interrupted = [r for r in result["lifecycle"] if r["phase"] == "provider_interrupted"]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]["provider_usage"], "unknown")
        self.assertLess(result["requests"][1]["at"] - result["requests"][0]["at"], 5.0)

    def test_python_cannot_consume_the_final_action_reserve(self):
        result = run_game("python")
        self.assertEqual(len(result["actions"]), 1)
        self.assertLess(result["receipt"]["elapsed_seconds"], 12)
        self.assertFalse(result["receipt"]["unresolved_provider_usage"])

    def test_trickling_success_body_cannot_restart_deadline(self):
        result = run_game("trickle")
        self.assertEqual([r["final"] for r in result["requests"]], [False, True])
        self.assertEqual(len(result["actions"]), 1)
        self.assertTrue(result["receipt"]["unresolved_provider_usage"])

    def test_trickling_error_body_cannot_restart_deadline(self):
        result = run_game("error_trickle")
        self.assertEqual([r["final"] for r in result["requests"]], [False, True])
        self.assertEqual(len(result["actions"]), 1)
        self.assertTrue(result["receipt"]["unresolved_provider_usage"])

    def test_refused_final_actions_do_not_invent_an_environment_action(self):
        result = run_game("refuse")
        self.assertEqual([r["final"] for r in result["requests"]], [False, True, True])
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["receipt"]["stop_reason"], "error")
        self.assertTrue(result["receipt"]["unresolved_provider_usage"])


if __name__ == "__main__":
    unittest.main()
