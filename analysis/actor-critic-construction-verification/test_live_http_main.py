"""Real launch main/relay/worker/game path, explicitly scripted upstream only.

No OpenRouter connection is permitted. Credentials are synthetic and never loaded.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "analysis/actor-critic-construction-live-launch"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing_usage", [False, True])
def test_actual_live_harness_scripted_upstream(monkeypatch, tmp_path, missing_usage):
    monkeypatch.syspath_prepend(str(LIVE))
    harness = load(LIVE / "harness.py", "coordinator_live_harness")
    fixture = load(ROOT / "analysis/actor-critic-construction-integrated/tests/test_host.py", "coordinator_candidate_fixture")
    received = []
    actor_calls = 0

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            nonlocal actor_calls
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            names = {t["function"]["name"] for t in body.get("tools", [])}
            if "submit_candidate" in names:
                calls = [{
                    "id": f"ctor-{len(received)}",
                    "type": "function",
                    "function": {
                        "name": "submit_candidate",
                        "arguments": json.dumps({"work_item": fixture.WORK, "candidate_source": fixture.CANDIDATE}),
                    },
                }]
                message = {"role": "assistant", "content": None, "tool_calls": calls}
            elif body.get("max_tokens") == 192:
                message = {"role": "assistant", "content": '{"claim_ref":"","question":"","reason":""}'}
            else:
                actor_calls += 1
                if actor_calls < 9:
                    code = f'print("inspection {actor_calls}")'
                elif actor_calls == 9:
                    code = 'print(self.submit_actions(["UP"], rationale="prefix"))'
                elif actor_calls == 10:
                    code = 'print(self.request_construction()); print(self.query_frozen_predict("UP"))'
                elif actor_calls == 11:
                    code = 'print(self.submit_actions(["UP"], rationale="after frozen prediction"))'
                elif actor_calls == 12:
                    code = 'print(self.assimilate_frozen())'
                else:
                    code = 'print(self.submit_actions(["UP"], rationale="fixture continuation"))'
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"actor-{actor_calls}", "type": "function", "function": {"name": "execute_python", "arguments": json.dumps({"code": code})}}]}
            response = {
                "id": f"coordinator-fixture-{len(received)}", "object": "chat.completion", "created": 1,
                "model": "qwen/qwen3.8-27b", "provider": "CoreWeave",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130, "cost": 0.00013, "completion_tokens_details": {"reasoning_tokens": 0}},
            }
            if missing_usage:
                response.pop("usage")
            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_open = urllib.request.urlopen

    def fixture_only(request, *args, **kwargs):
        if not isinstance(request, urllib.request.Request) or request.full_url != "https://openrouter.ai/api/v1/chat/completions":
            raise AssertionError("Unexpected coordinator network target")
        redirected = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", data=request.data, headers={"Content-Type": "application/json"}, method="POST")
        return original_open(redirected, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fixture_only)
    monkeypatch.setattr(harness, "load_key", lambda: "coordinator-synthetic-key")
    monkeypatch.setattr(harness, "MAX_ACTIONS", 3)  # bounded fixture, production cap remains 8
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(sys, "argv", [str(LIVE / "harness.py"), "--execute", "--out", str(raw_dir)])
    # Restore environment changes made by main after this isolated test.
    import os
    old_environment = dict(os.environ)
    try:
        returncode = harness.main()
        assert received, "worker never reached real HTTP relay"
        for body in received:
            assert "parallel_tool_calls" not in body
            assert "chat_template_kwargs" not in body
            assert body["provider"]["only"] == ["coreweave/fp8"]
        if missing_usage:
            assert returncode != 0
            assert len(received) == 1, "missing usage must stop further provider attempts"
        else:
            assert returncode == 0, (raw_dir / "RESULTS.json").read_text()
            actual_steps = [json.loads(line) for line in (raw_dir / "environment-steps.jsonl").read_text().splitlines()]
            transition_rows = [json.loads(line) for path in raw_dir.rglob("transitions.jsonl") for line in path.read_text().splitlines() if line.strip()]
            action_id_to_name = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACEBAR", 6: "CLICK", 7: "UNDO"}

            def norm(action):
                if isinstance(action, str) and action.startswith("ACTION") and action[6:].isdigit():
                    return action_id_to_name[int(action[6:])]
                return action

            assert [norm(row.get("action")) for row in actual_steps] == [norm(row.get("action")) for row in transition_rows]
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
