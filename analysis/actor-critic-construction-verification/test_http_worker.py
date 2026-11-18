"""Independent real worker test with explicit HTTP constructor configuration.

All model responses are scripted over loopback HTTP. No external model is called.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "analysis/actor-critic-construction-integrated"


def test_http_constructor_receives_live_worker_prefix_and_returns_to_actor(monkeypatch, tmp_path):
    sys.path[:0] = [str(CANDIDATE), str(ROOT / "deployment/kaggle/tests"), str(ROOT / "deployment/kaggle")]
    spec = importlib.util.spec_from_file_location("coordinator_loopback_fixture", CANDIDATE / "tests/test_loopback_join.py")
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    import test_join_qualification as qualification

    received = []
    payload = fixture.response([fixture.call("submit_candidate", {"work_item": fixture.WORK, "candidate_source": fixture.CANDIDATE})])

    class ConstructorHandler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    constructor = HTTPServer(("127.0.0.1", 0), ConstructorHandler)
    constructor_thread = threading.Thread(target=constructor.serve_forever, daemon=True)
    constructor_thread.start()
    original_build = qualification.build_agent
    actual_command = []

    def build_with_http(*args, **kwargs):
        command = json.loads(os.environ["NOOA_AGENT_COMMAND"])
        filtered = []
        index = 0
        while index < len(command):
            if command[index] in {"--constructor-transport", "--constructor-fixture", "--constructor-url"}:
                index += 2
            else:
                filtered.append(command[index])
                index += 1
        filtered.extend(["--constructor-transport", "http", "--constructor-url", f"http://127.0.0.1:{constructor.server_port}/constructor"])
        actual_command.extend(filtered)
        monkeypatch.setenv("NOOA_AGENT_COMMAND", json.dumps(filtered))
        return original_build(*args, **kwargs)

    monkeypatch.setattr(qualification, "build_agent", build_with_http)
    actor_server = actor_thread = agent = None
    try:
        # Empty fixture file deliberately cannot provide a successful candidate.
        agent, actor_server, actor_thread, seen, snapshot, frames = fixture._start_join(monkeypatch, tmp_path, [])
        for index, frame in enumerate(frames):
            if index:
                agent.frames.append(frame)
            assert agent.choose_action(agent.frames, frame).name == "ACTION1"
            agent.action_counter += 1  # official launcher increments after each action
        from test_my_agent import frame as make_frame
        agent.frames.append(make_frame([[0, 0], [0, 1]]))
        assert len(received) == 1
        request = received[0]
        user_messages = [message["content"] for message in request["messages"] if message["role"] == "user"]
        packet = json.loads(user_messages[0])["visible_case"]
        assert packet["initial_observation"]["grid"] == [[0, 1], [0, 0]]
        assert packet["current_observation"]["grid"] == [[1, 0], [0, 0]]
        assert packet["frontier"] == 1
        assert "[[0,0],[1,0]]" not in json.dumps(packet, separators=(",", ":")), "future observation leaked to constructor"
        actor_calls, critic_calls, saw_prediction, saw_assimilation = snapshot()
        assert critic_calls >= 1 and saw_prediction and saw_assimilation
        assert actor_calls <= 14
        records = [json.loads(path.read_text()) for path in tmp_path.rglob("RESULT.json")]
        assert any(record["frozen"] and record["transport_label"] == "real-http-not-claimed-live-model" for record in records)
        (tmp_path / "coordinator-http-proof.json").write_text(json.dumps({
            "scope": "scripted loopback transport through real worker; no live model",
            "constructor_requests": len(received), "constructor_visible_case": packet,
            "actor_calls": actor_calls, "critic_calls": critic_calls,
            "actor_received_prediction": saw_prediction, "actor_received_assimilation": saw_assimilation,
            "command": actual_command,
        }, indent=2) + "\n")
    finally:
        try:
            if agent is not None:
                agent.cleanup(scorecard=None)
        finally:
            if actor_server is not None:
                actor_server.shutdown()
                actor_server.server_close()
            if actor_thread is not None:
                actor_thread.join(timeout=2)
            constructor.shutdown()
            constructor.server_close()
            constructor_thread.join(timeout=2)
