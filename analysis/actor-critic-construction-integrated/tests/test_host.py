from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from host import FixtureTransport, HttpTransport, run_episode  # noqa: E402  # identity path
from prefix import project_transitions  # noqa: E402

CANDIDATE = '''
def initialize(observation):
    return {"grid": observation["grid"]}
def predict(state, action):
    return [{"state": {"grid": state["grid"], "last_action": action}, "commitments": {"marker": "join-fixture-alpha", "action": action}, "declared_dependencies": ["grid:observation"]}]
def assimilate(predicted_state, observation, verdict):
    return {"grid": observation["grid"], "verdict": verdict, "last_action": predicted_state.get("last_action")}
'''

WORK = {
    "decision": "next move",
    "obstacle": "transition uncertainty",
    "operation": "test",
    "claim": "grid successor",
    "evidence_refs": [1],
    "dependencies": ["grid:observation"],
    "completion_checks": [1],
    "remaining_scope": "unseen actions",
    "reasoning_need": "needs_computation",
}


def call(name, args):
    return {"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def response(calls):
    return {
        "id": "fixture",
        "provider": "fixture/provider",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 0}},
    }


def case_from(tmp_path: Path):
    path = tmp_path / "transitions.jsonl"
    path.write_text(json.dumps({
        "step": 1, "turn": 0, "action": "RIGHT",
        "before": {"grid": [[12, 4, 4]], "state": "NOT_FINISHED", "levels_completed": 0, "available_actions": ["RIGHT"]},
        "after": {"grid": [[4, 12, 4]], "state": "NOT_FINISHED", "levels_completed": 0, "available_actions": ["RIGHT"]},
    }) + "\n")
    return project_transitions(path)


def test_scripted_constructor_freezes_visible_replay_only(tmp_path: Path) -> None:
    transport = FixtureTransport([response([call("submit_candidate", {"work_item": WORK, "candidate_source": CANDIDATE})])])
    record = run_episode(case_from(tmp_path), transport, tmp_path / "out")
    assert record["construction_status"] == "CANDIDATE_READY"
    assert record["frozen"] is True
    assert record["supported_model"] is False
    assert record["prospective"] == "not_run"
    assert record["evidence_claim"] == "ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL"
    frozen = json.loads((tmp_path / "out" / "FROZEN.json").read_text())
    assert frozen["supported_model"] is False
    assert frozen["candidate_source"] == CANDIDATE


def test_request_evidence_does_not_freeze(tmp_path: Path) -> None:
    transport = FixtureTransport([response([call("request_evidence", {
        "obstacle": "need another observation",
        "needed_observation": "a second executed transition",
        "decision_relevance": "cannot close the successor",
    })])])
    record = run_episode(case_from(tmp_path), transport, tmp_path / "out")
    assert record["construction_status"] == "NEEDS_EVIDENCE"
    assert record["frozen"] is False
    assert not (tmp_path / "out" / "FROZEN.json").exists()


def test_http_transport_is_real_http_not_live_model(tmp_path: Path) -> None:
    payload = response([call("request_evidence", {
        "obstacle": "need another observation",
        "needed_observation": "a second executed transition",
        "decision_relevance": "cannot close the successor",
    })])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        transport = HttpTransport(f"http://127.0.0.1:{server.server_port}/v1")
        record = run_episode(case_from(tmp_path), transport, tmp_path / "out")
        assert transport.label == "real-http-not-claimed-live-model"
        assert record["construction_status"] == "NEEDS_EVIDENCE"
        assert record["frozen"] is False
    finally:
        server.shutdown()
        thread.join(timeout=2)
