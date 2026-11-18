"""Real v11 adapter + our headless + loopback critic/actor. TRANSPORT is loopback, not a live model."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
V11 = ROOT / "deployment/kaggle/submission-runtime-20260912-v11"
sys.path.insert(0, str(HERE))
from tests.test_host import CANDIDATE, WORK, call, response  # noqa: E402  # identity path


def _start_join(monkeypatch, tmp_path, fixture_responses):
    sys.path[:0] = [str(ROOT / "deployment/kaggle/tests"), str(ROOT / "deployment/kaggle")]
    import test_my_agent
    import test_nooa_join
    import test_join_qualification as qual

    spec = importlib.util.spec_from_file_location("runtime.local_solver", V11 / "shared_local_solver.py")
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "runtime.local_solver", runtime)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(test_my_agent, "AGENT_PATH", V11 / "adapter.py")
    monkeypatch.setattr(test_nooa_join, "CANDIDATE", V11)

    seen: list[dict] = []
    actor_index = 0
    critic_index = 0
    saw_marker_in_later_request = False
    saw_unscored_in_later_request = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal actor_index, critic_index, saw_marker_in_later_request, saw_unscored_in_later_request
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            rendered = json.dumps(body.get("messages"))
            if "join-fixture-alpha" in rendered:
                saw_marker_in_later_request = True
            if "unscored" in rendered:
                saw_unscored_in_later_request = True
            critic = body.get("max_tokens") == 192
            if critic:
                critic_index += 1
                assert "tools" not in body
                packed = json.loads(body["messages"][1]["content"])
                assert "constructor" not in json.dumps(packed)
                message = {"role": "assistant", "content": json.dumps({"claim_ref": "", "question": "", "reason": ""})}
                finish = "stop"
            else:
                actor_index += 1
                if actor_index > 14:
                    self.send_error(429, "scripted transport cap")
                    return
                if actor_index < 9:
                    code = f'print("inspection {actor_index}")'
                elif actor_index == 9:
                    code = 'print(self.submit_actions(["UP"], rationale="prefix probe"))'
                elif actor_index == 10:
                    code = 'print(self.request_construction()); print(self.query_frozen_predict("UP"))'
                elif actor_index == 11:
                    code = 'print(self.submit_actions(["UP"], rationale="join-fixture-alpha from freeze"))'
                elif actor_index == 12:
                    code = 'print(self.assimilate_frozen())'
                else:
                    code = 'print(self.submit_actions(["UP"], rationale="after assimilate"))'
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call{actor_index}", "type": "function",
                    "function": {"name": "execute_python", "arguments": json.dumps({"code": code})},
                }]}
                finish = "tool_calls"
            raw = json.dumps({
                "id": "join-fixture", "object": "chat.completion", "created": 1,
                "model": "qwen/qwen3.8-27b",
                "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
                          "completion_tokens_details": {"reasoning_tokens": 0}},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fixture = tmp_path / "constructor-fixture.json"
    fixture.write_text(json.dumps(fixture_responses))
    settings = {
        "ARC_DISABLE_DOTENV": "1", "ARC_LLM_MODEL": "qwen/qwen3.8-27b",
        "ARC_LLM_API_KEY": "fixture", "ARC_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "ARC_LLM_EXTRA_BODY": '{"reasoning":{"enabled":false}}',
        "ARC_LLM_MAX_TOKENS": "16384", "ARC_SOLVER_ARM": "loop",
        "ARC_DISCOVERY_ARM": "a", "ARC_CONTINGENT_ARM": "off",
        "NOOA_JOIN_ROOT": str(V11),
        "NOOA_AGENT_COMMAND": json.dumps([
            sys.executable,
            str(HERE / "headless.py"),
            "--constructor-transport", "fixture",
            "--constructor-fixture", str(fixture),
            "--run-dir", ".",
        ]),
        "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "45", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TYCHO_KAGGLE_MAX_ACTIONS": "3",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    frames = [
        test_my_agent.frame([[0, 1], [0, 0]]),
        test_my_agent.frame([[1, 0], [0, 0]]),
        test_my_agent.frame([[0, 0], [1, 0]]),
    ]
    agent = qual.build_agent(monkeypatch, tmp_path, frames[0])
    return agent, server, thread, seen, lambda: (actor_index, critic_index, saw_marker_in_later_request, saw_unscored_in_later_request), frames


def _drive(agent, frames):
    for index, frame in enumerate(frames):
        if index:
            agent.frames.append(frame)
        assert agent.choose_action(agent.frames, frame).name == "ACTION1"
        agent.action_counter += 1


def _cleanup(agent, server, thread, extra):
    import test_my_agent
    try:
        agent.frames.append(test_my_agent.frame(extra))
        agent.cleanup(scorecard=None)
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_headless_choose_action_constructs_then_queries_freeze(monkeypatch, tmp_path):
    fixture = [response([call("submit_candidate", {"work_item": WORK, "candidate_source": CANDIDATE})])]
    agent, server, thread, seen, snapshot, frames = _start_join(monkeypatch, tmp_path, fixture)
    try:
        _drive(agent, frames)
        actor_index, critic_index, saw_marker, saw_unscored = snapshot()
        frozen = list(tmp_path.rglob("FROZEN.json"))
        assert frozen, "request_construction did not freeze from this run's transitions"
        assert json.loads(frozen[0].read_text())["supported_model"] is False
        calls = [json.loads(line) for path in tmp_path.rglob("model_calls.jsonl") for line in path.read_text().splitlines()]
        blob = json.dumps(calls)
        assert "critic" in [row.get("request_purpose") for row in calls]
        assert critic_index >= 1
        assert actor_index <= 14
        assert "join-fixture-alpha" in blob
        assert saw_marker, "next actor request never saw the freeze commitment"
        assert "unscored" in blob
        assert saw_unscored, "later actor request never saw assimilate unscored output"
        assert "request_construction" in json.dumps(seen)
    finally:
        _cleanup(agent, server, thread, [[0, 0], [0, 1]])


def test_failed_construction_still_submits(monkeypatch, tmp_path):
    fixture = [response([call("request_evidence", {
        "obstacle": "need another observation",
        "needed_observation": "a second executed transition",
        "decision_relevance": "cannot close the successor",
    })])]
    agent, server, thread, seen, snapshot, frames = _start_join(monkeypatch, tmp_path, fixture)
    try:
        _drive(agent, frames[:2])
        assert not list(tmp_path.rglob("FROZEN.json"))
        actor_index, _, _, _ = snapshot()
        assert actor_index <= 14
        assert "request_construction" in json.dumps(seen)
    finally:
        _cleanup(agent, server, thread, [[0, 0], [1, 0]])

