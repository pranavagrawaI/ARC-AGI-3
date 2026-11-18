"""An Actor DONE on an unfinished game is refused and the state redelivered.

The 2026-09-12 objection-closure public run: the Actor declared LS20 "stuck"
with 152s and 193 actions remaining, returned DONE, the worker fell silent, the
host's decision timeout fired, and the three-game run ended with two games
unplayed. This test drives the real Actor-to-worker-to-host path.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


@pytest.mark.asyncio
async def test_done_on_an_unfinished_game_is_refused_and_the_actor_must_submit(
        monkeypatch, tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    sys.path[:0] = [str(ROOT / "deployment/kaggle/tests"), str(ROOT / "deployment/kaggle")]
    import test_my_agent
    import test_nooa_join
    import test_join_qualification as qual

    spec = importlib.util.spec_from_file_location("runtime.local_solver", HERE / "shared_local_solver.py")
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "runtime.local_solver", runtime)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(test_my_agent, "AGENT_PATH", HERE / "adapter.py")
    monkeypatch.setattr(test_nooa_join, "CANDIDATE", HERE)
    actor_index = 0
    refused_seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal actor_index
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body.get("max_tokens") != 192, "no critic boundary is reached in three calls"
            actor_index += 1
            rendered = json.dumps(body["messages"])
            if actor_index == 1:
                code = 'print("inspection")'
            elif actor_index == 2:
                # The v9 public-run failure, verbatim in shape: a confident,
                # wrong verdict that the game is over.
                code = ('self.message("Game stuck at level 0. Budget exhausted.")\n'
                        'return_result(RespondReason.DONE, explanation="Game stuck - budget exhausted")')
            else:
                assert "HOST: the game is not finished" in rendered
                assert "Your DONE was not accepted" in rendered
                refused_seen.append(True)
                code = 'self.submit_actions(["UP"], rationale="the host said so; act")'
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call{actor_index}", "type": "function",
                "function": {"name": "execute_python", "arguments": json.dumps({"code": code})},
            }]}
            raw = json.dumps({
                "id": "done-fixture", "object": "chat.completion", "created": 1,
                "model": "qwen/qwen3.8-27b", "choices": [{"index": 0,
                    "finish_reason": "tool_calls", "message": message}],
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
    settings = {
        "ARC_DISABLE_DOTENV": "1", "ARC_LLM_MODEL": "qwen/qwen3.8-27b",
        "ARC_LLM_API_KEY": "fixture", "ARC_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "ARC_LLM_EXTRA_BODY": '{"reasoning":{"enabled":false}}',
        "ARC_LLM_MAX_TOKENS": "16384", "ARC_SOLVER_ARM": "loop",
        "ARC_DISCOVERY_ARM": "a", "ARC_CONTINGENT_ARM": "off",
        "NOOA_JOIN_ROOT": str(HERE),
        "NOOA_AGENT_COMMAND": json.dumps([sys.executable, str(HERE / "headless.py"), "--run-dir", "."]),
        "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "60", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TYCHO_KAGGLE_MAX_ACTIONS": "1",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    first = test_my_agent.frame([[0, 1], [0, 0]])
    agent = qual.build_agent(monkeypatch, tmp_path, first)
    try:
        action = agent.choose_action(agent.frames, first)
        assert action.name == "ACTION1", "the run continued to a real action"
    finally:
        agent.cleanup(scorecard=None)
        server.shutdown()
        thread.join(timeout=2)
    assert refused_seen == [True]
    refused = [json.loads(line) for path in tmp_path.rglob("done_refused.jsonl")
               for line in path.read_text().splitlines()]
    assert len(refused) == 1
    assert refused[0]["state"] == "NOT_FINISHED"
    assert "budget exhausted" in refused[0]["explanation"]
    calls = [json.loads(line) for path in tmp_path.rglob("model_calls.jsonl")
             for line in path.read_text().splitlines()]
    assert [row["request_purpose"] for row in calls] == ["actor"] * 3
