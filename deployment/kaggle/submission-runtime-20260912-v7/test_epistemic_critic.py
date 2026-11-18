import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_contract():
    spec = importlib.util.spec_from_file_location(
        "epistemic_critic_test", HERE / "chassis/epistemic_critic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_cadence_and_source_binding():
    c = load_contract()
    assert not c.review_due(actor_calls=1, step=0, last_actor_calls=0, last_step=0, reviews=0)
    assert c.review_due(actor_calls=2, step=0, last_actor_calls=0, last_step=0, reviews=0)
    assert c.review_due(actor_calls=1, step=2, last_actor_calls=1, last_step=0, reviews=0)
    messages = [{"role": "assistant", "content": "One click proves all clicks are useless."}]
    critique = c.validate(json.dumps({
        "assessment": "concern", "tempo_verdict": "ACT_NOW",
        "actor_quote": "all clicks are useless",
        "inference_gap": "One target does not establish the action class.",
        "decision_consequence": "Untested targets would be discarded.",
        "minimal_revision": "Restrict the claim and probe another target.",
    }))
    match = c.bind_quote(critique, messages)["matches"][0]
    assert (match["message"], match["field"], match["start"], match["end"]) == (
        0, "content", 17, 39
    )


def test_no_concern_is_empty_continue():
    c = load_contract()
    good = {"assessment": "no_concern", "tempo_verdict": "CONTINUE",
            "actor_quote": "", "inference_gap": "",
            "decision_consequence": "", "minimal_revision": ""}
    assert c.validate(json.dumps(good)) == good
    good["tempo_verdict"] = "ACT_NOW"
    with pytest.raises(ValueError, match="empty CONTINUE"):
        c.validate(json.dumps(good))


@pytest.mark.asyncio
async def test_live_worker_injects_toolless_review_before_third_actor_call(monkeypatch, tmp_path):
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
    seen = []
    actor_index = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal actor_index
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            critic = body.get("max_tokens") == 256
            if critic:
                assert "tools" not in body
                packed = json.loads(body["messages"][1]["content"])
                rendered = json.dumps(packed["actor_request_as_evidence"])
                assert "inspection one" in rendered and "inspection two" in rendered
                assert "HOST TEMPO CONTRACT" in rendered
                content = json.dumps({
                    "assessment": "concern", "tempo_verdict": "ACT_NOW",
                    "actor_quote": "inspection two result",
                    "inference_gap": "Further restatement creates no new evidence.",
                    "decision_consequence": "The reversible probe is delayed.",
                    "minimal_revision": "Act now and use the successor as evidence.",
                })
                message = {"role": "assistant", "content": content}
                finish = "stop"
            else:
                actor_index += 1
                if actor_index < 3:
                    code = ('print("inspection one")' if actor_index == 1
                            else 'print("inspection" + " two result")')
                else:
                    rendered = json.dumps(body["messages"])
                    assert "HOST EPISTEMIC REVIEW critic:1" in rendered
                    code = 'self.submit_actions(["UP"], rationale="reversible evidence probe")'
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call{actor_index}", "type": "function",
                    "function": {"name": "execute_python", "arguments": json.dumps({"code": code})},
                }]}
                finish = "tool_calls"
            raw = json.dumps({
                "id": "critic-fixture", "object": "chat.completion", "created": 1,
                "model": "qwen/qwen3.8-27b", "choices": [{"index": 0,
                    "finish_reason": finish, "message": message}],
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
        "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "30", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TYCHO_KAGGLE_MAX_ACTIONS": "1",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    first = test_my_agent.frame([[0, 1], [0, 0]])
    agent = qual.build_agent(monkeypatch, tmp_path, first)
    try:
        action = agent.choose_action(agent.frames, first)
        assert action.name == "ACTION1"
    finally:
        agent.cleanup(scorecard=None)
        server.shutdown()
        thread.join(timeout=2)
    assert len([request for request in seen if request.get("max_tokens") == 256]) == 1
    reviews = [json.loads(line) for path in tmp_path.rglob("critic_reviews.jsonl")
               for line in path.read_text().splitlines()]
    assert reviews[0]["status"] == "completed"
    calls = [json.loads(line) for path in tmp_path.rglob("model_calls.jsonl")
             for line in path.read_text().splitlines()]
    assert [row["request_purpose"] for row in calls] == ["actor", "actor", "critic", "actor"]
