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


def trigger(c, **updates):
    args = dict(
        actor_calls=1, step=0, last_actor_calls=0, last_step=0,
        levels_completed=0, last_levels_completed=0,
        mismatch_count=0, last_mismatch_count=0,
        pending_sequence_length=0, pending_sequence_step=None,
    )
    args.update(updates)
    return c.review_trigger(**args)


def test_event_driven_review_boundaries_coalesce_without_a_count_budget():
    c = load_contract()
    assert trigger(c, actor_calls=7) is None
    assert trigger(c, actor_calls=8) == "eight_actor_calls"
    assert trigger(c, actor_calls=100_000, last_actor_calls=99_993) is None
    assert trigger(c, step=3) == "three_actions_observed"
    assert trigger(c, step=11, pending_sequence_length=2,
                   pending_sequence_step=10) == "action_sequence_observed"
    assert trigger(c, levels_completed=1) == "level_advanced"
    assert trigger(c, mismatch_count=1) == "prediction_mismatch"
    # One review is enough when several boundaries coincide.
    assert trigger(c, actor_calls=8, step=3, levels_completed=1,
                   mismatch_count=1) == "level_advanced"


def test_only_actor_authored_events_can_be_source_bound():
    c = load_contract()
    messages = [
        {"role": "system", "content": "HOST: clicks never work."},
        {"role": "user", "content": "Try to solve the game."},
        {"role": "assistant", "content": "One click proves all clicks are useless."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "function": {"name": "execute_python", "arguments":
                         '{"code":"print(\\"all clicks are useless\\")"}'},
        }]},
        {"role": "tool", "content": "Environment says clicks never work."},
    ]
    evidence = c.actor_evidence(messages)
    assert [row["claim_ref"] for row in evidence] == [
        "actor-message-2-content", "actor-message-3-tool-0"
    ]
    rendered = json.dumps(evidence)
    assert "HOST: clicks never work" not in rendered
    assert "Environment says clicks never work" not in rendered
    question = c.validate(json.dumps({
        "claim_ref": "actor-message-2-content",
        "question": "How does one tested target establish the entire click action class?",
        "reason": "Discarding all clicks removes untested interactions from inquiry.",
    }))
    assert c.bind_claim_ref(question, evidence)["match"] == evidence[0]
    question["claim_ref"] = "host-message-0-content"
    with pytest.raises(ValueError, match="0 source matches"):
        c.bind_claim_ref(question, evidence)


def test_question_contract_is_small_and_exact():
    c = load_contract()
    empty = {"claim_ref": "", "question": "", "reason": ""}
    assert c.validate(json.dumps(empty)) == empty
    with pytest.raises(ValueError, match="all three"):
        c.validate(json.dumps({"claim_ref": "actor-message-1-content",
                               "question": "What supports this?", "reason": ""}))
    with pytest.raises(ValueError, match=r"end with \?"):
        c.validate(json.dumps({"claim_ref": "actor-message-1-content",
                               "question": "What supports this",
                               "reason": "It controls the next commitment."}))


@pytest.mark.asyncio
async def test_live_worker_retries_invalid_review_then_injects_source_bound_question(
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
    seen = []
    actor_index = 0
    critic_index = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal actor_index, critic_index
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            critic = body.get("max_tokens") == 192
            if critic:
                critic_index += 1
                assert "tools" not in body
                packed = json.loads(body["messages"][1]["content"])
                evidence = packed["actor_evidence"]
                assert packed["review_trigger"] == "eight_actor_calls"
                assert all("HOST TEMPO CONTRACT" not in row["text"] for row in evidence)
                assert "HOST TEMPO CONTRACT" in json.dumps(
                    packed["full_context_for_interpretation_only"]
                )
                if critic_index == 1:
                    content = '{"wrong":"shape"}'
                else:
                    content = json.dumps({
                        "claim_ref": evidence[-1]["claim_ref"],
                        "question": "What new evidence will another inspection produce?",
                        "reason": "Repeated inspection delays a reversible evidence-producing action.",
                    })
                message = {"role": "assistant", "content": content}
                finish = "stop"
            else:
                actor_index += 1
                if actor_index < 10:
                    code = f'print("inspection {actor_index}")'
                else:
                    rendered = json.dumps(body["messages"])
                    assert "HOST EPISTEMIC QUESTION critic:2" in rendered
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
    critic_requests = [request for request in seen if request.get("max_tokens") == 192]
    assert len(critic_requests) == 2
    reviews = [json.loads(line) for path in tmp_path.rglob("critic_reviews.jsonl")
               for line in path.read_text().splitlines()]
    assert [row["status"] for row in reviews] == ["invalid_output", "completed"]
    assert all(row["review_trigger"] == "eight_actor_calls" for row in reviews)
    calls = [json.loads(line) for path in tmp_path.rglob("model_calls.jsonl")
             for line in path.read_text().splitlines()]
    assert [row["request_purpose"] for row in calls] == (
        ["actor"] * 8 + ["critic", "actor", "critic", "actor"]
    )
