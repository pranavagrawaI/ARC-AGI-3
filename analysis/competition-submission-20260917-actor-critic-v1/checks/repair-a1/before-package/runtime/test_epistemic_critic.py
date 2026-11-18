import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


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


def sample_packet(c, *, claims=6, transitions=3):
    evidence = [{"claim_ref": f"actor-message-{i}-content", "kind": "assistant_content",
                 "text": f"turn {i}: the bar gained one cell, so filling it must be the goal. "
                         + "x" * 4000}
                for i in range(claims)]
    return c.critic_packet(
        trigger="prediction_mismatch", evidence=evidence,
        commitment={"turn": 12, "step": 14, "actions": ["CLICK 32 32"] * 10,
                    "rationale": "10 bar clicks, each +1 (19-28).",
                    "predicted_path": [{"expect": "move"}] * 10},
        transitions=[{"step": i, "action": "CLICK 32 32", "changed_cells": 1,
                      "shape_changed": False, "changed_bbox": [63, 60, 63, 60],
                      "state": "NOT_FINISHED", "levels_completed": 0}
                     for i in range(transitions)],
        budget=c.budget_view(wall_remaining=112.5, actions_executed=14, wall_elapsed=487.5),
        ledger={"pending": None, "reminders": 0, "retired_claims": [], "closed": []},
        state={"turn": 12, "step": 14, "level": 0, "state": "NOT_FINISHED",
               "levels_completed": 0, "available_actions": ["RESET", "CLICK"]})


def test_the_packet_is_source_bound_and_far_smaller_than_the_transcript():
    """Critic prompts reached 117k tokens and 1.32x the Actor's own context."""
    c = load_contract()
    packet = sample_packet(c, claims=40)
    transcript = json.dumps([{"role": "assistant", "content": "x" * 4000} for _ in range(40)])
    rendered = json.dumps(packet)
    assert len(rendered) < len(transcript) * 0.12
    refs = [row["claim_ref"] for row in packet["actor_claims"]]
    assert len(refs) == c.PACKET_CLAIMS == len(set(refs))
    assert refs == ["actor-message-37-content", "actor-message-38-content",
                    "actor-message-39-content"], "the newest claims are the live ones"
    # Truncated for transport, but every cited ref still resolves against the
    # complete ledger, so a bound question is never bound to a partial record.
    evidence = [{"claim_ref": f"actor-message-{i}-content", "kind": "assistant_content",
                 "text": "full " * 500} for i in range(40)]
    question = c.validate(json.dumps({
        "claim_ref": "actor-message-39-content",
        "question": "What observation links a filled bar to level completion?",
        "reason": "It decides whether 46 more clicks are evidence or waste."}))
    assert c.bind_claim_ref(question, evidence)["match"]["text"] == "full " * 500


def test_the_host_supplies_measured_change_without_cell_values():
    """CD82 spent two of seven reviews on which pixel moved; its input allowed that.

    The invariant is on what the host supplies. Actor claim text is Actor-authored
    and routinely mentions ``grid_rows`` because the Actor's own code calls it, so
    a substring scan over the whole packet would be the wrong check.
    """
    c = load_contract()
    packet = sample_packet(c)
    assert "full_context_for_interpretation_only" not in packet
    assert not any(key in packet for key in ("messages", "full_context", "transcript"))
    assert set(packet["measured_transitions"][0]) == {
        "step", "action", "changed_cells", "shape_changed", "changed_bbox",
        "state", "levels_completed"}
    host_supplied = {k: v for k, v in packet.items()
                     if k not in ("actor_claims", "commitment")}
    rendered = json.dumps(host_supplied)
    assert '"role"' not in rendered, "no message stream reaches the critic"
    for banned in ("grid_rows", "before_grid", "after_grid", "color_changes"):
        assert banned not in rendered, f"{banned} would invite mechanics debugging"
    # Actor code that names grid_rows must still be citable, not scrubbed.
    with_code = c.critic_packet(
        trigger="three_actions_observed",
        evidence=[{"claim_ref": "actor-message-3-tool-0", "kind": "assistant_tool_call",
                   "text": "g = self.grid_array(self.latest_state()['grid_rows'])"}],
        commitment=None, transitions=[], ledger={}, state={},
        budget=c.budget_view(wall_remaining=10.0, actions_executed=1, wall_elapsed=30.0))
    assert "grid_rows" in with_code["actor_claims"][0]["text"]


def test_the_packet_prices_the_budget_in_seconds_and_never_as_a_countdown():
    """All three v8 games died on wall clock with over 90% of their actions unspent,
    so the critic needs the price. But the v9 public run showed what a projected
    count does: the critic restated "affordable_actions_at_this_rate: 2" as "only
    two wall-clock actions remaining", the Actor adopted it as a rule and declared
    the game over with 152s and 193 actions left."""
    c = load_contract()
    budget = sample_packet(c)["budget"]
    assert budget["wall_seconds_remaining"] == 112.5
    assert budget["seconds_per_executed_action"] == 34.8
    assert "affordable_actions_at_this_rate" not in budget
    assert not any("afford" in key or "remaining_actions" in key for key in budget)
    assert "The host, not the Actor, ends a game" in budget["meaning"]


def test_recent_commitments_are_citable_and_carry_the_strategic_claim():
    """packet-live-probe-v2: the claim window filled with transition dumps and
    batch-permission repair; the objective-level claim lived in an earlier
    submit's rationale and the critic lost it (3 of 12 objective questions)."""
    c = load_contract()
    submitted = [
        {"turn": 5, "step": 5, "actions": ["CLICK 32 32"],
         "rationale": "if the bar still advances, then ANY click fills the bar"},
        {"turn": 6, "step": 6, "actions": ["CLICK 32 32"],
         "rationale": "fill the bar — click background center"},
        {"turn": 10, "step": 10, "actions": ["CLICK 32 32"] * 8,
         "rationale": "batch of 8 bar-filling clicks"},
    ]
    claims = c.commitment_claims(submitted)
    assert [row["claim_ref"] for row in claims] == [
        "actor-commitment-turn-5", "actor-commitment-turn-6", "actor-commitment-turn-10"]
    assert claims[-1]["text"].startswith("[step 10; submitted 8x CLICK 32 32]")
    question = c.validate(json.dumps({
        "claim_ref": "actor-commitment-turn-6",
        "question": "What observation links a filled bar to level completion?",
        "reason": "It decides whether the batch is evidence or waste."}))
    assert c.bind_claim_ref(question, claims)["match"]["kind"] == "actor_commitment"
    packet = c.critic_packet(
        trigger="three_actions_observed", evidence=[], commitment=None, transitions=[],
        ledger={}, state={}, commitments=claims,
        budget=c.budget_view(wall_remaining=10.0, actions_executed=1, wall_elapsed=30.0))
    assert packet["recent_commitments"] == claims
    assert len(c.commitment_claims(submitted * 4)) == c.PACKET_COMMITMENTS


def test_a_bounded_claim_window_still_carries_the_live_commitment():
    c = load_contract()
    packet = sample_packet(c, claims=40, transitions=20)
    assert packet["commitment"]["actions"] == ["CLICK 32 32"] * 10
    assert len(packet["measured_transitions"]) == c.PACKET_TRANSITIONS
    assert packet["measured_transitions"][-1]["step"] == 19


@pytest.mark.asyncio
async def test_live_worker_opens_one_objection_then_restates_it_without_a_model_call(
        monkeypatch, tmp_path):
    """The whole contract on the real Actor-to-critic-to-action path.

    S5I5 spent 93 of 600 wall seconds on four model-generated restatements of one
    objection and ended 32 seconds short of settling it. Here the second review
    boundary reaches a pending objection and is answered locally, a commitment
    that ignores the objection is refused, and a budgeted probe closes it.
    """
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
    refused = []

    OBJECTION = "the bar filling proves nothing about level completion"

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
                # The packet, not the transcript: no roles, no grids, no cell values.
                assert set(packed) >= {"commitment", "actor_claims", "measured_transitions",
                                       "budget", "objections"}
                assert "full_context_for_interpretation_only" not in packed
                rendered = json.dumps(packed)
                assert "HOST TEMPO CONTRACT" not in rendered
                assert '"role"' not in rendered
                claims = packed["actor_claims"]
                assert claims and all(row["claim_ref"].startswith("actor-message-")
                                      for row in claims)
                if critic_index == 1:
                    content = '{"wrong":"shape"}'
                else:
                    content = json.dumps({
                        "claim_ref": claims[-1]["claim_ref"],
                        "question": f"What observation shows that {OBJECTION} is false?",
                        "reason": "It decides whether the next batch is evidence or waste.",
                    })
                message = {"role": "assistant", "content": content}
                finish = "stop"
            else:
                actor_index += 1
                rendered = json.dumps(body["messages"])
                if actor_index < 10:
                    code = f'print("inspection {actor_index}")'
                elif actor_index == 10:
                    assert "HOST EPISTEMIC OBJECTION critic:2" in rendered
                    assert "is now OPEN" in rendered
                    code = 'print("inspection 10 — objection noted, not answered")'
                elif actor_index < 18:
                    code = f'print("inspection {actor_index}")'
                elif actor_index == 18:
                    # Eight further Actor calls reached a review boundary. It must
                    # have been answered locally, not by a second critic request.
                    assert "is STILL OPEN after 1 reminder(s); no new review was spent" in rendered
                    code = ('print(self.submit_actions(["UP"], '
                            'rationale="ignore the objection and act anyway"))')
                elif actor_index == 19:
                    assert "REJECTED: epistemic objection critic:2" in rendered
                    refused.append(True)
                    code = ('print(self.submit_actions(["UP"], rationale="settle it by acting",'
                            ' closes={"review_id":"critic:2","disposition":"probe",'
                            '"evidence":"run the bar to the end and watch for a level change",'
                            '"actions_budgeted":40}))')
                else:
                    code = 'print("post-submit")'
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
        "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "60", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
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
    assert len(critic_requests) == 2, "the second boundary must not buy a second review"
    assert refused == [True], "a commitment that ignores an open objection is refused"
    reviews = [json.loads(line) for path in tmp_path.rglob("critic_reviews.jsonl")
               for line in path.read_text().splitlines()]
    assert [row["status"] for row in reviews] == ["invalid_output", "completed", "restated"]
    assert reviews[-1]["reminders"] == 1
    objections = [json.loads(line) for path in tmp_path.rglob("objections.jsonl")
                  for line in path.read_text().splitlines()]
    assert [row["event"] for row in objections] == ["closed"]
    assert objections[0]["disposition"] == "probe"
    assert objections[0]["actions_budgeted"] == 40
    calls = [json.loads(line) for path in tmp_path.rglob("model_calls.jsonl")
             for line in path.read_text().splitlines()]
    assert [row["request_purpose"] for row in calls] == (
        ["actor"] * 8 + ["critic", "actor", "critic"] + ["actor"] * 10)
