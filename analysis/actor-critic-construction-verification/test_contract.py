"""Coordinator-owned behavioral regressions for the construction join.

Scripted constructor transport is explicit. These are not live-model results.
The real actor/critic worker path is verified separately by integration tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "analysis/actor-critic-construction-integrated"
sys.path.insert(0, str(CANDIDATE))


def capture_constructor_request(tmp_path):
    from host import run_episode
    from prefix import project_transitions

    transition = {
        "step": 1, "turn": 0, "action": "RIGHT",
        "before": {"grid": [[12, 4, 4]], "levels_completed": 0, "state": "NOT_FINISHED"},
        "after": {"grid": [[4, 12, 4]], "levels_completed": 0, "state": "NOT_FINISHED"},
    }
    path = tmp_path / "transitions.jsonl"
    path.write_text(json.dumps(transition) + "\n")
    requests = []

    def transport(body):
        requests.append(body)
        return {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "coordinator-evidence-request", "type": "function",
                "function": {"name": "request_evidence", "arguments": json.dumps({
                    "obstacle": "One move does not establish a general rule.",
                    "needed_observation": "Another transition.",
                    "decision_relevance": "Determine whether the first displacement repeats.",
                })},
            }]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }

    result = run_episode(project_transitions(path), transport, tmp_path / "construction")
    assert requests
    assert result["construction_status"] == "NEEDS_EVIDENCE"
    assert not (tmp_path / "construction/FROZEN.json").exists()
    return requests[0]


def test_default_constructor_can_access_observed_grids(tmp_path):
    request = capture_constructor_request(tmp_path)
    tool_names = {tool["function"]["name"] for tool in request.get("tools", [])}
    rendered = json.dumps(request["messages"], separators=(",", ":"))
    has_grids = "[[12,4,4]]" in rendered and "[[4,12,4]]" in rendered
    assert has_grids or "execute_python" in tool_names, "No grids in constructor prompt and no inspection tool: construction cannot consolidate observed information."


def test_generic_constructor_does_not_require_ls20_semantics(tmp_path):
    request = capture_constructor_request(tmp_path)
    system = "\n".join(message["content"] for message in request["messages"] if message["role"] == "system")
    assert "bar_left=integer" not in system
    assert "jointly specify block=" not in system


def test_tampered_freeze_is_rejected_before_query_execution(tmp_path, monkeypatch):
    from frozen_client import FrozenModelClient

    path = tmp_path / "FROZEN.json"
    path.write_text(json.dumps({
        "schema": "actor-critic-construction-frozen-v1",
        "candidate_source": "tampered source must never reach the candidate runner",
        "source_digest": "0" * 64,
    }))
    client = FrozenModelClient(tmp_path)
    invoked = []

    class UnexpectedRPC:
        def initialize(self, observation):
            invoked.append("initialize")
            return {}

        def predict(self, state, action):
            invoked.append("predict")
            return [{"state": {}, "commitments": {}}]

    monkeypatch.setattr(client, "_rpc", lambda source: UnexpectedRPC())
    try:
        result = client.query_frozen_predict("RIGHT", {"grid": [[12, 4, 4]]})
    except (ValueError, RuntimeError):
        result = "rejected"
    assert not invoked, "Digest mismatch reached candidate execution"
    assert "predicted" not in str(result)


def test_empty_prefix_does_not_consume_construction_opportunity(tmp_path, monkeypatch):
    import construction_service as service

    ipc = tmp_path / "ipc"
    ipc.mkdir()
    path = ipc / "transitions.jsonl"
    path.write_text("")
    first = service.construct_from_run(tmp_path, ipc)
    assert first.get("playable") is True
    assert not first.get("capped", False)
    assert not (tmp_path / service.MARKER).exists()


def test_constructor_transport_failure_is_playable(tmp_path, monkeypatch):
    import construction_service as service
    from host import FixtureTransport

    ipc = tmp_path / "ipc"
    ipc.mkdir()
    (ipc / "transitions.jsonl").write_text(json.dumps({
        "step": 1, "turn": 0, "action": "UP",
        "before": {"grid": [[0]], "levels_completed": 0},
        "after": {"grid": [[1]], "levels_completed": 0},
    }) + "\n")
    monkeypatch.setattr(service, "load_transport", lambda: FixtureTransport([]))
    result = service.construct_from_run(tmp_path, ipc)
    assert result["construction_status"] == "FAILED"
    assert result["playable"] is True
    assert not (tmp_path / "FROZEN.json").exists()


def test_join_import_does_not_mutate_donor_constructor():
    import subprocess

    code = (
        "import sys; "
        f"sys.path.insert(0, {str(CANDIDATE.parent / 'construction-policy-live-20260914T093351Z')!r}); "
        "import constructor; before=(constructor.SYSTEM, constructor.evaluate_candidate); "
        f"sys.path.insert(0, {str(CANDIDATE)!r}); "
        "import host; "
        "assert before == (constructor.SYSTEM, constructor.evaluate_candidate), 'donor constructor mutated on import'"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_abi_replay_never_reports_unmeasured_agreement(tmp_path):
    from evaluate_visible import evaluate_visible_abi
    from prefix import project_transitions

    path = tmp_path / "transitions.jsonl"
    path.write_text(json.dumps({"step": 1, "turn": 0, "action": "UP",
                                "before": {"grid": [[0]]}, "after": {"grid": [[1]]}}) + "\n")
    verdicts = []

    class Boundary:
        def invoke(self, source, method, args):
            if method == "initialize":
                return {"grid": [[0]]}
            if method == "predict":
                return [{"state": {"grid": [[0]]}, "commitments": {}, "declared_dependencies": []}]
            verdicts.append(args[2])
            return {"grid": args[1]["grid"]}

    result = evaluate_visible_abi("unused fixture source", project_transitions(path), boundary=Boundary())
    assert result["visible"]["status"] == "passed"
    assert verdicts and all(verdict.get("status") == "unscored" for verdict in verdicts)


def test_abi_replay_does_not_silently_discard_prediction_branches(tmp_path):
    from evaluate_visible import evaluate_visible_abi
    from prefix import project_transitions

    path = tmp_path / "transitions.jsonl"
    path.write_text(json.dumps({"step": 1, "turn": 0, "action": "UP",
                                "before": {"grid": [[0]]}, "after": {"grid": [[1]]}}) + "\n")
    assimilated = []

    class Boundary:
        def invoke(self, source, method, args):
            if method == "initialize":
                return {}
            if method == "predict":
                return [{"state": {"branch": i}, "commitments": {}, "declared_dependencies": []} for i in range(2)]
            assimilated.append(args[0]["branch"])
            return args[0]

    result = evaluate_visible_abi("unused fixture source", project_transitions(path), boundary=Boundary())
    assert result["visible"]["status"] == "failed" or set(assimilated) == {0, 1}
