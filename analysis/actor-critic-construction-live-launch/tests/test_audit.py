from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

LIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIVE))
from audit import classify  # noqa: E402


def test_docs_and_proposed_code_are_not_success(tmp_path: Path) -> None:
    (tmp_path / "FROZEN.json").write_text(json.dumps({"candidate_source": "pass"}))
    (tmp_path / "requests.jsonl").write_text(json.dumps({
        "purpose": "actor",
        "request": {"messages": [{"role": "system", "content": "API: request_construction query_frozen_predict assimilate_frozen"}]},
    }) + "\n")
    assert classify(tmp_path)["success"] is False
    (tmp_path / "responses.jsonl").write_text(json.dumps({
        "response": {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "execute_python",
            "arguments": json.dumps({"code": "self.query_frozen_predict('UP'); self.assimilate_frozen()"}),
        }}]}}]},
    }) + "\n")
    assert classify(tmp_path)["success"] is False


def test_claimed_events_without_provider_construction_cannot_succeed(tmp_path: Path) -> None:
    source = "def initialize(observation):\n    return observation\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    ipc = tmp_path / "workspaces" / "g" / "ipc"
    ipc.mkdir(parents=True)
    (tmp_path / "workspaces" / "g" / "FROZEN.json").write_text(json.dumps({"candidate_source": source, "source_digest": digest}))
    (ipc / "transitions.jsonl").write_text(
        json.dumps({"step": 2, "action": "UP", "before": {"grid": [[1]]}, "after": {"grid": [[2]]}}) + "\n"
        + json.dumps({"step": 3, "action": "UP", "before": {"grid": [[2]]}, "after": {"grid": [[3]]}}) + "\n"
    )
    (tmp_path / "environment-steps.jsonl").write_text(
        json.dumps({"action": "ACTION1"}) + "\n" + json.dumps({"action": "ACTION1"}) + "\n"
    )
    (tmp_path / "host-events.jsonl").write_text(
        json.dumps({
            "event": "query_frozen_predict", "status": "COMPLETED", "action": "UP", "step": 1,
            "source_digest": digest, "tool_result": json.dumps({"status": "predicted"}),
        }) + "\n"
        + json.dumps({
            "event": "assimilate_frozen", "status": "COMPLETED", "action": "UP", "step": 2,
            "source_digest": digest, "tool_result": json.dumps({"status": "assimilated"}),
        }) + "\n"
    )
    req = {"call": 1, "purpose": "critic", "request": {"tools": None, "max_tokens": 192, "messages": []}}
    resp = {"call": 1, "response": {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}}
    (tmp_path / "requests.jsonl").write_text(json.dumps(req) + "\n")
    (tmp_path / "responses.jsonl").write_text(json.dumps(resp) + "\n")
    result = classify(tmp_path)
    assert result["success"] is False, result
