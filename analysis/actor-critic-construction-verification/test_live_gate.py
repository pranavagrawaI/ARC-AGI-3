"""Coordinator regressions: API documentation is not executed loop evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_audit():
    path = ROOT / "analysis/actor-critic-construction-live-launch/audit.py"
    spec = importlib.util.spec_from_file_location("independent_live_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_names_and_arbitrary_freeze_do_not_establish_use(tmp_path):
    (tmp_path / "FROZEN.json").write_text(json.dumps({"candidate_source": "pass"}))
    (tmp_path / "requests.jsonl").write_text(json.dumps({
        "purpose": "actor",
        "request": {"messages": [{"role": "system", "content":
            "API: request_construction query_frozen_predict assimilate_frozen"}]},
    }) + "\n")
    result = load_audit().classify(tmp_path)
    assert not result["success"], result


def test_requested_methods_without_completed_results_do_not_establish_use(tmp_path):
    (tmp_path / "FROZEN.json").write_text(json.dumps({"candidate_source": "pass"}))
    (tmp_path / "responses.jsonl").write_text(json.dumps({
        "response": {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "execute_python", "arguments": json.dumps({"code":
                "self.query_frozen_predict('UP'); self.assimilate_frozen()"}),
        }}]}}]},
    }) + "\n")
    result = load_audit().classify(tmp_path)
    assert not result["success"], result


def test_purpose_tag_and_completed_without_tool_result_are_not_success(tmp_path):
    source = "def initialize(observation):\n    return observation\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    ipc = tmp_path / "workspaces" / "g" / "ipc"
    ipc.mkdir(parents=True)
    (tmp_path / "workspaces" / "g" / "FROZEN.json").write_text(
        json.dumps({"candidate_source": source, "source_digest": digest})
    )
    (ipc / "request_lifecycle.jsonl").write_text(
        json.dumps({"event": "completed", "request_purpose": "critic"}) + "\n"
    )
    (ipc / "transitions.jsonl").write_text(json.dumps({"step": 2, "action": "UP"}) + "\n")
    (tmp_path / "environment-steps.jsonl").write_text(json.dumps({"action": "ACTION1"}) + "\n")
    (tmp_path / "host-events.jsonl").write_text(
        json.dumps({
            "event": "query_frozen_predict", "status": "COMPLETED",
            "action": "UP", "step": 1, "source_digest": digest,
        }) + "\n"
        + json.dumps({
            "event": "assimilate_frozen", "status": "COMPLETED",
            "action": "UP", "step": 2, "source_digest": digest,
        }) + "\n"
    )
    (tmp_path / "requests.jsonl").write_text(json.dumps({
        "call": 1, "purpose": "critic", "request": {"messages": []},
    }) + "\n")
    (tmp_path / "responses.jsonl").write_text(json.dumps({
        "call": 1,
        "response": {"choices": [{"message": {}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    }) + "\n")
    result = load_audit().classify(tmp_path)
    assert not result["success"], result
