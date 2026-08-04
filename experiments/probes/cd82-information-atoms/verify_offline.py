"""Zero-call end-to-end verification for the cd82 information-atom probe."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

from probe import (
    SOURCE_PAYLOAD_SHA256,
    bytes_sha256,
    execute_bundle,
    load_conditions,
    load_frozen_branch,
    payload_sha256,
)


class FixtureResponse:
    def __init__(self, payload: dict, call: int) -> None:
        self.message = {
            "role": "assistant",
            "content": json.dumps(
                {"action": call, "data": None, "reason": f"offline fixture {call}"}
            ),
        }
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.response_id = f"offline-{call}"
        self.request_payload = deepcopy(payload)
        self.request_payload_sha256 = payload_sha256(payload)
        self.request_payload_encoding = "json-utf8-sort-keys-compact-v1"
        self.response_metadata = {"provider": "fixture"}
        self.response_body = {
            "id": self.response_id,
            "provider": "fixture",
            "choices": [{"message": self.message}],
            "usage": self.usage,
        }
        self.response_body_raw = json.dumps(self.response_body, sort_keys=True)
        self.response_body_sha256 = bytes_sha256(self.response_body_raw.encode())


def main() -> int:
    branch = load_frozen_branch()
    calls = []

    def complete(payload: dict):
        calls.append(deepcopy(payload))
        return FixtureResponse(payload, len(calls))

    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "bundle"
        result = execute_bundle(output_dir=output, complete_payload=complete)
        archived = {
            path.relative_to(output).as_posix(): bytes_sha256(path.read_bytes())
            for path in output.rglob("*")
            if path.is_file() and path.name != "result.json"
        }
        checks = {
            "source_payload_bound": branch["source_payload_sha256"] == SOURCE_PAYLOAD_SHA256,
            "four_conditions": list(load_conditions()) == result["conditions"],
            "four_calls": len(calls) == result["model_calls"] == 4,
            "zero_environment_actions": result["environment_actions"] == 0,
            "archived_evidence_matches": archived == result["evidence_hashes"],
            "semantic_verdict_deferred": result["semantic_verdict"] == "pending_human_review",
        }
    status = "passed" if all(checks.values()) else "failed"
    print(json.dumps({"schema": "cd82-information-atoms-offline-verification-v1", "status": status, "checks": checks}, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
