"""Freeze all captured requests, including failures; never contacts a model."""

import argparse
import hashlib
import json
from pathlib import Path
from corpus import validate
from runner import inspect_response

HERE = Path(__file__).resolve().parent
CAPTURE = (
    HERE.parents[2]
    / "analysis/deployment-probes/20260906-actor-capture/raw/actor-capture-20260906"
)


def freeze():
    assert len(list(CAPTURE.glob("*-request.json"))) == 14
    server_info = json.loads((CAPTURE / "server-info.json").read_text())
    assert server_info.get("server_args", server_info)["sampling_defaults"] == "model"
    events = [
        json.loads(line) for line in (CAPTURE / "events.jsonl").read_text().splitlines()
    ]
    records = []
    for index in range(14):
        request_path = CAPTURE / f"{index:04d}-request.json"
        response_path = CAPTURE / f"{index:04d}-response.json"
        raw = request_path.read_bytes()
        response_raw = response_path.read_bytes()
        request, response = json.loads(raw), json.loads(response_raw)
        request_hash = hashlib.sha256(raw).hexdigest()
        response_hash = hashlib.sha256(response_raw).hexdigest()
        before = [
            e
            for e in events
            if e.get("attempt") == index and e["event"] == "recorded_before_forward"
        ]
        after = [
            e
            for e in events
            if e.get("attempt") == index and e["event"] == "upstream_response_complete"
        ]
        assert len(before) == len(after) == 1
        assert before[0]["sha256"] == request_hash
        assert after[0]["sha256"] == response_hash and after[0]["status"] == 200
        records.append(
            {
                "id": f"capture-{index:04d}",
                "kind": "tool-continuation",
                "request_body": raw.decode(),
                "request": request,
                "captured_quality": inspect_response(response, request),
                "captured_usage": response["usage"],
                "provenance": {
                    "capture": "20260906-actor-capture",
                    "attempt": index,
                    "request_sha256": request_hash,
                    "response_sha256": response_hash,
                },
            }
        )
    value = {
        "schema": "actor-replay-v2",
        "records": records,
        "generation_config": json.loads(
            (CAPTURE / "generation_config.json").read_text()
        ),
        "captured_sampling_defaults": "model",
        "fixed_diagnostic_case": max(
            records, key=lambda r: r["captured_usage"]["prompt_tokens"]
        )["id"],
        "coverage": {
            "early_tool_continuations": 14,
            "long_history": False,
            "forced_action": False,
            "thinking_enabled": False,
            "invalid_tool_response_attempts": [4, 12, 13],
        },
        "sampling_policy": "Preserve omitted request fields; pinned engine with sampling-defaults=model and verified generation_config. No injected natural output cap.",
    }
    validate(value)
    return (json.dumps(value, indent=2) + "\n").encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = freeze()
    destination = HERE / "corpus.json"
    if args.check:
        assert destination.read_bytes() == data
    else:
        destination.write_bytes(data)
    print("14 paired captures verified and sealed")
