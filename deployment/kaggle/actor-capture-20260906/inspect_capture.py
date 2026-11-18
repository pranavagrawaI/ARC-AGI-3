"""Inspect captured attempts without labelling missing coverage as present."""

import argparse
import hashlib
import json
from pathlib import Path


def inspect(root):
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    rows = []
    for path in sorted(root.glob("*-request.json")):
        index = int(path.name.split("-")[0])
        raw = path.read_bytes()
        request = json.loads(raw)
        response_path = root / f"{index:04d}-response.json"
        response = (
            json.loads(response_path.read_text()) if response_path.exists() else {}
        )
        complete = [
            e
            for e in events
            if e.get("attempt") == index and e["event"] == "upstream_response_complete"
        ]
        admitted = [
            e
            for e in events
            if e.get("attempt") == index and e["event"] == "recorded_before_forward"
        ]
        assert (
            len(admitted) == 1
            and admitted[0]["sha256"] == hashlib.sha256(raw).hexdigest()
        )
        if complete:
            assert (
                len(complete) == 1
                and complete[0]["sha256"]
                == hashlib.sha256(response_path.read_bytes()).hexdigest()
            )
        rows.append(
            {
                "attempt": index,
                "request_sha256": hashlib.sha256(raw).hexdigest(),
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                "messages": len(request["messages"]),
                "tool_result_present": any(
                    m["role"] == "tool" for m in request["messages"]
                ),
                "tool_choice": request.get("tool_choice"),
                "tools": [
                    t.get("function", {}).get("name") for t in request.get("tools", [])
                ],
                "explicit_sampling": {
                    k: request[k]
                    for k in [
                        "temperature",
                        "top_p",
                        "top_k",
                        "max_tokens",
                        "chat_template_kwargs",
                    ]
                    if k in request
                },
                "http_status": complete[0]["status"] if complete else None,
                "response_id": response.get("id"),
            }
        )
    return {
        "status": "REVIEW_REQUIRED",
        "requests": rows,
        "failures": [e for e in events if e["event"] == "failure"],
        "note": "Resolve omitted sampling from retained configuration; reconcile with model_calls and server logs. No automatic long-history or forced-action coverage claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.capture_directory), indent=2))
