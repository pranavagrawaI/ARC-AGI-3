"""Local request evidence, separate from gameplay sessions and gateway billing."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from local_wire import LOCAL_MODEL, valid_response, validate_identity, usage_receipt


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.is_file() else []


def accounting(root):
    root = Path(root)
    requests, responses = rows(root / "requests.jsonl"), rows(root / "responses.jsonl")
    valid_identity = False
    try:
        identity = json.loads((root / "MODEL-IDENTITY.json").read_text())
        validate_identity(identity["model"], identity["mounted_weights"], identity["startup_argv"])
        valid_identity = True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    valid = []
    role_counts = dict.fromkeys(("actor", "summarizer", "constructor", "critic"), 0)
    for request, response in zip(requests, responses):
        call = request.get("call")
        raw_path = root / "raw-http" / f"{call:02d}.body" if type(call) is int else root / "missing"
        parsed = response.get("response")
        if (call != response.get("call") or not valid_response(parsed)
                or request.get("request", {}).get("model") != LOCAL_MODEL
                or not raw_path.is_file() or json.loads(raw_path.read_bytes()) != parsed):
            continue
        role = request.get("purpose")
        if role not in role_counts:
            continue
        role_counts[role] += 1
        valid.append({"call": call, "purpose": role, "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "usage": usage_receipt(parsed)})
    integral = [r.get("call") for r in requests] == list(range(1, len(requests) + 1))
    complete = bool(requests) and valid_identity and integral and len(valid) == len(requests) == len(responses)
    complete = complete and not any(rows(root / name) for name in ("errors.jsonl", "delivery.jsonl", "local-rejections.jsonl"))
    return {"schema": "local-model-request-accounting-v1", "model": LOCAL_MODEL,
            "identity_status": "ESTABLISHED" if valid and valid_identity else "NOT_ESTABLISHED",
            "admitted": len(requests), "receipted": len(valid), "unreceipted_usage": "unknown" if len(valid) != len(requests) else "none",
            "roles": role_counts, "receipts": valid, "complete": complete,
            "known_usage": {k: sum(r["usage"][k] for r in valid) for k in ("prompt_tokens", "completion_tokens")}}


def expected_requests(result):
    record = result.get("local_requests")
    if result.get("status") != "passed" or not isinstance(record, dict) or record.get("schema") != "local-model-request-accounting-v1":
        raise ValueError("missing successful local request evidence")
    n = record.get("admitted")
    roles, receipts = record.get("roles"), record.get("receipts")
    if (record.get("complete") is not True or record.get("model") != LOCAL_MODEL
            or record.get("identity_status") != "ESTABLISHED" or type(n) is not int or not 1 <= n <= 23
            or record.get("receipted") != n or not isinstance(receipts, list) or len(receipts) != n
            or not isinstance(roles, dict) or set(roles) != {"actor", "summarizer", "constructor", "critic"}
            or any(type(v) is not int or v < 0 for v in roles.values()) or sum(roles.values()) != n
            or roles["actor"] + roles["summarizer"] > 20 or not roles["constructor"] or not roles["critic"]
            or [r.get("call") for r in receipts] != list(range(1, n + 1))):
        raise ValueError("local request accounting does not reconcile")
    if not result.get("audit", {}).get("success") or result.get("error") or result.get("fatal"):
        raise ValueError("local integration rejected")
    return n
