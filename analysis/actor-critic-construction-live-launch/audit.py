"""Classify freeze/query/assimilate/critic from host events. Docs and proposed code are not use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REJECT_STATUSES = {
    "NO_FROZEN_MODEL",
    "TAMPERED_FREEZE",
    "PREDICT_WITHOUT_ASSIMILATE",
    "NO_FROZEN_STATE",
    "PREDICT_BRANCH_COUNT_UNSUPPORTED",
    "AMBIGUOUS_PENDING",
    "REJECT_UNKNOWN_TRANSITION",
    "REJECT_MISMATCH",
}

# GameAction.ACTION{n}.name is not the worker/transitions label. Id n is ARC name.
ARC_ACTION_ID_TO_NAME = {
    0: "RESET",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "SPACEBAR",
    6: "CLICK",
    7: "UNDO",
}
REQUIRED_TOOL_STATUS = {
    "query_frozen_predict": "predicted",
    "assimilate_frozen": "assimilated",
}


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_jsonl(run_dir: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direct = run_dir / name
    rows.extend(_jsonl(direct))
    for path in run_dir.rglob(name):
        if path != direct:
            rows.extend(_jsonl(path))
    return rows


def freeze_record(run_dir: Path) -> dict[str, Any] | None:
    seen: list[dict[str, Any]] = []
    paths = [run_dir / "FROZEN.json", *run_dir.rglob("FROZEN.json")]
    unique = []
    for path in paths:
        if path in unique or not path.is_file():
            continue
        unique.append(path)
        raw = _read_json(path)
        if not isinstance(raw, dict):
            continue
        source = raw.get("candidate_source")
        digest = raw.get("source_digest")
        if not isinstance(source, str) or not source.strip():
            continue
        expected = hashlib.sha256(source.encode()).hexdigest()
        if digest != expected:
            continue
        seen.append({"source_digest": expected, "candidate_source": source, "path": str(path)})
    digests = {row["source_digest"] for row in seen}
    if len(digests) != 1:
        return None
    return seen[0]


def normalize_action(action: Any) -> str | None:
    if not isinstance(action, str) or not action:
        return None
    if action.startswith("ACTION") and action[6:].isdigit():
        return ARC_ACTION_ID_TO_NAME.get(int(action[6:]))
    if action in ARC_ACTION_ID_TO_NAME.values():
        return action
    return None


def _tool_result_status(row: dict[str, Any]) -> str | None:
    raw = row.get("tool_result")
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def completed_host_events(run_dir: Path, event: str, digest: str) -> list[dict[str, Any]]:
    matched = []
    required = REQUIRED_TOOL_STATUS.get(event)
    for row in _all_jsonl(run_dir, "host-events.jsonl"):
        if row.get("event") != event:
            continue
        if row.get("status") != "COMPLETED":
            continue
        if row.get("source_digest") != digest:
            continue
        if not isinstance(row.get("action"), str) or not row["action"]:
            continue
        if type(row.get("step")) is not int:
            continue
        if required is not None and _tool_result_status(row) != required:
            continue
        matched.append(row)
    return matched


def critic_acall_completed(run_dir: Path) -> bool:
    responses = {row.get("call"): row for row in _jsonl(run_dir / "responses.jsonl")}
    for request in _jsonl(run_dir / "requests.jsonl"):
        body = request.get("request")
        if not isinstance(body, dict):
            continue
        if body.get("tools") is not None:
            continue
        if body.get("max_tokens") != 192:
            continue
        response = responses.get(request.get("call"))
        if not isinstance(response, dict):
            continue
        parsed = response.get("response")
        if isinstance(parsed, dict) and parsed.get("choices"):
            return True
    return False


def request_response_accounted(run_dir: Path) -> bool:
    requests = _jsonl(run_dir / "requests.jsonl")
    responses = _jsonl(run_dir / "responses.jsonl")
    if not requests or len(requests) != len(responses):
        return False
    if _jsonl(run_dir / "errors.jsonl"):
        return False
    for request, response in zip(requests, responses, strict=True):
        if request.get("call") != response.get("call"):
            return False
        parsed = response.get("response")
        if not isinstance(parsed, dict) or not parsed.get("choices"):
            return False
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            return False
        if type(usage.get("prompt_tokens")) is not int or type(usage.get("completion_tokens")) is not int:
            return False
        if min(usage["prompt_tokens"], usage["completion_tokens"]) < 0:
            return False
        if parsed.get("model") != "qwen/qwen3.8-27b" or parsed.get("provider") != "CoreWeave":
            return False
    return True


def bound_use(queries: list[dict[str, Any]], assimilations: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> bool:
    """A same-digest query/assimilate pair may precede later play steps."""
    executed = {(normalize_action(row.get("action")), row.get("step")) for row in transitions}
    for query in queries:
        for assim in assimilations:
            action = normalize_action(query["action"])
            if action is None or action != normalize_action(assim["action"]):
                continue
            if query["step"] + 1 != assim["step"]:
                continue
            if (action, assim["step"]) in executed:
                return True
    return False


def env_matches_transitions(env_steps: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> bool:
    if len(env_steps) != len(transitions):
        return False
    for step, transition in zip(env_steps, transitions, strict=True):
        env_action = normalize_action(step.get("action"))
        if env_action is None or env_action != normalize_action(transition.get("action")):
            return False
    return True


def provider_constructed_source(run_dir: Path, digest: str) -> bool:
    requests = {row.get("call"): row.get("request", {}) for row in _jsonl(run_dir / "requests.jsonl")}
    for receipt in _jsonl(run_dir / "responses.jsonl"):
        request = requests.get(receipt.get("call"), {})
        names = {(item.get("function") or {}).get("name") for item in (request.get("tools") or [])}
        if "submit_candidate" not in names:
            continue
        for choice in receipt.get("response", {}).get("choices", []):
            for call in (choice.get("message", {}).get("tool_calls") or []):
                function = call.get("function", {})
                if function.get("name") != "submit_candidate":
                    continue
                try:
                    arguments = json.loads(function.get("arguments", ""))
                except (ValueError, TypeError):
                    continue
                source = arguments.get("candidate_source") if isinstance(arguments, dict) else None
                if isinstance(source, str) and hashlib.sha256(source.encode()).hexdigest() == digest:
                    return True
    return False


def classify(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    frozen = freeze_record(run_dir)
    result = {}
    for path in [run_dir / "RESULT.json", *run_dir.rglob("RESULT.json")]:
        loaded = _read_json(path)
        if isinstance(loaded, dict) and loaded:
            result = loaded
            break
    transitions = _all_jsonl(run_dir, "transitions.jsonl")
    env_steps = _jsonl(run_dir / "environment-steps.jsonl") or _all_jsonl(run_dir, "environment-steps.jsonl")
    queries = completed_host_events(run_dir, "query_frozen_predict", frozen["source_digest"]) if frozen else []
    assimilations = completed_host_events(run_dir, "assimilate_frozen", frozen["source_digest"]) if frozen else []
    constructions = completed_host_events(run_dir, "request_construction", frozen["source_digest"]) if frozen else []
    critic_seen = critic_acall_completed(run_dir)
    accounted = request_response_accounted(run_dir)
    bound = bound_use(queries, assimilations, transitions) if frozen else False
    audit = {
        "frozen": frozen is not None,
        "result_status": result.get("construction_status"),
        "executed_transitions": len(transitions),
        "environment_steps": len(env_steps),
        "provider_requests": len(_jsonl(run_dir / "requests.jsonl")),
        "provider_responses": len(_jsonl(run_dir / "responses.jsonl")),
        "queried_frozen_predict": bool(queries),
        "assimilated_frozen": bool(assimilations),
        "requested_construction": bool(constructions),
        "provider_constructed_source": provider_constructed_source(run_dir, frozen["source_digest"]) if frozen else False,
        "critic_exercised": critic_seen,
        "critic_unexercised": not critic_seen,
        "request_response_accounted": accounted,
        "action_step_digest_bound": bound,
        "env_matches_transitions": env_matches_transitions(env_steps, transitions),
        "success": False,
    }
    if _jsonl(run_dir / "errors.jsonl"):
        audit["verdict"] = "TRANSPORT_FAILED"
    elif frozen is None:
        audit["verdict"] = "NO_FREEZE"
        audit["note"] = "A finished game without a digest-valid FROZEN.json is not integration success."
    elif not audit["provider_constructed_source"] or not constructions or result.get("construction_status") != "CANDIDATE_READY":
        audit["verdict"] = "CONSTRUCTION_RECEIPT_INCOMPLETE"
    elif not accounted:
        audit["verdict"] = "ACCOUNTING_INCOMPLETE"
    elif not transitions or not env_steps:
        audit["verdict"] = "NO_EXECUTED_TRANSITIONS"
    elif not audit["env_matches_transitions"]:
        audit["verdict"] = "TRANSITION_ENV_MISMATCH"
    elif not queries or not assimilations:
        audit["verdict"] = "FREEZE_WITHOUT_USE"
        audit["note"] = "API documentation or proposed execute_python code is not completed use."
    elif not bound:
        audit["verdict"] = "USE_NOT_BOUND_TO_EXECUTED_TRANSITION"
    elif not critic_seen:
        audit["verdict"] = "CRITIC_UNEXERCISED"
    else:
        audit["verdict"] = "INTEGRATION_EVENTS_PRESENT"
        audit["success"] = True
    return audit
