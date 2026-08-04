"""Prepare and execute the cd82 information-atom shadow probe.

The probe replays one archived model request under four cumulative information
conditions. It never constructs or steps a game environment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
from urllib.request import urlopen

from dotenv import load_dotenv


PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[2]
AGENT_ROOT = REPO / "experiments" / "agents" / "0002-llm-repl-agent"
for import_root in (REPO, AGENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_repl_agent.client import (  # noqa: E402
    ChatResponse,
    OpenAICompatibleChatClient,
    serialize_request_payload,
)
from llm_repl_agent.config import (  # noqa: E402
    OPENROUTER_DEFAULT_ENDPOINT,
    resolve_model_environment,
)


SCHEMA = "cd82-information-atoms-v1"
SOURCE_PAYLOAD_SHA256 = "3c7f703169b26456b4918281e8085facb70beb39ef46f5bcbe5f5bdd83a0600f"
MODEL = "google/gemma-4-31b-it"
PROVIDER_TAG = "modelrun/fp4"
EXECUTION_CONFIRMATION = "EXECUTE CD82 INFORMATION ATOMS V1"
CONDITION_FILES = {
    "c0-control": None,
    "c1-belief": "conditions/c1-belief.txt",
    "c2-linked-result": "conditions/c2-linked-result.txt",
    "c3-contradiction": "conditions/c3-contradiction.txt",
}
SOURCE_COMMITMENT_PATHS = (
    "experiments/probes/cd82-information-atoms/probe.py",
    "experiments/probes/cd82-information-atoms/branch.json",
    "experiments/probes/cd82-information-atoms/claim.md",
    "experiments/probes/cd82-information-atoms/review-rubric.json",
    "experiments/probes/cd82-information-atoms/conditions/c1-belief.txt",
    "experiments/probes/cd82-information-atoms/conditions/c2-linked-result.txt",
    "experiments/probes/cd82-information-atoms/conditions/c3-contradiction.txt",
    "experiments/agents/0002-llm-repl-agent/llm_repl_agent/client.py",
)


def bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return bytes_sha256(path.read_bytes())


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return bytes_sha256(serialize_request_payload(payload))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _process_events(path: Path) -> dict[int, dict[str, Any]]:
    events: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if isinstance(event, dict) and isinstance(event.get("event_index"), int):
            events[event["event_index"]] = event
    return events


def load_frozen_branch() -> dict[str, Any]:
    branch = _load_object(PROBE_ROOT / "branch.json")
    run_dir = REPO / str(branch["source_run"])
    manifest = _load_object(run_dir / "manifest.json")
    process_path = run_dir / "process.jsonl"
    source_relative = str(branch["source_request_payload"])
    source_path = run_dir / source_relative
    evidence_hashes = manifest.get("evidence_hashes")
    if not isinstance(evidence_hashes, Mapping):
        raise ValueError("source manifest has no evidence hashes")
    if evidence_hashes.get("process.jsonl") != file_sha256(process_path):
        raise ValueError("source process evidence drifted")
    if evidence_hashes.get(source_relative) != file_sha256(source_path):
        raise ValueError("source request evidence drifted")
    source_payload = _load_object(source_path)
    expected_hash = str(branch["source_payload_sha256"])
    if expected_hash != SOURCE_PAYLOAD_SHA256:
        raise ValueError("branch and code source commitments disagree")
    if file_sha256(source_path) != expected_hash or payload_sha256(source_payload) != expected_hash:
        raise ValueError("source payload does not match its canonical commitment")

    events = _process_events(process_path)
    response_event = events[int(branch["source_process_event"])]
    decision_event = events[int(branch["preceding_decision_event"])]
    transition_event = events[int(branch["preceding_transition_event"])]
    if response_event.get("type") != "model_response":
        raise ValueError("source branch event is not a model response")
    if response_event.get("request_payload_archive") != source_relative:
        raise ValueError("source branch event names a different request archive")
    if response_event.get("request_payload_sha256") != expected_hash:
        raise ValueError("source branch event request hash drifted")
    if decision_event.get("type") != "decision" or decision_event.get("action") != 3:
        raise ValueError("preceding decision is not LEFT")
    reason = decision_event.get("reasoning", {}).get("reason")
    if not isinstance(reason, str) or "Moving LEFT" not in reason:
        raise ValueError("preceding decision lacks the committed LEFT prediction")
    if (
        transition_event.get("type") != "transition"
        or transition_event.get("action") != 3
        or transition_event.get("changed_pixels") != 0
        or transition_event.get("before_digest") != transition_event.get("after_digest")
    ):
        raise ValueError("preceding transition is not the frozen zero-change result")

    result = deepcopy(branch)
    result["source_payload"] = source_payload
    result["source_run_dir"] = run_dir
    return result


def load_conditions() -> dict[str, str | None]:
    conditions: dict[str, str | None] = {}
    for condition_id, relative in CONDITION_FILES.items():
        if relative is None:
            conditions[condition_id] = None
            continue
        text = (PROBE_ROOT / relative).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"empty information condition: {condition_id}")
        conditions[condition_id] = text
    return conditions


def _observation_location(payload: Mapping[str, Any]) -> tuple[int, int, dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("payload has no messages")
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "text":
                continue
            try:
                value = json.loads(str(part.get("text")))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("environment_observation"), dict):
                candidates.append((message_index, part_index, value))
    if not candidates:
        raise ValueError("payload has no environment-observation message")
    return candidates[-1]


def current_observation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(_observation_location(payload)[2])


def build_condition_payload(
    source_payload: Mapping[str, Any], information: str | None
) -> dict[str, Any]:
    payload = deepcopy(dict(source_payload))
    if information is None:
        return payload
    message_index, part_index, observation = _observation_location(payload)
    if "information_probe" in observation:
        raise ValueError("source observation already has information_probe")
    observation["information_probe"] = information
    payload["messages"][message_index]["content"][part_index]["text"] = json.dumps(
        observation, separators=(",", ":"), sort_keys=True
    )
    return payload


def json_diff_paths(old: Any, new: Any, path: str = "") -> list[str]:
    if type(old) is not type(new):
        return [path]
    if isinstance(old, dict):
        result: list[str] = []
        for key in sorted(set(old) | set(new)):
            child = f"{path}.{key}" if path else str(key)
            if key not in old or key not in new:
                result.append(child)
            else:
                result.extend(json_diff_paths(old[key], new[key], child))
        return result
    if isinstance(old, list):
        if len(old) != len(new):
            return [path]
        result = []
        for index, (old_item, new_item) in enumerate(zip(old, new, strict=True)):
            result.extend(json_diff_paths(old_item, new_item, f"{path}[{index}]"))
        return result
    if isinstance(old, str) and isinstance(new, str) and path.endswith(".text"):
        try:
            old_json = json.loads(old)
            new_json = json.loads(new)
        except json.JSONDecodeError:
            return [] if old == new else [path]
        return json_diff_paths(old_json, new_json, path)
    return [] if old == new else [path]


def _response_selection(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return {"kind": "text", "content": content}
        if isinstance(value, dict) and isinstance(value.get("action"), int):
            return {
                "kind": "action",
                "action": value["action"],
                "data": value.get("data"),
                "reason": value.get("reason"),
            }
        return {"kind": "json", "value": value}
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        names = []
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, Mapping) else {}
            names.append(function.get("name"))
        return {"kind": "tool", "tool_names": names}
    return {"kind": "empty"}


def execute_bundle(
    *,
    output_dir: Path,
    complete_payload: Callable[[dict[str, Any]], ChatResponse],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"shadow output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    branch = load_frozen_branch()
    source = branch["source_payload"]
    conditions = load_conditions()
    records = []
    try:
        for index, (condition_id, information) in enumerate(conditions.items()):
            payload = build_condition_payload(source, information)
            request_path = output_dir / "requests" / f"{index:02d}-{condition_id}.json"
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_bytes(serialize_request_payload(payload))
            response = complete_payload(payload)
            if response.request_payload is not None and payload_sha256(
                response.request_payload
            ) != payload_sha256(payload):
                raise ValueError("transport request payload differs from the frozen branch")
            raw = response.response_body_raw
            if raw is None:
                raw = json.dumps(response.response_body or {}, separators=(",", ":"), sort_keys=True)
            response_path = output_dir / "responses" / f"{index:02d}-{condition_id}.raw"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(raw, encoding="utf-8")
            records.append(
                {
                    "condition": condition_id,
                    "information": information,
                    "request_sha256": file_sha256(request_path),
                    "response_sha256": file_sha256(response_path),
                    "response_id": response.response_id,
                    "selection": _response_selection(response.message),
                    "usage": dict(response.usage),
                    "response_metadata": dict(response.response_metadata or {}),
                }
            )
    except Exception as error:
        _write_json(
            output_dir / "failure.json",
            {
                "schema": SCHEMA,
                "status": "failed",
                "completed_conditions": len(records),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    evidence_hashes = {
        path.relative_to(output_dir).as_posix(): file_sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "conditions": list(conditions),
        "model_calls": len(records),
        "environment_actions": 0,
        "records": records,
        "evidence_hashes": evidence_hashes,
        "semantic_verdict": "pending_human_review",
        **dict(metadata or {}),
    }
    _write_json(output_dir / "result.json", result)
    return result


def _canonical_endpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(record.get(key))
        for key in (
            "name",
            "provider_name",
            "tag",
            "quantization",
            "context_length",
            "max_completion_tokens",
            "supported_parameters",
        )
    }


def fetch_endpoint_record() -> dict[str, Any]:
    url = f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints"
    with urlopen(url, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    endpoints = body.get("data", {}).get("endpoints", [])
    matches = [record for record in endpoints if record.get("tag") == PROVIDER_TAG]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {PROVIDER_TAG} endpoint; got {len(matches)}")
    return _canonical_endpoint(matches[0])


def _participating_source_hashes() -> dict[str, str]:
    return {relative: file_sha256(REPO / relative) for relative in SOURCE_COMMITMENT_PATHS}


def build_plan(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    branch = load_frozen_branch()
    conditions = load_conditions()
    source = branch["source_payload"]
    return {
        "schema": "cd82-information-atoms-plan-v1",
        "claim": "Find the minimum prefix-visible information that changes belief revision and next experiment selection after a contradicted LEFT prediction.",
        "source": {
            "run": str(branch["source_run"]),
            "process_event": branch["source_process_event"],
            "request_payload": branch["source_request_payload"],
            "request_payload_sha256": SOURCE_PAYLOAD_SHA256,
        },
        "conditions": [
            {
                "id": condition_id,
                "information": information,
                "payload_sha256": payload_sha256(build_condition_payload(source, information)),
            }
            for condition_id, information in conditions.items()
        ],
        "runtime": {
            "model": source["model"],
            "provider": source["provider"],
            "seed": source.get("seed"),
            "temperature": source.get("temperature"),
            "max_completion_tokens": source.get("max_completion_tokens"),
            "model_calls": len(conditions),
            "environment_actions": 0,
            "transport_retries": 0,
        },
        "endpoint": deepcopy(dict(endpoint)),
        "source_sha256": _participating_source_hashes(),
        "execution_confirmation": EXECUTION_CONFIRMATION,
        "automatic_semantic_acceptance": False,
    }


def prepare(plan_path: Path) -> dict[str, Any]:
    if plan_path.exists():
        raise ValueError(f"plan already exists: {plan_path}")
    plan = build_plan(fetch_endpoint_record())
    _write_json(plan_path, plan)
    return plan


def validate_execution_confirmation(
    *, confirmation: str, supplied_plan_sha256: str, actual_plan_sha256: str
) -> None:
    if confirmation != EXECUTION_CONFIRMATION:
        raise ValueError(f"confirmation must exactly equal {EXECUTION_CONFIRMATION!r}")
    if supplied_plan_sha256 != actual_plan_sha256:
        raise ValueError("confirmed plan sha256 does not match the frozen plan")


def resolve_openrouter_environment(environ: Mapping[str, str]):
    """Resolve the pinned model against the resolver's official OpenRouter URL."""

    return resolve_model_environment(
        environ,
        model=MODEL,
        endpoint=OPENROUTER_DEFAULT_ENDPOINT,
        profile="openrouter-generic",
    )


def execute(
    *,
    plan_path: Path,
    output_dir: Path,
    confirmation: str,
    confirmed_plan_sha256: str,
) -> dict[str, Any]:
    actual_plan_sha256 = file_sha256(plan_path)
    validate_execution_confirmation(
        confirmation=confirmation,
        supplied_plan_sha256=confirmed_plan_sha256,
        actual_plan_sha256=actual_plan_sha256,
    )
    plan = _load_object(plan_path)
    current_plan = build_plan(fetch_endpoint_record())
    if plan != current_plan:
        raise ValueError("current source, branch, conditions, or endpoint differs from frozen plan")
    load_dotenv(REPO / ".env")
    environment = resolve_openrouter_environment(os.environ)
    if not environment.api_key:
        raise RuntimeError("OpenRouter API key is unavailable")
    client = OpenAICompatibleChatClient(
        base_url=environment.endpoint,
        model=MODEL,
        api_key=environment.api_key,
        max_tokens=1,
        seed=0,
        rate_limit_retries=0,
    )
    return execute_bundle(
        output_dir=output_dir,
        complete_payload=lambda payload: client.complete_payload(payload),
        metadata={
            "plan_sha256": actual_plan_sha256,
            "endpoint": plan["endpoint"],
            "executed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--plan", type=Path, default=PROBE_ROOT / "artifacts" / "plan.json")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--plan", type=Path, default=PROBE_ROOT / "artifacts" / "plan.json")
    execute_parser.add_argument("--output", type=Path, default=PROBE_ROOT / "artifacts" / "bundle-v1")
    execute_parser.add_argument("--confirm", required=True)
    execute_parser.add_argument("--confirm-plan-sha256", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        plan = prepare(args.plan.resolve())
        print(json.dumps({"plan": str(args.plan), "plan_sha256": file_sha256(args.plan), "model_calls": plan["runtime"]["model_calls"], "environment_actions": 0}, sort_keys=True))
        return 0
    result = execute(
        plan_path=args.plan.resolve(),
        output_dir=args.output.resolve(),
        confirmation=args.confirm,
        confirmed_plan_sha256=args.confirm_plan_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
