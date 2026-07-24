"""Preflight or execute one frozen post-game shadow intervention bundle.

This CLI never creates or steps a game environment. Ten model calls are impossible
without both ``--execute-shadow`` and exact candidate/hash confirmations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from dotenv import load_dotenv


AGENT_ROOT = Path(__file__).resolve().parent
REPO = AGENT_ROOT.parents[2]
for import_root in (REPO, AGENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_repl_agent.client import OpenAICompatibleChatClient, serialize_request_payload
from llm_repl_agent.config import resolve_model_environment
from llm_repl_agent.experiment_protocol import (
    canonical_endpoint_metadata,
    endpoint_metadata_sha256,
    fetch_endpoint_metadata,
    load_frozen_runtime,
    participating_source_hashes,
    validate_endpoint_metadata,
)
from llm_repl_agent.intervention import (
    build_branch_payloads,
    build_shadow_schedule,
    candidate_id,
    execute_shadow_bundle,
    response_signature,
)


load_dotenv(REPO / ".env")
SHADOW_RUN_SCHEMA = "scientist-game-designer-player-shadow-run-v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _inside(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("candidate source archive escapes the run directory")
    return path


def _process_event(path: Path, event_index: int) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict) and value.get("event_index") == event_index:
            return value
    raise ValueError(f"process event {event_index} does not exist")


def load_shadow_preflight(
    *,
    run_dir: Path,
    candidate_path: Path,
    baseline_endpoint_path: Path,
) -> dict[str, Any]:
    manifest = _load_object(run_dir / "manifest.json")
    candidate = _load_object(candidate_path)
    baseline_endpoint = _load_object(baseline_endpoint_path)
    if candidate.get("schema") != "scientist-game-designer-player-candidate-workflow-v1":
        raise ValueError("candidate workflow has the wrong schema")
    if candidate.get("condition") != "enabled":
        raise ValueError("shadow intervention is available only for enabled runs")
    if candidate.get("second_candidate_permitted") is not False:
        raise ValueError("candidate workflow does not enforce the one-candidate rule")
    selector = candidate.get("selector")
    auditor = candidate.get("auditor")
    if not isinstance(selector, Mapping) or not isinstance(auditor, Mapping):
        raise ValueError("candidate workflow is missing selector or auditor records")
    if auditor.get("status") != "accepted":
        return {
            "schema": SHADOW_RUN_SCHEMA,
            "status": "auditor_rejected",
            "gameplay_invalidated": False,
            "calls_authorized": False,
        }
    if auditor.get("source_reconstruction_matches_selected_content") is not True:
        raise ValueError("auditor did not verify the selected reconstruction content")

    experiment = manifest.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("run manifest has no experiment commitment")
    generation = experiment.get("generation")
    run_id = manifest.get("run_id")
    if not isinstance(generation, str) or not isinstance(run_id, str):
        raise ValueError("run manifest is missing generation or run identity")
    if experiment.get("source_hashes") != participating_source_hashes():
        raise ValueError("current participating sources differ from the frozen gameplay run")
    if experiment.get("endpoint_metadata_sha256") != endpoint_metadata_sha256(
        baseline_endpoint
    ):
        raise ValueError("baseline endpoint does not match the frozen gameplay run")
    if selector.get("run_id") != run_id:
        raise ValueError("candidate run identity does not match the source run")
    event_index = selector.get("selected_process_event_index")
    tool_call_id = selector.get("selected_tool_call_id")
    if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 1:
        raise ValueError("candidate process event index is invalid")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("candidate tool call identity is invalid")
    for role, evidence in (
        ("selector", selector.get("evidence_event_indices")),
        ("auditor", auditor.get("evidence_event_indices")),
    ):
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 1
                or index > event_index
                for index in evidence
            )
        ):
            raise ValueError(f"{role} evidence must cite only events through the candidate decision")
    expected_candidate_id = candidate_id(
        generation=generation,
        run_id=run_id,
        event_index=event_index,
        tool_call_id=tool_call_id,
    )
    if selector.get("candidate_id") != expected_candidate_id:
        raise ValueError("candidate identity does not match the frozen construction")

    evidence_hashes = candidate.get("evidence_hashes")
    manifest_hashes = manifest.get("evidence_hashes")
    if evidence_hashes != manifest_hashes or not isinstance(evidence_hashes, Mapping):
        raise ValueError("candidate evidence hashes do not match the gameplay manifest")
    process_path = run_dir / "process.jsonl"
    if evidence_hashes.get("process.jsonl") != _sha256_file(process_path):
        raise ValueError("candidate process evidence hash does not match the immutable file")
    event = _process_event(process_path, event_index)
    if event.get("type") != "model_response":
        raise ValueError("selected candidate event is not a model response")
    archive_relative = selector.get("source_request_payload_archive")
    if not isinstance(archive_relative, str):
        raise ValueError("candidate does not identify its source request archive")
    if event.get("request_payload_archive") != archive_relative:
        raise ValueError("candidate source archive does not match the selected event")
    archive_path = _inside(run_dir, archive_relative)
    archive_hash = _sha256_file(archive_path)
    if evidence_hashes.get(archive_relative) != archive_hash:
        raise ValueError("source request archive is absent from immutable evidence hashes")
    if event.get("request_payload_sha256") != archive_hash:
        raise ValueError("source request archive does not match its pre-redaction commitment")
    if selector.get("source_request_payload_sha256") != archive_hash:
        raise ValueError("candidate source commitment does not match the selected event")
    source_payload = _load_object(archive_path)
    if sha256(serialize_request_payload(source_payload)).hexdigest() != archive_hash:
        raise ValueError("source payload canonical encoding does not match its commitment")

    selected_path = selector.get("selected_result_content_path")
    if not isinstance(selected_path, str):
        raise ValueError("candidate selected-result path is invalid")
    path_tokens = selected_path.split("/")
    if len(path_tokens) < 4 or path_tokens[1] != "messages":
        raise ValueError("candidate selected-result path is invalid")
    message_index = int(path_tokens[2])
    messages = source_payload.get("messages")
    if not isinstance(messages, list) or not (0 <= message_index < len(messages)):
        raise ValueError("candidate selected tool message is outside the source payload")
    selected_message = messages[message_index]
    if not isinstance(selected_message, Mapping) or selected_message.get("tool_call_id") != tool_call_id:
        raise ValueError("candidate tool call does not match the selected source message")
    decision_state = event.get("decision_state")
    if not isinstance(decision_state, Mapping):
        raise ValueError("selected model response has no committed production decision state")
    original_signature = selector.get("original_signature")
    if original_signature != response_signature(
        event.get("response", {}), decision_state=decision_state
    ):
        raise ValueError("candidate original signature does not match the selected response")
    changed_signature = selector.get("expected_changed_signature")
    if not isinstance(changed_signature, str):
        raise ValueError("candidate expected changed signature is invalid")

    schedule = build_shadow_schedule(candidate_id=expected_candidate_id, generation=generation)
    build_branch_payloads(
        source_payload,
        selected_result_content_path=selected_path,
        intact_content=selector.get("selected_result_content"),
        perturbed_content=selector.get("grounded_perturbation"),
        seed=int(schedule[0]["seed"]),
    )
    config = load_frozen_runtime()
    validate_endpoint_metadata(baseline_endpoint, config)
    current_endpoint = canonical_endpoint_metadata(
        fetch_endpoint_metadata(config),
        model=config.model,
        tag=config.provider.only[0],
    )
    if current_endpoint != baseline_endpoint:
        raise ValueError("current endpoint metadata differs from the frozen gameplay endpoint")
    return {
        "schema": SHADOW_RUN_SCHEMA,
        "status": "preflight_passed",
        "calls_authorized": True,
        "gameplay_invalidated": False,
        "candidate_id": expected_candidate_id,
        "candidate_sha256": _sha256_file(candidate_path),
        "generation": generation,
        "run_id": run_id,
        "source_payload": source_payload,
        "source_payload_sha256": archive_hash,
        "selected_result_content_path": selected_path,
        "intact_content": selector.get("selected_result_content"),
        "perturbed_content": selector.get("grounded_perturbation"),
        "original_signature": original_signature,
        "expected_changed_signature": changed_signature,
        "decision_state": dict(decision_state),
        "endpoint_metadata": baseline_endpoint,
        "schedule": schedule,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-endpoint-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-shadow", action="store_true")
    parser.add_argument("--confirm-candidate")
    parser.add_argument("--confirm-candidate-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise ValueError(f"shadow output already exists: {args.output}")
    preflight = load_shadow_preflight(
        run_dir=args.run_dir.resolve(),
        candidate_path=args.candidate.resolve(),
        baseline_endpoint_path=args.baseline_endpoint_metadata.resolve(),
    )
    public_preflight = {key: value for key, value in preflight.items() if key != "source_payload"}
    if preflight["status"] == "auditor_rejected" or not args.execute_shadow:
        public_preflight["executed_at"] = datetime.now(timezone.utc).isoformat()
        public_preflight["shadow_executed"] = False
        _write_json(args.output, public_preflight)
        return 0
    if args.confirm_candidate != preflight["candidate_id"]:
        raise ValueError("--confirm-candidate must exactly match the frozen candidate ID")
    if args.confirm_candidate_sha256 != preflight["candidate_sha256"]:
        raise ValueError("--confirm-candidate-sha256 must exactly match the candidate file")

    config = load_frozen_runtime()
    environment = resolve_model_environment(
        os.environ,
        model=config.model,
        endpoint=config.endpoint,
        profile=config.model_profile,
    )
    client = OpenAICompatibleChatClient(
        base_url=environment.endpoint,
        model=environment.model,
        api_key=environment.api_key,
        max_tokens=1,
        seed=0,
    )
    result = execute_shadow_bundle(
        source_payload=preflight["source_payload"],
        source_payload_sha256=preflight["source_payload_sha256"],
        candidate_id=preflight["candidate_id"],
        generation=preflight["generation"],
        selected_result_content_path=preflight["selected_result_content_path"],
        intact_content=preflight["intact_content"],
        perturbed_content=preflight["perturbed_content"],
        original_signature=preflight["original_signature"],
        expected_changed_signature=preflight["expected_changed_signature"],
        decision_state=preflight["decision_state"],
        complete_payload=lambda payload: client.complete_payload(payload),
    )
    result.update({
        "candidate_id": preflight["candidate_id"],
        "candidate_sha256": preflight["candidate_sha256"],
        "generation": preflight["generation"],
        "run_id": preflight["run_id"],
        "endpoint_metadata": preflight["endpoint_metadata"],
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "shadow_executed": True,
        "gameplay_invalidated": False,
    })
    _write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
