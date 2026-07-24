"""Fail-closed authority artifact validation for consequential experiment stages."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .evidence import validate_run_evidence
from .experiment_protocol import (
    RunOutcome,
    assess_pair,
    endpoint_metadata_sha256,
    frozen_pair_schedule,
    gameplay_attempt_validity,
    load_frozen_runtime,
    validate_endpoint_metadata,
    validate_pair_commitments,
)


AGENT_ROOT = Path(__file__).resolve().parent.parent
REPO = AGENT_ROOT.parents[2]
PILOT_DECISION_SCHEMA = "scientist-game-designer-player-pilot-inspection-v2"
FINAL_FREEZE_SCHEMA = "scientist-game-designer-player-final-freeze-v2"
RELEASE_GATE_SCHEMA = "scientist-game-designer-player-release-gate-v1"
RELEASE_CANDIDATE_SCHEMA = "scientist-game-designer-player-release-candidate-v1"
PILOT_APPROVAL_REQUEST_SCHEMA = (
    "scientist-game-designer-player-execution-approval-request-v2"
)
PILOT_APPROVAL_SCHEMA = "scientist-game-designer-player-execution-approval-v2"
AUTHORITY_CONSUMPTION_SCHEMA = "scientist-game-designer-player-authority-consumption-v1"
MEMBER_AUTHORITY_CONSUMPTION_SCHEMA = (
    "scientist-game-designer-player-member-authority-consumption-v1"
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def resolve_artifact_reference(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validated_attempt(
    attempts: Any,
    *,
    valid_attempt: int,
    maximum_attempts: int,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(attempts, list) or not 1 <= valid_attempt <= maximum_attempts:
        raise ValueError(f"{label} has no authorized attempt ledger")
    if len(attempts) != valid_attempt:
        raise ValueError(f"{label} attempt ledger does not stop at the first valid pair")
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping) or attempt.get("attempt") != index:
            raise ValueError(f"{label} attempt ledger is duplicated or out of order")
        expected_valid = index == valid_attempt
        if attempt.get("valid") is not expected_valid:
            raise ValueError(f"{label} attempt validity ledger is inconsistent")
    return attempts[-1]


def _manifest_outcome(manifest: Mapping[str, Any]) -> RunOutcome:
    levels = manifest.get("levels_completed")
    completions = manifest.get("completion_action_counts")
    if (
        isinstance(levels, bool)
        or not isinstance(levels, int)
        or levels < 0
        or not isinstance(completions, list)
    ):
        raise ValueError("member outcome telemetry is malformed")

    def completion(index: int) -> int | None:
        if len(completions) <= index:
            return None
        value = completions[index]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("member completion telemetry is malformed")
        return value

    return RunOutcome(
        levels_completed=levels,
        level_1_completion_action=completion(0),
        level_2_completion_action=completion(1),
    )


def validate_pilot_decision(
    decision_path: Path,
    *,
    pilot_pair_dir: Path,
    generation: str,
) -> dict[str, Any]:
    """Resolve an independently produced decision and require exact pilot acceptance.

    Legacy inspection records are deliberately not promotable: the v2 decision binds the
    accepted pilot, the authority record, included-run scope, and worst-case exposure.
    """

    decision_path = decision_path.resolve()
    pilot_pair_dir = pilot_pair_dir.resolve()
    decision = load_object(decision_path, label="pilot decision")
    if (
        decision.get("schema") != PILOT_DECISION_SCHEMA
        or decision.get("decision") != "accepted"
        or decision.get("accepted_for_final_freeze") is not True
    ):
        raise ValueError("final freeze requires an accepted pilot decision artifact")
    if decision_path.parent != pilot_pair_dir:
        raise ValueError("pilot decision must be stored beside the pilot it decides")
    if (
        decision.get("generation") != generation
        or decision.get("pair_id") != "P"
        or decision.get("excluded_pilot") is not True
    ):
        raise ValueError("accepted pilot decision identifies another pilot")
    referenced_pair_dir = resolve_artifact_reference(
        decision.get("pilot_pair_dir"), label="pilot decision pair"
    )
    if referenced_pair_dir != pilot_pair_dir:
        raise ValueError("accepted pilot decision references another pair directory")

    pilot_result_path = pilot_pair_dir / "pair-result.json"
    pilot = load_object(pilot_result_path, label="pilot pair result")
    if decision.get("pilot_pair_result_sha256") != file_sha256(pilot_result_path):
        raise ValueError("accepted pilot decision does not bind the exact pair result")
    if (
        pilot.get("schema") != "scientist-game-designer-player-pair-v1"
        or pilot.get("generation") != generation
        or pilot.get("pair_id") != "P"
        or pilot.get("included") is not False
        or pilot.get("status") != "pair_complete"
        or pilot.get("gameplay_executed") is not True
    ):
        raise ValueError("accepted pilot decision references an invalid excluded pilot")

    blockers = decision.get("blocking_findings")
    if not isinstance(blockers, list) or blockers:
        raise ValueError("accepted pilot decision must have no blocking findings")
    if not _nonblank(decision.get("authority")) or not _nonblank(
        decision.get("approval_record")
    ):
        raise ValueError("accepted pilot decision lacks its promotion authority record")
    if decision.get("scope") != {
        "accept_pilot": True,
        "create_final_freeze": True,
        "execute_included_pairs": True,
        "execute_shadow_interventions": False,
    }:
        raise ValueError("accepted pilot decision does not grant the exact next stage")

    exposure = decision.get("maximum_exposure")
    if not isinstance(exposure, Mapping):
        raise ValueError("accepted pilot decision has no included-run exposure ceiling")
    attempts = exposure.get("maximum_pair_attempts_per_pair")
    if isinstance(attempts, bool) or attempts not in {1, 2}:
        raise ValueError("accepted pilot decision has an invalid included-pair attempt ceiling")
    config = load_frozen_runtime()
    expected_exposure = {
        "included_pairs": 3,
        "members_per_pair": 2,
        "maximum_pair_attempts_per_pair": attempts,
        "environment_actions": (
            3 * 2 * attempts * config.budgets.total_actions_invariant
        ),
        "model_calls": 3 * 2 * attempts * config.runtime.max_model_calls,
    }
    if dict(exposure) != expected_exposure:
        raise ValueError("accepted pilot decision exposure does not match frozen runtime limits")
    return decision


def validate_release_gate(
    release_gate_dir: Path,
    *,
    source_hashes: Mapping[str, str],
    endpoint_metadata_hash: str | None = None,
) -> dict[str, Any]:
    """Recompute the complete release bundle used by later authority artifacts."""

    release_gate_dir = release_gate_dir.resolve()
    release_path = release_gate_dir / "release-gate-results.json"
    candidate_path = release_gate_dir / "release-candidate-manifest.json"
    release = load_object(release_path, label="release gate results")
    candidate = load_object(candidate_path, label="release candidate manifest")
    if (
        release.get("schema") != RELEASE_GATE_SCHEMA
        or release.get("passed") is not True
        or release.get("pilot_authorized_by_gate") is not True
        or release.get("source_hashes") != dict(source_hashes)
    ):
        raise ValueError("release gate does not bind the exact passing source set")
    compatibility = release.get("compatibility")
    if not isinstance(compatibility, Mapping) or not _nonblank(
        compatibility.get("endpoint_metadata_sha256")
    ):
        raise ValueError("release gate does not bind endpoint compatibility evidence")
    if (
        endpoint_metadata_hash is not None
        and compatibility.get("endpoint_metadata_sha256") != endpoint_metadata_hash
    ):
        raise ValueError("release gate endpoint commitment mismatch")
    if (
        candidate.get("schema") != RELEASE_CANDIDATE_SCHEMA
        or candidate.get("release_gate_passed") is not True
        or candidate.get("release_gate_results_sha256") != file_sha256(release_path)
        or candidate.get("participating_source_hashes") != dict(source_hashes)
    ):
        raise ValueError("release candidate does not bind the exact release gate")

    artifacts = release.get("artifacts")
    committed = candidate.get("generated_artifact_hashes")
    expected_logical = {
        "endpoint_metadata",
        "sandbox_results",
        "synthetic_test_results",
        "intervention_test_vectors",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_logical:
        raise ValueError("release supporting artifact inventory mismatch")
    if not isinstance(committed, Mapping):
        raise ValueError("release candidate is missing generated artifact hashes")
    expected_filenames = {"release-gate-results.json"}
    for logical_name in sorted(expected_logical):
        relative = artifacts.get(logical_name)
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"invalid release supporting artifact path: {logical_name}")
        path = (release_gate_dir / relative).resolve()
        if path.parent != release_gate_dir or not path.is_file():
            raise ValueError(f"missing release supporting artifact: {logical_name}")
        expected_filenames.add(path.name)
        if committed.get(path.name) != file_sha256(path):
            raise ValueError(f"release supporting artifact hash mismatch: {logical_name}")
    if set(committed) != expected_filenames:
        raise ValueError("release candidate generated artifact inventory mismatch")
    if committed.get("release-gate-results.json") != file_sha256(release_path):
        raise ValueError("release gate results hash mismatch")
    endpoint_path = release_gate_dir / str(artifacts["endpoint_metadata"])
    endpoint = load_object(endpoint_path, label="release endpoint metadata")
    validate_endpoint_metadata(endpoint)
    if endpoint_metadata_sha256(endpoint) != compatibility.get("endpoint_metadata_sha256"):
        raise ValueError("release endpoint metadata hash mismatch")
    return release


def validate_pilot_approval(
    approval_path: Path,
    *,
    generation: str,
    release_gate_dir: Path,
    source_hashes: Mapping[str, str],
    endpoint_metadata_hash: str,
) -> dict[str, Any]:
    """Validate a human pilot grant against its exact request and machine gate."""

    approval_path = approval_path.resolve()
    approval = load_object(approval_path, label="pilot approval")
    if approval.get("schema") != PILOT_APPROVAL_SCHEMA:
        raise ValueError("pilot gameplay requires a v2 pilot approval artifact")
    request_name = approval.get("approval_request")
    if (
        not isinstance(request_name, str)
        or not request_name
        or Path(request_name).is_absolute()
        or Path(request_name).name != request_name
    ):
        raise ValueError("pilot approval request must be a sibling artifact")
    request_path = approval_path.parent / request_name
    request = load_object(request_path, label="pilot approval request")
    if approval.get("approval_request_sha256") != file_sha256(request_path):
        raise ValueError("pilot approval request hash mismatch")

    release_gate_dir = release_gate_dir.resolve()
    release = validate_release_gate(
        release_gate_dir,
        source_hashes=source_hashes,
        endpoint_metadata_hash=endpoint_metadata_hash,
    )
    release_path = release_gate_dir / "release-gate-results.json"
    candidate_path = release_gate_dir / "release-candidate-manifest.json"
    try:
        schedule = next(row for row in frozen_pair_schedule() if row.pair_id == "P")
    except StopIteration as error:  # pragma: no cover - frozen protocol invariant
        raise ValueError("frozen pilot schedule is missing") from error
    config = load_frozen_runtime()
    request_release_dir = resolve_artifact_reference(
        request.get("release_gate_dir"), label="approval request release gate"
    )
    request_exposure = request.get("maximum_exposure")
    if not isinstance(request_exposure, Mapping):
        raise ValueError("pilot approval request has no exposure ceiling")
    maximum_attempts = request_exposure.get("maximum_pair_attempts")
    if isinstance(maximum_attempts, bool) or maximum_attempts not in {1, 2}:
        raise ValueError("pilot approval request has an invalid attempt ceiling")
    expected_exposure = {
        "maximum_pair_attempts": maximum_attempts,
        "members_per_attempt": 2,
        "environment_actions": (
            2 * maximum_attempts * config.budgets.total_actions_invariant
        ),
        "model_calls": 2 * maximum_attempts * config.runtime.max_model_calls,
    }
    request_valid = (
        request.get("schema") == PILOT_APPROVAL_REQUEST_SCHEMA
        and request.get("status") == "awaiting_explicit_human_approval"
        and request.get("requested_action") == "execute_excluded_pilot_pair_P"
        and request.get("generation") == generation
        and request.get("pair_id") == "P"
        and request.get("included_in_confirmatory_analysis") is False
        and request.get("execution_order") == list(schedule.order)
        and request.get("environment_seed") == schedule.environment_seed
        and request.get("requested_model_seed") == schedule.requested_seed
        and request.get("endpoint") == config.endpoint
        and request.get("model") == config.model
        and request.get("provider") == config.provider.only[0]
        and request_release_dir == release_gate_dir
        and request.get("release_gate_results_sha256") == file_sha256(release_path)
        and request.get("release_candidate_manifest_sha256")
        == file_sha256(candidate_path)
        and request.get("endpoint_metadata_sha256") == endpoint_metadata_hash
        and request.get("source_hashes") == dict(source_hashes)
        and dict(request_exposure) == expected_exposure
        and release.get("source_hashes") == dict(source_hashes)
    )
    if not request_valid:
        raise ValueError("pilot approval request does not bind the exact execution")
    expected_scope = {
        "execute_excluded_pilot": True,
        "accept_pilot": False,
        "create_final_freeze": False,
        "execute_included_pairs": False,
        "execute_shadow_interventions": False,
    }
    approval_valid = (
        approval.get("status") == "approved"
        and _nonblank(approval.get("authority"))
        and _nonblank(approval.get("approval_record"))
        and approval.get("generation") == generation
        and approval.get("pair_id") == "P"
        and approval.get("scope") == expected_scope
        and approval.get("maximum_exposure") == expected_exposure
        and approval.get("release_gate_results_sha256") == file_sha256(release_path)
        and approval.get("release_candidate_manifest_sha256")
        == file_sha256(candidate_path)
        and approval.get("endpoint_metadata_sha256") == endpoint_metadata_hash
        and approval.get("source_hashes") == dict(source_hashes)
    )
    if not approval_valid:
        raise ValueError("pilot approval does not grant the exact bounded execution")
    return {
        "kind": "pilot_approval",
        "path": str(approval_path),
        "sha256": file_sha256(approval_path),
        "maximum_pair_attempts": maximum_attempts,
        "approval": approval,
        "request": request,
    }


def _write_new_json(path: Path, value: Mapping[str, Any], *, conflict: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(conflict) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def consume_pair_authority(
    *,
    generation_root: Path,
    pair_root: Path,
    generation: str,
    pair_id: str,
    authority_path: Path,
    authority_kind: str,
    maximum_pair_attempts: int,
) -> Path:
    """Atomically burn one pair grant and bind every permitted member run path."""

    generation_root = generation_root.resolve()
    pair_root = pair_root.resolve()
    authority_path = authority_path.resolve()
    if (
        not generation.strip()
        or pair_id not in {"P", "1", "2", "3"}
        or authority_kind not in {"pilot_approval", "final_freeze"}
        or isinstance(maximum_pair_attempts, bool)
        or maximum_pair_attempts not in {1, 2}
        or pair_root.parent != generation_root
        or not authority_path.is_file()
    ):
        raise ValueError("invalid pair authority consumption request")
    allowed_runs = [
        {
            "attempt": attempt,
            "condition": condition,
            "run_dir": str(pair_root / f"attempt-{attempt}" / condition),
        }
        for attempt in range(1, maximum_pair_attempts + 1)
        for condition in ("enabled", "disabled")
    ]
    record = {
        "schema": AUTHORITY_CONSUMPTION_SCHEMA,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
        "generation": generation,
        "pair_id": pair_id,
        "pair_root": str(pair_root),
        "authority_kind": authority_kind,
        "authority_path": str(authority_path),
        "authority_sha256": file_sha256(authority_path),
        "maximum_pair_attempts": maximum_pair_attempts,
        "allowed_runs": allowed_runs,
    }
    record_path = generation_root / ".authority" / f"pair-{pair_id}.json"
    _write_new_json(
        record_path,
        record,
        conflict=f"pair {pair_id} authority already consumed",
    )
    return record_path


def _validate_pair_commitment_artifact(
    pair: Mapping[str, Any],
    *,
    pair_root: Path,
    generation: str,
    pair_id: str,
    source_hashes: Mapping[str, str],
    endpoint_metadata_hash: str,
) -> dict[str, Mapping[str, Any]]:
    commitments_path = pair_root / "pair-commitments.json"
    if (
        not commitments_path.is_file()
        or pair.get("pair_commitments_sha256") != file_sha256(commitments_path)
    ):
        raise ValueError("pair commitments hash mismatch")
    document = load_object(commitments_path, label="pair commitments")
    commitments = document.get("commitments")
    if (
        document.get("schema") != "scientist-game-designer-player-pair-v1"
        or set(document) != {"schema", "commitments"}
        or not isinstance(commitments, list)
        or len(commitments) != 2
    ):
        raise ValueError("pair commitments artifact is incomplete")
    by_condition = {
        row.get("condition"): row
        for row in commitments
        if isinstance(row, Mapping) and row.get("condition") in {"enabled", "disabled"}
    }
    if set(by_condition) != {"enabled", "disabled"}:
        raise ValueError("pair commitments omit a condition")
    enabled = by_condition.get("enabled")
    disabled = by_condition.get("disabled")
    if not isinstance(enabled, Mapping) or not isinstance(disabled, Mapping):
        raise ValueError("pair commitments omit a condition")
    validate_pair_commitments(enabled, disabled)
    for commitment in (enabled, disabled):
        if (
            commitment.get("generation") != generation
            or commitment.get("pair_id") != pair_id
            or commitment.get("source_hashes") != dict(source_hashes)
            or commitment.get("endpoint_metadata_sha256") != endpoint_metadata_hash
        ):
            raise ValueError("pair commitments do not bind the exact authorized execution")
    return {"enabled": enabled, "disabled": disabled}


def _validate_pilot_pair_result(
    pilot_pair_dir: Path,
    *,
    generation: str,
    source_hashes: Mapping[str, str],
    endpoint_metadata_hash: str,
    release_gate_results_hash: str,
    release_gate_dir: Path,
) -> dict[str, Any]:
    pilot_pair_dir = pilot_pair_dir.resolve()
    result_path = pilot_pair_dir / "pair-result.json"
    pilot = load_object(result_path, label="pilot pair result")
    if (
        pilot.get("schema") != "scientist-game-designer-player-pair-v1"
        or pilot.get("generation") != generation
        or pilot.get("pair_id") != "P"
        or pilot.get("included") is not False
        or pilot.get("status") != "pair_complete"
        or pilot.get("gameplay_executed") is not True
        or pilot.get("source_hashes") != dict(source_hashes)
        or pilot.get("endpoint_metadata_sha256") != endpoint_metadata_hash
        or pilot.get("release_gate_results_sha256") != release_gate_results_hash
    ):
        raise ValueError("pilot result is not a complete current-generation excluded pilot")
    commitments_by_condition = _validate_pair_commitment_artifact(
        pilot,
        pair_root=pilot_pair_dir,
        generation=generation,
        pair_id="P",
        source_hashes=source_hashes,
        endpoint_metadata_hash=endpoint_metadata_hash,
    )
    valid_attempt = pilot.get("valid_attempt")
    attempts = pilot.get("attempts")
    if isinstance(valid_attempt, bool) or not isinstance(valid_attempt, int):
        raise ValueError("pilot result has no valid pair attempt")
    consumption_path = resolve_artifact_reference(
        pilot.get("authority_consumption"), label="pilot authority consumption"
    )
    consumption = load_object(consumption_path, label="pilot authority consumption")
    authority_path = resolve_artifact_reference(
        consumption.get("authority_path"), label="pilot approval"
    )
    if (
        pilot.get("authority_consumption_sha256") != file_sha256(consumption_path)
        or consumption.get("schema") != AUTHORITY_CONSUMPTION_SCHEMA
        or consumption.get("generation") != generation
        or consumption.get("pair_id") != "P"
        or consumption.get("authority_kind") != "pilot_approval"
        or resolve_artifact_reference(
            consumption.get("pair_root"), label="pilot consumed pair root"
        )
        != pilot_pair_dir
        or consumption.get("authority_sha256") != file_sha256(authority_path)
    ):
        raise ValueError("pilot result does not bind its consumed approval")
    validated_approval = validate_pilot_approval(
        authority_path,
        generation=generation,
        release_gate_dir=release_gate_dir,
        source_hashes=source_hashes,
        endpoint_metadata_hash=endpoint_metadata_hash,
    )
    maximum_attempts = validated_approval["maximum_pair_attempts"]
    expected_allowed_runs = [
        {
            "attempt": attempt,
            "condition": condition,
            "run_dir": str(pilot_pair_dir / f"attempt-{attempt}" / condition),
        }
        for attempt in range(1, maximum_attempts + 1)
        for condition in ("enabled", "disabled")
    ]
    if (
        consumption.get("maximum_pair_attempts") != maximum_attempts
        or consumption.get("allowed_runs") != expected_allowed_runs
        or not 1 <= valid_attempt <= maximum_attempts
    ):
        raise ValueError("pilot result exceeds its approved execution slots")
    valid_attempt_record = _validated_attempt(
        attempts,
        valid_attempt=valid_attempt,
        maximum_attempts=maximum_attempts,
        label="pilot result",
    )
    members = valid_attempt_record.get("members")
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("pilot valid attempt does not preserve both members")
    by_condition = {
        member.get("condition"): member
        for member in members
        if isinstance(member, Mapping)
        and member.get("condition") in {"enabled", "disabled"}
    }
    if set(by_condition) != {"enabled", "disabled"}:
        raise ValueError("pilot valid attempt does not preserve both conditions")
    for condition in ("enabled", "disabled"):
        member = by_condition[condition]
        commitment = commitments_by_condition[condition]
        expected_run_dir = pilot_pair_dir / f"attempt-{valid_attempt}" / condition
        run_dir = resolve_artifact_reference(member.get("run_dir"), label="pilot member run")
        if run_dir != expected_run_dir.resolve():
            raise ValueError("pilot member run path does not match the attempt ledger")
        manifest_path = run_dir / "manifest.json"
        manifest = load_object(manifest_path, label="pilot member manifest")
        evidence = manifest.get("evidence_hashes")
        experiment = manifest.get("experiment")
        member_consumption_path = resolve_artifact_reference(
            experiment.get("member_authority_consumption")
            if isinstance(experiment, Mapping)
            else None,
            label="pilot member authority consumption",
        )
        member_consumption = load_object(
            member_consumption_path, label="pilot member authority consumption"
        )
        if (
            member.get("validity") != "valid_contestant_result"
            or member.get("manifest_sha256") != file_sha256(manifest_path)
            or not isinstance(evidence, Mapping)
            or not validate_run_evidence(run_dir, evidence)
            or gameplay_attempt_validity(manifest, commitment, run_dir=run_dir)
            != "valid_contestant_result"
            or not isinstance(experiment, Mapping)
            or experiment.get("generation") != generation
            or experiment.get("pair_id") != "P"
            or experiment.get("condition") != condition
            or experiment.get("included") is not False
            or experiment.get("source_hashes") != dict(source_hashes)
            or experiment.get("endpoint_metadata_sha256") != endpoint_metadata_hash
            or resolve_artifact_reference(
                experiment.get("pair_authority_consumption"),
                label="pilot pair authority consumption",
            )
            != consumption_path
            or experiment.get("pair_authority_consumption_sha256")
            != file_sha256(consumption_path)
            or experiment.get("member_authority_consumption_sha256")
            != file_sha256(member_consumption_path)
            or member_consumption.get("schema")
            != MEMBER_AUTHORITY_CONSUMPTION_SCHEMA
            or member_consumption.get("generation") != generation
            or member_consumption.get("pair_id") != "P"
            or member_consumption.get("attempt") != valid_attempt
            or member_consumption.get("condition") != condition
            or resolve_artifact_reference(
                member_consumption.get("run_dir"), label="authorized pilot run"
            )
            != run_dir
            or resolve_artifact_reference(
                member_consumption.get("pair_authority_consumption"),
                label="authorized pilot pair consumption",
            )
            != consumption_path
            or member_consumption.get("pair_authority_consumption_sha256")
            != file_sha256(consumption_path)
        ):
            raise ValueError(f"pilot {condition} raw evidence does not validate")
    return pilot


def build_final_freeze(
    *,
    generation: str,
    pilot_pair_dir: Path,
    pilot_decision_path: Path,
    release_gate_dir: Path,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build a freeze only from an accepted decision and its exact transitive evidence."""

    if not generation.strip():
        raise ValueError("generation must not be blank")
    pilot_pair_dir = pilot_pair_dir.resolve()
    pilot_decision_path = pilot_decision_path.resolve()
    release_gate_dir = release_gate_dir.resolve()
    decision = validate_pilot_decision(
        pilot_decision_path,
        pilot_pair_dir=pilot_pair_dir,
        generation=generation,
    )
    endpoint_hash = decision.get("endpoint_metadata_sha256")
    if not _nonblank(endpoint_hash):
        raise ValueError("accepted pilot decision has no endpoint commitment")
    release = validate_release_gate(
        release_gate_dir,
        source_hashes=source_hashes,
        endpoint_metadata_hash=str(endpoint_hash),
    )
    release_path = release_gate_dir / "release-gate-results.json"
    candidate_path = release_gate_dir / "release-candidate-manifest.json"
    if (
        decision.get("source_hashes") != dict(source_hashes)
        or decision.get("release_gate_results_sha256") != file_sha256(release_path)
    ):
        raise ValueError("accepted pilot decision does not bind the exact release")
    pilot = _validate_pilot_pair_result(
        pilot_pair_dir,
        generation=generation,
        source_hashes=source_hashes,
        endpoint_metadata_hash=str(endpoint_hash),
        release_gate_results_hash=file_sha256(release_path),
        release_gate_dir=release_gate_dir,
    )
    pilot_path = pilot_pair_dir / "pair-result.json"
    artifacts = {
        "pilot_pair_result": str(pilot_path),
        "pilot_decision": str(pilot_decision_path),
        "release_gate_results": str(release_path),
        "release_candidate_manifest": str(candidate_path),
    }
    paths = {name: Path(path) for name, path in artifacts.items()}
    return {
        "schema": FINAL_FREEZE_SCHEMA,
        "final_freeze": True,
        "generation": generation,
        "accepted_pilot": {
            "pair_id": "P",
            "status": "accepted",
            "valid_attempt": pilot.get("valid_attempt"),
        },
        "artifacts": artifacts,
        "artifact_hashes": {
            name: file_sha256(path) for name, path in sorted(paths.items())
        },
        "source_hashes": dict(source_hashes),
        "endpoint_metadata_sha256": str(endpoint_hash),
        "release_gate_results_sha256": file_sha256(release_path),
        "authorization": {
            "authority": decision["authority"],
            "approval_record": decision["approval_record"],
            "scope": decision["scope"],
            "maximum_exposure": decision["maximum_exposure"],
        },
        "release_gate_schema": release["schema"],
    }


def validate_final_freeze(
    freeze_path: Path,
    *,
    generation: str,
    release_gate_dir: Path,
    source_hashes: Mapping[str, str],
    endpoint_metadata_hash: str,
) -> dict[str, Any]:
    """Resolve and recompute the full authority closure of a final freeze."""

    freeze = load_object(freeze_path.resolve(), label="final freeze")
    artifacts = freeze.get("artifacts")
    hashes = freeze.get("artifact_hashes")
    expected_artifact_names = {
        "pilot_pair_result",
        "pilot_decision",
        "release_gate_results",
        "release_candidate_manifest",
    }
    if (
        freeze.get("schema") != FINAL_FREEZE_SCHEMA
        or freeze.get("final_freeze") is not True
        or freeze.get("generation") != generation
        or freeze.get("source_hashes") != dict(source_hashes)
        or freeze.get("endpoint_metadata_sha256") != endpoint_metadata_hash
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != expected_artifact_names
        or not isinstance(hashes, Mapping)
        or set(hashes) != expected_artifact_names
    ):
        raise ValueError("included gameplay requires an exact accepted final freeze")
    resolved = {
        name: resolve_artifact_reference(artifacts[name], label=f"final freeze {name}")
        for name in sorted(expected_artifact_names)
    }
    release_gate_dir = release_gate_dir.resolve()
    if (
        resolved["release_gate_results"]
        != release_gate_dir / "release-gate-results.json"
        or resolved["release_candidate_manifest"]
        != release_gate_dir / "release-candidate-manifest.json"
    ):
        raise ValueError("final freeze references another release gate")
    for name, path in resolved.items():
        if not path.is_file() or hashes.get(name) != file_sha256(path):
            raise ValueError(f"final freeze artifact hash mismatch: {name}")

    decision = validate_pilot_decision(
        resolved["pilot_decision"],
        pilot_pair_dir=resolved["pilot_pair_result"].parent,
        generation=generation,
    )
    release = validate_release_gate(
        release_gate_dir,
        source_hashes=source_hashes,
        endpoint_metadata_hash=endpoint_metadata_hash,
    )
    release_hash = file_sha256(resolved["release_gate_results"])
    _validate_pilot_pair_result(
        resolved["pilot_pair_result"].parent,
        generation=generation,
        source_hashes=source_hashes,
        endpoint_metadata_hash=endpoint_metadata_hash,
        release_gate_results_hash=release_hash,
        release_gate_dir=release_gate_dir,
    )
    if (
        freeze.get("release_gate_results_sha256") != release_hash
        or decision.get("release_gate_results_sha256") != release_hash
        or freeze.get("release_gate_schema") != release.get("schema")
        or freeze.get("accepted_pilot")
        != {
            "pair_id": "P",
            "status": "accepted",
            "valid_attempt": load_object(
                resolved["pilot_pair_result"], label="pilot pair result"
            ).get("valid_attempt"),
        }
        or freeze.get("authorization")
        != {
            "authority": decision["authority"],
            "approval_record": decision["approval_record"],
            "scope": decision["scope"],
            "maximum_exposure": decision["maximum_exposure"],
        }
    ):
        raise ValueError("final freeze authority closure mismatch")
    return freeze


def consume_member_authority(
    consumption_path: Path,
    *,
    run_dir: Path,
    generation: str,
    pair_id: str,
    condition: str,
    release_gate_dir: Path,
    source_hashes: Mapping[str, str],
    endpoint_metadata_hash: str,
) -> Path:
    """Revalidate the underlying grant and atomically consume one allowed member slot."""

    consumption_path = consumption_path.resolve()
    run_dir = run_dir.resolve()
    pair_root = run_dir.parent.parent
    generation_root = pair_root.parent
    expected_generation_root = (
        AGENT_ROOT / "artifacts" / "generations" / generation
    ).resolve()
    expected_consumption_path = generation_root / ".authority" / f"pair-{pair_id}.json"
    if (
        generation_root != expected_generation_root
        or consumption_path != expected_consumption_path
        or condition not in {"enabled", "disabled"}
    ):
        raise ValueError("member execution is outside the canonical authority ledger")
    record = load_object(consumption_path, label="pair authority consumption")
    authority_path = resolve_artifact_reference(
        record.get("authority_path"), label="consumed pair authority"
    )
    allowed_runs = record.get("allowed_runs")
    matching = [
        row
        for row in allowed_runs
        if isinstance(allowed_runs, list)
        and isinstance(row, Mapping)
        and row.get("condition") == condition
        and resolve_artifact_reference(row.get("run_dir"), label="allowed member run")
        == run_dir
    ] if isinstance(allowed_runs, list) else []
    if (
        record.get("schema") != AUTHORITY_CONSUMPTION_SCHEMA
        or record.get("generation") != generation
        or record.get("pair_id") != pair_id
        or resolve_artifact_reference(record.get("pair_root"), label="consumed pair root")
        != pair_root
        or not authority_path.is_file()
        or record.get("authority_sha256") != file_sha256(authority_path)
        or len(matching) != 1
    ):
        raise ValueError("pair authority consumption does not permit this member run")
    maximum_attempts = record.get("maximum_pair_attempts")
    if isinstance(maximum_attempts, bool) or maximum_attempts not in {1, 2}:
        raise ValueError("pair authority consumption has an invalid attempt ceiling")
    authority_kind = record.get("authority_kind")
    if pair_id == "P" and authority_kind == "pilot_approval":
        validated = validate_pilot_approval(
            authority_path,
            generation=generation,
            release_gate_dir=release_gate_dir,
            source_hashes=source_hashes,
            endpoint_metadata_hash=endpoint_metadata_hash,
        )
        authorized_attempts = validated["maximum_pair_attempts"]
    elif pair_id in {"1", "2", "3"} and authority_kind == "final_freeze":
        freeze = validate_final_freeze(
            authority_path,
            generation=generation,
            release_gate_dir=release_gate_dir,
            source_hashes=source_hashes,
            endpoint_metadata_hash=endpoint_metadata_hash,
        )
        authorized_attempts = freeze["authorization"]["maximum_exposure"][
            "maximum_pair_attempts_per_pair"
        ]
    else:
        raise ValueError("pair authority kind cannot authorize this pair")
    if authorized_attempts != maximum_attempts:
        raise ValueError("pair authority consumption escalates the approved attempt ceiling")
    attempt = matching[0].get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise ValueError("allowed member run has no attempt identity")
    marker_path = (
        generation_root
        / ".authority"
        / "members"
        / f"pair-{pair_id}-attempt-{attempt}-{condition}.json"
    )
    _write_new_json(
        marker_path,
        {
            "schema": MEMBER_AUTHORITY_CONSUMPTION_SCHEMA,
            "consumed_at": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
            "pair_id": pair_id,
            "attempt": attempt,
            "condition": condition,
            "run_dir": str(run_dir),
            "pair_authority_consumption": str(consumption_path),
            "pair_authority_consumption_sha256": file_sha256(consumption_path),
        },
        conflict=(
            f"pair {pair_id} attempt {attempt} condition {condition} authority already consumed"
        ),
    )
    return marker_path


def validate_included_pair_result(
    pair_result_path: Path,
    *,
    final_freeze_path: Path,
    generation: str,
    release_gate_dir: Path,
    source_hashes: Mapping[str, str],
    endpoint_metadata_hash: str,
) -> dict[str, Any]:
    """Validate one completed included pair through freeze, slot, and raw evidence closure."""

    final_freeze_path = final_freeze_path.resolve()
    freeze = validate_final_freeze(
        final_freeze_path,
        generation=generation,
        release_gate_dir=release_gate_dir,
        source_hashes=source_hashes,
        endpoint_metadata_hash=endpoint_metadata_hash,
    )
    pair_result_path = pair_result_path.resolve()
    pair_root = pair_result_path.parent
    pair = load_object(pair_result_path, label="included pair result")
    pair_id = pair.get("pair_id")
    if (
        pair.get("schema") != "scientist-game-designer-player-pair-v1"
        or pair.get("generation") != generation
        or not isinstance(pair_id, str)
        or pair_id not in {"1", "2", "3"}
        or pair.get("included") is not True
        or pair.get("status") != "pair_complete"
        or pair.get("gameplay_executed") is not True
        or pair.get("source_hashes") != dict(source_hashes)
        or pair.get("endpoint_metadata_sha256") != endpoint_metadata_hash
        or pair.get("release_gate_results_sha256")
        != freeze.get("release_gate_results_sha256")
    ):
        raise ValueError("included pair result is outside the final freeze")
    commitments_by_condition = _validate_pair_commitment_artifact(
        pair,
        pair_root=pair_root,
        generation=generation,
        pair_id=pair_id,
        source_hashes=source_hashes,
        endpoint_metadata_hash=endpoint_metadata_hash,
    )

    consumption_path = resolve_artifact_reference(
        pair.get("authority_consumption"), label="included pair authority consumption"
    )
    consumption = load_object(consumption_path, label="included pair authority consumption")
    maximum_attempts = freeze["authorization"]["maximum_exposure"][
        "maximum_pair_attempts_per_pair"
    ]
    expected_allowed_runs = [
        {
            "attempt": attempt,
            "condition": condition,
            "run_dir": str(pair_root / f"attempt-{attempt}" / condition),
        }
        for attempt in range(1, maximum_attempts + 1)
        for condition in ("enabled", "disabled")
    ]
    if (
        pair.get("authority_consumption_sha256") != file_sha256(consumption_path)
        or consumption.get("schema") != AUTHORITY_CONSUMPTION_SCHEMA
        or consumption.get("generation") != generation
        or consumption.get("pair_id") != pair_id
        or consumption.get("authority_kind") != "final_freeze"
        or resolve_artifact_reference(
            consumption.get("pair_root"), label="included consumed pair root"
        )
        != pair_root
        or resolve_artifact_reference(
            consumption.get("authority_path"), label="included consumed freeze"
        )
        != final_freeze_path
        or consumption.get("authority_sha256") != file_sha256(final_freeze_path)
        or consumption.get("maximum_pair_attempts") != maximum_attempts
        or consumption.get("allowed_runs") != expected_allowed_runs
    ):
        raise ValueError("included pair does not bind its consumed final freeze")
    valid_attempt = pair.get("valid_attempt")
    attempts = pair.get("attempts")
    if (
        isinstance(valid_attempt, bool)
        or not isinstance(valid_attempt, int)
    ):
        raise ValueError("included pair has no authorized valid attempt")
    valid_attempt_record = _validated_attempt(
        attempts,
        valid_attempt=valid_attempt,
        maximum_attempts=maximum_attempts,
        label="included pair",
    )
    members = valid_attempt_record.get("members")
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("included pair valid attempt is ambiguous")
    by_condition = {
        row.get("condition"): row
        for row in members
        if isinstance(row, Mapping) and row.get("condition") in {"enabled", "disabled"}
    }
    if set(by_condition) != {"enabled", "disabled"}:
        raise ValueError("included pair valid attempt omits a member")
    outcomes: dict[str, RunOutcome] = {}
    for condition in ("enabled", "disabled"):
        member = by_condition[condition]
        run_dir = resolve_artifact_reference(member.get("run_dir"), label="included run")
        if run_dir != (pair_root / f"attempt-{valid_attempt}" / condition).resolve():
            raise ValueError("included member run path is outside its attempt")
        manifest_path = run_dir / "manifest.json"
        manifest = load_object(manifest_path, label="included member manifest")
        experiment = manifest.get("experiment")
        evidence = manifest.get("evidence_hashes")
        commitment = commitments_by_condition[condition]
        if not isinstance(experiment, Mapping):
            raise ValueError("included member has no experiment commitments")
        member_consumption_path = resolve_artifact_reference(
            experiment.get("member_authority_consumption"),
            label="included member authority consumption",
        )
        member_consumption = load_object(
            member_consumption_path, label="included member authority consumption"
        )
        if (
            member.get("validity") != "valid_contestant_result"
            or member.get("manifest_sha256") != file_sha256(manifest_path)
            or not isinstance(evidence, Mapping)
            or not validate_run_evidence(run_dir, evidence)
            or gameplay_attempt_validity(manifest, commitment, run_dir=run_dir)
            != "valid_contestant_result"
            or experiment.get("generation") != generation
            or experiment.get("pair_id") != pair_id
            or experiment.get("condition") != condition
            or experiment.get("included") is not True
            or experiment.get("source_hashes") != dict(source_hashes)
            or experiment.get("endpoint_metadata_sha256") != endpoint_metadata_hash
            or experiment.get("prompt_sha256") != commitment.get("prompt_sha256")
            or resolve_artifact_reference(
                experiment.get("pair_authority_consumption"),
                label="included pair authority consumption",
            )
            != consumption_path
            or experiment.get("pair_authority_consumption_sha256")
            != file_sha256(consumption_path)
            or experiment.get("member_authority_consumption_sha256")
            != file_sha256(member_consumption_path)
            or member_consumption.get("schema")
            != MEMBER_AUTHORITY_CONSUMPTION_SCHEMA
            or member_consumption.get("generation") != generation
            or member_consumption.get("pair_id") != pair_id
            or member_consumption.get("attempt") != valid_attempt
            or member_consumption.get("condition") != condition
            or resolve_artifact_reference(
                member_consumption.get("run_dir"), label="included authorized run"
            )
            != run_dir
            or resolve_artifact_reference(
                member_consumption.get("pair_authority_consumption"),
                label="included authorized pair consumption",
            )
            != consumption_path
            or member_consumption.get("pair_authority_consumption_sha256")
            != file_sha256(consumption_path)
        ):
            raise ValueError(f"included {pair_id}/{condition} evidence does not validate")
        outcomes[condition] = _manifest_outcome(manifest)
    expected_sign = assess_pair(outcomes["enabled"], outcomes["disabled"])
    if pair.get("pair_sign") != expected_sign:
        raise ValueError("included pair sign does not match validated member outcomes")
    return pair
