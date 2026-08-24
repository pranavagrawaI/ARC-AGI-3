"""Single-arm diagnostic for the sterile CodeAct compatibility frontier."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.compatibility_authorization import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
)
from tycho_nooa.compatibility_experiment import (
    CODEACT_COMPATIBILITY_PROMPT,
    CompatibilityArmObservation,
    _run_codeact,
)
from tycho_nooa.compatibility_launch import AuthorizedCompatibilityLLM
from tycho_nooa.compatibility_manifest import sha256_file, sha256_text
from tycho_nooa.execution_events import ExecutionProvenanceEvent
from tycho_nooa.parity_compatibility import load_compatibility_gate
from tycho_nooa.parity_manifest import FileCommitment, ModelRoute


class CodeActRunwayCaps(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    calls_total: Literal[6] = 6
    input_tokens_per_call: int = Field(ge=1)
    output_tokens_per_call: int = Field(ge=1)
    wall_seconds: float = Field(gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class CodeActRunwayManifest(BaseModel):
    """Frozen one-variable diagnostic; execution remains separately authorized."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    execution_authorized: bool
    runtime_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    package_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    gate: FileCommitment
    prior_native_control: FileCommitment
    prior_codeact_trace: FileCommitment
    runner: FileCommitment
    gate_executor: FileCommitment
    authorization_schema: FileCommitment
    protocol: FileCommitment
    experiment_card: FileCommitment
    codeact_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_route: ModelRoute
    caps: CodeActRunwayCaps
    output_root: str

    @model_validator(mode="after")
    def authorized_manifest_is_complete(self) -> CodeActRunwayManifest:
        if self.execution_authorized:
            _validate_enabled_manifest(self)
        return self

    def require_execution_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError("paid CodeAct runway execution is disabled")


class CodeActRunwayMilestones(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_discovered_call: int | None = None
    evidence_disclosed_call: int | None = None
    computation_completed_call: int | None = None
    correct_answer_observed_call: int | None = None
    return_result_call: int | None = None


class CodeActRunwayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str
    passed: bool
    interpretation: str
    prior_native_control: dict[str, Any]
    codeact: CompatibilityArmObservation
    milestones: CodeActRunwayMilestones


async def execute_codeact_runway_arm(
    gate: Any,
    *,
    repository_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_llm_calls: int,
    max_output_tokens: int = 4096,
    session_timeout_seconds: float = 300.0,
    cell_timeout_seconds: float = 5.0,
) -> CompatibilityArmObservation:
    """Adapt the frozen gate executor without changing its committed bytes."""
    try:
        async with asyncio.timeout(session_timeout_seconds):
            return await _run_codeact(
                gate,
                repository_root=repository_root,
                destination=destination,
                llm=llm,
                max_llm_calls=max_llm_calls,
                max_output_tokens=max_output_tokens,
                cell_timeout_seconds=cell_timeout_seconds,
            )
    except BaseException as error:
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(
            destination / "failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def load_codeact_runway_manifest(
    path: Path, *, repository_root: Path
) -> tuple[CodeActRunwayManifest, Any]:
    manifest = CodeActRunwayManifest.model_validate_json(path.read_text())
    for commitment in (
        manifest.gate,
        manifest.prior_native_control,
        manifest.prior_codeact_trace,
        manifest.runner,
        manifest.gate_executor,
        manifest.authorization_schema,
        manifest.protocol,
        manifest.experiment_card,
    ):
        _validate_file(commitment, repository_root)
    if manifest.codeact_prompt_sha256 != sha256_text(CODEACT_COMPATIBILITY_PROMPT):
        raise ValueError("CodeAct prompt commitment mismatch")
    card = repository_root / manifest.experiment_card.path
    if len(card.read_bytes()) > 1024:
        raise ValueError("experiment card exceeds 1 KiB")
    gate = load_compatibility_gate(
        repository_root / manifest.gate.path, repository_root=repository_root
    )
    if gate.task.gate_id != "g3-composition":
        raise ValueError("runway diagnostic requires the frozen composition gate")
    _validate_native_control(repository_root / manifest.prior_native_control.path)
    return manifest, gate


async def execute_authorized_codeact_runway(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    destination: Path,
    llm: UnifiedLLM,
) -> CodeActRunwayReport:
    """Run exactly one six-call CodeAct arm under the durable paid-call ledger."""
    manifest, gate, authorization, authorization_hash = preflight_codeact_runway(
        manifest_path,
        authorization_path,
        repository_root=repository_root,
        llm=llm,
    )
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    (destination / "experiment.json").write_bytes(manifest_path.read_bytes())
    (destination / "authorization.json").write_bytes(authorization_path.read_bytes())
    ledger = DurableAuthorizationLedger.create(
        destination / "authorization-events.jsonl",
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    gate_id = gate.task.gate_id
    ledger.start_gate(gate_id)
    guarded_llm = AuthorizedCompatibilityLLM(
        llm, gate_id=gate_id, arm="codeact", ledger=ledger
    )
    try:
        observation = await execute_codeact_runway_arm(
            gate,
            repository_root=repository_root,
            destination=destination / "codeact",
            llm=guarded_llm,
            max_llm_calls=manifest.caps.calls_total,
            max_output_tokens=manifest.caps.output_tokens_per_call,
            session_timeout_seconds=manifest.caps.wall_seconds,
            cell_timeout_seconds=min(5.0, manifest.caps.wall_seconds),
        )
        report = _build_report(manifest, observation, destination, repository_root)
        report_value = report.model_dump(mode="json")
        _write_json(destination / "report.json", report_value)
        ledger.complete_gate(gate_id, passed=report.passed, report=report_value)
        ledger.finish_ladder(report.interpretation, report=report_value)
        return report
    except BaseException as error:
        if ledger.state.run_state == "armed":
            ledger.stop(f"launcher_failure:{type(error).__name__}")
        _write_json(
            destination / "launch-failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def preflight_codeact_runway(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    llm: UnifiedLLM,
) -> tuple[CodeActRunwayManifest, Any, CompatibilityAuthorization, str]:
    manifest, gate = load_codeact_runway_manifest(
        manifest_path, repository_root=repository_root
    )
    manifest.require_execution_authorized()
    authorization = CompatibilityAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    authorization.require_current()
    manifest_hash = sha256_file(manifest_path)
    if authorization.experiment_manifest_sha256 != manifest_hash:
        raise ValueError("authorization does not bind this experiment manifest")
    _validate_route_and_caps(manifest, authorization, llm)
    return manifest, gate, authorization, sha256_file(authorization_path)


def extract_runway_milestones(path: Path) -> CodeActRunwayMilestones:
    events = [
        ExecutionProvenanceEvent.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    call_by_event = _provider_call_indices(events)
    capability = _first_event_call(events, call_by_event, "CapabilityRequested")
    disclosed = _first_event_call(events, call_by_event, "CapabilityDisclosed")
    computation = _first_computation_call(events, call_by_event, disclosed)
    correct = _first_correct_answer_call(events, call_by_event)
    terminal = _first_event_call(events, call_by_event, "ReturnAccepted")
    return CodeActRunwayMilestones(
        capability_discovered_call=capability,
        evidence_disclosed_call=disclosed,
        computation_completed_call=computation,
        correct_answer_observed_call=correct,
        return_result_call=terminal,
    )


def _provider_call_indices(
    events: list[ExecutionProvenanceEvent],
) -> dict[int, int]:
    call = 0
    mapping: dict[int, int] = {}
    for event in events:
        if event.event_type in {
            "CellDispatched",
            "ProviderTurnRejected",
        }:
            call += 1
        mapping[event.event_seq] = call
    return mapping


def _first_event_call(
    events: list[ExecutionProvenanceEvent],
    mapping: dict[int, int],
    event_type: str,
) -> int | None:
    event = next((row for row in events if row.event_type == event_type), None)
    return None if event is None else mapping[event.event_seq]


def _first_computation_call(
    events: list[ExecutionProvenanceEvent],
    mapping: dict[int, int],
    disclosed_call: int | None,
) -> int | None:
    if disclosed_call is None:
        return None
    for event in events:
        if event.event_type != "CellCompleted":
            continue
        call = mapping[event.event_seq]
        if call > disclosed_call:
            return call
    return None


def _first_correct_answer_call(
    events: list[ExecutionProvenanceEvent], mapping: dict[int, int]
) -> int | None:
    for event in events:
        if event.event_type not in {"CellCompleted", "ReturnProposed"}:
            continue
        visible = json.dumps(event.payload) + _cell_source(events, event.cell_seq)
        if "11->5" in visible and "L0:T1" in visible:
            return mapping[event.event_seq]
    return None


def _cell_source(
    events: list[ExecutionProvenanceEvent], cell_seq: int | None
) -> str:
    for event in events:
        if event.event_type == "CellDispatched" and event.cell_seq == cell_seq:
            source = event.payload.get("source", "")
            return source if isinstance(source, str) else ""
    return ""


def _build_report(
    manifest: CodeActRunwayManifest,
    observation: CompatibilityArmObservation,
    destination: Path,
    repository_root: Path,
) -> CodeActRunwayReport:
    milestones = extract_runway_milestones(destination / "codeact/provenance.jsonl")
    return CodeActRunwayReport(
        experiment_id=manifest.experiment_id,
        passed=observation.passed,
        interpretation=_interpret(observation, milestones),
        prior_native_control=json.loads(
            (repository_root / manifest.prior_native_control.path).read_text()
        ),
        codeact=observation,
        milestones=milestones,
    )


def _interpret(
    observation: CompatibilityArmObservation, milestones: CodeActRunwayMilestones
) -> str:
    terminal = milestones.return_result_call
    if observation.passed:
        return {
            4: "deadline_truncation",
            5: "compatible_turn_hungry",
            6: "compatible_fragile",
        }.get(terminal, "compatible_before_prior_ceiling")
    return (
        "interface_policy_mismatch"
        if milestones.correct_answer_observed_call is not None
        else "no_compatible_completion"
    )


def _validate_route_and_caps(
    manifest: CodeActRunwayManifest,
    authorization: CompatibilityAuthorization,
    llm: UnifiedLLM,
) -> None:
    route = manifest.model_route
    route_values = (
        authorization.requested_model != route.model
        or authorization.expected_reported_model != route.reported_model,
        authorization.allowed_provider != route.provider,
        authorization.allowed_reasoning_effort != route.reasoning_effort,
        authorization.allowed_temperature != route.temperature,
    )
    if any(route_values):
        raise ValueError("authorization route does not match diagnostic route")
    caps = manifest.caps
    cap_values = (
        authorization.maximum_calls != caps.calls_total
        or authorization.max_input_tokens_per_call != caps.input_tokens_per_call,
        authorization.max_output_tokens_per_call != caps.output_tokens_per_call,
        authorization.maximum_spend_usd != Decimal(str(caps.max_cost_usd)),
    )
    if any(cap_values):
        raise ValueError("authorization caps do not match diagnostic caps")
    configured_route = (
        llm.model,
        _configured_effort(llm),
        llm.config.get("temperature"),
    )
    expected_route = (route.model, route.reasoning_effort, route.temperature)
    if configured_route != expected_route:
        raise ValueError("configured LLM does not match diagnostic route")


def _validate_native_control(path: Path) -> None:
    control = json.loads(path.read_text())
    observed = (control.get("passed"), control.get("metrics", {}).get("return_call_index"))
    if observed != (True, 2):
        raise ValueError("prior Native control must pass on call 2")


def _validate_file(commitment: FileCommitment, repository_root: Path) -> None:
    path = repository_root / commitment.path
    observed = sha256_file(path) if path.is_file() else None
    if observed != commitment.sha256:
        raise ValueError(f"file commitment mismatch: {commitment.path}")


def _validate_enabled_manifest(manifest: CodeActRunwayManifest) -> None:
    route = manifest.model_route
    route_is_complete = all(
        (route.model, route.reported_model, route.provider, route.reasoning_effort)
    )
    if not route_is_complete or manifest.caps.max_cost_usd is None:
        raise ValueError("authorized execution requires route and cost cap")


def _configured_effort(llm: UnifiedLLM) -> Any:
    effort = llm.config.get("reasoning_effort")
    reasoning = llm.config.get("reasoning")
    if effort is not None or not isinstance(reasoning, dict):
        return effort
    return reasoning.get("effort") or (
        "enabled" if reasoning.get("enabled") is True else None
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
