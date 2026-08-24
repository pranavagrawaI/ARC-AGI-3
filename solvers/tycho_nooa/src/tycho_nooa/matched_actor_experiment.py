"""Mutation-free Native-versus-CodeAct Actor comparison and cost vector."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tycho_nooa.cognitive_parity import (
    CognitiveParityPacket,
    ParityEvaluation,
    evaluate_answer,
)
from tycho_nooa.evidence import EvidenceCapability, WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import (
    MatchedCodeActController,
    MatchedNativeActorController,
    ModelAction,
    cognitive_answer,
    return_model_for_task,
)

TreatmentArm = Literal["native", "codeact"]


class ActorResourceVector(BaseModel):
    """No scalar hides which resource one control architecture consumed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    dollar_cost: float = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    python_cells: int = Field(ge=0)
    evidence_disclosures: int = Field(ge=0)
    invalid_interactions: int = Field(ge=0)
    repair_interactions: int = Field(ge=0)


class ActorMilestones(BaseModel):
    """Call indices for observed progress; absent means not established."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_record_disclosed: int | None = None
    representation_decoded: int | None = None
    target_value_identified: int | None = None
    provenance_bound: int | None = None
    candidate_answer_formed: int | None = None
    terminal_result_accepted: int | None = None


class ActorTreatmentObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    packet_id: str
    arm: TreatmentArm
    answer: dict[str, Any]
    evaluation: ParityEvaluation
    resources: ActorResourceVector
    milestones: ActorMilestones
    provenance_path: str | None = None


class MatchedActorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    packet_id: str
    native: ActorTreatmentObservation
    codeact: ActorTreatmentObservation


class RecordingEvidenceCapability:
    """Retain immutable disclosures without changing their semantics."""

    def __init__(self, delegate: EvidenceCapability):
        self._delegate = delegate
        self.disclosures: list[dict[str, Any]] = []

    def query(self, raw_query: dict[str, Any]) -> dict[str, Any]:
        value = self._delegate.query(raw_query)
        self.disclosures.append(json.loads(json.dumps(value)))
        return value


async def execute_matched_actor_packet(
    packet: CognitiveParityPacket,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
    max_provider_calls: int,
    max_output_tokens: int,
    session_timeout_seconds: float,
    cell_timeout_seconds: float,
) -> MatchedActorReport:
    """Run both no-authority arms under the same numeric resource envelope."""
    destination.mkdir(parents=True, exist_ok=False)
    evidence_root = repository_root / packet.checkpoint_root
    async with asyncio.timeout(session_timeout_seconds):
        native = await _run_native(
            packet,
            evidence_root=evidence_root,
            destination=destination / "native",
            llm=native_llm,
            max_provider_calls=max_provider_calls,
            max_output_tokens=max_output_tokens,
            cell_timeout_seconds=cell_timeout_seconds,
        )
    async with asyncio.timeout(session_timeout_seconds):
        codeact = await _run_codeact(
            packet,
            evidence_root=evidence_root,
            destination=destination / "codeact",
            llm=codeact_llm,
            max_provider_calls=max_provider_calls,
            max_output_tokens=max_output_tokens,
            cell_timeout_seconds=cell_timeout_seconds,
        )
    report = MatchedActorReport(
        packet_id=packet.task.packet_id,
        native=native,
        codeact=codeact,
    )
    _write_json(destination / "report.json", report.model_dump(mode="json"))
    return report


async def _run_native(
    packet: CognitiveParityPacket,
    *,
    evidence_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_provider_calls: int,
    max_output_tokens: int,
    cell_timeout_seconds: float,
) -> ActorTreatmentObservation:
    destination.mkdir()
    evidence = RecordingEvidenceCapability(
        WorkspaceEvidenceCapability.from_root(evidence_root)
    )
    controller = MatchedNativeActorController(
        llm=llm,
        packet=packet,
        evidence=evidence,
        max_llm_calls=max_provider_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    started = time.monotonic()
    matched = await controller.run()
    latency = time.monotonic() - started
    answer = cognitive_answer(packet, matched)
    observation = _observation(
        packet,
        "native",
        answer.model_dump(mode="json"),
        usage=[turn.usage for turn in controller.provider_turns],
        actions=controller.model_actions,
        evidence=evidence,
        python_cells=controller.python_cells,
        invalid_interactions=controller.invalid_returns,
        latency_seconds=latency,
        terminal_call=controller.terminal_return_call,
        successful_decode_calls=controller.successful_decode_calls,
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation


async def _run_codeact(
    packet: CognitiveParityPacket,
    *,
    evidence_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_provider_calls: int,
    max_output_tokens: int,
    cell_timeout_seconds: float,
) -> ActorTreatmentObservation:
    destination.mkdir()
    evidence = RecordingEvidenceCapability(
        WorkspaceEvidenceCapability.from_root(evidence_root)
    )
    provenance_path = destination / "provenance.jsonl"
    log = ExecutionProvenanceLog(
        provenance_path,
        actor_session_id=f"matched:{packet.task.packet_id}:codeact",
    )
    controller = MatchedCodeActController(
        llm=llm,
        packet=packet,
        evidence=evidence,
        provenance=InvocationProvenance(
            log, invocation_id=f"{packet.task.packet_id}:codeact"
        ),
        max_llm_calls=max_provider_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    started = time.monotonic()
    matched = await controller.run()
    latency = time.monotonic() - started
    answer = cognitive_answer(packet, matched)
    invalid = sum(
        event.event_type
        in {"ProviderTurnRejected", "CapabilityRejected", "CellFailed", "ReturnRejected"}
        for event in log.events
    )
    successful_decode_calls = _codeact_successful_decode_calls(
        controller.model_actions, log.events
    )
    observation = _observation(
        packet,
        "codeact",
        answer.model_dump(mode="json"),
        usage=[turn.usage for turn in controller.provider_turns],
        actions=controller.model_actions,
        evidence=evidence,
        python_cells=sum(action.name == "execute_python" for action in controller.model_actions),
        invalid_interactions=invalid,
        latency_seconds=latency,
        terminal_call=controller.terminal_return_call,
        successful_decode_calls=successful_decode_calls,
        provenance_path="provenance.jsonl",
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation


def _observation(
    packet: CognitiveParityPacket,
    arm: TreatmentArm,
    answer: dict[str, Any],
    *,
    usage: list[dict[str, Any] | None],
    actions: tuple[ModelAction, ...],
    evidence: RecordingEvidenceCapability,
    python_cells: int,
    invalid_interactions: int,
    latency_seconds: float,
    terminal_call: int | None,
    successful_decode_calls: tuple[int, ...],
    provenance_path: str | None = None,
) -> ActorTreatmentObservation:
    evaluation = evaluate_answer(packet, answer)
    totals = _usage_totals(usage)
    candidate_call = _first_correct_candidate(packet, actions)
    provenance_call = _first_provenance_binding(packet, actions)
    decode_call = min(successful_decode_calls, default=None)
    return ActorTreatmentObservation(
        packet_id=packet.task.packet_id,
        arm=arm,
        answer=answer,
        evaluation=evaluation,
        resources=ActorResourceVector(
            provider_calls=len(usage),
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            dollar_cost=totals["dollar_cost"],
            latency_seconds=latency_seconds,
            python_cells=python_cells,
            evidence_disclosures=len(evidence.disclosures),
            invalid_interactions=invalid_interactions,
            repair_interactions=(invalid_interactions if terminal_call is not None else 0),
        ),
        milestones=ActorMilestones(
            target_record_disclosed=_first_target_disclosure(packet, evidence, actions),
            representation_decoded=decode_call,
            target_value_identified=candidate_call,
            provenance_bound=provenance_call,
            candidate_answer_formed=candidate_call,
            terminal_result_accepted=terminal_call,
        ),
        provenance_path=provenance_path,
    )


def _first_correct_candidate(
    packet: CognitiveParityPacket, actions: tuple[ModelAction, ...]
) -> int | None:
    for action in actions:
        if action.name != "return_result":
            continue
        try:
            model = return_model_for_task(packet)
            answer = cognitive_answer(packet, model.model_validate(action.arguments))
            if _evaluation_is_correct(evaluate_answer(packet, answer.model_dump(mode="json"))):
                return action.call_index
        except (ValidationError, ValueError, TypeError):
            continue
    return None


def _evaluation_is_correct(value: ParityEvaluation) -> bool:
    checks = [
        value.evidence_precision == 1,
        value.evidence_recall == 1,
    ]
    checks.extend(
        item is not False
        for item in (
            value.verdict_correct,
            value.falsification_correct,
            value.invariant_correct,
            value.scope_correct,
            value.action_legal,
        )
    )
    if value.invariant_atom_recall is not None:
        checks.append(value.invariant_atom_recall == 1)
    return all(checks)


def _first_provenance_binding(
    packet: CognitiveParityPacket, actions: tuple[ModelAction, ...]
) -> int | None:
    required = _required_ref_pairs(packet)
    if not required:
        return None
    for action in actions:
        if action.name == "return_result" and required.issubset(
            _proposed_ref_pairs(action)
        ):
            return action.call_index
    return None


def _required_ref_pairs(packet: CognitiveParityPacket) -> set[tuple[int, int]]:
    return {
        (ref.level, ref.turn) for ref in packet.expected.required_evidence_refs
    }


def _proposed_ref_pairs(action: ModelAction) -> set[tuple[Any, Any]]:
    refs = action.arguments.get("evidence_refs")
    if not isinstance(refs, list):
        return set()
    return {
        (item.get("level"), item.get("turn"))
        for item in refs
        if isinstance(item, dict)
    }


def _codeact_successful_decode_calls(
    actions: tuple[ModelAction, ...], events: tuple[Any, ...]
) -> tuple[int, ...]:
    completed_cells = _completed_cell_sequences(events)
    action_calls = {action.tool_call_id: action.call_index for action in actions}
    return tuple(
        action_calls[tool_call_id]
        for event in events
        if _is_completed_decode_dispatch(event, completed_cells)
        if isinstance((tool_call_id := event.payload.get("tool_call_id")), str)
        if tool_call_id in action_calls
    )


def _completed_cell_sequences(events: tuple[Any, ...]) -> set[int]:
    return {
        event.cell_seq
        for event in events
        if event.event_type == "CellCompleted" and event.cell_seq is not None
    }


def _is_completed_decode_dispatch(event: Any, completed_cells: set[int]) -> bool:
    source = event.payload.get("source")
    return (
        event.event_type == "CellDispatched"
        and event.cell_seq in completed_cells
        and isinstance(source, str)
        and "rle.decode_row" in source
    )


def _first_target_disclosure(
    packet: CognitiveParityPacket,
    evidence: RecordingEvidenceCapability,
    actions: tuple[ModelAction, ...],
) -> int | None:
    targets = {
        f"L{ref.level}:T{ref.turn}" for ref in packet.expected.required_evidence_refs
    }
    if not targets:
        return None
    query_calls = _evidence_call_indices(actions)
    for index, disclosure in enumerate(evidence.disclosures):
        if targets & _disclosed_refs(disclosure):
            return _call_for_disclosure(index, query_calls)
    return None


def _evidence_call_indices(actions: tuple[ModelAction, ...]) -> list[int]:
    native = [
        action.call_index for action in actions if action.name == "evidence_query"
    ]
    if native:
        return native
    return [action.call_index for action in actions if action.name == "execute_python"]


def _disclosed_refs(disclosure: dict[str, Any]) -> set[Any]:
    return {record.get("ref") for record in disclosure.get("records", [])}


def _call_for_disclosure(index: int, query_calls: list[int]) -> int | None:
    if not query_calls:
        return None
    return query_calls[min(index, len(query_calls) - 1)]


def _usage_totals(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "dollar_cost": 0.0,
    }
    for row in rows:
        if not row:
            continue
        totals["input_tokens"] += _usage_int(row, "prompt_tokens", "input_tokens")
        totals["output_tokens"] += _usage_int(
            row, "completion_tokens", "output_tokens"
        )
        totals["dollar_cost"] += _usage_float(row, "cost", "observed_cost_usd")
    return totals


def _usage_int(row: dict[str, Any], *names: str) -> int:
    for name in names:
        value = row.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0


def _usage_float(row: dict[str, Any], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
    return 0.0


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
