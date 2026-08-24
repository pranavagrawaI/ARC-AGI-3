"""Zero-cost six-packet proof for the shared Context/Evidence Plane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from pydantic import BaseModel, ConfigDict

from tycho_nooa.cognitive_parity import CognitiveParityPacket, evaluate_answer
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import cognitive_answer
from tycho_nooa.matched_actor_audit import _flat_expected_answer
from tycho_nooa.matched_actor_context import (
    ContextCodeActController,
    ContextNativeActorController,
)
from tycho_nooa.matched_actor_corpus import ActorCognitiveStratum
from tycho_nooa.matched_actor_experiment import RecordingEvidenceCapability
from tycho_nooa.matched_actor_experiment_v4 import _context_evidence

_COMPUTATION = """\
records = [record for page in pages for record in page.records]
refs = sorted(record.ref for record in records)
checksum = 0
for record in records:
    for field in ('grid', 'prev', 'next'):
        grid = getattr(record, field, None)
        if grid:
            for row in grid:
                checksum += sum(rle.decode_row(row))
print(len(records), len(refs), checksum)
"""


class ContextActorReachabilityAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    stratum: ActorCognitiveStratum
    evidence_disclosures_identical: bool
    computation_stdout_identical: bool
    native_answer_correct: bool
    codeact_answer_correct: bool
    bulk_evidence_absent_from_provider_context: bool
    immutable_handle_semantics_shared: bool
    native_python_has_evidence_acquisition_authority: bool
    canonical_write_authority: bool
    arc_action_authority: bool


@dataclass(frozen=True)
class _NativeAuditRun:
    controller: ContextNativeActorController
    value: BaseModel
    evidence: RecordingEvidenceCapability
    llm: FakeLLMClient


@dataclass(frozen=True)
class _CodeActAuditRun:
    controller: ContextCodeActController
    value: BaseModel
    evidence: RecordingEvidenceCapability
    log: ExecutionProvenanceLog


async def audit_context_actor_packet(
    packet: CognitiveParityPacket,
    *,
    stratum: ActorCognitiveStratum,
    repository_root: Path,
    destination: Path,
) -> ContextActorReachabilityAudit:
    """Drive both controllers with evaluator-authored reachability scripts."""
    destination.mkdir(parents=True, exist_ok=False)
    root = repository_root / packet.checkpoint_root
    specs, pages = _targeted_evidence_script(packet, root)
    final = _flat_expected_answer(packet)
    compute = stratum != "computation_unnecessary" and bool(specs)

    native = await _run_native_script(packet, root, specs, final, compute=compute)
    codeact = await _run_codeact_script(
        packet, root, destination, specs, final, compute=compute
    )
    native_answer = cognitive_answer(packet, native.value).model_dump(mode="json")
    codeact_answer = cognitive_answer(packet, codeact.value).model_dump(mode="json")
    native_stdout = _native_stdout(native.controller)
    codeact_stdout = _codeact_stdout(codeact.log)
    provider_contexts = [
        native.llm.last_messages,
        codeact.controller.reconstructed_messages(),
    ]
    same_disclosures = (
        native.evidence.disclosures == codeact.evidence.disclosures == pages
    )
    same_computation = not compute or native_stdout == codeact_stdout
    audit = ContextActorReachabilityAudit(
        packet_id=packet.task.packet_id,
        stratum=stratum,
        evidence_disclosures_identical=same_disclosures,
        computation_stdout_identical=same_computation,
        native_answer_correct=_correct(packet, native_answer),
        codeact_answer_correct=_correct(packet, codeact_answer),
        bulk_evidence_absent_from_provider_context=all(
            not _contains_value(context, page)
            for context in provider_contexts
            for page in pages
        ),
        immutable_handle_semantics_shared=same_disclosures and same_computation,
        native_python_has_evidence_acquisition_authority=False,
        canonical_write_authority=False,
        arc_action_authority=False,
    )
    (destination / "audit.json").write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return audit


async def _run_native_script(
    packet: CognitiveParityPacket,
    root: Path,
    specs: list[dict[str, Any]],
    final: dict[str, Any],
    *,
    compute: bool,
) -> _NativeAuditRun:
    plane, evidence = _context_evidence(packet, root, arm="native")
    responses = [_response("evidence_query", spec) for spec in specs]
    if compute:
        handles = [f"ev_native_{index:04d}" for index in range(1, len(specs) + 1)]
        responses.append(
            _response(
                "run_python",
                {
                    "code": (
                        f"handles={handles!r}\n"
                        "pages=[evidence_cache.load(handle) for handle in handles]\n"
                        + _COMPUTATION
                    ),
                    "values": {},
                },
            )
        )
    responses.append(_response("return_result", final))
    llm = FakeLLMClient(scripted_responses=responses)
    controller = ContextNativeActorController(
        llm=llm,
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        max_llm_calls=len(responses),
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    return _NativeAuditRun(controller, await controller.run(), evidence, llm)


async def _run_codeact_script(
    packet: CognitiveParityPacket,
    root: Path,
    destination: Path,
    specs: list[dict[str, Any]],
    final: dict[str, Any],
    *,
    compute: bool,
) -> _CodeActAuditRun:
    plane, evidence = _context_evidence(packet, root, arm="codeact")
    responses = _codeact_responses(specs, final, compute=compute)
    llm = FakeLLMClient(scripted_responses=responses)
    log = ExecutionProvenanceLog(
        destination / "codeact-provenance.jsonl",
        actor_session_id=f"audit:{packet.task.packet_id}",
    )
    controller = ContextCodeActController(
        llm=llm,
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        provenance=InvocationProvenance(
            log, invocation_id=f"audit:{packet.task.packet_id}:codeact"
        ),
        max_llm_calls=len(responses),
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    return _CodeActAuditRun(controller, await controller.run(), evidence, log)


def _codeact_responses(
    specs: list[dict[str, Any]], final: dict[str, Any], *, compute: bool
) -> list[LLMResponse]:
    responses: list[LLMResponse] = []
    if specs:
        code = "receipts=[]\n" + "\n".join(
            f"receipts.append(evidence.query(**{spec!r}))" for spec in specs
        )
        if compute:
            code += (
                "\npages=[evidence_cache.load(receipt.handle) "
                "for receipt in receipts]\n" + _COMPUTATION
            )
        responses.append(_response("execute_python", {"code": code}))
    return [*responses, _response("return_result", final)]


def _native_stdout(controller: ContextNativeActorController) -> str:
    if not controller.computation_records:
        return ""
    return controller.computation_records[0].stdout


def _codeact_stdout(log: ExecutionProvenanceLog) -> str:
    return next(
        (
            str(event.payload.get("stdout", ""))
            for event in log.events
            if event.event_type == "CellCompleted" and event.payload.get("stdout")
        ),
        "",
    )


def _contains_value(container: Any, target: dict[str, Any]) -> bool:
    if container == target:
        return True
    if isinstance(container, dict):
        return any(_contains_value(value, target) for value in container.values())
    if isinstance(container, (list, tuple)):
        return any(_contains_value(value, target) for value in container)
    if isinstance(container, str):
        encoded = json.dumps(target, sort_keys=True, separators=(",", ":"))
        return encoded in container
    return False


def _targeted_evidence_script(
    packet: CognitiveParityPacket, root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select only cited records; this proves reachability, not model discovery."""
    from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability

    refs = packet.expected.required_evidence_refs
    if not refs:
        refs = packet.expected.acceptable_evidence_refs[:1]
    capability = WorkspaceEvidenceCapability.from_root(root)
    specs: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for ref in refs:
        identity = (ref.level, ref.turn)
        if identity in seen:
            continue
        seen.add(identity)
        kind = "frames" if ref.turn == 0 else "transitions"
        spec: dict[str, Any] = {
            "kind": kind,
            "level": ref.level,
            "limit": 1,
            "projection": "full",
            "grid_encoding": "rle",
        }
        if ref.turn > 0:
            spec["after"] = {"level": ref.level, "turn": ref.turn - 1}
        page = capability.query(spec)
        if [record["ref"] for record in page["records"]] != [
            f"L{ref.level}:T{ref.turn}"
        ]:
            raise RuntimeError(f"targeted evidence query missed {ref}")
        specs.append(spec)
        pages.append(page)
    return specs, pages


def _correct(packet: CognitiveParityPacket, answer: dict[str, Any]) -> bool:
    value = evaluate_answer(packet, answer)
    checks = [value.evidence_precision == 1, value.evidence_recall == 1]
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


def _response(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"scripted-{name}",
                name=name,
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    )
