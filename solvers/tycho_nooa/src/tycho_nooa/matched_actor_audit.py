"""Zero-cost scripted proof that both Actor treatments can reach the same world."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from pydantic import BaseModel, ConfigDict

from tycho_nooa.cognitive_parity import CognitiveParityPacket, evaluate_answer
from tycho_nooa.computational_evidence import describe_evidence_page
from tycho_nooa.evidence import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import (
    MatchedCodeActController,
    MatchedNativeActorController,
    cognitive_answer,
)
from tycho_nooa.matched_actor_experiment import RecordingEvidenceCapability
from tycho_nooa.models import ActionDecision

_COMPUTATION = (
    "records = [record for page in pages for record in page['records']]\n"
    "refs = sorted(record['ref'] for record in records)\n"
    "checksum = 0\n"
    "for record in records:\n"
    "    for field in ('grid', 'prev', 'next'):\n"
    "        for row in record.get(field, ()):\n"
    "            checksum += sum(rle.decode_row(row))\n"
    "print(len(records), len(refs), checksum)"
)


class ActorReachabilityAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    evidence_disclosures_identical: bool
    computation_stdout_identical: bool
    native_answer_correct: bool
    codeact_answer_correct: bool
    native_python_has_evidence_authority: bool
    canonical_write_authority: bool
    arc_action_authority: bool


async def audit_matched_actor_packet(
    packet: CognitiveParityPacket,
    *,
    repository_root: Path,
    destination: Path,
) -> ActorReachabilityAudit:
    """Execute one successful scripted trajectory through each real controller."""
    destination.mkdir(parents=True, exist_ok=False)
    root = repository_root / packet.checkpoint_root
    specs, pages = _evidence_script(packet, root)
    final = _flat_expected_answer(packet)

    native_evidence = RecordingEvidenceCapability(
        WorkspaceEvidenceCapability.from_root(root)
    )
    native_llm = FakeLLMClient(
        scripted_responses=[
            *[_response("evidence_query", spec) for spec in specs],
            _response(
                "run_python",
                {
                    "code": _COMPUTATION,
                    "values": {
                        "pages": [describe_evidence_page(page) for page in pages]
                    },
                },
            ),
            _response("return_result", final),
        ]
    )
    native = MatchedNativeActorController(
        llm=native_llm,
        packet=packet,
        evidence=native_evidence,
        max_llm_calls=len(specs) + 2,
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    native_value = await native.run()

    codeact_evidence = RecordingEvidenceCapability(
        WorkspaceEvidenceCapability.from_root(root)
    )
    code = "pages = []\n" + "\n".join(
        f"pages.append(evidence.query(**{spec!r}))" for spec in specs
    )
    code = code + "\n" + _COMPUTATION
    codeact_llm = FakeLLMClient(
        scripted_responses=[
            _response("execute_python", {"code": code}),
            _response("return_result", final),
        ]
    )
    provenance_path = destination / "codeact-provenance.jsonl"
    log = ExecutionProvenanceLog(
        provenance_path, actor_session_id=f"audit:{packet.task.packet_id}"
    )
    codeact = MatchedCodeActController(
        llm=codeact_llm,
        packet=packet,
        evidence=codeact_evidence,
        provenance=InvocationProvenance(
            log, invocation_id=f"audit:{packet.task.packet_id}:codeact"
        ),
        max_llm_calls=2,
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    codeact_value = await codeact.run()

    codeact_stdout = next(
        event.payload["stdout"]
        for event in log.events
        if event.event_type == "CellCompleted" and event.payload.get("stdout")
    )
    native_answer = cognitive_answer(packet, native_value)
    codeact_answer = cognitive_answer(packet, codeact_value)
    audit = ActorReachabilityAudit(
        packet_id=packet.task.packet_id,
        evidence_disclosures_identical=(
            native_evidence.disclosures == codeact_evidence.disclosures == pages
        ),
        computation_stdout_identical=(
            native.computation_records[0].stdout == codeact_stdout
        ),
        native_answer_correct=_correct(packet, native_answer.model_dump(mode="json")),
        codeact_answer_correct=_correct(
            packet, codeact_answer.model_dump(mode="json")
        ),
        native_python_has_evidence_authority=False,
        canonical_write_authority=False,
        arc_action_authority=False,
    )
    (destination / "audit.json").write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return audit


def _evidence_script(
    packet: CognitiveParityPacket, root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs = (
        *packet.expected.required_evidence_refs,
        *packet.expected.acceptable_evidence_refs,
    )
    if not refs:
        return [], []
    capability = WorkspaceEvidenceCapability.from_root(root)
    levels = sorted({ref.level for ref in refs})
    specs: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    for kind in _evidence_kinds(refs):
        for level in levels:
            for spec, page in _evidence_pages(capability, kind=kind, level=level):
                specs.append(spec)
                pages.append(page)
    return specs, pages


def _evidence_kinds(refs: tuple[Any, ...]) -> tuple[str, ...]:
    if any(ref.turn == 0 for ref in refs):
        return "transitions", "frames"
    return ("transitions",)


def _evidence_pages(
    capability: WorkspaceEvidenceCapability, *, kind: str, level: int
):
    after: dict[str, int] | None = None
    while True:
        spec: dict[str, Any] = {
            "kind": kind,
            "level": level,
            "limit": 32,
            "projection": "full",
            "grid_encoding": "rle",
        }
        if after is not None:
            spec["after"] = after
        page = capability.query(spec)
        yield spec, page
        if not page["truncated"]:
            return
        after = page["next_after"]


def _flat_expected_answer(packet: CognitiveParityPacket) -> dict[str, Any]:
    expected = packet.expected
    refs = expected.required_evidence_refs or expected.acceptable_evidence_refs[:1]
    evidence_refs = [ref.model_dump(mode="json") for ref in refs]
    kind = packet.task.task_kind
    if kind == "retrieve":
        return {"conclusion": expected.rationale, "evidence_refs": evidence_refs}
    if kind == "discriminate":
        return {
            "conclusion": expected.rationale,
            "verdict": expected.accepted_verdicts[0],
            "falsified": list(expected.falsified_hypotheses),
            "evidence_refs": evidence_refs,
        }
    if kind == "recover_invariant":
        return {
            "invariant_id": expected.accepted_invariant_ids[0],
            "invariant": expected.rationale,
            "invariant_atoms": expected.invariant_atoms,
            "evidence_refs": evidence_refs,
            "scope": expected.accepted_scopes[0],
        }
    if kind == "propose_action":
        decision = expected.native_action or ActionDecision(
            action=expected.legal_actions[0]
        )
        return {
            "decision": decision.model_dump(mode="json"),
            "evidence_refs": evidence_refs,
            "expected_effect": expected.rationale,
            "information_goal": None,
        }
    raise ValueError(f"unsupported audit task kind: {kind}")


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
