"""Zero-provider three-arm proof for remote-history orchestration."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from typing import Literal

from nooa.unifiedllm import FakeLLMClient
from pydantic import BaseModel, ConfigDict

from tycho_nooa.context_evidence_v2 import (
    EvidenceAccessContext,
    EvidenceAdmissionAuthority,
    EvidenceCache,
    EvidenceGrantRegistry,
    EvidenceReadBroker,
)
from tycho_nooa.context_evidence_v3 import ContextEvidencePlane
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import cognitive_answer
from tycho_nooa.matched_actor_context_v2 import (
    ContextCodeActController,
    ContextNativeActorController,
)
from tycho_nooa.matched_actor_context_v3 import ContextNativeAcquisitionController
from tycho_nooa.matched_actor_contract_golden_paths import (
    _FACTS_PREFIX,
    _build_plan,
    _correct,
    _facts_verified,
    _response,
)
from tycho_nooa.matched_actor_experiment import RecordingEvidenceCapability
from tycho_nooa.repaired_checkpoint import load_repaired_pilot_sample

StressArm = Literal["native_paged", "codeact", "native_acquire"]


class RemoteArmProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: StressArm
    provider_calls: int
    model_evidence_acquisitions: int
    source_pages: int
    python_cells: int
    terminal_call: int
    answer_correct: bool
    facts_verified: bool
    canonical_write_authority: Literal[False] = False
    arc_action_authority: Literal[False] = False


class RemoteHistoryStressProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: Literal["remote-history-stress-v1"] = "remote-history-stress-v1"
    provider_calls: Literal[0] = 0
    arc_actions: Literal[0] = 0
    packet_id: Literal["remote-failure-composition"]
    exact_source_disclosures_identical: bool
    packet_specific_helper_present: Literal[False] = False
    arms: tuple[RemoteArmProof, RemoteArmProof, RemoteArmProof]


async def prove_remote_history_stress(
    *, repository_root: Path, destination: Path
) -> RemoteHistoryStressProof:
    destination.mkdir(parents=True, exist_ok=False)
    _, packets = load_repaired_pilot_sample(
        repository_root / "analysis/matched-actor-v1-pilot/sample.json",
        repository_root / "analysis/matched-actor-v5-pilot/checkpoints.json",
        repository_root=repository_root,
    )
    packet = next(
        item for item in packets if item.task.packet_id == "remote-failure-composition"
    )
    root = repository_root / packet.checkpoint_root
    plan = _build_plan(packet.task, root)
    native, native_pages = await _prove_native_paged(packet, root, plan)
    codeact, codeact_pages = await _prove_codeact(
        packet, root, plan, destination
    )
    acquire, acquire_pages = await _prove_native_acquire(packet, root, plan)
    proof = RemoteHistoryStressProof(
        packet_id=packet.task.packet_id,
        exact_source_disclosures_identical=(
            native_pages == codeact_pages == acquire_pages
        ),
        arms=(native, codeact, acquire),
    )
    (destination / "report.json").write_text(
        json.dumps(proof.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return proof


async def _prove_native_paged(packet, root: Path, plan):
    plane, source = _v2_plane(packet, root, arm="native")
    controller = ContextNativeActorController(
        llm=FakeLLMClient(
            scripted_responses=[
                _response(step.name, step.arguments) for step in plan.native_steps
            ]
        ),
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        max_llm_calls=len(plan.native_steps),
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    value = await controller.run()
    stdout = tuple(record.stdout for record in controller.computation_records)
    proof = RemoteArmProof(
        arm="native_paged",
        provider_calls=len(controller.provider_turns),
        model_evidence_acquisitions=len(source.disclosures),
        source_pages=len(source.disclosures),
        python_cells=controller.python_cells,
        terminal_call=controller.terminal_return_call or 0,
        answer_correct=_correct(
            packet, cognitive_answer(packet, value).model_dump(mode="json")
        ),
        facts_verified=_facts_verified(stdout, plan.facts),
    )
    return proof, source.disclosures


async def _prove_codeact(packet, root: Path, plan, destination: Path):
    plane, source = _v2_plane(packet, root, arm="codeact")
    log = ExecutionProvenanceLog(
        destination / "codeact-provenance.jsonl",
        actor_session_id="remote-history:codeact",
    )
    controller = ContextCodeActController(
        llm=FakeLLMClient(
            scripted_responses=[
                _response("execute_python", {"code": plan.codeact_code}),
                _response("return_result", plan.answer),
            ]
        ),
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        provenance=InvocationProvenance(
            log, invocation_id="remote-history:codeact"
        ),
        max_llm_calls=2,
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    value = await controller.run()
    stdout = tuple(
        str(event.payload.get("stdout", ""))
        for event in log.events
        if event.event_type == "CellCompleted"
    )
    proof = RemoteArmProof(
        arm="codeact",
        provider_calls=len(controller.provider_turns),
        model_evidence_acquisitions=len(source.disclosures),
        source_pages=len(source.disclosures),
        python_cells=1,
        terminal_call=controller.terminal_return_call or 0,
        answer_correct=_correct(
            packet, cognitive_answer(packet, value).model_dump(mode="json")
        ),
        facts_verified=_facts_verified(stdout, plan.facts),
    )
    return proof, source.disclosures


async def _prove_native_acquire(packet, root: Path, plan):
    plane, source = _v3_plane(packet, root)
    handle = "ev_native_acquire_0001"
    code = (
        f"page=evidence_cache.load({handle!r})\n"
        "records=page.records\n"
        "no_change=[]\n"
        "for record in records:\n"
        "    before=[rle.decode_row(row) for row in record.prev]\n"
        "    after=[rle.decode_row(row) for row in record.next]\n"
        "    if before == after:\n"
        "        no_change.append(record.ref)\n"
        f"print({_FACTS_PREFIX!r}, "
        "{'scanned_records':len(records),'no_change_refs':no_change})"
    )
    controller = ContextNativeAcquisitionController(
        llm=FakeLLMClient(
            scripted_responses=[
                _response(
                    "evidence_acquire",
                    {
                        "kind": "transitions",
                        "level": 0,
                        "projection": "full",
                        "grid_encoding": "rle",
                    },
                ),
                _response("run_python", {"code": code, "values": {}}),
                _response("return_result", plan.answer),
            ]
        ),
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        max_llm_calls=3,
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    value = await controller.run()
    stdout = tuple(record.stdout for record in controller.computation_records)
    proof = RemoteArmProof(
        arm="native_acquire",
        provider_calls=len(controller.provider_turns),
        model_evidence_acquisitions=controller.evidence_queries,
        source_pages=len(source.disclosures),
        python_cells=controller.python_cells,
        terminal_call=controller.terminal_return_call or 0,
        answer_correct=_correct(
            packet, cognitive_answer(packet, value).model_dump(mode="json")
        ),
        facts_verified=_facts_verified(stdout, plan.facts),
    )
    return proof, source.disclosures


def _v2_plane(packet, root: Path, *, arm: str):
    from tycho_nooa.matched_actor_contract_golden_paths import _context_evidence

    return _context_evidence(packet, root, arm=arm)


def _v3_plane(packet, root: Path):
    source = RecordingEvidenceCapability(WorkspaceEvidenceCapability.from_root(root))
    context = EvidenceAccessContext(
        session_id=f"remote-history:{packet.task.packet_id}:native_acquire",
        invocation_id=f"{packet.task.packet_id}:native_acquire",
        role="actor",
        frontier=packet.task.evidence_frontier.model_dump(mode="json"),
    )
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    names = count(1)
    plane = ContextEvidencePlane(
        source,
        EvidenceAdmissionAuthority(
            cache,
            grants,
            handle_factory=lambda: f"ev_native_acquire_{next(names):04d}",
        ),
        EvidenceReadBroker(cache, grants),
        context,
    )
    return plane, source
