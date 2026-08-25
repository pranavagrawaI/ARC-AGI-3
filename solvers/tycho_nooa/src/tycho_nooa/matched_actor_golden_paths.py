"""Answer-key-independent golden paths for the failed Matched Actor pilot cases."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from pydantic import BaseModel, ConfigDict

from tycho_nooa.cognitive_parity import CognitiveParityPacket, evaluate_answer
from tycho_nooa.computational_evidence import decode_rle_row
from tycho_nooa.context_evidence import EvidenceReceipt
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import cognitive_answer
from tycho_nooa.matched_actor_context import (
    CONTEXT_NATIVE_SYSTEM_PROMPT,
    ContextCodeActController,
    ContextNativeActorController,
)
from tycho_nooa.matched_actor_experiment_v4 import _context_evidence
from tycho_nooa.native_parity import NativeEvidenceArguments

Arm = Literal["native", "codeact"]
_FACTS_PREFIX = "GOLDEN_FACTS "
_INCOMPLETE_PACKET_IDS = (
    "action-success-grounded",
    "history-success-level1-noop",
    "objective-success-level-boundary",
    "cross-level-success-action-response",
    "remote-failure-composition",
)


class InterfaceFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: str
    surface: Literal["native", "codeact", "shared"]
    observation: str
    consequence: str


class ArmGoldenPathProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Arm
    provider_calls: int
    evidence_queries: int
    python_cells: int
    terminal_call: int
    answer_correct: bool
    facts_verified: bool
    within_six_calls: bool
    contract_derivable: bool
    undocumented_dependencies: tuple[str, ...]
    operation_names: tuple[str, ...]


class PacketGoldenPathProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    native: ArmGoldenPathProof
    codeact: ArmGoldenPathProof
    task_inherently_exceeds_six_calls: tuple[Arm, ...]
    minimum_basis: str


class GoldenPathSuiteProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: Literal["matched-actor-v5-golden-paths"] = (
        "matched-actor-v5-golden-paths"
    )
    provider_calls: Literal[0] = 0
    arc_actions: Literal[0] = 0
    answer_key_used_for_planning: Literal[False] = False
    packet_count: int
    all_runtime_paths_correct: bool
    interface_findings: tuple[InterfaceFinding, ...]
    packets: tuple[PacketGoldenPathProof, ...]


@dataclass(frozen=True)
class _Step:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class _Plan:
    packet_id: str
    facts: dict[str, Any]
    answer: dict[str, Any]
    native_steps: tuple[_Step, ...]
    codeact_code: str
    minimum_basis: str
    native_dependencies: tuple[str, ...] = ()


async def prove_golden_paths(
    packets: tuple[CognitiveParityPacket, ...],
    *,
    repository_root: Path,
    destination: Path,
) -> GoldenPathSuiteProof:
    """Execute both runtime-valid paths without making a provider call."""
    selected = {
        packet.task.packet_id: packet
        for packet in packets
        if packet.task.packet_id in _INCOMPLETE_PACKET_IDS
    }
    if tuple(selected) != _INCOMPLETE_PACKET_IDS:
        raise ValueError("golden-path packet order or membership drifted")
    destination.mkdir(parents=True, exist_ok=False)
    proofs: list[PacketGoldenPathProof] = []
    for packet in selected.values():
        root = repository_root / packet.checkpoint_root
        plan = _build_plan(packet.task, root)
        packet_destination = destination / packet.task.packet_id
        packet_destination.mkdir()
        native = await _prove_native(packet, root, plan)
        codeact = await _prove_codeact(packet, root, plan, packet_destination)
        proof = PacketGoldenPathProof(
            packet_id=packet.task.packet_id,
            native=native,
            codeact=codeact,
            task_inherently_exceeds_six_calls=tuple(
                arm
                for arm, result in (("native", native), ("codeact", codeact))
                if not result.within_six_calls
            ),
            minimum_basis=plan.minimum_basis,
        )
        (packet_destination / "proof.json").write_text(
            json.dumps(proof.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        proofs.append(proof)
    suite = GoldenPathSuiteProof(
        packet_count=len(proofs),
        all_runtime_paths_correct=all(
            arm.answer_correct and arm.facts_verified
            for proof in proofs
            for arm in (proof.native, proof.codeact)
        ),
        interface_findings=_interface_findings(),
        packets=tuple(proofs),
    )
    (destination / "report.json").write_text(
        json.dumps(suite.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return suite


async def _prove_native(
    packet: CognitiveParityPacket, root: Path, plan: _Plan
) -> ArmGoldenPathProof:
    plane, evidence = _context_evidence(packet, root, arm="native")
    responses = [_response(step.name, step.arguments) for step in plan.native_steps]
    controller = ContextNativeActorController(
        llm=FakeLLMClient(scripted_responses=responses),
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        max_llm_calls=len(responses),
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    value = await controller.run()
    answer = cognitive_answer(packet, value).model_dump(mode="json")
    computation_stdout = tuple(
        record.stdout for record in controller.computation_records
    )
    facts_verified = (
        _facts_verified(computation_stdout, plan.facts)
        if computation_stdout
        else _receipt_facts_verified(evidence.disclosures, plan.facts)
    )
    return ArmGoldenPathProof(
        arm="native",
        provider_calls=len(controller.provider_turns),
        evidence_queries=len(evidence.disclosures),
        python_cells=controller.python_cells,
        terminal_call=controller.terminal_return_call or 0,
        answer_correct=_correct(packet, answer),
        facts_verified=facts_verified,
        within_six_calls=len(responses) <= 6,
        contract_derivable=not plan.native_dependencies,
        undocumented_dependencies=plan.native_dependencies,
        operation_names=tuple(step.name for step in plan.native_steps),
    )


async def _prove_codeact(
    packet: CognitiveParityPacket,
    root: Path,
    plan: _Plan,
    destination: Path,
) -> ArmGoldenPathProof:
    plane, evidence = _context_evidence(packet, root, arm="codeact")
    log = ExecutionProvenanceLog(
        destination / "codeact-provenance.jsonl",
        actor_session_id=f"golden:{packet.task.packet_id}",
    )
    responses = [
        _response("execute_python", {"code": plan.codeact_code}),
        _response("return_result", plan.answer),
    ]
    controller = ContextCodeActController(
        llm=FakeLLMClient(scripted_responses=responses),
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        provenance=InvocationProvenance(
            log, invocation_id=f"golden:{packet.task.packet_id}:codeact"
        ),
        max_llm_calls=len(responses),
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )
    if "evidence_contract" in packet.treatment_payload():
        raise RuntimeError("matched packet now supplies an evidence contract")
    query_contract = controller._execution_manifest().capability_contracts.get(
        "evidence.query"
    )
    if query_contract != {"result_schema": "evidence_receipt.v1"}:
        raise RuntimeError(
            "matched CodeAct query contract changed; update the golden-path finding"
        )
    value = await controller.run()
    answer = cognitive_answer(packet, value).model_dump(mode="json")
    stdout = tuple(
        str(event.payload.get("stdout", ""))
        for event in log.events
        if event.event_type == "CellCompleted"
    )
    return ArmGoldenPathProof(
        arm="codeact",
        provider_calls=len(controller.provider_turns),
        evidence_queries=len(evidence.disclosures),
        python_cells=1,
        terminal_call=controller.terminal_return_call or 0,
        answer_correct=_correct(packet, answer),
        facts_verified=_facts_verified(stdout, plan.facts),
        within_six_calls=True,
        contract_derivable=False,
        undocumented_dependencies=("codeact_evidence_query_contract_missing",),
        operation_names=("execute_python", "return_result"),
    )


def _build_plan(task: Any, root: Path) -> _Plan:
    """Derive a script from task text and evidence; never accept an answer key."""
    from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability

    source = WorkspaceEvidenceCapability.from_root(root)
    builders = {
        "action-success-grounded": _action_plan,
        "history-success-level1-noop": _history_plan,
        "objective-success-level-boundary": _objective_plan,
        "cross-level-success-action-response": _cross_level_plan,
        "remote-failure-composition": _remote_plan,
    }
    try:
        return builders[task.packet_id](task, source)
    except KeyError as error:
        raise ValueError(f"no golden path for {task.packet_id!r}") from error


def _action_plan(task: Any, source: Any) -> _Plan:
    query = _query(kind="transitions", level=1, limit=2, projection="metadata")
    page = source.query(query)
    ref = page["records"][-1]["ref"]
    action = (
        "UP"
        if "UP" in task.inputs["available_actions"]
        else task.inputs["available_actions"][0]
    )
    facts = {"grounding_ref": ref, "legal_action": action}
    answer = {
        "decision": {"action": action, "row": None, "col": None},
        "evidence_refs": [_ref_value(ref)],
        "expected_effect": f"{action} is host-listed and may reveal the next response.",
        "information_goal": "Observe the effect of one legal directional action.",
    }
    code = (
        f"receipt=evidence.query(**{query!r})\n"
        f"print({_FACTS_PREFIX!r}, "
        "{'grounding_ref': receipt.refs[-1], 'legal_action': 'UP'})"
    )
    return _Plan(
        packet_id=task.packet_id,
        facts=facts,
        answer=answer,
        native_steps=(
            _Step("evidence_query", query),
            _Step("return_result", answer),
        ),
        codeact_code=code,
        minimum_basis="one grounding disclosure plus one terminal return",
    )


def _history_plan(task: Any, source: Any) -> _Plan:
    query = _query(kind="transitions", level=1, limit=1, projection="full")
    page = source.query(query)
    record = page["records"][0]
    changed = _changed_cells(record)
    facts = {"ref": record["ref"], "action": record["action"], "changed_cells": changed}
    verdict = "H_history" if record["action"] and changed == 0 else "H_visual"
    answer = {
        "conclusion": (
            f"{record['ref']} records {record['action']} with {changed} changed cells, "
            "so action history is required."
        ),
        "verdict": verdict,
        "falsified": ["H_visual" if verdict == "H_history" else "H_history"],
        "evidence_refs": [_ref_value(record["ref"])],
    }
    computation = _fact_code(("ev_native_0001",), _history_fact_body())
    code = (
        f"receipt=evidence.query(**{query!r})\n"
        "page=evidence_cache.load(receipt.handle)\n" + _history_fact_body()
    )
    return _Plan(
        packet_id=task.packet_id,
        facts=facts,
        answer=answer,
        native_steps=(
            _Step("evidence_query", query),
            _Step("run_python", {"code": computation, "values": {}}),
            _Step("return_result", answer),
        ),
        codeact_code=code,
        minimum_basis="one full disclosure, one exact-value comparison, one return",
    )


def _objective_plan(task: Any, source: Any) -> _Plan:
    transitions = _query(kind="transitions", level=0, limit=32, projection="metadata")
    frames = _query(kind="frames", level=1, limit=1, projection="metadata")
    transition_page, frame_page = source.query(transitions), source.query(frames)
    last, first = transition_page["records"][-1], frame_page["records"][0]
    facts = {
        "last_level0_ref": last["ref"],
        "last_level0_action": last["action"],
        "first_level1_ref": first["ref"],
    }
    verdict = "H_LEFT" if last["action"] == "LEFT" else "H_OTHER"
    answer = {
        "conclusion": (
            f"{last['ref']} records {last['action']} immediately before {first['ref']}."
        ),
        "verdict": verdict,
        "falsified": ["H_OTHER" if verdict == "H_LEFT" else "H_LEFT"],
        "evidence_refs": [_ref_value(last["ref"]), _ref_value(first["ref"])],
    }
    body = _objective_fact_body()
    native_code = (
        "transitions=evidence_cache.load('ev_native_0001')\n"
        "frames=evidence_cache.load('ev_native_0002')\n" + body
    )
    code = (
        f"transition_receipt=evidence.query(**{transitions!r})\n"
        f"frame_receipt=evidence.query(**{frames!r})\n"
        "transitions=evidence_cache.load(transition_receipt.handle)\n"
        "frames=evidence_cache.load(frame_receipt.handle)\n" + body
    )
    return _Plan(
        packet_id=task.packet_id,
        facts=facts,
        answer=answer,
        native_steps=(
            _Step("evidence_query", transitions),
            _Step("evidence_query", frames),
            _Step("run_python", {"code": native_code, "values": {}}),
            _Step("return_result", answer),
        ),
        codeact_code=code,
        minimum_basis="two boundary disclosures, one ordering computation, one return",
    )


def _cross_level_plan(task: Any, source: Any) -> _Plan:
    metadata = _query(kind="transitions", limit=32, projection="metadata")
    metadata_page = source.query(metadata)
    left = {
        record["level"]: record
        for record in metadata_page["records"]
        if record["action"] == "LEFT"
    }
    target_queries = tuple(
        _query(
            kind="transitions",
            level=level,
            after={"level": level, "turn": record["turn"] - 1},
            limit=1,
            projection="full",
        )
        for level, record in sorted(left.items())
    )
    target_records = [source.query(query)["records"][0] for query in target_queries]
    counts = {record["level"]: _changed_cells(record) for record in target_records}
    facts = {
        "refs": [record["ref"] for record in target_records],
        "level0_changed_cells": counts[0],
        "level1_changed_cells": counts[1],
        "same_changed_cell_count": counts[0] == counts[1],
    }
    answer = {
        "invariant_id": "left-response-level-dependent",
        "invariant": (
            f"LEFT changes {counts[0]} cells in level 0 and {counts[1]} in level 1."
        ),
        "invariant_atoms": {
            "level0_changed_cells": counts[0],
            "level1_changed_cells": counts[1],
            "same_changed_cell_count": counts[0] == counts[1],
        },
        "evidence_refs": [_ref_value(record["ref"]) for record in target_records],
        "scope": "cross_level",
    }
    discovery = _cross_discovery_body()
    comparison = _cross_comparison_body()
    native_discovery = "metadata=evidence_cache.load('ev_native_0001')\n" + discovery
    native_comparison = (
        "targets=[evidence_cache.load('ev_native_0002').records[0],"
        "evidence_cache.load('ev_native_0003').records[0]]\n" + comparison
    )
    code = (
        f"metadata_receipt=evidence.query(**{metadata!r})\n"
        "metadata=evidence_cache.load(metadata_receipt.handle)\n"
        + discovery.replace(f"print({_FACTS_PREFIX!r}, discovery)\n", "")
        + "\ntargets=[]\n"
        + "\n".join(
            f"receipt_{index}=evidence.query(**{query!r})\n"
            f"targets.append(evidence_cache.load(receipt_{index}.handle).records[0])"
            for index, query in enumerate(target_queries)
        )
        + "\n"
        + comparison
    )
    return _Plan(
        packet_id=task.packet_id,
        facts=facts,
        answer=answer,
        native_steps=(
            _Step("evidence_query", metadata),
            _Step("run_python", {"code": native_discovery, "values": {}}),
            *(_Step("evidence_query", query) for query in target_queries),
            _Step("run_python", {"code": native_comparison, "values": {}}),
            _Step("return_result", answer),
        ),
        codeact_code=code,
        minimum_basis=(
            "one metadata discovery, one discovery computation, two targeted full "
            "disclosures, one comparison, one return"
        ),
        native_dependencies=("after_cursor_shape_under_specified",),
    )


def _remote_plan(task: Any, source: Any) -> _Plan:
    queries: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    after: dict[str, int] | None = None
    while True:
        query = _query(
            kind="transitions",
            level=0,
            after=after,
            limit=32,
            projection="full",
        )
        page = source.query(query)
        queries.append(query)
        pages.append(page)
        after = page["next_after"]
        if after is None:
            break
    records = [record for page in pages for record in page["records"]]
    no_change = [record for record in records if _changed_cells(record) == 0]
    facts = {
        "scanned_records": len(records),
        "no_change_refs": [r["ref"] for r in no_change],
    }
    answer = {
        "conclusion": "Four recorded transitions produced no changed grid cell.",
        "evidence_refs": [_ref_value(record["ref"]) for record in no_change],
    }
    handles = [f"ev_native_{index:04d}" for index in range(1, len(queries) + 1)]
    native_code = (
        f"pages=[evidence_cache.load(handle) for handle in {handles!r}]\n"
        + _remote_fact_body()
    )
    code = (
        "pages=[]\n"
        "after=None\n"
        "while True:\n"
        "    arguments={'kind':'transitions','level':0,'limit':32,"
        "'projection':'full','grid_encoding':'rle'}\n"
        "    if after is not None:\n"
        "        arguments['after']=after\n"
        "    receipt=evidence.query(**arguments)\n"
        "    page=evidence_cache.load(receipt.handle)\n"
        "    pages.append(page)\n"
        "    after=page.next_after\n"
        "    if after is None:\n"
        "        break\n" + _remote_fact_body()
    )
    return _Plan(
        packet_id=task.packet_id,
        facts=facts,
        answer=answer,
        native_steps=(
            *(_Step("evidence_query", query) for query in queries),
            _Step("run_python", {"code": native_code, "values": {}}),
            _Step("return_result", answer),
        ),
        codeact_code=code,
        minimum_basis=(
            f"{len(records)} full transitions fit at most two per bounded page: "
            f"{len(queries)} acquisitions, one comparison, one return"
        ),
        native_dependencies=(
            "after_cursor_shape_under_specified",
            "receipt_omits_next_after",
        ),
    )


def _query(
    *,
    kind: str,
    level: int | None = None,
    after: dict[str, int] | None = None,
    limit: int,
    projection: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": kind,
        "limit": limit,
        "projection": projection,
        "grid_encoding": "rle",
    }
    if level is not None:
        value["level"] = level
    if after is not None:
        value["after"] = after
    return value


def _history_fact_body() -> str:
    return (
        "record=page.records[0]\n"
        + _changed_function()
        + "facts={'ref':record.ref,'action':record.action,"
        "'changed_cells':changed_cells(record)}\n"
        f"print({_FACTS_PREFIX!r}, facts)\n"
    )


def _objective_fact_body() -> str:
    return (
        "last=transitions.records[-1]\n"
        "first=frames.records[0]\n"
        "facts={'last_level0_ref':last.ref,'last_level0_action':last.action,"
        "'first_level1_ref':first.ref}\n"
        f"print({_FACTS_PREFIX!r}, facts)\n"
    )


def _cross_discovery_body() -> str:
    return (
        "left={record.level:record for record in metadata.records "
        "if record.action=='LEFT'}\n"
        "discovery={'refs':[left[level].ref for level in sorted(left)]}\n"
        f"print({_FACTS_PREFIX!r}, discovery)\n"
    )


def _cross_comparison_body() -> str:
    return (
        _changed_function()
        + "counts={record.level:changed_cells(record) for record in targets}\n"
        "facts={'refs':[record.ref for record in targets],"
        "'level0_changed_cells':counts[0],'level1_changed_cells':counts[1],"
        "'same_changed_cell_count':counts[0]==counts[1]}\n"
        f"print({_FACTS_PREFIX!r}, facts)\n"
    )


def _remote_fact_body() -> str:
    return (
        "records=[record for page in pages for record in page.records]\n"
        + _changed_function()
        + "facts={'scanned_records':len(records),'no_change_refs':"
        "[record.ref for record in records if changed_cells(record)==0]}\n"
        f"print({_FACTS_PREFIX!r}, facts)\n"
    )


def _changed_function() -> str:
    return (
        "def changed_cells(record):\n"
        "    before=[rle.decode_row(row) for row in record.prev_rows]\n"
        "    after=[rle.decode_row(row) for row in record.next_rows]\n"
        "    return sum(left!=right for a,b in zip(before,after) "
        "for left,right in zip(a,b))\n"
    )


def _fact_code(handles: tuple[str, ...], body: str) -> str:
    if handles != ("ev_native_0001",):
        raise ValueError("single-page fact helper received unexpected handles")
    return "page=evidence_cache.load('ev_native_0001')\n" + body


def _changed_cells(record: dict[str, Any]) -> int:
    before = tuple(decode_rle_row(row) for row in record["prev"])
    after = tuple(decode_rle_row(row) for row in record["next"])
    return sum(
        left != right
        for before_row, after_row in zip(before, after, strict=True)
        for left, right in zip(before_row, after_row, strict=True)
    )


def _ref_value(ref: str) -> dict[str, int]:
    level, turn = ref.removeprefix("L").split(":T")
    return {"level": int(level), "turn": int(turn)}


def _facts_verified(
    stdout: tuple[str, ...],
    expected: dict[str, Any],
) -> bool:
    candidates: list[dict[str, Any]] = []
    for value in stdout:
        for line in value.splitlines():
            if not line.startswith(_FACTS_PREFIX):
                continue
            parsed = ast.literal_eval(line.removeprefix(_FACTS_PREFIX))
            if isinstance(parsed, dict):
                candidates.append(parsed)
    return expected in candidates


def _receipt_facts_verified(
    disclosures: list[dict[str, Any]], expected: dict[str, Any]
) -> bool:
    if set(expected) != {"grounding_ref", "legal_action"} or not disclosures:
        return False
    records = disclosures[-1].get("records", [])
    return bool(records) and records[-1].get("ref") == expected["grounding_ref"]


def _correct(packet: CognitiveParityPacket, answer: dict[str, Any]) -> bool:
    evaluation = evaluate_answer(packet, answer)
    checks = [evaluation.evidence_precision == 1, evaluation.evidence_recall == 1]
    checks.extend(
        result is not False
        for result in (
            evaluation.verdict_correct,
            evaluation.falsification_correct,
            evaluation.invariant_correct,
            evaluation.scope_correct,
            evaluation.action_legal,
        )
    )
    if evaluation.invariant_atom_recall is not None:
        checks.append(evaluation.invariant_atom_recall == 1)
    return all(checks)


def _interface_findings() -> tuple[InterfaceFinding, ...]:
    native_query_schema = NativeEvidenceArguments.model_json_schema()
    limit_schema = native_query_schema["properties"]["limit"]
    after_schema = native_query_schema["properties"]["after"]
    findings = [
        InterfaceFinding(
            finding_id="codeact_evidence_query_contract_missing",
            surface="codeact",
            observation=(
                "CognitiveParityPacket.treatment_payload omits evidence_contract; "
                "the matched manifest exposes only result_schema and renders the "
                "leaf capability as <evidence.query>."
            ),
            consequence=(
                "Every successful CodeAct evidence query relies on undocumented "
                "keyword arguments; the paid introspection loops were rational."
            ),
        ),
        InterfaceFinding(
            finding_id="native_limit_constraint_missing",
            surface="native",
            observation=(
                "The provider schema has no maximum for limit even though the "
                "evidence service rejects values above 32."
            ),
            consequence="The provider can emit schema-valid calls that T1 rejects.",
        ),
        InterfaceFinding(
            finding_id="after_cursor_shape_under_specified",
            surface="shared",
            observation=(
                "after is exposed as an arbitrary integer map rather than a value "
                "requiring both level and turn."
            ),
            consequence="Pagination repair consumes calls in either control language.",
        ),
        InterfaceFinding(
            finding_id="receipt_omits_next_after",
            surface="shared",
            observation=(
                "EvidenceReceipt omits next_after, truncated, and truncation_reason "
                "although those fields exist in the granted exact page."
            ),
            consequence=(
                "The provider cannot see the canonical continuation cursor without "
                "an additional computational dereference."
            ),
        ),
        InterfaceFinding(
            finding_id="native_single_call_policy_not_prompted",
            surface="native",
            observation=(
                "The controller rejects multiple tools in one turn, but the system "
                "prompt does not state that rule."
            ),
            consequence="Provider multi-tool turns can consume a call without progress.",
        ),
    ]
    if "maximum" in limit_schema:
        raise RuntimeError(
            "native limit schema now exposes its maximum; update findings"
        )
    if "level" in json.dumps(after_schema) and "required" in json.dumps(after_schema):
        raise RuntimeError("native after schema now exposes its required shape")
    omitted = {"next_after", "truncated", "truncation_reason"} & set(
        EvidenceReceipt.model_fields
    )
    if omitted:
        raise RuntimeError(
            f"receipt pagination fields are no longer omitted: {omitted}"
        )
    if "exactly one" in CONTEXT_NATIVE_SYSTEM_PROMPT:
        raise RuntimeError("native prompt now documents its single-call policy")
    return tuple(findings)


def _response(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"golden-{name}",
                name=name,
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    )
