from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.context_evidence_v2 import (
    EvidenceAccessContext,
    EvidenceAdmissionAuthority,
    EvidenceCache,
    EvidenceGrantRegistry,
    EvidenceReadBroker,
)
from tycho_nooa.context_evidence_v3 import (
    MAX_ACQUIRED_RECORDS,
    ContextEvidencePlane,
    EvidenceAcquireArguments,
)
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability
from tycho_nooa.matched_actor import cognitive_answer
from tycho_nooa.matched_actor_context_v3 import ContextNativeAcquisitionController
from tycho_nooa.matched_actor_contract_golden_paths import _build_plan
from tycho_nooa.matched_actor_experiment import RecordingEvidenceCapability
from tycho_nooa.repaired_checkpoint import load_repaired_pilot_sample

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SAMPLE = REPOSITORY_ROOT / "analysis/matched-actor-v1-pilot/sample.json"
CHECKPOINTS = REPOSITORY_ROOT / "analysis/matched-actor-v5-pilot/checkpoints.json"
FRONTIER = {"level": 0, "turn": 3}


class _PagedSource:
    def __init__(self) -> None:
        self.queries: list[dict] = []

    def query(self, raw_query: dict) -> dict:
        self.queries.append(raw_query)
        after = raw_query.get("after")
        turn = 1 if after is None else after["turn"] + 1
        truncated = turn < 3
        return {
            "kind": raw_query["kind"],
            "frontier": FRONTIER,
            "records": [
                {
                    "type": "transition_metadata",
                    "level": 0,
                    "turn": turn,
                    "ref": f"L0:T{turn}",
                    "action": "LEFT",
                    "row": None,
                    "col": None,
                }
            ],
            "next_after": {"level": 0, "turn": turn} if truncated else None,
            "truncated": truncated,
            "truncation_reason": "record_limit" if truncated else None,
            "grid_encoding": raw_query["grid_encoding"],
            "projection": raw_query["projection"],
            "encoding_legend": None,
        }


def _plane(source, *, frontier: dict, handle: str = "ev_bulk"):
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    clock = lambda: datetime(2026, 8, 25, tzinfo=UTC)
    context = EvidenceAccessContext(
        session_id="bulk-test",
        invocation_id="bulk-invocation",
        role="actor",
        frontier=frontier,
    )
    return ContextEvidencePlane(
        source,
        EvidenceAdmissionAuthority(
            cache,
            grants,
            clock=clock,
            handle_factory=lambda: handle,
        ),
        EvidenceReadBroker(cache, grants, clock=clock),
        context,
    )


def _response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"bulk-{name}",
                name=name,
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    )


def test_acquisition_materializes_a_complete_set_behind_one_grant() -> None:
    source = _PagedSource()
    plane = _plane(source, frontier=FRONTIER)

    receipt = plane.acquire(
        {
            "kind": "transitions",
            "level": 0,
            "projection": "metadata",
            "grid_encoding": "rle",
        }
    )
    exact = plane.read({"handle": receipt["handle"]})

    assert receipt["record_count"] == 3
    assert receipt["refs"] == ["L0:T1", "L0:T2", "L0:T3"]
    assert receipt["next_after"] is None
    assert receipt["truncated"] is False
    assert [record["ref"] for record in exact["records"]] == receipt["refs"]
    assert len(source.queries) == 3
    assert source.queries[0].get("after") is None
    assert source.queries[1]["after"] == {"level": 0, "turn": 1}
    assert source.queries[2]["after"] == {"level": 0, "turn": 2}
    assert all(query["limit"] == 32 for query in source.queries)


def test_acquisition_contract_is_generic_and_host_bounded() -> None:
    schema = EvidenceAcquireArguments.model_json_schema()

    assert set(schema["properties"]) == {
        "kind",
        "level",
        "frontier",
        "grid_encoding",
        "projection",
    }
    assert "after" not in schema["properties"]
    assert "limit" not in schema["properties"]
    assert MAX_ACQUIRED_RECORDS == 256


@pytest.mark.asyncio
async def test_remote_native_acquisition_golden_path_takes_three_provider_turns(
) -> None:
    _, packets = load_repaired_pilot_sample(
        SAMPLE,
        CHECKPOINTS,
        repository_root=REPOSITORY_ROOT,
    )
    packet = next(
        item for item in packets if item.task.packet_id == "remote-failure-composition"
    )
    root = REPOSITORY_ROOT / packet.checkpoint_root
    plan = _build_plan(packet.task, root)
    source = RecordingEvidenceCapability(WorkspaceEvidenceCapability.from_root(root))
    names = count(1)
    plane = _plane(
        source,
        frontier=packet.task.evidence_frontier.model_dump(mode="json"),
        handle=f"ev_bulk_{next(names):04d}",
    )
    code = (
        "page=evidence_cache.load('ev_bulk_0001')\n"
        "changed=[]\n"
        "for record in page.records:\n"
        "    before=[rle.decode_row(row) for row in record.prev]\n"
        "    after=[rle.decode_row(row) for row in record.next]\n"
        "    if before == after:\n"
        "        changed.append(record.ref)\n"
        "print(changed)"
    )
    llm = FakeLLMClient(
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
    )
    controller = ContextNativeAcquisitionController(
        llm=llm,
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        max_llm_calls=3,
        max_output_tokens=256,
        cell_timeout_seconds=5,
    )

    result = await controller.run()
    answer = cognitive_answer(packet, result).model_dump(mode="json")

    assert len(controller.provider_turns) == 3
    assert controller.terminal_return_call == 3
    assert controller.evidence_queries == 1
    assert len(source.disclosures) == 30
    assert answer["conclusion"] == plan.answer["conclusion"]
    assert answer["evidence_refs"] == plan.answer["evidence_refs"]
    stdout = controller.computation_records[0].stdout
    assert all(
        f"L{ref['level']}:T{ref['turn']}" in stdout
        for ref in plan.answer["evidence_refs"]
    )
    rendered = json.dumps(llm.last_messages)
    assert '"prev"' not in rendered
    assert '"next"' not in rendered
