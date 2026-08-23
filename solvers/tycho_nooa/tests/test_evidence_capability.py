from __future__ import annotations

import json
from pathlib import Path

import pytest
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.evidence import (
    EvidenceFrontierMismatchError,
    EvidenceQuery,
    EvidenceRef,
    WorkspaceEvidenceCapability,
)
from tycho_nooa.models import ActionDecision
from tycho_nooa.sterile_execution import ExecutionManifest, SterileExecutor


def recorded_workspace(tmp_path: Path) -> GameWorkspace:
    workspace = GameWorkspace(
        "evidence-game",
        root=str(tmp_path),
        render=False,
        available_actions=["UP", "DOWN"],
    )
    workspace.record([[0, 1], [2, 3]], level=0, turn_in_level=0)
    workspace.record(
        [[0, 1], [3, 2]],
        level=0,
        turn_in_level=1,
        action="UP",
        state="NOT_FINISHED",
        available=["UP", "DOWN"],
    )
    workspace.record(
        [[1, 0], [3, 2]],
        level=0,
        turn_in_level=2,
        action="DOWN",
        state="NOT_FINISHED",
        available=["UP", "DOWN"],
    )
    return workspace


async def test_evidence_query_crosses_as_cached_values_not_workspace_references(
    tmp_path: Path,
) -> None:
    workspace = recorded_workspace(tmp_path)
    capability = WorkspaceEvidenceCapability(workspace)
    manifest = ExecutionManifest(
        invocation_id="actor-evidence-1",
        role="actor",
        capabilities=("evidence.query",),
    )

    async with SterileExecutor(
        manifest,
        return_type=ActionDecision,
        capability_handlers={"evidence.query": capability.query},
    ) as executor:
        result = await executor.run_cell(
            """
first = evidence.query(kind='transitions', level=0)
assert first['frontier'] == {'level': 0, 'turn': 2}
assert [item['action'] for item in first['records']] == ['UP', 'DOWN']
first['records'][0]['prev'][0][0] = 15
second = evidence.query(kind='transitions', level=0)
assert second['records'][0]['prev'][0][0] == 0
return_result(action='UP')
"""
        )

    assert result.return_value == ActionDecision(action="UP")
    assert len(result.capability_records) == 1
    record = result.capability_records[0]
    assert record.name == "evidence.query"
    assert record.arguments == {"kind": "transitions", "level": 0}
    assert record.status == "complete"
    serialized = json.dumps(record.result)
    assert str(workspace.dir) not in serialized


def test_query_is_paginated_and_bound_to_one_evidence_frontier(tmp_path: Path) -> None:
    workspace = recorded_workspace(tmp_path)
    capability = WorkspaceEvidenceCapability(workspace)

    first = capability.query({"kind": "frames", "limit": 2})
    assert [item["turn"] for item in first["records"]] == [0, 1]
    assert first["next_after"] == {"level": 0, "turn": 1}
    assert first["truncated"] is True

    second = capability.query(
        {
            "kind": "frames",
            "after": first["next_after"],
            "frontier": first["frontier"],
            "limit": 2,
        }
    )
    assert [item["turn"] for item in second["records"]] == [2]
    assert second["next_after"] is None
    assert second["truncated"] is False

    workspace.record([[9]], level=1, turn_in_level=0)
    with pytest.raises(EvidenceFrontierMismatchError, match="requested frontier"):
        capability.query(
            EvidenceQuery(
                kind="frames",
                frontier=EvidenceRef(level=0, turn=2),
            )
        )


async def test_manifest_and_handlers_must_describe_the_same_capabilities() -> None:
    manifest = ExecutionManifest(
        invocation_id="actor-evidence-mismatch",
        role="actor",
        capabilities=("evidence.query",),
    )

    with pytest.raises(ValueError, match="manifest capabilities"):
        SterileExecutor(manifest, return_type=ActionDecision)


async def test_rejected_query_is_audited_and_does_not_end_the_repl(
    tmp_path: Path,
) -> None:
    capability = WorkspaceEvidenceCapability(recorded_workspace(tmp_path))
    manifest = ExecutionManifest(
        invocation_id="actor-evidence-rejection",
        role="actor",
        capabilities=("evidence.query",),
    )

    async with SterileExecutor(
        manifest,
        return_type=ActionDecision,
        capability_handlers={"evidence.query": capability.query},
    ) as executor:
        rejected = await executor.run_cell(
            "evidence.query(kind='frames', frontier={'level': 9, 'turn': 9})"
        )
        accepted = await executor.run_cell(
            "page = evidence.query(kind='frames'); return_result(action='DOWN')"
        )

    assert rejected.status == "error"
    assert rejected.exception is not None
    assert rejected.exception.type == "_CapabilityError"
    assert len(rejected.capability_records) == 1
    capability_error = rejected.capability_records[0]
    assert capability_error.status == "error"
    assert capability_error.exception is not None
    assert capability_error.exception.type == "EvidenceFrontierMismatchError"
    assert accepted.return_value == ActionDecision(action="DOWN")


async def test_evidence_name_is_absent_without_manifest_capability() -> None:
    manifest = ExecutionManifest(invocation_id="actor-no-evidence", role="actor")

    async with SterileExecutor(manifest, return_type=ActionDecision) as executor:
        result = await executor.run_cell("evidence.query(kind='frames')")

    assert result.status == "error"
    assert result.exception is not None
    assert result.exception.type == "NameError"
