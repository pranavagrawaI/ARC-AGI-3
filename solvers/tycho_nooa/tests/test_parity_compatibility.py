from __future__ import annotations

import json
from pathlib import Path

import pytest
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.cognitive_parity import RetrievalAnswer, tree_sha256
from tycho_nooa.evidence import EvidenceRef, WorkspaceEvidenceCapability
from tycho_nooa.parity_compatibility import (
    CompatibilityEvidenceCapability,
    CompatibilityEvidencePolicy,
    CompatibilityExpected,
    CompatibilityGate,
    CompatibilityTask,
    evaluate_compatibility_answer,
    load_compatibility_gate,
)


def termination_gate() -> CompatibilityGate:
    return CompatibilityGate(
        task=CompatibilityTask(
            gate_id="g0-termination",
            gate_kind="termination",
            instruction="Return READY.",
            inputs={"supplied_answer": "READY"},
        ),
        evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
        expected=CompatibilityExpected(conclusion="READY"),
    )


def metadata_policy() -> CompatibilityEvidencePolicy:
    return CompatibilityEvidencePolicy(
        enabled=True,
        kind="transitions",
        level=0,
        limit=4,
        projection="metadata",
        grid_encoding="raw",
        max_queries=1,
    )


def test_treatment_payload_excludes_checkpoint_and_answer_key() -> None:
    gate = termination_gate()

    visible = gate.treatment_payload()

    assert visible["task"]["inputs"] == {"supplied_answer": "READY"}
    assert visible["evidence_contract"]["enabled"] is False
    assert "expected" not in visible
    assert "checkpoint_root" not in visible


def test_gate_topology_requires_checkpoint_exactly_when_evidence_is_enabled() -> None:
    with pytest.raises(ValueError, match="requires a frozen checkpoint"):
        CompatibilityGate(
            task=CompatibilityTask(
                gate_id="g1", gate_kind="metadata_query", instruction="Query once."
            ),
            evidence=metadata_policy(),
            expected=CompatibilityExpected(conclusion="LEFT"),
        )

    with pytest.raises(ValueError, match="cannot carry"):
        CompatibilityGate(
            checkpoint_root="somewhere",
            checkpoint_sha256="sha256:" + "0" * 64,
            task=termination_gate().task,
            evidence=termination_gate().evidence,
            expected=termination_gate().expected,
        )


def test_evaluator_scores_exact_value_and_reference() -> None:
    gate = termination_gate()

    accepted = evaluate_compatibility_answer(
        gate,
        RetrievalAnswer(evidence_refs=(), conclusion="READY"),
    )
    rejected = evaluate_compatibility_answer(
        gate,
        RetrievalAnswer(
            evidence_refs=(EvidenceRef(level=0, turn=1),), conclusion="ready"
        ),
    )

    assert accepted.passed is True
    assert rejected.model_dump() == {
        "gate_id": "g0-termination",
        "valid_typed_return": True,
        "conclusion_correct": False,
        "evidence_refs_correct": False,
        "passed": False,
    }


def test_capability_allows_one_exact_metadata_query(tmp_path: Path) -> None:
    workspace = GameWorkspace(
        "compatibility", root=str(tmp_path), render=False, available_actions=["LEFT"]
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    workspace.record(
        [[2]],
        level=0,
        turn_in_level=1,
        action="LEFT",
        state="NOT_FINISHED",
        available=["LEFT"],
    )
    subject = CompatibilityEvidenceCapability(
        WorkspaceEvidenceCapability(workspace), metadata_policy()
    )

    page = subject.query(metadata_policy().query_value())

    assert page["records"] == [
        {
            "type": "transition_metadata",
            "level": 0,
            "turn": 1,
            "action": "LEFT",
            "row": None,
            "col": None,
        }
    ]
    assert subject.queries == 1
    with pytest.raises(ValueError, match="cap exceeded"):
        subject.query(metadata_policy().query_value())


def test_capability_rejects_nonidentical_query_shape(tmp_path: Path) -> None:
    workspace = GameWorkspace(
        "compatibility", root=str(tmp_path), render=False, available_actions=["LEFT"]
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    subject = CompatibilityEvidenceCapability(
        WorkspaceEvidenceCapability(workspace), metadata_policy()
    )

    with pytest.raises(ValueError, match="requires exact"):
        subject.query({"kind": "transitions", "projection": "metadata"})


def test_loaded_gate_binds_checkpoint_and_constructs_capability(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    workspace = GameWorkspace(
        "compatibility",
        root=str(checkpoint),
        render=False,
        available_actions=["LEFT"],
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    gate = CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256=f"sha256:{tree_sha256(checkpoint)}",
        task=CompatibilityTask(
            gate_id="g1", gate_kind="metadata_query", instruction="Query once."
        ),
        evidence=metadata_policy(),
        expected=CompatibilityExpected(conclusion="LEFT"),
    )
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate.model_dump(mode="json")))

    loaded = load_compatibility_gate(path, repository_root=tmp_path)
    capability = CompatibilityEvidenceCapability.from_gate(
        loaded, repository_root=tmp_path
    )

    assert capability is not None
    assert capability.queries == 0

    workspace.record([[2]], level=0, turn_in_level=1, action="LEFT")
    with pytest.raises(ValueError, match="commitment mismatch"):
        load_compatibility_gate(path, repository_root=tmp_path)


def test_checkpoint_path_cannot_escape_repository(tmp_path: Path) -> None:
    gate = CompatibilityGate(
        checkpoint_root="../escape",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(
            gate_id="g1", gate_kind="metadata_query", instruction="Query once."
        ),
        evidence=metadata_policy(),
        expected=CompatibilityExpected(conclusion="LEFT"),
    )
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate.model_dump(mode="json")))

    with pytest.raises(ValueError, match="escapes"):
        load_compatibility_gate(path, repository_root=tmp_path)
