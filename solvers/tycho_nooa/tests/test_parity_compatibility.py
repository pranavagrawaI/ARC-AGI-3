from __future__ import annotations

import json
from pathlib import Path

import pytest
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.cognitive_parity import tree_sha256
from tycho_nooa.compatibility_return import (
    CompositionReturn,
    IdentifierBindingReturn,
    RepresentationReturn,
    TerminationReturn,
)
from tycho_nooa.evidence import WorkspaceEvidenceCapability
from tycho_nooa.parity_compatibility import (
    CompatibilityEvidenceCapability,
    CompatibilityEvidencePolicy,
    CompatibilityExpected,
    CompatibilityGate,
    CompatibilityTask,
    canonicalize_evidence_query,
    evaluate_compatibility_answer,
    load_compatibility_gate,
)


def termination_gate() -> CompatibilityGate:
    return CompatibilityGate(
        task=CompatibilityTask(
            gate_id="g0-termination",
            gate_kind="termination",
            instruction="Return READY.",
            inputs={"supplied_answer": True},
        ),
        evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
        expected=CompatibilityExpected(ok=True),
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
        max_query_attempts=2,
    )


def test_treatment_payload_excludes_checkpoint_and_answer_key() -> None:
    gate = termination_gate()

    visible = gate.treatment_payload()

    assert visible["task"]["inputs"] == {"supplied_answer": True}
    assert visible["evidence_contract"]["enabled"] is False
    assert visible["evidence_contract"]["max_successful_disclosures"] == 0
    assert "max_queries" not in visible["evidence_contract"]
    assert "expected" not in visible
    assert "checkpoint_root" not in visible

    metadata_gate = CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(
            gate_id="g1", gate_kind="capability_retrieval", instruction="Query metadata."
        ),
        evidence=metadata_policy(),
        expected=CompatibilityExpected(evidence_ref="L0:T1"),
    )
    assert "grid_encoding" not in metadata_gate.treatment_payload()[
        "evidence_contract"
    ]


def test_gate_topology_requires_checkpoint_exactly_when_evidence_is_enabled() -> None:
    with pytest.raises(ValueError, match="requires a frozen checkpoint"):
        CompatibilityGate(
            task=CompatibilityTask(
                gate_id="g1", gate_kind="capability_retrieval", instruction="Query once."
            ),
            evidence=metadata_policy(),
            expected=CompatibilityExpected(evidence_ref="L0:T1"),
        )

    with pytest.raises(ValueError, match="cannot carry"):
        CompatibilityGate(
            checkpoint_root="somewhere",
            checkpoint_sha256="sha256:" + "0" * 64,
            task=termination_gate().task,
            evidence=termination_gate().evidence,
            expected=termination_gate().expected,
        )


def test_evaluator_scores_gate_value_and_identifier_interference() -> None:
    accepted = evaluate_compatibility_answer(
        termination_gate(), TerminationReturn(ok=True), evidence_queries=0
    )
    gate = CompatibilityGate(
        task=CompatibilityTask(
            gate_id="g2a",
            gate_kind="identifier_binding",
            instruction="Bind the record ref.",
            inputs={
                "record": {"ref": "L0:T1"},
                "frontier": "L1:T2",
                "next_after": "L0:T1",
            },
        ),
        evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
        expected=CompatibilityExpected(
            evidence_ref="L0:T1",
            frontier_substitution="L1:T2",
            turn_number_leakage="L0:T2",
        ),
    )
    leakage = evaluate_compatibility_answer(
        gate, IdentifierBindingReturn(evidence_ref="L0:T2"), evidence_queries=0
    )

    assert accepted.passed is True
    assert leakage.passed is False
    assert leakage.identifier_classification == "turn_number_leakage"


def test_identifier_binding_requires_exact_immutable_interference_input() -> None:
    with pytest.raises(ValueError, match="record, frontier, and next_after"):
        CompatibilityGate(
            task=CompatibilityTask(
                gate_id="g2a",
                gate_kind="identifier_binding",
                instruction="Bind the record ref.",
                inputs={"record": {"ref": "L0:T1"}},
            ),
            evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
            expected=CompatibilityExpected(
                evidence_ref="L0:T1",
                frontier_substitution="L1:T2",
                turn_number_leakage="L0:T2",
            ),
        )


def test_representation_and_composition_are_independently_scored() -> None:
    representation = CompatibilityGate(
        task=CompatibilityTask(
            gate_id="g2b", gate_kind="representation_rle", instruction="Decode."
        ),
        evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
        expected=CompatibilityExpected(decoded_value=5),
    )
    composition = CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(
            gate_id="g3", gate_kind="composition", instruction="Compose."
        ),
        evidence=metadata_policy().model_copy(
            update={"projection": "full", "grid_encoding": "rle"}
        ),
        expected=CompatibilityExpected(conclusion="11->5", evidence_ref="L0:T1"),
    )

    assert evaluate_compatibility_answer(
        representation, RepresentationReturn(decoded_value=5), evidence_queries=0
    ).passed
    wrong_ref = evaluate_compatibility_answer(
        composition,
        CompositionReturn(conclusion="11->5", evidence_ref="L0:T2"),
        evidence_queries=1,
    )
    assert wrong_ref.value_correct is False
    assert wrong_ref.capability_use_correct is True


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
            "ref": "L0:T1",
            "action": "LEFT",
            "row": None,
            "col": None,
        }
    ]
    assert subject.queries == 1
    with pytest.raises(ValueError, match="cap exceeded"):
        subject.query(metadata_policy().query_value())


def test_metadata_normalization_erases_grid_encoding(tmp_path: Path) -> None:
    policy = metadata_policy().model_copy(
        update={"max_queries": 3, "max_query_attempts": 3}
    )
    workspace = GameWorkspace(
        "compatibility", root=str(tmp_path), render=False, available_actions=["LEFT"]
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    subject = CompatibilityEvidenceCapability(
        WorkspaceEvidenceCapability(workspace), policy
    )
    base = {"kind": "transitions", "level": 0, "limit": 4, "projection": "metadata"}

    pages = [
        subject.query(base),
        subject.query({**base, "grid_encoding": "raw"}),
        subject.query({**base, "grid_encoding": "rle"}),
    ]

    assert pages[0] == pages[1] == pages[2]
    assert len({audit.semantic_sha256 for audit in subject.audits}) == 1
    assert all(audit.semantic_query.grid_encoding is None for audit in subject.audits)


def test_semantic_cache_uses_one_entry_for_metadata_spellings() -> None:
    class CountingEvidence:
        def __init__(self) -> None:
            self.calls = 0

        def query(self, query):
            self.calls += 1
            return {"frontier": {"level": 0, "turn": 0}, "query": query}

    evidence = CountingEvidence()
    policy = metadata_policy().model_copy(
        update={"max_queries": 3, "max_query_attempts": 3}
    )
    subject = CompatibilityEvidenceCapability(evidence, policy)
    base = {"kind": "transitions", "level": 0, "limit": 4, "projection": "metadata"}

    subject.query(base)
    subject.query({**base, "grid_encoding": "raw"})
    subject.query({**base, "grid_encoding": "rle"})

    assert evidence.calls == 1


def test_rejected_attempt_can_be_corrected_without_another_disclosure(
    tmp_path: Path,
) -> None:
    workspace = GameWorkspace(
        "compatibility", root=str(tmp_path), render=False, available_actions=["LEFT"]
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    subject = CompatibilityEvidenceCapability(
        WorkspaceEvidenceCapability(workspace), metadata_policy()
    )

    with pytest.raises(ValueError, match="requires semantic"):
        subject.query({"kind": "transitions", "projection": "metadata"})

    page = subject.query(metadata_policy().query_value())

    assert page["projection"] == "metadata"
    assert subject.query_attempts == 2
    assert subject.successful_disclosures == 1
    assert subject.audits[0].semantic_query is None
    assert subject.audits[1].disclosed is True


def test_second_successful_disclosure_is_blocked(tmp_path: Path) -> None:
    workspace = GameWorkspace(
        "compatibility", root=str(tmp_path), render=False, available_actions=["LEFT"]
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    subject = CompatibilityEvidenceCapability(
        WorkspaceEvidenceCapability(workspace), metadata_policy()
    )
    query = metadata_policy().query_value()

    subject.query(query)
    with pytest.raises(ValueError, match="disclosure cap exceeded"):
        subject.query(query)

    assert subject.query_attempts == 2
    assert subject.successful_disclosures == 1
    assert subject.audits[-1].disclosed is False


def test_full_projection_keeps_grid_encoding_semantic() -> None:
    policy = metadata_policy().model_copy(
        update={"projection": "full", "grid_encoding": "raw"}
    )

    with pytest.raises(ValueError, match="requires semantic"):
        canonicalize_evidence_query(
            {
                "kind": "transitions",
                "level": 0,
                "limit": 4,
                "projection": "full",
                "grid_encoding": "rle",
            },
            policy,
        )


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
            gate_id="g1", gate_kind="capability_retrieval", instruction="Query once."
        ),
        evidence=metadata_policy(),
        expected=CompatibilityExpected(evidence_ref="L0:T1"),
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
            gate_id="g1", gate_kind="capability_retrieval", instruction="Query once."
        ),
        evidence=metadata_policy(),
        expected=CompatibilityExpected(evidence_ref="L0:T1"),
    )
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate.model_dump(mode="json")))

    with pytest.raises(ValueError, match="escapes"):
        load_compatibility_gate(path, repository_root=tmp_path)
