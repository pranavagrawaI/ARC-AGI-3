from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tycho_nooa.cognitive_parity import (
    ActionProposal,
    CognitiveParityPacket,
    CognitiveTask,
    ExpectedAnswer,
    HypothesisAnswer,
    InvariantAnswer,
    evaluate_answer,
    load_packet,
    tree_sha256,
    write_packet,
)
from tycho_nooa.evidence import EvidenceRef
from tycho_nooa.models import ActionDecision


def packet(
    checkpoint: Path,
    *,
    task_kind: str = "discriminate",
    expected: ExpectedAnswer | None = None,
) -> CognitiveParityPacket:
    task = CognitiveTask(
        packet_id="old-transition-falsifier",
        checkpoint_sha256=f"sha256:{tree_sha256(checkpoint)}",
        evidence_frontier=EvidenceRef(level=0, turn=18),
        task_kind=task_kind,
        instruction="Choose the hypothesis contradicted by recorded evidence.",
        inputs={"hypotheses": ["H1", "H2"]},
    )
    return CognitiveParityPacket(
        checkpoint_root="checkpoint",
        task=task,
        expected=expected
        or ExpectedAnswer(
            required_evidence_refs=(EvidenceRef(level=0, turn=4),),
            accepted_verdicts=("H2",),
        ),
    )


def test_treatment_payload_never_contains_the_answer_key(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "turn.json").write_text("observed")
    benchmark = packet(checkpoint)

    visible = json.dumps(benchmark.treatment_payload())

    assert "expected" not in visible
    assert "required_evidence_refs" not in visible
    assert "accepted_verdicts" not in visible


def test_packet_reload_fails_if_checkpoint_bytes_change(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    turn = checkpoint / "turn.json"
    turn.write_text("observed")
    benchmark = packet(checkpoint)
    packet_path = tmp_path / "packet.json"
    write_packet(packet_path, benchmark)

    assert load_packet(packet_path, repository_root=tmp_path) == benchmark
    turn.write_text("mutated")

    with pytest.raises(ValueError, match="checkpoint commitment mismatch"):
        load_packet(packet_path, repository_root=tmp_path)


def test_discrimination_separates_retrieval_from_verdict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    benchmark = packet(checkpoint)
    answer = HypothesisAnswer(
        supported="H1",
        falsified=("H2",),
        evidence_refs=(EvidenceRef(level=0, turn=4),),
        conclusion="The right transition was found, but interpreted backward.",
    )

    result = evaluate_answer(benchmark, answer)

    assert result.evidence_precision == 1
    assert result.evidence_recall == 1
    assert result.verdict_correct is False


def test_invariant_scores_abstraction_and_scope_separately(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    benchmark = packet(
        checkpoint,
        task_kind="recover_invariant",
        expected=ExpectedAnswer(
            required_evidence_refs=(EvidenceRef(level=0, turn=3),),
            accepted_invariant_ids=("movement-step-3",),
            accepted_scopes=("trajectory",),
        ),
    )
    answer = InvariantAnswer(
        invariant_id="movement-step-3",
        invariant="Movement advances by three cells.",
        evidence_refs=(EvidenceRef(level=0, turn=3),),
        scope="level",
    )

    result = evaluate_answer(benchmark, answer)

    assert result.invariant_correct is True
    assert result.scope_correct is False


def test_action_disagreement_is_parity_not_accuracy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    benchmark = packet(
        checkpoint,
        task_kind="propose_action",
        expected=ExpectedAnswer(
            required_evidence_refs=(EvidenceRef(level=0, turn=18),),
            native_action=ActionDecision(action="UP"),
        ),
    )
    answer = ActionProposal(
        decision=ActionDecision(action="RIGHT"),
        evidence_refs=(EvidenceRef(level=0, turn=18),),
        expected_effect="Test horizontal movement.",
        information_goal="Distinguish H1 from H2.",
    )

    result = evaluate_answer(benchmark, answer)

    assert result.action_legal is True
    assert result.native_action_agreement is False
    assert result.decision_quality_scored is False


def test_wrong_answer_kind_and_illegal_action_fail_before_scoring(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    benchmark = packet(checkpoint)

    with pytest.raises(ValueError, match="task_kind does not match"):
        evaluate_answer(
            benchmark,
            InvariantAnswer(
                invariant_id="x",
                invariant="x",
                evidence_refs=(),
                scope="level",
            ),
        )
    with pytest.raises(ValidationError, match="CLICK requires row and col"):
        ActionProposal.model_validate(
            {
                "decision": {"action": "CLICK"},
                "evidence_refs": [],
                "expected_effect": "inspect",
            }
        )
