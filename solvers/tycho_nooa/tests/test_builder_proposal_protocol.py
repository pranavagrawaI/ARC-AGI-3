from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tycho_nooa.builder_proposal_protocol import (
    BlobReference,
    BuilderReport,
    CandidateSubmission,
    ProposalLedger,
    ProposalValidationError,
    canonical_json,
    canonicalize_source,
)
from tycho_nooa.theory_evaluation import (
    EvaluationRunStatus,
    EvidenceFrontier,
    ObligationKind,
    ObligationResult,
    ObligationStatus,
    PublicationDisposition,
    TheoryEvaluation,
)
from tycho_nooa.theory_transaction import TheoryTransaction


class FixtureEvaluator:
    evaluator_sha256 = "fixture-evaluator"

    def __init__(self, results: dict[bytes, tuple[ObligationStatus, ...]]) -> None:
        self.results = results

    def evaluate(self, model: bytes, frontier: EvidenceFrontier) -> TheoryEvaluation:
        statuses = self.results[model]
        return TheoryEvaluation(
            model_sha256=hashlib.sha256(model).hexdigest(),
            evidence_frontier_sha256=frontier.sha256,
            evaluator_sha256=self.evaluator_sha256,
            run_status=EvaluationRunStatus.COMPLETE,
            obligations=tuple(
                ObligationResult(
                    obligation_id=f"T{index}",
                    kind=ObligationKind.TRANSITION,
                    status=status,
                    description=f"transition {index}",
                    diagnostics=(
                        ("counterexample", {"path": [index, "edge"]}),
                    ),
                )
                for index, status in enumerate(statuses, 1)
            ),
            error="observation-only evaluator note",
        )


class AlternatingEvaluator(FixtureEvaluator):
    def __init__(self) -> None:
        super().__init__({})
        self.calls = 0

    def evaluate(self, model: bytes, frontier: EvidenceFrontier) -> TheoryEvaluation:
        self.calls += 1
        status = (
            ObligationStatus.PASSED
            if self.calls % 2 == 0
            else ObligationStatus.FAILED
        )
        return TheoryEvaluation(
            model_sha256=hashlib.sha256(model).hexdigest(),
            evidence_frontier_sha256=frontier.sha256,
            evaluator_sha256=self.evaluator_sha256,
            run_status=EvaluationRunStatus.COMPLETE,
            obligations=(
                ObligationResult(
                    obligation_id="T1",
                    kind=ObligationKind.TRANSITION,
                    status=status,
                    description="transition 1",
                ),
            ),
        )


def test_canonical_json_is_stable_and_rejects_ambiguous_values() -> None:
    assert canonical_json({"z": 1, "a": [True, None]}) == (
        b'{"a":[true,null],"z":1}'
    )
    with pytest.raises(TypeError, match="floating-point"):
        canonical_json({"cost": 0.1})
    with pytest.raises(ValueError, match="NFC"):
        canonical_json({"name": "e\N{COMBINING ACUTE ACCENT}"})


def test_source_canonicalization_is_byte_exact_except_transport_variance() -> None:
    assert canonicalize_source(b"print('x')\r\n") == b"print('x')\n"
    assert canonicalize_source(b"print('x')") == b"print('x')\n"
    assert canonicalize_source(b"# keep trailing spaces  \n\n") == (
        b"# keep trailing spaces  \n\n"
    )
    with pytest.raises(ValueError, match="byte-order mark"):
        canonicalize_source(b"\xef\xbb\xbfprint('x')")


def test_frozen_case_commits_all_builder_visible_inputs(tmp_path: Path) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {b"root": (ObligationStatus.FAILED,)},
    )
    first = transaction.freeze_build_case(metadata={"focus": "repair"})
    second = transaction.freeze_build_case(metadata={"focus": "explain"})

    assert first.case_sha256 != second.case_sha256
    assert isinstance(first.incumbent_source, bytes)
    assert isinstance(first.evidence, tuple)
    assert first.incumbent_source == b"root\n"
    assert not hasattr(first, "retained_candidates")
    assert first.evaluator_sha256 == evaluator.evaluator_sha256

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    passed_root = tmp_path / "passed"
    passed_root.mkdir()
    failed, _ = begin_transaction(
        failed_root,
        b"root",
        {b"root": (ObligationStatus.FAILED,)},
    )
    passed, _ = begin_transaction(
        passed_root,
        b"root",
        {b"root": (ObligationStatus.PASSED,)},
    )
    assert failed.freeze_build_case().case_sha256 != (
        passed.freeze_build_case().case_sha256
    )


def test_frozen_json_keeps_arrays_and_objects_unambiguous(tmp_path: Path) -> None:
    transaction, _ = begin_transaction(
        tmp_path,
        b"root",
        {b"root": (ObligationStatus.FAILED,)},
    )
    case = transaction.freeze_build_case(
        metadata={"empty_array": [], "empty_object": {}, "pairs": [["a", 1]]}
    )

    metadata = case.builder_view()["metadata"]
    assert isinstance(metadata, dict)
    assert {key: metadata[key] for key in ("empty_array", "empty_object", "pairs")} == {
        "empty_array": [],
        "empty_object": {},
        "pairs": [["a", 1]],
    }


def test_submission_is_bound_to_the_exact_parent_preimage(tmp_path: Path) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED,),
            b"candidate\n": (ObligationStatus.PASSED,),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    stale = CandidateSubmission.replace(
        case=case,
        parent_source=b"xxxx\n",
        candidate_source=b"candidate\n",
        declared_parent_source_sha256=case.incumbent_source_sha256,
        provenance={"reason": "stale local copy"},
    )

    with pytest.raises(ProposalValidationError, match="parent preimage"):
        ledger.evaluate(stale)


def test_evaluate_is_pure_and_returns_an_immutable_receipt(tmp_path: Path) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED,),
            b"candidate\n": (ObligationStatus.PASSED,),
        },
    )
    workspace = transaction.workspace
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    submission = CandidateSubmission.replace(
        case=case,
        parent_source=case.incumbent_source,
        candidate_source=b"candidate",
        provenance={"reason": "repair T1"},
    )

    receipt = ledger.evaluate(submission)

    assert (workspace / "world_model.py").read_bytes() == b"root\n"
    assert receipt.semantic.evaluation.obligations[0].status is ObligationStatus.PASSED
    assert receipt.semantic.case_sha256 == case.case_sha256
    assert ledger.candidate_bytes(receipt.candidate_source_sha256) == b"candidate\n"
    with pytest.raises(AttributeError):
        receipt.semantic.case_sha256 = "tampered"  # type: ignore[misc]

    resumed = ProposalLedger.resume(case, evaluator, tmp_path / "proposal")
    assert resumed.receipts == ledger.receipts
    assert resumed.candidate_bytes(receipt.candidate_source_sha256) == b"candidate\n"


def test_repeated_semantic_mismatch_marks_determinism_fault(tmp_path: Path) -> None:
    transaction, _ = begin_transaction(
        tmp_path,
        b"root",
        {b"root": (ObligationStatus.FAILED,)},
    )
    evaluator = AlternatingEvaluator()
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    submission = CandidateSubmission.replace(
        case=case,
        parent_source=case.incumbent_source,
        candidate_source=b"candidate",
    )

    ledger.evaluate(submission)
    second = ledger.evaluate(submission)

    assert second.determinism_fault is True
    assert ledger.determinism_faults == (submission.submission_sha256,)


def test_repeated_deterministic_evaluation_keeps_one_semantic_receipt(
    tmp_path: Path,
) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED,),
            b"candidate": (ObligationStatus.PASSED,),
        },
    )
    case = transaction.freeze_build_case()
    store = tmp_path / "proposal"
    ledger = ProposalLedger(case, evaluator, store)
    submission = CandidateSubmission.replace(
        case=case,
        parent_source=case.incumbent_source,
        candidate_source=b"candidate",
    )

    first = ledger.evaluate(submission)
    second = ledger.evaluate(submission)

    assert first.semantic.semantic_result_sha256 == second.semantic.semantic_result_sha256
    assert ledger.determinism_faults == ()
    assert len(tuple((store / "receipts").glob("*.json"))) == 1
    assert len(tuple((store / "observations").glob("*.json"))) == 2


def test_lineage_cycle_is_rejected(tmp_path: Path) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED,),
            b"child": (ObligationStatus.PASSED,),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    child = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"child",
        )
    )
    cycle = CandidateSubmission.replace(
        case=case,
        parent_source=ledger.candidate_bytes(child.candidate_source_sha256),
        candidate_source=case.incumbent_source,
        declared_parent_source_sha256=child.candidate_source_sha256,
    )

    with pytest.raises(ProposalValidationError, match="lineage contains a cycle"):
        ledger.evaluate(cycle)


def test_finish_publishes_only_a_report_nominated_pareto_safe_candidate(
    tmp_path: Path,
) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED, ObligationStatus.PASSED),
            b"repair\n": (ObligationStatus.PASSED, ObligationStatus.PASSED),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    receipt = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"repair",
        )
    )
    report = BuilderReport.create(
        case=case,
        nominated_candidate_source_sha256=receipt.candidate_source_sha256,
        judgment="repairs T1 without losing T2",
    )

    decision = transaction.finish(report=report, receipts=ledger)

    assert decision.disposition is PublicationDisposition.PUBLISHABLE
    assert (transaction.workspace / "world_model.py").read_bytes() == b"repair\n"


def test_coupled_local_repair_cannot_hide_root_regression(tmp_path: Path) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (
                ObligationStatus.FAILED,
                ObligationStatus.PASSED,
                ObligationStatus.PASSED,
            ),
            b"local\n": (
                ObligationStatus.PASSED,
                ObligationStatus.FAILED,
                ObligationStatus.PASSED,
            ),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    receipt = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"local",
        )
    )
    report = BuilderReport.create(
        case=case,
        nominated_candidate_source_sha256=receipt.candidate_source_sha256,
        judgment="tempting local repair",
    )

    decision = transaction.finish(report=report, receipts=ledger)

    assert decision.disposition is PublicationDisposition.REJECTED_REGRESSION
    assert decision.lost == ("T2",)
    assert (transaction.workspace / "world_model.py").read_bytes() == b"root\n"


def test_equal_observed_replay_does_not_reward_unsupported_generality(
    tmp_path: Path,
) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED, ObligationStatus.PASSED),
            b"all-colors\n": (ObligationStatus.PASSED, ObligationStatus.PASSED),
            b"observed-colors\n": (
                ObligationStatus.PASSED,
                ObligationStatus.PASSED,
            ),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    broad = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"all-colors",
        )
    )
    bounded = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"observed-colors",
        )
    )
    report = BuilderReport.create(case=case, judgment="evidence cannot distinguish")

    assert broad.semantic.obligation_outcome_sha256 == (
        bounded.semantic.obligation_outcome_sha256
    )
    decision = transaction.finish(report=report, receipts=ledger)
    assert decision.disposition is PublicationDisposition.WITHHELD_AMBIGUOUS
    assert (transaction.workspace / "world_model.py").read_bytes() == b"root\n"


def test_explicit_support_scope_withholds_a_lone_broad_candidate(
    tmp_path: Path,
) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED, ObligationStatus.UNSUPPORTED),
            b"all-colors\n": (
                ObligationStatus.PASSED,
                ObligationStatus.UNSUPPORTED,
            ),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    broad = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"all-colors",
            provenance={"claim": "all colors move left"},
        )
    )
    report = BuilderReport.create(
        case=case,
        nominated_candidate_source_sha256=broad.candidate_source_sha256,
        judgment="broad rule replays observed colors but has unsupported scope",
    )

    decision = transaction.finish(report=report, receipts=ledger)

    assert decision.disposition is PublicationDisposition.WITHHELD_INCONCLUSIVE
    assert (transaction.workspace / "world_model.py").read_bytes() == b"root\n"


def test_finish_rejects_stale_context_without_overwriting_new_theory(
    tmp_path: Path,
) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED,),
            b"repair\n": (ObligationStatus.PASSED,),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    receipt = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"repair",
        )
    )
    report = BuilderReport.create(
        case=case,
        nominated_candidate_source_sha256=receipt.candidate_source_sha256,
        judgment="valid against old context",
    )
    (transaction.workspace / "world_model.py").write_bytes(b"new authority")

    decision = transaction.finish(report=report, receipts=ledger)

    assert decision.disposition is PublicationDisposition.WITHHELD_STALE_CONTEXT
    assert (transaction.workspace / "world_model.py").read_bytes() == b"new authority"


def test_internal_builder_differences_cannot_change_persistent_semantic_state(
    tmp_path: Path,
) -> None:
    outcomes: list[tuple[bytes, PublicationDisposition]] = []
    for name, scratch in (("first", b"arbitrary-a"), ("second", b"arbitrary-b")):
        root = tmp_path / name
        root.mkdir()
        transaction, evaluator = begin_transaction(
            root,
            b"root",
            {
                b"root": (ObligationStatus.FAILED,),
                b"repair\n": (ObligationStatus.PASSED,),
            },
        )
        case = transaction.freeze_build_case(metadata={"task": "same"})
        ledger = ProposalLedger(case, evaluator, root / "proposal")
        submission = CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"repair",
            provenance={"claim": "same"},
        )
        receipt = ledger.evaluate(submission)
        (root / "builder-scratch.bin").write_bytes(scratch)
        report = BuilderReport.create(
            case=case,
            nominated_candidate_source_sha256=receipt.candidate_source_sha256,
            judgment="same",
        )
        decision = transaction.finish(report=report, receipts=ledger)
        outcomes.append(
            ((transaction.workspace / "world_model.py").read_bytes(), decision.disposition)
        )

    assert outcomes == [
        (b"repair\n", PublicationDisposition.PUBLISHABLE),
        (b"repair\n", PublicationDisposition.PUBLISHABLE),
    ]


def test_recovery_selects_only_root_or_fully_committed_proposal(tmp_path: Path) -> None:
    transaction, evaluator = begin_transaction(
        tmp_path,
        b"root",
        {
            b"root": (ObligationStatus.FAILED,),
            b"repair": (ObligationStatus.PASSED,),
        },
    )
    case = transaction.freeze_build_case()
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    receipt = ledger.evaluate(
        CandidateSubmission.replace(
            case=case,
            parent_source=case.incumbent_source,
            candidate_source=b"repair",
        )
    )
    report = BuilderReport.create(
        case=case,
        nominated_candidate_source_sha256=receipt.candidate_source_sha256,
        judgment="publish",
    )
    active = (transaction.store_root / "active.json").read_bytes()
    transaction.finish(report=report, receipts=ledger)
    (transaction.store_root / "active.json").write_bytes(active)
    (transaction.workspace / "world_model.py").write_bytes(b"partial")

    recovered = TheoryTransaction.begin(
        workspace=transaction.workspace,
        evaluator=evaluator,
        frontier=transaction.frontier,
        cause="recover proposal publication",
        store_root=transaction.store_root,
    )

    assert (transaction.workspace / "world_model.py").read_bytes() == b"repair\n"
    recovered.abort("test cleanup")


def begin_transaction(
    tmp_path: Path,
    incumbent: bytes,
    results: dict[bytes, tuple[ObligationStatus, ...]],
) -> tuple[TheoryTransaction, FixtureEvaluator]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    canonical_incumbent = canonicalize_source(incumbent)
    (workspace / "world_model.py").write_bytes(canonical_incumbent)
    evidence = workspace / "level_0"
    evidence.mkdir()
    (evidence / "turn_000.txt").write_text("observation\n")
    evaluator = FixtureEvaluator(
        {canonicalize_source(source): statuses for source, statuses in results.items()}
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=EvidenceFrontier.freeze(workspace),
        cause="proposal protocol test",
        store_root=tmp_path / "transaction",
    )
    return transaction, evaluator


def test_blob_reference_rejects_tampered_content() -> None:
    reference = BlobReference.from_bytes(b"content", media_type="text/plain")
    with pytest.raises(ProposalValidationError, match="blob digest"):
        reference.verify(b"CONTENT")
