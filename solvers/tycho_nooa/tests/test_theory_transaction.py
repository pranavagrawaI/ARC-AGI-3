from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tycho.workspace.agent_tools import ToolExecutor
from tycho.workspace.sandbox import PythonSandbox
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.theory_evaluation import (
    EvaluationRunStatus,
    EvidenceFrontier,
    ObligationKind,
    ObligationResult,
    ObligationStatus,
    PublicationDisposition,
    SandboxTheoryEvaluator,
    TheoryEvaluation,
)
from tycho_nooa.theory_transaction import TheoryTransaction
from tycho_nooa.tools import BUILDER_TOOL_PROFILE, WorkspaceToolSurface
from tycho_nooa.wm_probe import ReadOnlyWorkspacePythonSandbox


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
                    description=f"transition T{index}",
                )
                for index, status in enumerate(statuses, start=1)
            ),
        )


def test_rejected_edit_restores_exact_parent_and_retains_candidate(
    tmp_path: Path,
) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"parent")
    evaluator = FixtureEvaluator(
        {
            b"parent": (ObligationStatus.PASSED, ObligationStatus.PASSED),
            b"regression": (ObligationStatus.FAILED, ObligationStatus.FAILED),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="test regression",
        store_root=tmp_path / "store",
    )

    result = transaction.apply_edit(
        "write_file",
        lambda: write_model(workspace, b"regression"),
    )

    assert (workspace / "world_model.py").read_bytes() == b"parent"
    assert result.decision.disposition is PublicationDisposition.REJECTED_REGRESSION
    assert result.record_id is not None
    assert transaction.candidate_bytes(result.record_id) == b"regression"
    assert "rejected_and_restored" in result.feedback
    assert "T1" in result.feedback


def test_accepted_edit_becomes_parent_for_later_regression(tmp_path: Path) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED, ObligationStatus.FAILED),
            b"one": (ObligationStatus.PASSED, ObligationStatus.FAILED),
            b"bad": (ObligationStatus.FAILED, ObligationStatus.PASSED),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="monotonic construction",
        store_root=tmp_path / "store",
    )

    accepted = transaction.apply_edit(
        "write_file", lambda: write_model(workspace, b"one")
    )
    rejected = transaction.apply_edit(
        "write_file", lambda: write_model(workspace, b"bad")
    )

    assert accepted.decision.disposition is PublicationDisposition.PUBLISHABLE
    assert rejected.decision.disposition is PublicationDisposition.REJECTED_REGRESSION
    assert (workspace / "world_model.py").read_bytes() == b"one"


def test_checkout_never_regresses_the_working_incumbent(tmp_path: Path) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED, ObligationStatus.FAILED),
            b"left": (ObligationStatus.PASSED, ObligationStatus.FAILED),
            b"right": (ObligationStatus.FAILED, ObligationStatus.PASSED),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="checkout comparison",
        store_root=tmp_path / "store",
    )
    transaction.apply_edit("write_file", lambda: write_model(workspace, b"left"))
    retained = transaction.apply_edit(
        "write_file", lambda: write_model(workspace, b"right")
    )

    checkout = transaction.checkout(retained.record_id or "")

    assert checkout.decision.disposition is PublicationDisposition.REJECTED_REGRESSION
    assert (workspace / "world_model.py").read_bytes() == b"left"


def test_checkout_copies_an_eligible_retained_body_into_the_workspace(
    tmp_path: Path,
) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    store = tmp_path / "store"
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    source = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="retain eligible theory",
        store_root=store,
    )
    retained = source.apply_edit("write_file", lambda: write_model(workspace, b"one"))
    source.abort("retain without publication")
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="explicit checkout",
        store_root=store,
    )

    checkout = transaction.checkout(retained.record_id or "")

    assert checkout.decision.disposition is PublicationDisposition.PUBLISHABLE
    assert (workspace / "world_model.py").read_bytes() == b"one"


def test_checkout_re_evaluates_a_retained_body_on_the_current_frontier(
    tmp_path: Path,
) -> None:
    workspace, old_frontier = workspace_with_model(tmp_path, b"zero")
    store = tmp_path / "store"
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    source = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=old_frontier,
        cause="older evidence",
        store_root=store,
    )
    retained = source.apply_edit("write_file", lambda: write_model(workspace, b"one"))
    source.abort("frontier advanced")
    (workspace / "level_0" / "turn_001.txt").write_text("1\n")
    current_frontier = EvidenceFrontier.freeze(workspace)
    assert current_frontier.sha256 != old_frontier.sha256
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=current_frontier,
        cause="current evidence checkout",
        store_root=store,
    )

    checkout = transaction.checkout(retained.record_id or "")

    assert checkout.decision.disposition is PublicationDisposition.PUBLISHABLE
    assert (workspace / "world_model.py").read_bytes() == b"one"
    new_record = json.loads(
        (store / "records" / f"{checkout.record_id}.json").read_text(encoding="utf-8")
    )
    assert new_record["source_record_id"] == retained.record_id
    assert new_record["evidence_frontier_sha256"] == current_frontier.sha256


def test_checkout_rejects_record_id_path_traversal(tmp_path: Path) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=FixtureEvaluator({b"zero": (ObligationStatus.FAILED,)}),
        frontier=frontier,
        cause="checkout containment",
        store_root=tmp_path / "store",
    )

    with pytest.raises(ValueError, match="invalid candidate record ID"):
        transaction.checkout("../../outside")


def test_checkout_rejects_a_tampered_candidate_record(tmp_path: Path) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    store = tmp_path / "store"
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="record commitment",
        store_root=store,
    )
    retained = transaction.apply_edit(
        "write_file", lambda: write_model(workspace, b"one")
    )
    record_path = store / "records" / f"{retained.record_id}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["operation"] = "tampered"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="record commitment mismatch"):
        transaction.checkout(retained.record_id or "")


def test_assessment_failure_restores_parent_bytes(tmp_path: Path) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")

    class RaisingEvaluator(FixtureEvaluator):
        def evaluate(
            self, model: bytes, frozen: EvidenceFrontier
        ) -> TheoryEvaluation:
            if model == b"bad":
                raise RuntimeError("judge crashed")
            return super().evaluate(model, frozen)

    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=RaisingEvaluator({b"zero": (ObligationStatus.PASSED,)}),
        frontier=frontier,
        cause="fail closed",
        store_root=tmp_path / "store",
    )

    with pytest.raises(RuntimeError, match="judge crashed"):
        transaction.apply_edit("write_file", lambda: write_model(workspace, b"bad"))

    assert (workspace / "world_model.py").read_bytes() == b"zero"


def test_candidate_blob_corruption_is_detected_not_overwritten(tmp_path: Path) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    store = tmp_path / "store"
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="immutable candidates",
        store_root=store,
    )
    transaction.apply_edit("write_file", lambda: write_model(workspace, b"one"))
    digest = hashlib.sha256(b"one").hexdigest()
    blob = store / "blobs" / f"{digest}.py"
    blob.write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="immutable theory path changed content"):
        transaction.apply_edit("write_file", lambda: write_model(workspace, b"one"))

    assert blob.read_bytes() == b"corrupt"
    assert (workspace / "world_model.py").read_bytes() == b"one"


def test_abort_restores_canonical_but_preserves_advisory_notes(tmp_path: Path) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="timeout",
        store_root=tmp_path / "store",
    )
    transaction.apply_edit("write_file", lambda: write_model(workspace, b"one"))
    notes = workspace / "notes" / "world_model.md"
    notes.parent.mkdir()
    notes.write_text("useful advisory finding")

    transaction.abort("deadline exceeded")

    assert (workspace / "world_model.py").read_bytes() == b"zero"
    assert notes.read_text() == "useful advisory finding"


def test_finish_publishes_working_incumbent_and_invalidates_stale_plan(
    tmp_path: Path,
) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    plan = workspace / "notes" / "validated_plan.json"
    plan.parent.mkdir()
    plan.write_text("stale")
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="publication",
        store_root=tmp_path / "store",
    )
    transaction.apply_edit("write_file", lambda: write_model(workspace, b"one"))

    decision = transaction.finish()

    assert decision.disposition is PublicationDisposition.PUBLISHABLE
    assert (workspace / "world_model.py").read_bytes() == b"one"
    assert not plan.exists()
    publication = json.loads(
        (tmp_path / "store" / "canonical.json").read_text(encoding="utf-8")
    )
    assert publication["parent_id"].startswith("theory-")
    assert publication["child_id"].startswith("theory-")
    assert publication["parent_model_sha256"] == hashlib.sha256(b"zero").hexdigest()
    assert publication["model_sha256"] == hashlib.sha256(b"one").hexdigest()
    assert publication["decision"]["gained"] == ["T1"]
    assert publication["parent_evaluation"]["obligations"][0]["status"] == "failed"
    assert publication["child_evaluation"]["obligations"][0]["status"] == "passed"
    assert publication["canonical_path"] == str(workspace / "world_model.py")


def test_begin_recovers_unfinished_transaction_to_its_canonical_bytes(
    tmp_path: Path,
) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    store = tmp_path / "store"
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    interrupted = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="interrupted construction",
        store_root=store,
    )
    interrupted.apply_edit("write_file", lambda: write_model(workspace, b"one"))
    assert (workspace / "world_model.py").read_bytes() == b"one"

    recovered = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="deadline recovery",
        store_root=store,
    )

    assert (workspace / "world_model.py").read_bytes() == b"zero"
    recovered.abort("test cleanup")


def test_begin_recovers_published_bytes_after_canonical_commit(
    tmp_path: Path,
) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    store = tmp_path / "store"
    evaluator = FixtureEvaluator(
        {
            b"zero": (ObligationStatus.FAILED,),
            b"one": (ObligationStatus.PASSED,),
        }
    )
    publishing = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="publication",
        store_root=store,
    )
    publishing.apply_edit("write_file", lambda: write_model(workspace, b"one"))
    active_record = (store / "active.json").read_bytes()
    publishing.finish()
    (store / "active.json").write_bytes(active_record)
    (workspace / "world_model.py").write_bytes(b"zero")

    recovered = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="post-commit recovery",
        store_root=store,
    )

    assert (workspace / "world_model.py").read_bytes() == b"one"
    recovered.abort("test cleanup")


def test_transactional_builder_denies_writes_outside_owned_artifacts(
    tmp_path: Path,
) -> None:
    workspace, frontier = workspace_with_model(tmp_path, b"zero")
    evaluator = FixtureEvaluator({b"zero": (ObligationStatus.FAILED,)})
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=frontier,
        cause="containment",
        store_root=tmp_path / "store",
    )
    game_workspace = GameWorkspace(
        "game", root=str(tmp_path / "surface"), available_actions=["RIGHT"]
    )
    surface = WorkspaceToolSurface(
        ToolExecutor(game_workspace),
        profile=BUILDER_TOOL_PROFILE,
        theory_transaction=transaction,
    )

    for protected in (
        "wmlib.py",
        "verify.py",
        "level_0/turn_000.txt",
        "candidates/world_model/index.json",
        "../store/ledger.jsonl",
    ):
        with pytest.raises(PermissionError, match="transactional Builder"):
            surface.write_file(protected, "tamper")


def test_read_only_builder_python_mount_cannot_write_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "probe.py"
    script.write_text("print('ok')")
    sandbox = object.__new__(ReadOnlyWorkspacePythonSandbox)
    sandbox.runtime = "docker"
    sandbox.image = "test-image"

    command = sandbox.command(workspace, script, name="test")

    mount = command[command.index("--mount") + 1]
    assert mount.endswith(",readonly")


def test_preserved_budget_awareness_8_of_16_to_0_of_16_is_restored(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    historical = (
        repo
        / "analysis/builder-continuation/20260822-matched-budget-awareness/arms"
        / "retained_candidate/workspace/ar25"
    )
    incumbent = (
        repo
        / "runs/tycho-nooa-3arm-gemma4-31b-ar25-approved7-20260821T145238Z"
        / "analysis/best-transient-world-model.py"
    ).read_bytes()
    regressed = (historical / "world_model.py").read_bytes()
    assert hashlib.sha256(incumbent).hexdigest() == (
        "4f78ef3b371d3e8f4d106710d19d03f94de6c60cfa399776818714fa1c5a86df"
    )
    assert hashlib.sha256(regressed).hexdigest() == (
        "f1c05db25fff661e028e29c4d255dc9c45e170c1e5c5388294e22fdef65fba75"
    )
    workspace = tmp_path / "workspace"
    shutil.copytree(historical, workspace)
    (workspace / "world_model.py").write_bytes(incumbent)
    evaluator = SandboxTheoryEvaluator(
        PythonSandbox(runtime="host"), allow_unsafe_host=True
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=EvidenceFrontier.freeze(workspace),
        cause="preserved 20260822 budget-awareness regression",
        store_root=tmp_path / "store",
    )

    result = transaction.apply_edit(
        "historical_broad_edit", lambda: write_model(workspace, regressed)
    )

    assert result.decision.disposition is PublicationDisposition.REJECTED_REGRESSION
    assert "parent:    8/16 exact" in result.feedback
    assert "candidate: 0/16 exact" in result.feedback
    assert (workspace / "world_model.py").read_bytes() == incumbent
    assert transaction.candidate_bytes(result.record_id or "") == regressed


def test_preserved_budget_awareness_0_of_16_to_8_of_16_is_publishable(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    historical = (
        repo
        / "analysis/builder-continuation/20260822-matched-budget-awareness/arms"
        / "retained_candidate/workspace/ar25"
    )
    improved = (
        repo
        / "runs/tycho-nooa-3arm-gemma4-31b-ar25-approved7-20260821T145238Z"
        / "analysis/best-transient-world-model.py"
    ).read_bytes()
    regressed = (historical / "world_model.py").read_bytes()
    workspace = tmp_path / "workspace"
    shutil.copytree(historical, workspace)
    evaluator = SandboxTheoryEvaluator(
        PythonSandbox(runtime="host"), allow_unsafe_host=True
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=EvidenceFrontier.freeze(workspace),
        cause="preserved 20260822 genuine improvement positive control",
        store_root=tmp_path / "store",
    )

    result = transaction.apply_edit(
        "historical_narrow_improvement", lambda: write_model(workspace, improved)
    )
    publication = transaction.finish()

    assert result.decision.disposition is PublicationDisposition.PUBLISHABLE
    assert publication.disposition is PublicationDisposition.PUBLISHABLE
    assert "parent:    0/16 exact" in result.feedback
    assert "candidate: 8/16 exact" in result.feedback
    assert (workspace / "world_model.py").read_bytes() == improved
    assert transaction.candidate_bytes(result.record_id or "") == improved
    assert regressed != improved


def workspace_with_model(tmp_path: Path, model: bytes) -> tuple[Path, EvidenceFrontier]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "world_model.py").write_bytes(model)
    level = workspace / "level_0"
    level.mkdir()
    (level / "turn_000.txt").write_text("0\n")
    return workspace, EvidenceFrontier.freeze(workspace)


def write_model(workspace: Path, content: bytes) -> str:
    (workspace / "world_model.py").write_bytes(content)
    return "wrote world_model.py"
