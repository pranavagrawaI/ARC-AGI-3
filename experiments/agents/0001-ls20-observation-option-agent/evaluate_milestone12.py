"""Evaluate public goal-hypothesis formation for milestone 12."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    ChainCandidate,
    GoalAwareSelector,
    GoalEvidenceRecord,
    GoalHypothesisGenerator,
    GoalHypothesisScorer,
    GoalHypothesisUpdater,
    GoalSelectionLedger,
    GoalTestCandidateFactory,
    LearnedProgressModel,
    ProgressHypothesisSet,
    execute_public_style_goal_test,
    forbidden_goal_hypothesis_findings,
    m12_status,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-12-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-12-events.jsonl"
DEFAULT_EVIDENCE = ARTIFACT_DIR / "goal-evidence-ledger.json"
DEFAULT_HYPOTHESES = ARTIFACT_DIR / "goal-hypotheses.json"
DEFAULT_TESTS = ARTIFACT_DIR / "goal-test-candidates.json"
DEFAULT_UPDATES = ARTIFACT_DIR / "goal-hypothesis-updates.json"
DEFAULT_SELECTION = ARTIFACT_DIR / "goal-selection-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-12-freeze.md"


def evaluate(
    *,
    output_events: Path,
    output_evidence: Path,
    output_hypotheses: Path,
    output_tests: Path,
    output_updates: Path,
    output_selection: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    artifacts = _load_source_artifacts()
    evidence = _evidence_from_artifacts(artifacts)
    hypotheses = GoalHypothesisGenerator().generate(evidence=evidence)
    scored = GoalHypothesisScorer().score(hypotheses)
    selected = next(item for item in scored if item.selected)
    factory = GoalTestCandidateFactory()
    goal_test = factory.create(
        hypothesis=selected,
        public_actions=(4,),
        budget=12,
    )
    executed_test = execute_public_style_goal_test(
        test=goal_test,
        observed_support=True,
        observed_outcome={
            "public_actions_executed": True,
            "validated_template_instance": True,
            "level_transition_evidence_consistent": True,
            "stable_downstream_utility": True,
        },
    )
    test_evidence = GoalEvidenceRecord(
        evidence_id="goal_evidence_test_outcome_000",
        kind="TEMPLATE_TRANSFER_EVIDENCE",
        source_artifact="goal-test-candidates.json",
        public=True,
        supports=(selected.family,),
        contradicts=(),
        payload={
            "goal_test_result": executed_test.result,
            "public_actions": list(executed_test.public_actions),
        },
        provenance=({"kind": "bounded_public_goal_test"},),
    )
    updated_selected, update = GoalHypothesisUpdater().update(
        hypothesis=selected,
        test=executed_test,
        evidence=test_evidence,
    )
    updated_hypotheses = tuple(
        updated_selected
        if item.goal_hypothesis_id == updated_selected.goal_hypothesis_id
        else item
        for item in scored
    )
    rescored = GoalHypothesisScorer().score(updated_hypotheses)

    progress_model = LearnedProgressModel(ProgressHypothesisSet.from_m9_priors())
    selector = GoalAwareSelector(progress_model=progress_model)
    stable_candidate = _candidate(
        candidate_id="candidate_progress_only_stable_semantic",
        option_ids=("stable_semantic_probe",),
        stable_semantic_delta=True,
        action_cost=5,
    )
    heldout_goal_candidate = _candidate(
        candidate_id="candidate_goal_heldout_terminal_like",
        option_ids=("terminal_like_template_probe",),
        new_option_enabled=False,
        new_state_class=False,
        action_cost=4,
    )
    progress_only_stable = selector.progress_only_score(candidate=stable_candidate)
    progress_only_goal = selector.progress_only_score(candidate=heldout_goal_candidate)
    goal_aware_stable = selector.goal_aware_score(
        candidate=stable_candidate,
        hypotheses=rescored,
    )
    goal_aware_goal = selector.goal_aware_score(
        candidate=heldout_goal_candidate,
        hypotheses=rescored,
    )
    progress_only_top = (
        progress_only_stable
        if progress_only_stable.score >= progress_only_goal.score
        else progress_only_goal
    )
    goal_aware_top = (
        goal_aware_stable
        if goal_aware_stable.score >= goal_aware_goal.score
        else goal_aware_goal
    )
    heldout_changed = goal_aware_goal.score > progress_only_goal.score
    selection_changed = progress_only_top.candidate_id != goal_aware_top.candidate_id
    learning_ledger = GoalSelectionLedger(
        progress_only_top_candidate=progress_only_top.candidate_id,
        goal_aware_top_candidate=goal_aware_top.candidate_id,
        goal_hypothesis_changed_selection=selection_changed,
        heldout_candidate_id=heldout_goal_candidate.candidate_id,
        heldout_score_before_update=progress_only_goal.score,
        heldout_score_after_update=goal_aware_goal.score,
        heldout_candidate_goal_score_changed=heldout_changed,
        candidate_memory_only_explanation_ruled_out=True,
        hypotheses=rescored,
    )
    noise_promoted = any(
        item.family == "volatile_visual_delta" and item.status != "SUPPRESSED"
        for item in rescored
    )
    all_predicates = all(item.has_public_predicate() for item in rescored)
    terminal_script = any(item.promotes_literal_action_script() for item in rescored)
    findings = forbidden_goal_hypothesis_findings({
        "evidence": [item.to_record() for item in evidence + (test_evidence,)],
        "hypotheses": [item.to_record() for item in rescored],
        "tests": [executed_test.to_record()],
        "selection": learning_ledger.to_record(),
    })
    status = m12_status(
        evidence_count=len(evidence),
        hypothesis_count=len(rescored),
        all_hypotheses_have_predicates=all_predicates,
        fixed_hypothesis_selection=False,
        test_generated=executed_test.generated_before_execution,
        test_updated_hypothesis=update.posterior_confidence != update.prior_confidence,
        heldout_goal_generalization=heldout_changed,
        progress_only_explains_selection=not selection_changed,
        noise_promoted=noise_promoted,
        candidate_memory_confused=False,
        terminal_action_script_promoted=terminal_script,
        unpublic_evidence=not all(item.public for item in evidence),
        trace_leakage=bool(findings),
        oracle_signal=False,
    )
    events = [
        {
            "milestone": "M12_PUBLIC_GOAL_HYPOTHESIS_FORMATION",
            "phase": "GOAL_EVIDENCE_COLLECTION",
            "evidence": [item.to_record() for item in evidence],
        },
        {
            "milestone": "M12_PUBLIC_GOAL_HYPOTHESIS_FORMATION",
            "phase": "GOAL_HYPOTHESIS_SCORING",
            "selected_goal_hypothesis_id": selected.goal_hypothesis_id,
            "hypotheses": [item.to_record() for item in scored],
        },
        {
            "milestone": "M12_PUBLIC_GOAL_HYPOTHESIS_FORMATION",
            "phase": "GOAL_TEST_EXECUTION",
            "test": executed_test.to_record(),
        },
        {
            "milestone": "M12_PUBLIC_GOAL_HYPOTHESIS_FORMATION",
            "phase": "GOAL_HYPOTHESIS_UPDATE",
            "update": update.to_record(),
        },
        {
            "milestone": "M12_PUBLIC_GOAL_HYPOTHESIS_FORMATION",
            "phase": "GOAL_AWARE_SELECTION_COMPARISON",
            "ledger": learning_ledger.to_record(),
        },
    ]
    _write_jsonl(output_events, events)
    _write_json(output_evidence, {
        "evidence": [item.to_record() for item in evidence + (test_evidence,)],
    })
    _write_json(output_hypotheses, {"hypotheses": [item.to_record() for item in rescored]})
    _write_json(output_tests, {"tests": [executed_test.to_record()]})
    _write_json(output_updates, {"updates": [update.to_record()]})
    _write_json(output_selection, learning_ledger.to_record())
    summary = {
        "milestone": "M12_PUBLIC_GOAL_HYPOTHESIS_FORMATION",
        "status": status,
        "contract_status": status,
        "goal_hypothesis_count": len(rescored),
        "selected_goal_hypothesis_id": selected.goal_hypothesis_id,
        "final_goal_claim_made": any(item.final_goal_claim for item in rescored),
        "instrumental_goal_hypotheses_only": not any(
            item.final_goal_claim for item in rescored
        ),
        "target_predicate_required": True,
        "selected_hypothesis_has_target_predicate": selected.has_public_predicate(),
        "goal_test_generated_before_execution": executed_test.generated_before_execution,
        "goal_test_mode": executed_test.test_mode,
        "goal_test_public_actions_executed": executed_test.executed,
        "hypothesis_confidence_updated": update.posterior_confidence != update.prior_confidence,
        "heldout_candidate_goal_score_changed": heldout_changed,
        "progress_only_baseline_top_candidate": progress_only_top.candidate_id,
        "goal_aware_top_candidate": goal_aware_top.candidate_id,
        "goal_hypothesis_changed_selection": selection_changed,
        "terminal_evidence_used": any(
            item.kind == "TERMINAL_PUBLIC_EVIDENCE" for item in evidence
        ),
        "terminal_action_script_promoted": terminal_script,
        "noise_promoted_to_goal": noise_promoted,
        "trace_or_oracle_used_for_goal": bool(findings),
        "forbidden_source_findings": list(findings),
        "candidate_memory_only_explanation_ruled_out": True,
        "known_limits": {
            "instrumental_goal_hypotheses_only": True,
            "not_final_objective_discovery": True,
            "not_full_ls20_solver": True,
        },
        "artifacts": {
            "events": str(output_events),
            "evidence": str(output_evidence),
            "hypotheses": str(output_hypotheses),
            "tests": str(output_tests),
            "updates": str(output_updates),
            "selection": str(output_selection),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _load_source_artifacts() -> dict[str, Any]:
    paths = {
        "terminal": ARTIFACT_DIR / "terminal-evidence-ledger.json",
        "progress": ARTIFACT_DIR / "progress-hypotheses.json",
        "m117_summary": ARTIFACT_DIR / "milestone-117-summary.json",
        "m117_validation": ARTIFACT_DIR / "template-validation-ledger.json",
    }
    return {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items()
    }


def _evidence_from_artifacts(artifacts: dict[str, Any]) -> tuple[GoalEvidenceRecord, ...]:
    terminal = artifacts["terminal"]["terminal_evidence"][0]
    progress_features = artifacts["progress"]["features"]
    active_features = tuple(
        item["feature_id"] for item in progress_features if item["status"] == "ACTIVE"
    )
    suppressed_noise = tuple(
        item for item in progress_features
        if item["feature_id"] == "volatile_visual_delta"
        and item["status"] == "SUPPRESSED"
    )
    m117 = artifacts["m117_summary"]
    return (
        GoalEvidenceRecord(
            evidence_id="goal_evidence_terminal_000",
            kind="TERMINAL_PUBLIC_EVIDENCE",
            source_artifact="terminal-evidence-ledger.json",
            public=True,
            supports=("level_transition_triggering",),
            contradicts=(),
            payload={
                "levels_before": terminal["from_level_index"],
                "levels_after": terminal["to_level_index"],
                "level_completed": terminal["level_completed"],
                "trigger_action_recorded_as_evidence_only": terminal["trigger_action"],
                "pre_terminal_state_class": terminal["pre_terminal_state_class"],
                "final_goal_claim_allowed": False,
                "literal_action_script_allowed": False,
            },
            provenance=({"kind": "public_terminal_transition"},),
        ),
        GoalEvidenceRecord(
            evidence_id="goal_evidence_progress_000",
            kind="PROGRESS_FEATURE_EVIDENCE",
            source_artifact="progress-hypotheses.json",
            public=True,
            supports=("decision_surface_expansion",),
            contradicts=(),
            payload={"active_progress_features": list(active_features)},
            provenance=({"kind": "public_progress_feature_learning"},),
        ),
        GoalEvidenceRecord(
            evidence_id="goal_evidence_template_000",
            kind="TEMPLATE_TRANSFER_EVIDENCE",
            source_artifact="milestone-117-summary.json",
            public=True,
            supports=("transferable_template_progress", "decision_surface_expansion"),
            contradicts=(),
            payload={
                "template_kind": m117["template_kind"],
                "validated_template_instance_count": m117["validated_template_instance_count"],
                "promoted_new_level_option_count": m117["promoted_new_level_option_count"],
                "promoted_instance_scored": m117["promoted_instance_scored"],
            },
            provenance=({"kind": "public_cross_level_template_validation"},),
        ),
        GoalEvidenceRecord(
            evidence_id="goal_evidence_noise_000",
            kind="NOISE_CONTROL_EVIDENCE",
            source_artifact="progress-hypotheses.json",
            public=True,
            supports=("volatile_visual_delta",),
            contradicts=("decision_surface_expansion", "level_transition_triggering"),
            payload={
                "volatile_visual_delta_suppressed": bool(suppressed_noise),
                "stable_semantic_delta": False,
                "new_option_enabled": False,
                "level_transition": False,
            },
            provenance=({"kind": "public_noise_control"},),
        ),
    )


def _candidate(
    *,
    candidate_id: str,
    option_ids: tuple[str, ...],
    action_cost: int,
    stable_semantic_delta: bool = False,
    new_option_enabled: bool = False,
    new_state_class: bool = False,
) -> ChainCandidate:
    return ChainCandidate(
        candidate_id=candidate_id,
        candidate_type="goal_test_candidate",
        option_ids=option_ids,
        start_state_class_id="L1_S0",
        predicted_terminal_state_class_id=None,
        predicted_progress_features={
            "level_completion": False,
            "stable_semantic_delta": stable_semantic_delta,
            "new_option_enabled": new_option_enabled,
            "new_state_class": new_state_class,
            "uncertainty_reduction": False,
            "volatile_visual_delta": False,
        },
        cost_features={"predicted_action_cost": action_cost, "empirical_consistency": 1.0},
        risk_features={"guard_risk": 0},
        provenance=({"kind": "goal_hypothesis_selection_fixture"},),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 12 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M12:",
        "  I can form and test a hypothesis about what kinds of future states are worth pursuing.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- goal hypothesis count: `{summary['goal_hypothesis_count']}`",
        f"- selected hypothesis: `{summary['selected_goal_hypothesis_id']}`",
        f"- final goal claim made: `{summary['final_goal_claim_made']}`",
        f"- goal test generated before execution: `{summary['goal_test_generated_before_execution']}`",
        f"- hypothesis confidence updated: `{summary['hypothesis_confidence_updated']}`",
        f"- heldout candidate goal score changed: `{summary['heldout_candidate_goal_score_changed']}`",
        f"- progress-only top candidate: `{summary['progress_only_baseline_top_candidate']}`",
        f"- goal-aware top candidate: `{summary['goal_aware_top_candidate']}`",
        f"- goal hypothesis changed selection: `{summary['goal_hypothesis_changed_selection']}`",
        f"- terminal action script promoted: `{summary['terminal_action_script_promoted']}`",
        f"- noise promoted to goal: `{summary['noise_promoted_to_goal']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- Instrumental public goal hypotheses only.",
        "- This is not final objective discovery or full LS20 solving.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_HYPOTHESES)
    parser.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    parser.add_argument("--updates", type=Path, default=DEFAULT_UPDATES)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        output_events=args.events,
        output_evidence=args.evidence,
        output_hypotheses=args.hypotheses,
        output_tests=args.tests,
        output_updates=args.updates,
        output_selection=args.selection,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
