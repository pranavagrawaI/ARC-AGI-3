"""Evaluate M12.6.1 public motor-constraint learning."""

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
    ActionEffectConstraintModel,
    AttemptMemory,
    ChainCandidate,
    MotorAwareSelector,
    SelectionPolicy,
    action_effect_observations_from_events,
    forbidden_motor_constraint_findings,
    m1261_status,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_SOURCE_EVENTS = ARTIFACT_DIR / "milestone-117-events.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-1261-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-1261-events.jsonl"
DEFAULT_MOTOR_AUDIT = ARTIFACT_DIR / "motor-action-effect-audit.json"
DEFAULT_SELECTION = ARTIFACT_DIR / "motor-aware-selection-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-1261-freeze.md"


def evaluate(
    *,
    source_events: Path,
    output_events: Path,
    output_motor_audit: Path,
    output_selection: Path,
    output_freeze: Path,
    minimum_support: int,
    no_op_rate_threshold: float,
) -> dict[str, Any]:
    source_records = _read_jsonl(source_events)
    observations = action_effect_observations_from_events(source_records)
    motor_model = ActionEffectConstraintModel(
        minimum_support=minimum_support,
        no_op_rate_threshold=no_op_rate_threshold,
    )
    for observation in observations:
        motor_model.observe(observation)

    candidates = _diagnostic_candidates()
    action_plans = {
        "option_repeated_no_op_risk": (3,),
        "option_observed_motion": (4,),
    }
    baseline_selection = SelectionPolicy().select(
        round_index=0,
        state_class_id="public_context",
        candidates=candidates,
        attempt_memory=AttemptMemory(),
    )
    motor_selection = MotorAwareSelector(motor_model=motor_model).select(
        round_index=0,
        state_class_id="public_context",
        candidates=candidates,
        action_plans=action_plans,
        attempt_memory=AttemptMemory(),
    )
    constraints = motor_model.constraints()
    constrained = tuple(constraint for constraint in constraints if constraint.constrained)
    no_op_candidate = _candidate_by_id(
        motor_selection,
        "z_candidate_repeated_no_op_risk",
    )
    repeated_no_op_deprioritized = bool(
        no_op_candidate
        and not no_op_candidate.selected
        and no_op_candidate.score_terms
        and no_op_candidate.score_terms.get("motor_no_op_penalty", 0.0) < 0.0
    )
    audit_payload = {
        "model": motor_model.to_record(),
        "baseline_selection": baseline_selection.to_record(),
        "motor_selection": motor_selection.to_record(),
    }
    findings = forbidden_motor_constraint_findings(audit_payload)
    selection_changed = (
        baseline_selection.selected_candidate_id != motor_selection.selected_candidate_id
    )
    status = m1261_status(
        observation_count=len(observations),
        constrained_action_count=len(constrained),
        selection_changed=selection_changed,
        repeated_no_op_deprioritized=repeated_no_op_deprioritized,
        named_wall_detector=False,
        oracle_or_private_state_used=bool(findings),
    )
    events = [
        {
            "milestone": "M12_6_1_PUBLIC_MOTOR_CONSTRAINTS",
            "phase": "PUBLIC_ACTION_EFFECT_AUDIT",
            "source_events": str(source_events),
            "observation_count": len(observations),
            "no_op_count": sum(1 for observation in observations if observation.no_op),
            "moved_count": sum(1 for observation in observations if observation.moved),
            "level_delta_count": sum(
                1 for observation in observations if observation.level_delta != 0
            ),
            "constraints": [constraint.to_record() for constraint in constraints],
        },
        {
            "milestone": "M12_6_1_PUBLIC_MOTOR_CONSTRAINTS",
            "phase": "MOTOR_AWARE_SELECTION",
            "baseline_selection": baseline_selection.to_record(),
            "motor_selection": motor_selection.to_record(),
            "selection_changed": selection_changed,
            "repeated_no_op_deprioritized": repeated_no_op_deprioritized,
        },
    ]
    _write_jsonl(output_events, events)
    _write_json(output_motor_audit, audit_payload)
    _write_json(output_selection, {
        "baseline": baseline_selection.to_record(),
        "motor_aware": motor_selection.to_record(),
    })
    summary = {
        "milestone": "M12_6_1_PUBLIC_MOTOR_CONSTRAINTS",
        "status": status,
        "contract_status": status,
        "source_events": str(source_events),
        "public_action_effect_observation_count": len(observations),
        "public_no_op_count": sum(1 for observation in observations if observation.no_op),
        "public_moved_count": sum(1 for observation in observations if observation.moved),
        "public_level_delta_count": sum(
            1 for observation in observations if observation.level_delta != 0
        ),
        "minimum_support": minimum_support,
        "no_op_rate_threshold": no_op_rate_threshold,
        "constrained_action_count": len(constrained),
        "constraints": [constraint.to_record() for constraint in constrained],
        "baseline_selected_candidate_id": baseline_selection.selected_candidate_id,
        "motor_aware_selected_candidate_id": motor_selection.selected_candidate_id,
        "selection_changed_due_to_motor_constraints": selection_changed,
        "repeated_no_op_deprioritized": repeated_no_op_deprioritized,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "public_context_constraint_not_object_identity": True,
            "not_a_general_planner": True,
            "not_terminal_causality": True,
        },
        "artifacts": {
            "events": str(output_events),
            "motor_audit": str(output_motor_audit),
            "selection": str(output_selection),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _diagnostic_candidates() -> tuple[ChainCandidate, ...]:
    base_features = {
        "level_completion": False,
        "stable_semantic_delta": False,
        "new_option_enabled": False,
        "new_state_class": True,
        "uncertainty_reduction": False,
        "volatile_visual_delta": False,
    }
    base = ChainCandidate(
        candidate_id="z_candidate_repeated_no_op_risk",
        candidate_type="single_option",
        option_ids=("option_repeated_no_op_risk",),
        start_state_class_id="public_context",
        predicted_terminal_state_class_id="public_context_next",
        predicted_progress_features=dict(base_features),
        cost_features={"predicted_action_cost": 1, "empirical_consistency": 1.0},
        risk_features={"guard_risk": 0},
        provenance=({"kind": "diagnostic_public_motor_candidate"},),
    )
    return (
        base,
        ChainCandidate(
            candidate_id="a_candidate_observed_motion",
            candidate_type="single_option",
            option_ids=("option_observed_motion",),
            start_state_class_id="public_context",
            predicted_terminal_state_class_id="public_context_next",
            predicted_progress_features=dict(base_features),
            cost_features={"predicted_action_cost": 1, "empirical_consistency": 1.0},
            risk_features={"guard_risk": 0},
            provenance=({"kind": "diagnostic_public_motor_candidate"},),
        ),
    )


def _candidate_by_id(selection, candidate_id: str) -> ChainCandidate | None:
    for candidate in selection.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        "# Milestone 12.6.1 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M12.6.1:",
        "  I can tell when my own actions are failing locally, learn a public action-effect constraint model, and stop selecting repeated no-op actions as progress.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- public action-effect observations: `{summary['public_action_effect_observation_count']}`",
        f"- public no-op count: `{summary['public_no_op_count']}`",
        f"- constrained action count: `{summary['constrained_action_count']}`",
        f"- baseline selected candidate: `{summary['baseline_selected_candidate_id']}`",
        f"- motor-aware selected candidate: `{summary['motor_aware_selected_candidate_id']}`",
        f"- selection changed: `{summary['selection_changed_due_to_motor_constraints']}`",
        f"- repeated no-op candidate deprioritized: `{summary['repeated_no_op_deprioritized']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- Constraints are public action-effect regularities, not object identity.",
        "- This is motor competence cleanup, not terminal causality or a full planner.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-events", type=Path, default=DEFAULT_SOURCE_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--motor-audit", type=Path, default=DEFAULT_MOTOR_AUDIT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--no-op-rate-threshold", type=float, default=0.5)
    args = parser.parse_args()
    summary = evaluate(
        source_events=args.source_events,
        output_events=args.events,
        output_motor_audit=args.motor_audit,
        output_selection=args.selection,
        output_freeze=args.freeze,
        minimum_support=args.minimum_support,
        no_op_rate_threshold=args.no_op_rate_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
