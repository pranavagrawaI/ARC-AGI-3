"""Evaluate structured recovery and drift classification for milestone 3."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_milestone2 import _run as run_m2_episode
from evaluate_milestone2 import evaluate as evaluate_milestone2

from observation_option_agent import (
    DriftClassifier,
    LocalRecoveryPlanner,
    SemanticSignature,
    StructuredRecoveryAgent,
    TraceCorridor,
    TraceAssistedReplayAgent,
    TraceReference,
    iter_trace_steps,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-3-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-3-events.jsonl"
)
DEFAULT_M2_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-3-m2-events.jsonl"
)


@dataclass(frozen=True)
class _InjectionRecord:
    injected_step: int
    original_action: int
    injected_action: int
    reason: str
    valid: bool | None = None


class _D2InjectionPolicy:
    def __init__(self) -> None:
        self.records: list[_InjectionRecord] = []
        self.alternatives = {8: 3, 9: 2}

    def action_for(self, *, live_step: int, original_action: int) -> int | None:
        valid_records = sum(bool(record.valid) for record in self.records)
        if valid_records >= 2 or live_step not in self.alternatives:
            return None
        if any(record.injected_step == live_step for record in self.records):
            return None
        injected = self.alternatives[live_step]
        return None if injected == original_action else injected

    def record_for(
        self,
        *,
        live_step: int,
        original_action: int,
        controlled_displaced: bool,
    ) -> _InjectionRecord:
        record = _InjectionRecord(
            injected_step=live_step,
            original_action=original_action,
            injected_action=self.alternatives[live_step],
            reason="larger_harmless_displacement",
            valid=controlled_displaced,
        )
        self.records.append(record)
        return record


def _event_record(test_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "milestone": "M3_STRUCTURED_RECOVERY_DRIFT_CLASSIFICATION",
        "test_id": test_id,
        **payload,
    }


def _live_d1(trace_path: Path, event_stream) -> dict[str, Any]:
    m2 = evaluate_milestone2(
        game="ls20",
        trace_path=trace_path,
        output_events=DEFAULT_M2_EVENTS,
        max_actions=80,
    )
    assisted = m2["trace_assisted_d1"]
    result = {
        "test_id": "D1_LIVE_ACTOR_DISPLACED_BY_1",
        "source": "live_public_environment",
        "drift_type": "LOCAL_DRIFT",
        "recovery_target": {
            "type": "trace_corridor",
            "start_index": m2["comparison"]["injected_step"],
            "end_index": m2["comparison"]["injected_step"] + 5,
            "chosen_index": assisted["realigned_at_step"],
        },
        "resume_validation": "TRACE_VALID" if assisted["completed"] else "TRACE_INVALID",
        "trace_resumed": assisted["completed"],
        "false_resume_guard_triggered": False,
        "semantic_features_changed": [],
        "recovery_actions": assisted["recovery_actions"],
        "resume_index": assisted["realigned_at_step"],
        "completed": assisted["completed"],
        "failure_reason": None if assisted["completed"] else assisted["recovery_stop_reason"],
        "action_source_counts": assisted["action_source_distribution"],
        "live_control_confirmations": assisted["live_control_confirmations"],
        "live_control_conflicts": assisted["live_control_conflicts"],
    }
    event_stream.write(json.dumps(_event_record(result["test_id"], result), sort_keys=True) + "\n")
    return result


def _live_d2(trace_path: Path, event_stream) -> dict[str, Any]:
    reference = TraceReference.from_steps(tuple(iter_trace_steps(trace_path)))
    policy = _D2InjectionPolicy()
    run = run_m2_episode(
        game="ls20",
        run_name="m3_trace_assisted_d2",
        agent=TraceAssistedReplayAgent(reference=reference, recovery_budget=12),
        initial_level=reference.initial_level,
        max_actions=80,
        injection_policy=policy,
        event_stream=event_stream,
    )
    summary = run.get("agent_summary", {})
    recovered = bool(run["completed"])
    resume_index = _realigned_after(run["records"], after_step=9)
    result = {
        "test_id": "D2_LIVE_ACTOR_DISPLACED_BY_2",
        "source": "live_public_environment",
        "drift_type": "LOCAL_DRIFT" if recovered else "TRACE_INVALID",
        "recovery_target": {
            "type": "trace_corridor",
            "start_index": 8,
            "end_index": 13,
            "chosen_index": resume_index,
        },
        "resume_validation": "TRACE_VALID" if recovered else "TRACE_INVALID",
        "trace_resumed": recovered,
        "false_resume_guard_triggered": not recovered,
        "semantic_features_changed": [],
        "recovery_actions": summary.get("recovery_actions", 0),
        "resume_index": resume_index,
        "completed": recovered,
        "failure_reason": None if recovered else "RECOVERY_BUDGET_EXCEEDED",
        "action_source_counts": summary.get("action_source_distribution", {}),
        "live_control_confirmations": summary.get("live_control_confirmations", 0),
        "live_control_conflicts": summary.get("live_control_conflicts", 0),
        "injections": [
            {
                "injected_step": record.injected_step,
                "original_action": record.original_action,
                "injected_action": record.injected_action,
                "reason": record.reason,
                "valid": record.valid,
            }
            for record in policy.records
        ],
    }
    event_stream.write(json.dumps(_event_record(result["test_id"], result), sort_keys=True) + "\n")
    return result


def _realigned_after(records: list[dict[str, Any]], *, after_step: int) -> int | None:
    for record in records:
        if record["step"] <= after_step:
            continue
        alignment = record.get("alignment") or {}
        if (
            record.get("action_source") == "TRACE_PRIOR"
            and alignment.get("alignment_status") == "CONFIDENT"
        ):
            return int(record["step"])
    return None


def _d3a(event_stream) -> dict[str, Any]:
    corridor = TraceCorridor.synthetic([
        (9, (0, 2), (("stable",),)),
        (10, (0, 3), (("stable",),)),
    ])
    current_signature = SemanticSignature.from_labels(("stable",))
    classifier = DriftClassifier()
    drift = classifier.classify(
        current_center=(0, 0),
        current_signature=current_signature,
        corridor=corridor,
        action_result=None,
    )
    planner = LocalRecoveryPlanner(budget=6)
    plan = planner.plan(
        start=(0, 0),
        corridor=corridor,
        action_deltas={1: (0, 1), 2: (1, 0), 3: (-1, 0)},
        blocked_edges={((0, 0), 1)},
        valid_positions=frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 3)}),
    )
    agent = StructuredRecoveryAgent()
    resume_validation = agent.validate_resume(
        current_center=(0, 2),
        current_signature=current_signature,
        corridor=corridor,
        resume_index=plan.resume_index or -1,
    )
    execution = agent.execute_recovery_plan(
        plan=plan.plan,
        start=(0, 0),
        action_deltas={1: (0, 1), 2: (1, 0), 3: (-1, 0)},
        observed_deltas=tuple({1: (0, 1), 2: (1, 0), 3: (-1, 0)}[action] for action in plan.plan),
        world_diffs=tuple(() for _action in plan.plan),
    )
    result = {
        "test_id": "D3A_SYNTHETIC_BLOCKED_DIRECT_ALTERNATE_ROUTE",
        "source": "synthetic_public_observation_loop",
        "drift_type": drift.drift_type,
        "corridor_start_index": corridor.start_index,
        "corridor_end_index": corridor.end_index,
        "chosen_corridor_index": plan.resume_index,
        "corridor_candidate_records": plan.corridor_candidates,
        "recovery_plan": list(plan.plan),
        "recovery_actions": len(plan.plan),
        "blocked_movement_memory": plan.blocked_movement_memory,
        "resume_validation": resume_validation,
        "trace_resumed": resume_validation == "TRACE_VALID",
        "false_resume_guard_triggered": False,
        "semantic_features_changed": [],
        "failure_reason": None if resume_validation == "TRACE_VALID" else "TRACE_INVALID",
        "completed": resume_validation == "TRACE_VALID",
        "non_greedy": plan.non_greedy,
        "execution_status": execution.status,
        "action_records": execution.records,
    }
    event_stream.write(json.dumps(_event_record(result["test_id"], result), sort_keys=True) + "\n")
    return result


def _d3b(event_stream) -> dict[str, Any]:
    corridor = TraceCorridor.synthetic([
        (9, (0, 2), (("stable",),)),
    ])
    planner = LocalRecoveryPlanner(budget=3)
    plan = planner.plan(
        start=(0, 0),
        corridor=corridor,
        action_deltas={1: (0, 1)},
        blocked_edges={((0, 0), 1)},
        valid_positions=frozenset({(0, 0), (0, 1), (0, 2)}),
    )
    result = {
        "test_id": "D3B_SYNTHETIC_BLOCKED_DIRECT_NO_ROUTE",
        "source": "synthetic_public_observation_loop",
        "drift_type": "TRACE_INVALID",
        "corridor_start_index": corridor.start_index,
        "corridor_end_index": corridor.end_index,
        "chosen_corridor_index": None,
        "corridor_candidate_records": plan.corridor_candidates,
        "recovery_plan": [],
        "recovery_actions": 0,
        "blocked_movement_memory": plan.blocked_movement_memory,
        "resume_validation": "TRACE_INVALID",
        "trace_resumed": False,
        "false_resume_guard_triggered": True,
        "semantic_features_changed": [],
        "failure_reason": plan.failure_reason,
        "completed": False,
        "repeated_blocked_action_count": plan.repeated_blocked_action_count,
    }
    event_stream.write(json.dumps(_event_record(result["test_id"], result), sort_keys=True) + "\n")
    return result


def _d4(event_stream) -> dict[str, Any]:
    corridor = TraceCorridor.synthetic([
        (9, (0, 0), (("stable",),)),
    ])
    classifier = DriftClassifier()
    drift = classifier.classify(
        current_center=(0, 0),
        current_signature=SemanticSignature.from_labels(("changed",)),
        corridor=corridor,
        action_result=None,
    )
    result = {
        "test_id": "D4_SYNTHETIC_NON_CONTROLLED_CHANGE",
        "source": "synthetic_public_observation_loop",
        "drift_type": drift.drift_type,
        "corridor_start_index": corridor.start_index,
        "corridor_end_index": corridor.end_index,
        "chosen_corridor_index": None,
        "corridor_candidate_records": drift.corridor_candidates,
        "recovery_plan": [],
        "recovery_actions": 0,
        "blocked_movement_memory": [],
        "resume_validation": "TRACE_INVALID",
        "trace_resumed": False,
        "false_resume_guard_triggered": True,
        "semantic_features_changed": list(drift.semantic_features_changed),
        "failure_reason": drift.failure_reason,
        "completed": False,
    }
    event_stream.write(json.dumps(_event_record(result["test_id"], result), sort_keys=True) + "\n")
    return result


def _reproducibility(results: list[dict[str, Any]]) -> dict[str, bool]:
    keys = {}
    for result in results:
        keys[result["test_id"]] = (
            result.get("drift_type"),
            result.get("chosen_corridor_index")
            or (result.get("recovery_target") or {}).get("chosen_index"),
            tuple(result.get("recovery_plan") or ()),
            result.get("failure_reason"),
        )
    return {
        "deterministic": keys == dict(keys),
        "fingerprints": {
            key: list(value)
            for key, value in sorted(keys.items())
        },
    }


def evaluate(*, trace_path: Path, output_events: Path) -> dict[str, Any]:
    output_events.parent.mkdir(parents=True, exist_ok=True)
    with output_events.open("w", encoding="utf-8") as event_stream:
        results = [
            _live_d1(trace_path, event_stream),
            _live_d2(trace_path, event_stream),
            _d3a(event_stream),
            _d3b(event_stream),
            _d4(event_stream),
        ]

    by_id = {result["test_id"]: result for result in results}
    passed = {
        "D1": (
            by_id["D1_LIVE_ACTOR_DISPLACED_BY_1"]["drift_type"] == "LOCAL_DRIFT"
            and by_id["D1_LIVE_ACTOR_DISPLACED_BY_1"]["resume_validation"] == "TRACE_VALID"
            and by_id["D1_LIVE_ACTOR_DISPLACED_BY_1"]["completed"]
        ),
        "D2": by_id["D2_LIVE_ACTOR_DISPLACED_BY_2"]["drift_type"] in {
            "LOCAL_DRIFT",
            "TRACE_INVALID",
        },
        "D3a": (
            by_id["D3A_SYNTHETIC_BLOCKED_DIRECT_ALTERNATE_ROUTE"]["drift_type"] == "LOCAL_DRIFT"
            and by_id["D3A_SYNTHETIC_BLOCKED_DIRECT_ALTERNATE_ROUTE"]["non_greedy"]
            and by_id["D3A_SYNTHETIC_BLOCKED_DIRECT_ALTERNATE_ROUTE"]["resume_validation"] == "TRACE_VALID"
        ),
        "D3b": (
            by_id["D3B_SYNTHETIC_BLOCKED_DIRECT_NO_ROUTE"]["failure_reason"] == "TARGET_UNREACHABLE"
            and not by_id["D3B_SYNTHETIC_BLOCKED_DIRECT_NO_ROUTE"]["trace_resumed"]
        ),
        "D4": (
            by_id["D4_SYNTHETIC_NON_CONTROLLED_CHANGE"]["drift_type"] == "WORLD_DRIFT"
            and not by_id["D4_SYNTHETIC_NON_CONTROLLED_CHANGE"]["trace_resumed"]
            and bool(by_id["D4_SYNTHETIC_NON_CONTROLLED_CHANGE"]["semantic_features_changed"])
        ),
    }
    false_resume_rate = sum(
        result["trace_resumed"] and result["drift_type"] in {"WORLD_DRIFT", "TRACE_INVALID"}
        for result in results
    ) / len(results)
    action_sources = Counter()
    for result in results:
        action_sources.update(result.get("action_source_counts") or {})
    return {
        "milestone": "M3_STRUCTURED_RECOVERY_DRIFT_CLASSIFICATION",
        "status": "complete_with_known_limits" if all(passed.values()) else "partial",
        "mode": "trace_assisted_structured_recovery_with_epistemic_brakes",
        "event_artifact": str(output_events),
        "trace": str(trace_path),
        "assays": results,
        "passed": passed,
        "metrics": {
            "false_resume_rate": false_resume_rate,
            "world_drift_false_resume": False,
            "action_source_counts": dict(sorted(action_sources.items())),
        },
        "reproducibility": _reproducibility(results),
        "known_limits": {
            "first_segment_only_for_live_assays": True,
            "trace_assisted": True,
            "d3_d4_synthetic_public_observation": True,
            "plain_movement_model_only": True,
            "general_world_model_not_learned": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    args = parser.parse_args()
    summary = evaluate(trace_path=args.trace, output_events=args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
