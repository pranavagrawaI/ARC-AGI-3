"""Evaluate first-segment trace alignment with recovery through public frames."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

from experiments.harness import Agent, Decision, Transition
from experiments.harness.runner import IllegalActionError, public_observation

from observation_option_agent import (
    ControlledEntityBelief,
    D1InjectionPolicy,
    EventDetector,
    ObjectTracker,
    ObservationParser,
    PureReplayAgent,
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
    / "milestone-2-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-2-events.jsonl"
)
TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


class PublicRunMonitor:
    def __init__(self) -> None:
        self.parser = ObservationParser()
        self.tracker = ObjectTracker()
        self.detector = EventDetector()
        self.control_belief = ControlledEntityBelief()
        self.actions = 0
        self._initialized = False

    def initialize(self, observation) -> None:
        if self._initialized:
            return
        scene = self.parser.parse(observation, frame_index=0)
        self.tracker.update(scene)
        self._initialized = True

    def observe(self, transition: Transition) -> dict[str, Any]:
        self.initialize(transition.before)
        self.actions += 1
        scene = self.parser.parse(transition.after, frame_index=self.actions)
        force_new = (
            transition.after.levels_completed != transition.before.levels_completed
            or transition.after.full_reset
        )
        tracks = self.tracker.update(scene, force_new=force_new)
        event = self.detector.detect(
            step=self.actions,
            before=transition.before,
            action=transition.decision.action,
            after=transition.after,
            tracks=tracks,
        )
        prior_control = self.control_belief.active
        controlled_displaced = False
        if prior_control is not None:
            controlled = set(prior_control.object_ids)
            controlled_displaced = any(
                delta.object_id in controlled and delta.moved
                for delta in event.object_deltas
            )
        control = self.control_belief.observe(event)
        return {
            "controlled_displaced": controlled_displaced,
            "control_source": control.source if control is not None else None,
            "control_object_ids": list(control.object_ids) if control is not None else [],
        }


def _validate_decision(observation, decision: Decision):
    if decision.action not in observation.available_actions:
        raise IllegalActionError(
            f"agent chose action {decision.action}; "
            f"available actions are {observation.available_actions}"
        )
    action = GameAction.from_id(decision.action)
    data = dict(decision.data) if decision.data is not None else None
    if action.is_complex() and data is None:
        raise IllegalActionError(f"action {decision.action} requires action data")
    if data is not None:
        try:
            action.validate_data(data)
        except ValueError as error:
            raise IllegalActionError(
                f"invalid data for action {decision.action}: {error}"
            ) from error
    return action, data


def _policy() -> D1InjectionPolicy:
    return D1InjectionPolicy(
        primary_step=8,
        backup_steps=(9, 10),
        alternatives={8: 3, 9: 3, 10: 3},
    )


def _run(
    *,
    game: str,
    run_name: str,
    agent: Agent,
    initial_level: int,
    max_actions: int,
    injection_policy: D1InjectionPolicy | None,
    event_stream,
) -> dict[str, Any]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    monitor = PublicRunMonitor()
    records: list[dict[str, Any]] = []
    stop_reason = "action_budget"

    while len(records) < max_actions:
        if current.state in TERMINAL_STATES:
            stop_reason = "terminal"
            break
        if current.levels_completed > initial_level:
            stop_reason = "first_segment_complete"
            break
        if not current.available_actions:
            stop_reason = "no_available_actions"
            break

        live_step = len(records) + 1
        decision = agent.act(current)
        original_action = decision.action
        injected = False
        injected_action = None
        if injection_policy is not None:
            injected_action = injection_policy.action_for(
                live_step=live_step,
                original_action=original_action,
            )
            if injected_action is not None:
                injected = True
                decision = Decision(
                    action=injected_action,
                    data=decision.data,
                    reasoning={
                        **dict(decision.reasoning or {}),
                        "injection": {
                            "reason": "harmless_displacement",
                            "original_action": original_action,
                            "injected_action": injected_action,
                        },
                    },
                )

        action, data = _validate_decision(current, decision)
        following = public_observation(
            env.step(action, data=data, reasoning=dict(decision.reasoning or {}))
        )
        transition = Transition(current, decision, following)
        agent.observe(transition)
        monitor_record = monitor.observe(transition)

        injection_record = None
        if injected and injection_policy is not None and injected_action is not None:
            injection_record = injection_policy.record_for(
                live_step=live_step,
                original_action=original_action,
                controlled_displaced=bool(monitor_record["controlled_displaced"]),
            )

        record = {
            "run": run_name,
            "step": live_step,
            "original_action": original_action,
            "action": decision.action,
            "injected": injected,
            "injection": _injection_record(injection_record),
            "action_source": (decision.reasoning or {}).get("action_source"),
            "levels_before": current.levels_completed,
            "levels_after": following.levels_completed,
            "state_after": following.state,
            "alignment": (decision.reasoning or {}).get("alignment"),
            "monitor": monitor_record,
        }
        records.append(record)
        event_stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        event_stream.write("\n")
        current = following

    completed = current.levels_completed > initial_level
    action_source_distribution = Counter(
        str(record.get("action_source"))
        for record in records
        if record.get("action_source") is not None
    )
    return {
        "name": run_name,
        "completed": completed,
        "actions": len(records),
        "stop_reason": stop_reason,
        "final_state": current.state,
        "levels_completed": current.levels_completed,
        "injections": [
            _injection_record(record)
            for record in (injection_policy.records if injection_policy is not None else ())
        ],
        "valid_injection": any(
            record.valid for record in (injection_policy.records if injection_policy else ())
        ),
        "action_source_distribution": dict(sorted(action_source_distribution.items())),
        "records": records,
        "agent_summary": agent.summary() if hasattr(agent, "summary") else {},
    }


def evaluate(
    *,
    game: str,
    trace_path: Path,
    output_events: Path,
    max_actions: int,
) -> dict[str, Any]:
    reference = TraceReference.from_steps(tuple(iter_trace_steps(trace_path)))
    output_events.parent.mkdir(parents=True, exist_ok=True)
    with output_events.open("w", encoding="utf-8") as event_stream:
        normal = _run(
            game=game,
            run_name="normal_trace_assisted",
            agent=TraceAssistedReplayAgent(reference=reference),
            initial_level=reference.initial_level,
            max_actions=max_actions,
            injection_policy=None,
            event_stream=event_stream,
        )
        pure = _run(
            game=game,
            run_name="pure_replay_d1",
            agent=PureReplayAgent(reference.segment_actions),
            initial_level=reference.initial_level,
            max_actions=max_actions,
            injection_policy=_policy(),
            event_stream=event_stream,
        )
        assisted = _run(
            game=game,
            run_name="trace_assisted_d1",
            agent=TraceAssistedReplayAgent(reference=reference),
            initial_level=reference.initial_level,
            max_actions=max_actions,
            injection_policy=_policy(),
            event_stream=event_stream,
        )

    comparison = _comparison(reference, pure, assisted)
    status = (
        "complete_with_known_limits"
        if normal["completed"]
        and assisted["completed"]
        and assisted["valid_injection"]
        and comparison["post_injection_action_differs_from_pure_replay"]
        and comparison["trace_assisted_used_recovery_or_probe"]
        else "partial"
    )
    return {
        "milestone": 2,
        "status": status,
        "game": game,
        "mode": "trace_assisted_observation_only_first_segment_alignment_with_recovery",
        "event_artifact": str(output_events),
        "trace": str(trace_path),
        "first_segment": {
            "initial_level": reference.initial_level,
            "completion_step": reference.completion_step,
            "trace_actions": len(reference.segment_actions),
            "deviation_taxonomy": {
                "D1": "harmless_displacement",
                "D2": "blocked_or_off_trace_action_recorded_for_future",
                "D3": "wrong_direction_near_contact_recorded_for_future",
            },
        },
        "normal_run": _compact_run(normal, reference),
        "pure_replay_d1": _compact_run(pure, reference),
        "trace_assisted_d1": _compact_run(assisted, reference),
        "comparison": comparison,
        "evidence": {
            "public_observations_only": True,
            "trace_used_as_visual_prior_not_index_script": comparison[
                "post_injection_action_differs_from_pure_replay"
            ],
            "alignment_based_on_live_visual_state": True,
            "control_model_source": "trace_bootstrap_live_verified",
            "closed_loop_recovery": comparison["trace_assisted_used_recovery_or_probe"],
        },
        "known_limits": {
            "only_first_segment": True,
            "trace_assisted": True,
            "completion_condition_not_modeled": True,
            "reset_regression_not_tested": True,
            "resource_model_not_proven": True,
        },
    }


def _comparison(
    reference: TraceReference,
    pure: dict[str, Any],
    assisted: dict[str, Any],
) -> dict[str, Any]:
    pure_records = pure["records"]
    assisted_records = assisted["records"]
    injected_steps = [
        record["injected_step"]
        for record in assisted["injections"]
        if record and record.get("valid")
    ]
    injected_step = injected_steps[0] if injected_steps else None
    post_differs = False
    if injected_step is not None:
        for pure_record, assisted_record in zip(pure_records, assisted_records):
            if assisted_record["step"] <= injected_step:
                continue
            if pure_record["action"] != assisted_record["action"]:
                post_differs = True
                break
    sources = Counter(
        str(record.get("action_source"))
        for record in assisted_records
        if record.get("action_source") is not None
    )
    used_recovery = bool(sources["LOCAL_RECOVERY"] or sources["ALIGNMENT_PROBE"])
    return {
        "injected_step": injected_step,
        "pure_replay_completed": pure["completed"],
        "trace_assisted_completed": assisted["completed"],
        "pure_replay_actions": pure["actions"],
        "trace_assisted_actions": assisted["actions"],
        "pure_replay_failed_or_worse": (
            not pure["completed"] or pure["actions"] >= assisted["actions"]
        ),
        "post_injection_action_differs_from_pure_replay": post_differs,
        "trace_assisted_used_recovery_or_probe": used_recovery,
        "extra_actions_vs_trace": assisted["actions"] - reference.completion_step,
    }


def _compact_run(run: dict[str, Any], reference: TraceReference) -> dict[str, Any]:
    agent_summary = run.get("agent_summary", {})
    injected_steps = [
        item["injected_step"]
        for item in run["injections"]
        if item is not None and item.get("valid")
    ]
    injected_step = injected_steps[0] if injected_steps else None
    return {
        "completed": run["completed"],
        "actions": run["actions"],
        "trace_actions": reference.completion_step,
        "extra_actions_vs_trace": run["actions"] - reference.completion_step,
        "stop_reason": run["stop_reason"],
        "final_state": run["final_state"],
        "levels_completed": run["levels_completed"],
        "valid_injection": run["valid_injection"],
        "injections": run["injections"],
        "action_source_distribution": (
            agent_summary.get("action_source_distribution")
            or run["action_source_distribution"]
        ),
        "recovery_actions": agent_summary.get("recovery_actions", 0),
        "recovery_stop_reason": agent_summary.get("recovery_stop_reason"),
        "max_alignment_uncertainty": agent_summary.get("max_alignment_uncertainty", 0.0),
        "live_control_confirmations": agent_summary.get("live_control_confirmations", 0),
        "live_control_conflicts": agent_summary.get("live_control_conflicts", 0),
        "realigned_at_step": _realigned_at_step(
            run["records"],
            after_step=injected_step,
        ),
    }


def _realigned_at_step(
    records: list[dict[str, Any]],
    *,
    after_step: int | None,
) -> int | None:
    for record in records:
        if after_step is not None and record["step"] <= after_step:
            continue
        if record.get("action_source") != "TRACE_PRIOR":
            continue
        alignment = record.get("alignment") or {}
        if alignment.get("alignment_status") == "CONFIDENT":
            return int(record["step"])
    return None


def _injection_record(record) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "injected_step": record.injected_step,
        "original_action": record.original_action,
        "injected_action": record.injected_action,
        "reason": record.reason,
        "valid": record.valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--max-actions", type=int, default=80)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        trace_path=args.trace,
        output_events=args.events,
        max_actions=args.max_actions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
