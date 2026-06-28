"""Run a trace-assisted live contact probe through the public harness boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

from experiments.harness import Transition
from experiments.harness.runner import IllegalActionError, public_observation

from observation_option_agent import (
    LiveContactProbeAgent,
    ProbeProposal,
    iter_trace_steps,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_TRANSITION_SUMMARY = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "transition-learning-summary.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "live-probe-summary.json"
)
TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


def _bootstrap_actions(trace_path: Path, count: int) -> tuple[int, ...]:
    return tuple(
        step.action
        for index, step in enumerate(iter_trace_steps(trace_path))
        if index < count
    )


def _proposal_from_summary(path: Path, *, index: int = 0) -> ProbeProposal:
    summary = json.loads(path.read_text(encoding="utf-8"))
    results = summary["replay_probe_assay"]["top_results"]
    if not results:
        raise ValueError("transition summary contains no replay probe proposals")
    item = results[index]
    return ProbeProposal(
        contact_signature=tuple(
            (part["value"], part["area"], part["shape_id"])
            for part in item["contact_signature"]
        ),
        predicted_effect=item["predicted_effect"],
        support=item["support"],
        effect_rate=item["effect_rate"],
        lift=item["lift"],
    )


def _validate_decision(observation, decision):
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


def evaluate(
    *,
    game: str,
    trace_path: Path,
    transition_summary_path: Path,
    bootstrap_steps: int,
    probe_budget: int,
) -> dict[str, object]:
    proposal = _proposal_from_summary(transition_summary_path)
    bootstrap = _bootstrap_actions(trace_path, bootstrap_steps)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    agent = LiveContactProbeAgent(
        proposal=proposal,
        bootstrap_actions=bootstrap,
        positive_attempt_limit=1,
        negative_control_limit=1,
    )
    current = public_observation(env.reset())
    records = []
    budget = len(bootstrap) + probe_budget

    while len(records) < budget and current.state not in TERMINAL_STATES and not agent.done:
        decision = agent.act(current)
        action, data = _validate_decision(current, decision)
        next_frame = env.step(
            action,
            data=data,
            reasoning=dict(decision.reasoning or {}),
        )
        following = public_observation(next_frame)
        transition = Transition(current, decision, following)
        agent.observe(transition)
        records.append({
            "step": len(records) + 1,
            "action": decision.action,
            "phase": (decision.reasoning or {}).get("phase"),
            "levels_before": current.levels_completed,
            "levels_after": following.levels_completed,
            "state_after": following.state,
        })
        current = following

    result = agent.result()
    live_summary = agent.summary()
    return {
        "agent": "0001-live-contact-probe",
        "game": game,
        "mode": "trace-assisted valid live probe",
        "bootstrap_steps_requested": bootstrap_steps,
        "bootstrap_actions_used": live_summary.bootstrap_actions,
        "probe_budget": probe_budget,
        "actions": live_summary.actions,
        "final_state": current.state,
        "levels_completed": current.levels_completed,
        "proposal": {
            "contact_signature": [
                {"value": value, "area": area, "shape_id": shape_id}
                for value, area, shape_id in proposal.contact_signature
            ],
            "predicted_effect": proposal.predicted_effect,
            "support": proposal.support,
            "effect_rate": proposal.effect_rate,
            "lift": proposal.lift,
        },
        "result": {
            "positive_attempts": result.positive_attempts,
            "positive_successes": result.positive_successes,
            "negative_controls": result.negative_controls,
            "negative_control_violations": result.negative_control_violations,
            "reproduced": result.reproduced,
            "negative_control_passed": result.negative_control_passed,
        },
        "completed": live_summary.completed,
        "notes": list(live_summary.notes),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--transition-summary", type=Path, default=DEFAULT_TRANSITION_SUMMARY)
    parser.add_argument("--bootstrap-steps", type=int, default=29)
    parser.add_argument("--probe-budget", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        trace_path=args.trace,
        transition_summary_path=args.transition_summary,
        bootstrap_steps=args.bootstrap_steps,
        probe_budget=args.probe_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
