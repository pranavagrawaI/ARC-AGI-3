"""0029 - Counterfactual descriptor replay.

Run from the repo root:
    uv run python teach/lessons/0029-counterfactual-descriptor-replay.py
    uv run python teach/lessons/0029-counterfactual-descriptor-replay.py --live --levels 1,2,3,4,5,6,7

0028 checked terminal overlap at reward-adjacent frames. This lesson checks the
descriptor-chasing half without pretending we can route yet:

    Given a desired supported remote effect from the successful trace, can the
    primitive planner choose an observed trigger affordance that would fire it?

The candidate actions are not movement plans. They are replay-grounded local fact
bundles such as ACTION2 + contact + trigger_shape=... . Routing to produce those
facts is the next layer.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0029-counterfactual-descriptor-replay"


def _load_0027():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0027-primitive-planner.py"
    spec = _ilu.spec_from_file_location("cdr29_pp27", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pp27 = _load_0027()
rcf18 = pp27.tp26.ri19.rcf18


@dataclass(frozen=True)
class CounterfactualDecision:
    frame: int
    desired_effect: tuple[str, str, str]
    planner_action: str | None
    trace_action: str
    planner_signals: tuple[tuple[str, str, str], ...]
    trace_signals: tuple[tuple[str, str, str], ...]
    signal_match: bool
    exact_action_match: bool


def _signals_for_action(
    action: pp27.PrimitiveAction,
    effects: tuple[pp27.LearnedTriggerEffect, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (effect.object_id, effect.slot, effect.delta)
        for effect in effects
        if set(effect.cause_clause) <= action.local_facts
    )


def _state_for_effect(effect: pp27.LearnedTriggerEffect) -> tuple[dict[str, object], dict[str, object]]:
    if "rot" in effect.delta:
        before: object = 0
    elif "->" in effect.delta:
        before = effect.delta.split("->", 1)[0]
    else:
        before = "before"
    after = pp27._apply_delta(before, effect.delta)
    return {effect.slot: before}, {effect.slot: after}


def evaluate_counterfactual_effect(
    desired: pp27.LearnedTriggerEffect,
    effects: tuple[pp27.LearnedTriggerEffect, ...],
    candidates: tuple[pp27.PrimitiveAction, ...],
    trace_action: pp27.PrimitiveAction,
    *,
    frame: int = 0,
) -> CounterfactualDecision:
    initial, target = _state_for_effect(desired)
    plan = pp27.plan_to_descriptor(
        initial,
        target,
        effects,
        candidates,
        target_object=desired.object_id,
        max_depth=1,
    )
    by_name = {action.name: action for action in candidates}
    planner_action = plan[0] if plan else None
    planner_signals = _signals_for_action(by_name[planner_action], effects) if planner_action else ()
    trace_signals = _signals_for_action(trace_action, effects)
    desired_signal = (desired.object_id, desired.slot, desired.delta)
    return CounterfactualDecision(
        frame=frame,
        desired_effect=desired_signal,
        planner_action=planner_action,
        trace_action=trace_action.name,
        planner_signals=planner_signals,
        trace_signals=trace_signals,
        signal_match=desired_signal in planner_signals and desired_signal in trace_signals,
        exact_action_match=(planner_action or "").split("|")[0] == trace_action.name.split("|")[0],
    )


def _candidate_actions_from_events(events) -> tuple[pp27.PrimitiveAction, ...]:
    seen: dict[tuple[str, tuple[str, ...]], pp27.PrimitiveAction] = {}
    for event in events:
        key = (event.action, tuple(sorted(event.local_facts)))
        if key not in seen:
            label = ",".join(key[1])
            seen[key] = pp27.PrimitiveAction(f"{event.action}|{label}", 0, 0, event.local_facts)
    return tuple(seen[key] for key in sorted(seen))


def _trace_action_for_event(event) -> pp27.PrimitiveAction:
    label = ",".join(sorted(event.local_facts))
    return pp27.PrimitiveAction(f"{event.action}|{label}", 0, 0, event.local_facts)


def _effect_present(event, signal: tuple[str, str, str]) -> bool:
    return rcf18.rca17._effect_present(event, signal)


def evaluate_live_counterfactual_replay(trace: Path | None, levels: set[int]) -> list[CounterfactualDecision]:
    events, meta, _counts = rcf18._live_step_events(trace, levels)
    ledger = rcf18.falsify_remote_hypotheses(events, max_depth=2, min_support=2)
    supported = tuple(entry for entry in ledger if entry.status == "SUPPORTED")
    effects = tuple(pp27.LearnedTriggerEffect.from_remote_hypothesis(entry.hypothesis) for entry in supported)
    candidates = _candidate_actions_from_events(events)
    decisions: list[CounterfactualDecision] = []

    for event, info in zip(events, meta, strict=True):
        trace_action = _trace_action_for_event(event)
        for effect in effects:
            signal = (effect.object_id, effect.slot, effect.delta)
            if not _effect_present(event, signal):
                continue
            if signal not in _signals_for_action(trace_action, effects):
                continue
            decision = evaluate_counterfactual_effect(
                effect,
                effects,
                candidates,
                trace_action,
                frame=info["trace_index"],
            )
            decisions.append(decision)
    return decisions


def _write_live_artifact(decisions: list[CounterfactualDecision], levels: set[int]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-counterfactual-decisions.jsonl"
    with path.open("w") as fh:
        for decision in decisions:
            fh.write(json.dumps({
                "desired_effect": decision.desired_effect,
                "exact_action_match": decision.exact_action_match,
                "frame": decision.frame,
                "planner_action": decision.planner_action,
                "planner_signals": decision.planner_signals,
                "signal_match": decision.signal_match,
                "trace_action": decision.trace_action,
                "trace_signals": decision.trace_signals,
            }, sort_keys=True) + "\n")
    return path


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    candidates = (
        pp27.PrimitiveAction("ACTION1|contact,trigger=color", 0, 0, frozenset({"contact", "trigger=color"})),
        pp27.PrimitiveAction("ACTION2|contact,trigger=orientation", 0, 0, frozenset({"contact", "trigger=orientation"})),
        pp27.PrimitiveAction("ACTION3|noise", 0, 0, frozenset({"noise"})),
    )
    effects = (
        pp27.LearnedTriggerEffect(("contact", "trigger=color"), "workspace", "color", "red->blue"),
        pp27.LearnedTriggerEffect(("contact", "trigger=orientation"), "workspace", "orientation", "any->rot+1mod4"),
    )
    color = evaluate_counterfactual_effect(effects[0], effects, candidates, candidates[0])
    checks.append(_check(
        "1 chooses affordance that fires desired color effect",
        color.signal_match and color.exact_action_match,
        f"decision={color}",
    ))
    orientation = evaluate_counterfactual_effect(effects[1], effects, candidates, candidates[1])
    checks.append(_check(
        "2 consumes live rotation delta format counterfactually",
        orientation.signal_match and orientation.exact_action_match,
        f"decision={orientation}",
    ))
    disagreement = evaluate_counterfactual_effect(effects[1], effects, candidates, candidates[2])
    checks.append(_check(
        "3 detects trace action with wrong causal signal",
        not disagreement.signal_match,
        f"decision={disagreement}",
    ))
    unreachable = evaluate_counterfactual_effect(effects[0], effects, candidates[1:], candidates[1])
    checks.append(_check(
        "4 fails when no candidate can fire desired effect",
        unreachable.planner_action is None and not unreachable.signal_match,
        f"decision={unreachable}",
    ))
    return all(checks)


def run_live(trace: Path | None, levels: set[int]) -> list[CounterfactualDecision]:
    decisions = evaluate_live_counterfactual_replay(trace, levels)
    artifact = _write_live_artifact(decisions, levels)
    counts = Counter((d.signal_match, d.exact_action_match) for d in decisions)
    print("\nLive ls20 counterfactual descriptor replay:")
    print(f"  decisions={len(decisions)} signal/exact_counts={dict(counts)}")
    if decisions:
        signal_rate = sum(d.signal_match for d in decisions) / len(decisions)
        exact_rate = sum(d.exact_action_match for d in decisions) / len(decisions)
        print(f"  signal_match_rate={signal_rate:.3f} exact_action_match_rate={exact_rate:.3f}")
        for decision in decisions[:10]:
            print(
                f"  frame={decision.frame} desired={decision.desired_effect} "
                f"planner={decision.planner_action} trace={decision.trace_action} "
                f"signal_match={decision.signal_match} exact={decision.exact_action_match}"
            )
    else:
        print("  no supported trace effects found for counterfactual replay")
    print(f"  counterfactual_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate supported remote effects on recorded trace")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    args = parser.parse_args()

    print("Offline checks (counterfactual descriptor replay):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using counterfactual descriptor replay.")
    print("\nCounterfactual descriptor replay can test whether planner-chosen affordances fire desired supported effects.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels)


if __name__ == "__main__":
    main()
