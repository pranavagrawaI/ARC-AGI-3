"""0028 - Static replay evaluator for primitive plans.

Run from the repo root:
    uv run python teach/lessons/0028-static-replay-evaluator.py
    uv run python teach/lessons/0028-static-replay-evaluator.py --live --levels 1,2,3,4,5,6,7

This is the first check for the agent-like loop from 0027. It does not control
the environment. It asks a narrower replay question:

    At a recorded state, would the planner choose an action with the same
    learned causal or terminal signal as the successful trace action?

Exact action equality is deliberately secondary. The first useful signal is
mechanistic agreement: same remote effect while chasing the descriptor, or same
next_overlap_smaller terminal predicate once matched.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0028-static-replay-evaluator"


def _load_0027():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0027-primitive-planner.py"
    spec = _ilu.spec_from_file_location("sre28_pp27", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pp27 = _load_0027()
tp26 = pp27.tp26
am24 = pp27.am24
ri19 = tp26.ri19


@dataclass(frozen=True)
class ReplayDecision:
    phase: str
    frame: int
    planner_action: str | None
    trace_action: str | None
    planner_signal: tuple | None
    trace_signal: tuple | None
    signal_match: bool
    exact_action_match: bool


def _effect_signal(
    action: pp27.PrimitiveAction,
    effects: tuple[pp27.LearnedTriggerEffect, ...],
    target_object: str,
) -> tuple[str, str, str] | None:
    for effect in effects:
        if effect.object_id != target_object:
            continue
        if set(effect.cause_clause) <= action.local_facts:
            return (effect.object_id, effect.slot, effect.delta)
    return None


def evaluate_descriptor_replay_step(
    initial: dict[str, object],
    target: dict[str, object],
    effects: tuple[pp27.LearnedTriggerEffect, ...],
    actions: tuple[pp27.PrimitiveAction, ...],
    *,
    target_object: str,
    trace_action: str,
    frame: int = 0,
) -> ReplayDecision:
    plan = pp27.plan_to_descriptor(initial, target, effects, actions, target_object=target_object)
    planner_action = plan[0] if plan else None
    by_name = {action.name: action for action in actions}
    planner_signal = _effect_signal(by_name[planner_action], effects, target_object) if planner_action else None
    trace_signal = _effect_signal(by_name[trace_action], effects, target_object) if trace_action in by_name else None
    return ReplayDecision(
        phase="descriptor",
        frame=frame,
        planner_action=planner_action,
        trace_action=trace_action,
        planner_signal=planner_signal,
        trace_signal=trace_signal,
        signal_match=planner_signal is not None and planner_signal == trace_signal,
        exact_action_match=planner_action == trace_action,
    )


def _terminal_signal(actors: list, small, large, action: pp27.PrimitiveAction) -> tuple[str] | None:
    facts = tp26.terminal_facts(actors, small, large, action.dr, action.dc)
    if "next_overlap_smaller" in facts:
        return ("next_overlap_smaller",)
    return None


def evaluate_terminal_replay_step(
    actors: list,
    small,
    large,
    actions: tuple[pp27.PrimitiveAction, ...],
    *,
    trace_action: str,
    frame: int = 0,
) -> ReplayDecision:
    planner_action = pp27.terminal_overlap_action(actors, small, actions)
    by_name = {action.name: action for action in actions}
    planner_signal = _terminal_signal(actors, small, large, by_name[planner_action]) if planner_action else None
    trace_signal = _terminal_signal(actors, small, large, by_name[trace_action]) if trace_action in by_name else None
    return ReplayDecision(
        phase="terminal",
        frame=frame,
        planner_action=planner_action,
        trace_action=trace_action,
        planner_signal=planner_signal,
        trace_signal=trace_signal,
        signal_match=planner_signal is not None and planner_signal == trace_signal,
        exact_action_match=planner_action == trace_action,
    )


def _primitive_actions_from_oc16(oc16, available_actions: list[int]) -> tuple[pp27.PrimitiveAction, ...]:
    return tuple(
        pp27.PrimitiveAction(
            name=f"ACTION{action}",
            dr=int(oc16.ACTION_DELTAS[action][0]),
            dc=int(oc16.ACTION_DELTAS[action][1]),
            local_facts=frozenset({f"action={action}"}),
        )
        for action in available_actions
        if action in oc16.ACTION_DELTAS
    )


def _write_live_artifact(decisions: list[ReplayDecision], levels: set[int]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-terminal-replay-decisions.jsonl"
    with path.open("w") as fh:
        for decision in decisions:
            fh.write(json.dumps({
                "exact_action_match": decision.exact_action_match,
                "frame": decision.frame,
                "phase": decision.phase,
                "planner_action": decision.planner_action,
                "planner_signal": decision.planner_signal,
                "signal_match": decision.signal_match,
                "trace_action": decision.trace_action,
                "trace_signal": decision.trace_signal,
            }, sort_keys=True) + "\n")
    return path


def evaluate_live_terminal_replay(trace: Path | None, levels: set[int], window: int) -> list[ReplayDecision]:
    oc16 = ri19.rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    decisions: list[ReplayDecision] = []

    for seg in oc16.segments_for_levels(segments, levels):
        belief = oc16.AgentBelief()
        reward_indices = set(am24.ras23._reward_indices(records, seg))
        reward_windows = {idx for inc in reward_indices for idx in range(max(seg.start, inc - window), inc)}
        for idx in range(seg.start, seg.end + 1):
            _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
            if idx in reward_windows and idx + 1 < len(records) and records[idx + 1].action_num is not None:
                actors = oc16._actor_views(parts, belief.controlled_sigs)
                actions = _primitive_actions_from_oc16(oc16, records[idx].available_actions)
                for left, right in tp26._attribute_match_pairs(parts, oc16):
                    small, large = (left, right) if left.area <= right.area else (right, left)
                    decision = evaluate_terminal_replay_step(
                        actors,
                        small,
                        large,
                        actions,
                        trace_action=records[idx + 1].action_id,
                        frame=idx,
                    )
                    if decision.planner_signal or decision.trace_signal:
                        decisions.append(decision)
            if idx + 1 <= seg.end and records[idx + 1].action_num is not None:
                _bg1, parts1 = oc16.il.views_from_grid(records[idx + 1].frame, frame=idx + 1)
                belief.learn_remote_dynamics(parts, parts1)
                belief.observe_transition(records[idx].frame, records[idx + 1].action_num, records[idx + 1].frame)
                belief.frame_idx += 1
                belief.perceive(records[idx + 1].frame)
    return decisions


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    actions = (
        pp27.PrimitiveAction("ACTION1", -1, 0, frozenset({"contact", "trigger=color"})),
        pp27.PrimitiveAction("ACTION2", 0, 1, frozenset({"contact", "trigger=orientation"})),
        pp27.PrimitiveAction("ACTION3", 1, 0, frozenset({"contact", "noise"})),
    )
    effects = (
        pp27.LearnedTriggerEffect(("contact", "trigger=color"), "workspace", "color", "red->blue"),
        pp27.LearnedTriggerEffect(("contact", "trigger=orientation"), "workspace", "orientation", "rot+1mod4"),
    )
    desc = evaluate_descriptor_replay_step(
        {"shape": "L", "color": "red", "orientation": 0},
        {"shape": "L", "color": "blue", "orientation": 1},
        effects,
        actions,
        target_object="workspace",
        trace_action="ACTION1",
    )
    checks.append(_check(
        "1 descriptor replay matches causal signal",
        desc.signal_match and desc.exact_action_match,
        f"decision={desc}",
    ))

    bad = evaluate_descriptor_replay_step(
        {"shape": "L", "color": "red", "orientation": 0},
        {"shape": "L", "color": "blue", "orientation": 1},
        effects,
        actions,
        target_object="workspace",
        trace_action="ACTION3",
    )
    checks.append(_check(
        "2 descriptor replay detects trace/planner disagreement",
        not bad.signal_match,
        f"decision={bad}",
    ))

    import numpy as np

    actor_grid = np.zeros((5, 5), dtype=bool)
    actor_grid[2, 2] = True
    small_grid = np.zeros((5, 5), dtype=bool)
    small_grid[1, 2] = True
    large_grid = np.zeros((5, 5), dtype=bool)
    large_grid[4, 4] = True
    actor = am24._FakeView(actor_grid, 9, 0)
    small = am24._FakeView(small_grid, 9, 0)
    large = am24._FakeView(large_grid, 9, 0)
    term = evaluate_terminal_replay_step([actor], small, large, actions, trace_action="ACTION1")
    checks.append(_check(
        "3 terminal replay matches overlap signal",
        term.signal_match and term.exact_action_match,
        f"decision={term}",
    ))
    return all(checks)


def run_live(trace: Path | None, levels: set[int], window: int) -> list[ReplayDecision]:
    decisions = evaluate_live_terminal_replay(trace, levels, window)
    artifact = _write_live_artifact(decisions, levels)
    counts = Counter((d.signal_match, d.exact_action_match) for d in decisions)
    print("\nLive ls20 static terminal replay evaluation:")
    print(f"  decisions={len(decisions)} signal/exact_counts={dict(counts)}")
    if decisions:
        signal_rate = sum(d.signal_match for d in decisions) / len(decisions)
        exact_rate = sum(d.exact_action_match for d in decisions) / len(decisions)
        print(f"  signal_match_rate={signal_rate:.3f} exact_action_match_rate={exact_rate:.3f}")
        for decision in decisions[:8]:
            print(
                f"  frame={decision.frame} planner={decision.planner_action}:{decision.planner_signal} "
                f"trace={decision.trace_action}:{decision.trace_signal} signal_match={decision.signal_match}"
            )
    else:
        print("  no terminal replay decisions found; perception did not produce matched pairs with overlap candidates")
    print(f"  replay_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate terminal planner choices on recorded reward windows")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=4, help="frames before reward to evaluate")
    args = parser.parse_args()

    print("Offline checks (static replay evaluator):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using static replay evaluator.")
    print("\nStatic replay evaluator can compare planner and trace by causal/terminal signal.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window)


if __name__ == "__main__":
    main()
