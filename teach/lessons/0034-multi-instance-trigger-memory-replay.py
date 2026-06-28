"""0034 - Multi-instance causal trigger memory replay.

Run from the repo root:
    uv run python teach/lessons/0034-multi-instance-trigger-memory-replay.py
    uv run python teach/lessons/0034-multi-instance-trigger-memory-replay.py --live --levels 1,2,3,4,5,6,7 --window 6

0033 proved that prior causal memory is the new bottleneck: if a remembered
trigger is available, routing usually works, but a single memory slot per cause
clause gives poor coverage.

This lesson removes that single-slot assumption. The agent can remember multiple
trigger instances for the same cause/effect, track each forward independently,
and choose the shortest routable remembered trigger for a desired cause.

Replay still supplies the evaluation windows and supported effects. The agent's
selection does not read the future contacted target; that target is used only to
score whether the selected memory was the same object the trace later contacted.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0034-multi-instance-trigger-memory-replay"


def _load_0033():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0033-causal-trigger-memory-replay.py"
    spec = _ilu.spec_from_file_location("mitm34_ctm33", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ctm33 = _load_0033()
ewr32 = ctm33.ewr32
ltr31 = ctm33.ltr31
pp27 = ctm33.pp27
rcf18 = ctm33.rcf18


@dataclass(frozen=True)
class TriggerMemoryEntry:
    entry_id: int
    cause_clause: tuple[str, ...]
    effect_signal: tuple[str, str, str]
    view: object
    support: int
    first_seen_frame: int
    last_seen_frame: int


@dataclass(frozen=True)
class MultiMemoryDecision:
    contact_frame: int
    start_frame: int
    window: int
    desired_effect: tuple[str, str, str]
    cause_clause: tuple[str, ...]
    candidates: int
    correct_candidate_available: bool
    selected_entry_id: int | None
    selected_correct: bool
    route_found: bool
    route_actions: tuple[int, ...]
    trace_window_actions: tuple[int, ...]
    first_action_match: bool
    last_action_match: bool
    common_prefix_len: int
    common_suffix_len: int
    exact_match: bool

    @property
    def route_len(self) -> int:
        return len(self.route_actions)

    @property
    def common_prefix_frac(self) -> float:
        return self.common_prefix_len / self.route_len if self.route_len else 0.0

    @property
    def common_suffix_frac(self) -> float:
        return self.common_suffix_len / self.route_len if self.route_len else 0.0


def _track_entries(
    entries: tuple[TriggerMemoryEntry, ...],
    parts: list,
    matcher: str,
    oc16,
    frame: int,
) -> tuple[TriggerMemoryEntry, ...]:
    tracked = []
    for entry in entries:
        nxt = ctm33._memory_match(entry.view, parts, matcher, oc16)
        if nxt is not None:
            tracked.append(replace(entry, view=nxt, last_seen_frame=frame))
    return tuple(tracked)


def _same_instance(a, b) -> bool:
    return bool(a is not None and b is not None and len(a.pixels & b.pixels) > 0)


def _remember_contact(
    entries: tuple[TriggerMemoryEntry, ...],
    *,
    cause_clause: tuple[str, ...],
    effect_signal: tuple[str, str, str],
    contacted,
    frame: int,
    next_id: int,
) -> tuple[tuple[TriggerMemoryEntry, ...], int]:
    refreshed = []
    matched = False
    for entry in entries:
        if (
            entry.cause_clause == cause_clause
            and entry.effect_signal == effect_signal
            and _same_instance(entry.view, contacted)
        ):
            refreshed.append(replace(
                entry,
                view=contacted,
                support=entry.support + 1,
                last_seen_frame=frame,
            ))
            matched = True
        else:
            refreshed.append(entry)
    if not matched:
        refreshed.append(TriggerMemoryEntry(
            entry_id=next_id,
            cause_clause=cause_clause,
            effect_signal=effect_signal,
            view=contacted,
            support=1,
            first_seen_frame=frame,
            last_seen_frame=frame,
        ))
        next_id += 1
    return tuple(refreshed), next_id


def _route_entry(records, start_frame: int, actor_views: list, entry: TriggerMemoryEntry, oc16):
    firing_action = ltr31._parse_action_clause(entry.cause_clause)
    return ewr32.route_to_trigger_grid(
        records[start_frame].frame,
        actor_views,
        entry.view,
        oc16.ACTION_DELTAS,
        records[start_frame].available_actions,
        final_action=firing_action,
        cell=oc16.CELL,
    )


def _choose_entry(
    records,
    start_frame: int,
    actor_views: list,
    entries: tuple[TriggerMemoryEntry, ...],
    cause_clause: tuple[str, ...],
    effect_signal: tuple[str, str, str],
    oc16,
) -> tuple[TriggerMemoryEntry | None, object | None]:
    candidates = [
        entry for entry in entries
        if entry.cause_clause == cause_clause and entry.effect_signal == effect_signal
    ]
    routed = []
    for entry in candidates:
        route = _route_entry(records, start_frame, actor_views, entry, oc16)
        if route is not None:
            routed.append((len(route.actions), -entry.support, entry.entry_id, entry, route))
    if routed:
        _length, _support, _entry_id, entry, route = min(routed)
        return entry, route
    if candidates:
        # Memory exists but no route. Pick the most supported entry for correctness
        # accounting, while reporting route_found=False.
        entry = max(candidates, key=lambda item: (item.support, -item.entry_id))
        return entry, None
    return None, None


def _decision(
    records,
    contact_frame: int,
    start_frame: int,
    signal: tuple[str, str, str],
    cause_clause: tuple[str, ...],
    entries: tuple[TriggerMemoryEntry, ...],
    actor_views: list,
    future_target,
    oc16,
) -> MultiMemoryDecision | None:
    trace_window = ewr32._trace_window(records, start_frame, contact_frame)
    if trace_window is None:
        return None
    candidates = tuple(
        entry for entry in entries
        if entry.cause_clause == cause_clause and entry.effect_signal == signal
    )
    selected, route = _choose_entry(records, start_frame, actor_views, entries, cause_clause, signal, oc16)
    route_actions = route.actions if route is not None else ()
    first, cpl, exact = ewr32.compare_route_to_window(route_actions, trace_window)
    last, csl = ewr32.compare_route_suffix(route_actions, trace_window)
    return MultiMemoryDecision(
        contact_frame=contact_frame,
        start_frame=start_frame,
        window=contact_frame - start_frame,
        desired_effect=signal,
        cause_clause=cause_clause,
        candidates=len(candidates),
        correct_candidate_available=any(_same_instance(entry.view, future_target) for entry in candidates),
        selected_entry_id=None if selected is None else selected.entry_id,
        selected_correct=_same_instance(None if selected is None else selected.view, future_target),
        route_found=route is not None,
        route_actions=route_actions,
        trace_window_actions=trace_window,
        first_action_match=first,
        last_action_match=last,
        common_prefix_len=cpl,
        common_suffix_len=csl,
        exact_match=exact,
    )


def evaluate_multi_memory_replay(
    trace: Path | None,
    levels: set[int],
    window: int,
    memory_matcher: str = "overlap",
) -> list[MultiMemoryDecision]:
    events, meta, _counts = rcf18._live_step_events(trace, levels)
    supported = ltr31._supported_effects(events)
    effects = tuple(pp27.LearnedTriggerEffect.from_remote_hypothesis(entry.hypothesis) for entry in supported)
    wanted_by_trace = ewr32._wanted_by_trace(events, meta, effects)

    oc16 = rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    decisions: list[MultiMemoryDecision] = []

    for seg in oc16.segments_for_levels(segments, levels):
        belief = oc16.AgentBelief()
        entries: tuple[TriggerMemoryEntry, ...] = ()
        next_id = 1
        entries_by_frame: dict[int, tuple[TriggerMemoryEntry, ...]] = {}
        actors_by_frame: dict[int, list] = {}
        future_target_by_event: dict[tuple[int, tuple[str, ...], tuple[str, str, str]], dict[int, object]] = {}

        for idx in range(seg.start, seg.end):
            grid = records[idx].frame
            _bg, parts = oc16.il.views_from_grid(grid, frame=idx)
            entries = _track_entries(entries, parts, memory_matcher, oc16, idx)
            actor_views = oc16._actor_views(parts, belief.controlled_sigs)
            entries_by_frame[idx] = entries
            actors_by_frame[idx] = actor_views

            if idx in wanted_by_trace:
                firing_action = records[idx + 1].action_num
                for signal, cause_clause in wanted_by_trace[idx]:
                    trigger_shape = ltr31._parse_trigger_shape(cause_clause)
                    contacted = ewr32._contacted_trigger(parts, actor_views, trigger_shape, firing_action, oc16)
                    if contacted is None:
                        continue
                    future_target_by_event[(idx, cause_clause, signal)] = ewr32._backward_chain(
                        records, idx, window, contacted, seg.start, oc16, "overlap"
                    )
                    entries, next_id = _remember_contact(
                        entries,
                        cause_clause=cause_clause,
                        effect_signal=signal,
                        contacted=contacted,
                        frame=idx,
                        next_id=next_id,
                    )

            if idx + 1 <= seg.end and records[idx + 1].action_num is not None:
                _bg_t, parts_t = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                _bg_t1, parts_t1 = oc16.il.views_from_grid(records[idx + 1].frame, frame=idx + 1)
                belief.learn_remote_dynamics(parts_t, parts_t1)
                belief.observe_transition(records[idx].frame, records[idx + 1].action_num, records[idx + 1].frame)
                belief.frame_idx += 1
                belief.perceive(records[idx + 1].frame)

        for contact_frame, wanted in wanted_by_trace.items():
            if not (seg.start <= contact_frame < seg.end):
                continue
            for signal, cause_clause in wanted:
                chain = future_target_by_event.get((contact_frame, cause_clause, signal), {})
                for w in range(0, window + 1):
                    start_frame = contact_frame - w
                    if start_frame < seg.start:
                        break
                    decision = _decision(
                        records,
                        contact_frame,
                        start_frame,
                        signal,
                        cause_clause,
                        entries_by_frame.get(start_frame, ()),
                        actors_by_frame.get(start_frame, []),
                        chain.get(start_frame),
                        oc16,
                    )
                    if decision is not None:
                        decisions.append(decision)
    return decisions


def _write_artifact(decisions: list[MultiMemoryDecision], levels: set[int]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-multi-memory-decisions.jsonl"
    with path.open("w") as fh:
        for d in decisions:
            fh.write(json.dumps({
                "candidates": d.candidates,
                "cause_clause": d.cause_clause,
                "common_prefix_len": d.common_prefix_len,
                "common_suffix_len": d.common_suffix_len,
                "contact_frame": d.contact_frame,
                "correct_candidate_available": d.correct_candidate_available,
                "desired_effect": d.desired_effect,
                "exact_match": d.exact_match,
                "first_action_match": d.first_action_match,
                "last_action_match": d.last_action_match,
                "route_actions": d.route_actions,
                "route_found": d.route_found,
                "selected_correct": d.selected_correct,
                "selected_entry_id": d.selected_entry_id,
                "start_frame": d.start_frame,
                "trace_window_actions": d.trace_window_actions,
                "window": d.window,
            }, sort_keys=True) + "\n")
    return path


def _table(decisions: list[MultiMemoryDecision]) -> dict[int, dict[str, float]]:
    by_window: dict[int, list[MultiMemoryDecision]] = defaultdict(list)
    for d in decisions:
        by_window[d.window].append(d)
    table = {}
    for w, rows in sorted(by_window.items()):
        n = len(rows)
        routed = [d for d in rows if d.route_found]
        with_candidates = [d for d in rows if d.candidates > 0]
        table[w] = {
            "n": n,
            "memory_rate": sum(d.candidates > 0 for d in rows) / n,
            "correct_available_rate": sum(d.correct_candidate_available for d in rows) / n,
            "selected_correct_rate": sum(d.selected_correct for d in rows) / n,
            "route_rate": sum(d.route_found for d in rows) / n,
            "route_given_memory": sum(d.route_found for d in with_candidates) / len(with_candidates) if with_candidates else 0.0,
            "suffix": sum(d.common_suffix_frac for d in routed) / len(routed) if routed else 0.0,
            "mean_candidates": sum(d.candidates for d in rows) / n,
        }
    return table


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    a = ltr31._FakeView({(0, 0), (0, 1)}, centroid=(0, 0), area=2)
    b = ltr31._FakeView({(5, 5), (5, 6)}, centroid=(5, 5), area=2)
    cause = ("cause",)
    signal = ("obj", "slot", "delta")
    entries: tuple[TriggerMemoryEntry, ...] = ()
    entries, next_id = _remember_contact(
        entries, cause_clause=cause, effect_signal=signal, contacted=a, frame=1, next_id=1
    )
    entries, next_id = _remember_contact(
        entries, cause_clause=cause, effect_signal=signal, contacted=b, frame=2, next_id=next_id
    )
    entries, next_id = _remember_contact(
        entries, cause_clause=cause, effect_signal=signal, contacted=a, frame=3, next_id=next_id
    )
    checks = [
        _check("1 keeps multiple instances for one cause", len(entries) == 2, f"entries={entries}"),
        _check("2 refreshes support for repeated same-instance contacts", max(e.support for e in entries) == 2, f"supports={[e.support for e in entries]}"),
    ]
    return all(checks)


def run_live(trace: Path | None, levels: set[int], window: int, memory_matcher: str) -> list[MultiMemoryDecision]:
    decisions = evaluate_multi_memory_replay(trace, levels, window, memory_matcher)
    artifact = _write_artifact(decisions, levels)
    print(f"\nLive ls20 multi-instance trigger memory replay (memory_matcher={memory_matcher}):")
    print(f"  decisions={len(decisions)} window=0..{window}")
    if decisions:
        print(f"    {'w':>2} {'n':>4} {'mem':>6} {'corr?':>6} {'pick':>6} {'route':>6} {'r|mem':>6} {'suffixΔ':>8} {'cand':>6}")
        for w, stats in _table(decisions).items():
            print(
                f"    {w:>2} {int(stats['n']):>4} {stats['memory_rate']:>6.3f} {stats['correct_available_rate']:>6.3f} "
                f"{stats['selected_correct_rate']:>6.3f} {stats['route_rate']:>6.3f} {stats['route_given_memory']:>6.3f} "
                f"{stats['suffix']:>8.3f} {stats['mean_candidates']:>6.2f}"
            )
    print(f"  memory_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate multi-instance causal trigger memory")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=6, help="max frames to step back before each trigger event")
    parser.add_argument("--memory-matcher", choices=("overlap", "continuity", "loose"), default="overlap")
    args = parser.parse_args()

    print("Offline checks (multi-instance trigger memory replay):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before live multi-instance memory replay.")
    print("\nMulti-instance memory can retain distinct trigger objects for one cause.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window, args.memory_matcher)


if __name__ == "__main__":
    main()
