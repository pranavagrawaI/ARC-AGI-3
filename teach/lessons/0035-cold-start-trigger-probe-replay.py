"""0035 - Cold-start trigger probe replay.

Run from the repo root:
    uv run python teach/lessons/0035-cold-start-trigger-probe-replay.py
    uv run python teach/lessons/0035-cold-start-trigger-probe-replay.py --live --levels 1,2,3,4,5,6,7 --window 6

0034 showed that routing to known causal memories is not the bottleneck. The gap
is cold start: before the agent has a remembered trigger instance, can it choose a
plausible object to probe?

This lesson stays replay-only and non-executing. At each future caused trigger
event, it looks several frames earlier and selects a probe target without using
the future contacted object:

    selector=shape         : routable small object matching the learned trigger_shape
    selector=nearest       : nearest routable small non-actor object
    selector=shape-nearest : shape if available, otherwise nearest exploration

The future contacted trigger is used only for scoring: did the chosen probe target
match the object the trace later contacted, and does the route's final approach
agree with the trace? This is not yet an agent. It is the cheapest falsifier for
the cold-start trigger selection problem.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0035-cold-start-trigger-probe-replay"


def _load_0034():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0034-multi-instance-trigger-memory-replay.py"
    spec = _ilu.spec_from_file_location("cstp35_mitm34", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mitm34 = _load_0034()
ewr32 = mitm34.ewr32
ltr31 = mitm34.ltr31
pp27 = mitm34.pp27
rcf18 = mitm34.rcf18


@dataclass(frozen=True)
class ProbeDecision:
    contact_frame: int
    start_frame: int
    window: int
    desired_effect: tuple[str, str, str]
    cause_clause: tuple[str, ...]
    selector: str
    candidates: int
    probe_found: bool
    probe_matches_future: bool
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


def _small_non_actor_candidates(parts: list, actor_views: list, oc16) -> list:
    max_area = 4 * getattr(oc16, "CELL", 5) ** 2
    actor_ids = {id(view) for view in actor_views}
    return [
        view for view in parts
        if id(view) not in actor_ids and 0 < getattr(view, "area", 0) <= max_area
    ]


def _target_matches_shape(view, trigger_shape: str | None, oc16) -> bool:
    return trigger_shape is not None and f"trigger_shape={oc16.ig.shape_literal(view)}" == trigger_shape


def _candidate_route(records, start_frame: int, actor_views: list, candidate, cause_clause: tuple[str, ...], oc16):
    final_action = ltr31._parse_action_clause(cause_clause)
    if final_action is None:
        return None
    return ewr32.route_to_trigger_grid(
        records[start_frame].frame,
        actor_views,
        candidate,
        oc16.ACTION_DELTAS,
        records[start_frame].available_actions,
        final_action=final_action,
        cell=oc16.CELL,
    )


def _select_probe(
    records,
    start_frame: int,
    parts: list,
    actor_views: list,
    cause_clause: tuple[str, ...],
    selector: str,
    oc16,
):
    trigger_shape = ltr31._parse_trigger_shape(cause_clause)
    candidates = _small_non_actor_candidates(parts, actor_views, oc16)
    if selector == "shape":
        candidates = [view for view in candidates if _target_matches_shape(view, trigger_shape, oc16)]
    elif selector == "shape-nearest":
        shape_candidates = [view for view in candidates if _target_matches_shape(view, trigger_shape, oc16)]
        if shape_candidates:
            candidates = shape_candidates
    elif selector != "nearest":
        raise ValueError(f"unknown selector: {selector!r}")

    routed = []
    for order, view in enumerate(candidates):
        route = _candidate_route(records, start_frame, actor_views, view, cause_clause, oc16)
        if route is not None:
            routed.append((len(route.actions), view.area, order, view, route))
    if not routed:
        return None, None, len(candidates)
    _length, _area, _order, view, route = min(routed)
    return view, route, len(candidates)


def _decision(
    records,
    start_frame: int,
    contact_frame: int,
    signal: tuple[str, str, str],
    cause_clause: tuple[str, ...],
    selector: str,
    selected,
    route,
    candidate_count: int,
    future_target,
) -> ProbeDecision | None:
    trace_window = ewr32._trace_window(records, start_frame, contact_frame)
    if trace_window is None:
        return None
    route_actions = route.actions if route is not None else ()
    first, cpl, exact = ewr32.compare_route_to_window(route_actions, trace_window)
    last, csl = ewr32.compare_route_suffix(route_actions, trace_window)
    return ProbeDecision(
        contact_frame=contact_frame,
        start_frame=start_frame,
        window=contact_frame - start_frame,
        desired_effect=signal,
        cause_clause=cause_clause,
        selector=selector,
        candidates=candidate_count,
        probe_found=selected is not None,
        probe_matches_future=mitm34._same_instance(selected, future_target),
        route_actions=route_actions,
        trace_window_actions=trace_window,
        first_action_match=first,
        last_action_match=last,
        common_prefix_len=cpl,
        common_suffix_len=csl,
        exact_match=exact,
    )


def evaluate_probe_replay(
    trace: Path | None,
    levels: set[int],
    window: int,
    selector: str,
) -> list[ProbeDecision]:
    events, meta, _counts = rcf18._live_step_events(trace, levels)
    supported = ltr31._supported_effects(events)
    effects = tuple(pp27.LearnedTriggerEffect.from_remote_hypothesis(entry.hypothesis) for entry in supported)
    wanted_by_trace = ewr32._wanted_by_trace(events, meta, effects)

    oc16 = rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    decisions: list[ProbeDecision] = []

    for seg in oc16.segments_for_levels(segments, levels):
        belief = oc16.AgentBelief()
        parts_by_frame: dict[int, list] = {}
        actors_by_frame: dict[int, list] = {}
        future_target_by_event: dict[tuple[int, tuple[str, ...], tuple[str, str, str]], dict[int, object]] = {}

        for idx in range(seg.start, seg.end):
            _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
            actor_views = oc16._actor_views(parts, belief.controlled_sigs)
            parts_by_frame[idx] = parts
            actors_by_frame[idx] = actor_views

            if idx in wanted_by_trace:
                firing_action = records[idx + 1].action_num
                for signal, cause_clause in wanted_by_trace[idx]:
                    trigger_shape = ltr31._parse_trigger_shape(cause_clause)
                    contacted = ewr32._contacted_trigger(parts, actor_views, trigger_shape, firing_action, oc16)
                    if contacted is not None:
                        future_target_by_event[(idx, cause_clause, signal)] = ewr32._backward_chain(
                            records, idx, window, contacted, seg.start, oc16, "overlap"
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
                    selected, route, candidate_count = _select_probe(
                        records,
                        start_frame,
                        parts_by_frame.get(start_frame, []),
                        actors_by_frame.get(start_frame, []),
                        cause_clause,
                        selector,
                        oc16,
                    )
                    decision = _decision(
                        records,
                        start_frame,
                        contact_frame,
                        signal,
                        cause_clause,
                        selector,
                        selected,
                        route,
                        candidate_count,
                        chain.get(start_frame),
                    )
                    if decision is not None:
                        decisions.append(decision)
    return decisions


def _write_artifact(decisions: list[ProbeDecision], levels: set[int], selector: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-{selector}-probe-decisions.jsonl"
    with path.open("w") as fh:
        for d in decisions:
            fh.write(json.dumps({
                "candidates": d.candidates,
                "cause_clause": d.cause_clause,
                "common_prefix_len": d.common_prefix_len,
                "common_suffix_len": d.common_suffix_len,
                "contact_frame": d.contact_frame,
                "desired_effect": d.desired_effect,
                "exact_match": d.exact_match,
                "first_action_match": d.first_action_match,
                "last_action_match": d.last_action_match,
                "probe_found": d.probe_found,
                "probe_matches_future": d.probe_matches_future,
                "route_actions": d.route_actions,
                "selector": d.selector,
                "start_frame": d.start_frame,
                "trace_window_actions": d.trace_window_actions,
                "window": d.window,
            }, sort_keys=True) + "\n")
    return path


def _table(decisions: list[ProbeDecision]) -> dict[int, dict[str, float]]:
    by_window: dict[int, list[ProbeDecision]] = defaultdict(list)
    for d in decisions:
        by_window[d.window].append(d)
    table = {}
    for w, rows in sorted(by_window.items()):
        n = len(rows)
        found = [d for d in rows if d.probe_found]
        table[w] = {
            "n": n,
            "probe_rate": sum(d.probe_found for d in rows) / n,
            "match_rate": sum(d.probe_matches_future for d in rows) / n,
            "match_given_probe": sum(d.probe_matches_future for d in found) / len(found) if found else 0.0,
            "last_rate": sum(d.last_action_match for d in rows) / n,
            "suffix": sum(d.common_suffix_frac for d in found) / len(found) if found else 0.0,
            "mean_candidates": sum(d.candidates for d in rows) / n,
        }
    return table


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    # The selector itself is live-perception dependent; keep the offline checks on
    # the small pure helpers so failures are local and interpretable.
    class FakeIg:
        @staticmethod
        def shape_literal(view):
            return view.shape

    class FakeOc16:
        CELL = 5
        ig = FakeIg()

    target = type("V", (), {"shape": "x", "area": 3})()
    decoy = type("V", (), {"shape": "y", "area": 3})()
    checks = [
        _check("1 shape predicate accepts matching trigger shape", _target_matches_shape(target, "trigger_shape=x", FakeOc16), "shape=x"),
        _check("2 shape predicate rejects decoys", not _target_matches_shape(decoy, "trigger_shape=x", FakeOc16), "shape=y"),
    ]
    return all(checks)


def run_live(trace: Path | None, levels: set[int], window: int, selector: str) -> list[ProbeDecision]:
    decisions = evaluate_probe_replay(trace, levels, window, selector)
    artifact = _write_artifact(decisions, levels, selector)
    print(f"\nLive ls20 cold-start trigger probe replay (selector={selector}):")
    print(f"  decisions={len(decisions)} window=0..{window}")
    if decisions:
        print(f"    {'w':>2} {'n':>4} {'probe':>6} {'match':>6} {'m|prb':>6} {'last':>6} {'suffixΔ':>8} {'cand':>6}")
        for w, stats in _table(decisions).items():
            print(
                f"    {w:>2} {int(stats['n']):>4} {stats['probe_rate']:>6.3f} {stats['match_rate']:>6.3f} "
                f"{stats['match_given_probe']:>6.3f} {stats['last_rate']:>6.3f} {stats['suffix']:>8.3f} {stats['mean_candidates']:>6.2f}"
            )
    print(f"  probe_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate cold-start trigger probe selection")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=6, help="max frames to step back before each trigger event")
    parser.add_argument("--selector", choices=("shape", "nearest", "shape-nearest"), default="shape")
    args = parser.parse_args()

    print("Offline checks (cold-start trigger probe replay):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before live probe replay.")
    print("\nCold-start probe helpers can filter by learned trigger shape.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window, args.selector)


if __name__ == "__main__":
    main()
