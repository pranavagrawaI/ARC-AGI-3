"""0036 - Information/risk cold-start trigger probe replay.

Run from the repo root:
    uv run python teach/lessons/0036-info-risk-cold-start-probe-replay.py
    uv run python teach/lessons/0036-info-risk-cold-start-probe-replay.py --live --levels 1,2,3,4,5,6,7 --window 6

0035 showed that naive cold-start probing is weak:

    shape   : precise when the learned trigger shape is visible, sparse otherwise
    nearest : almost always the wrong causal object in a crowded scene

This lesson adds the next minimal agent-ish selector. It keeps contact-outcome
memory by observed trigger shape, then scores fallback probes by:

    novelty      - prefer shapes with few prior contacts
    remote value - prefer shapes that previously produced remote effects
    risk penalty - avoid shapes whose contacts produced teleport-like displacement
    cost         - prefer shorter routes all else equal

It is still replay-only: the selector does not execute probes. It chooses from the
current frame without reading the future contacted target; that future target is
used only to score whether the proposed cold-start probe was equivalent to the
trace contact.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0036-info-risk-cold-start-probe-replay"


def _load_0035():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0035-cold-start-trigger-probe-replay.py"
    spec = _ilu.spec_from_file_location("ircsp36_cstp35", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cstp35 = _load_0035()
ewr32 = cstp35.ewr32
mitm34 = cstp35.mitm34
ltr31 = cstp35.ltr31
pp27 = cstp35.pp27
rcf18 = cstp35.rcf18


@dataclass
class ShapeContactStats:
    contacts: int = 0
    remote_effects: int = 0
    teleport_like: int = 0

    @property
    def risk_rate(self) -> float:
        return self.teleport_like / self.contacts if self.contacts else 0.0

    @property
    def remote_rate(self) -> float:
        return self.remote_effects / self.contacts if self.contacts else 0.0


@dataclass(frozen=True)
class InfoRiskDecision:
    contact_frame: int
    start_frame: int
    window: int
    selector: str
    desired_effect: tuple[str, str, str]
    cause_clause: tuple[str, ...]
    candidates: int
    selected_shape: str | None
    selected_contacts: int
    selected_risk: float
    selected_remote_rate: float
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
    def common_suffix_frac(self) -> float:
        return self.common_suffix_len / self.route_len if self.route_len else 0.0


def _actor_centroid(actor_views: list) -> tuple[float, float] | None:
    if not actor_views:
        return None
    return (
        sum(view.centroid[0] for view in actor_views) / len(actor_views),
        sum(view.centroid[1] for view in actor_views) / len(actor_views),
    )


def _view_ahead(parts: list, actor_views: list, action: int, oc16):
    center = _actor_centroid(actor_views)
    if center is None or action not in oc16.ACTION_DELTAS:
        return None
    dr, dc = oc16.ACTION_DELTAS[action]
    cell = (round(center[0] + dr), round(center[1] + dc))
    for view in parts:
        if cell in view.pixels:
            return view
    return None


def _actor_shift(prev_actor_views: list, next_actor_views: list) -> float | None:
    a = _actor_centroid(prev_actor_views)
    b = _actor_centroid(next_actor_views)
    if a is None or b is None:
        return None
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _update_shape_stats(
    stats_by_shape: dict[str, ShapeContactStats],
    *,
    records,
    frame: int,
    parts_t: list,
    parts_t1: list,
    actor_views_t: list,
    actor_views_t1: list,
    supported_signals: set[tuple[str, str, str]],
    oc16,
) -> None:
    action = records[frame + 1].action_num
    if action is None:
        return
    target = _view_ahead(parts_t, actor_views_t, action, oc16)
    if target is None or oc16.ig.is_actor_view(target, set()):
        return
    max_area = 4 * getattr(oc16, "CELL", 5) ** 2
    if target.area > max_area:
        return
    shape = oc16.ig.shape_literal(target)
    stats = stats_by_shape[shape]
    stats.contacts += 1
    event_signals = set()
    # Reconstruct only the supported/remote-present bit cheaply: if this frame is
    # a supported trace event, the caller supplies its signals in supported_signals.
    event_signals.update(supported_signals)
    stats.remote_effects += int(bool(event_signals))
    shift = _actor_shift(actor_views_t, actor_views_t1)
    if shift is None or shift > oc16.CELL * 1.6:
        stats.teleport_like += 1


def _shape_literal(view, oc16) -> str:
    return oc16.ig.shape_literal(view)


def _score_info_risk(
    *,
    view,
    route,
    cause_clause: tuple[str, ...],
    stats_by_shape: dict[str, ShapeContactStats],
    oc16,
) -> tuple[float, tuple[float, int, int]]:
    shape = _shape_literal(view, oc16)
    stats = stats_by_shape.get(shape, ShapeContactStats())
    trigger_shape = ltr31._parse_trigger_shape(cause_clause)
    shape_match_bonus = 3.0 if cstp35._target_matches_shape(view, trigger_shape, oc16) else 0.0
    novelty = 1.0 / (1 + stats.contacts)
    remote_value = stats.remote_rate
    risk = stats.risk_rate
    cost = len(route.actions)
    score = shape_match_bonus + novelty + remote_value - (2.0 * risk) - (0.05 * cost)
    # Tie-break toward short, small probes deterministically.
    return score, (-cost, -view.area, -id(view))


def _select_probe(
    records,
    start_frame: int,
    parts: list,
    actor_views: list,
    cause_clause: tuple[str, ...],
    selector: str,
    stats_by_shape: dict[str, ShapeContactStats],
    oc16,
):
    if selector in {"shape", "nearest", "shape-nearest"}:
        return cstp35._select_probe(records, start_frame, parts, actor_views, cause_clause, selector, oc16)
    if selector != "info-risk":
        raise ValueError(f"unknown selector: {selector!r}")

    candidates = cstp35._small_non_actor_candidates(parts, actor_views, oc16)
    ranked = []
    for order, view in enumerate(candidates):
        route = cstp35._candidate_route(records, start_frame, actor_views, view, cause_clause, oc16)
        if route is None:
            continue
        score, tie = _score_info_risk(
            view=view,
            route=route,
            cause_clause=cause_clause,
            stats_by_shape=stats_by_shape,
            oc16=oc16,
        )
        ranked.append((score, tie, -order, view, route))
    if not ranked:
        return None, None, len(candidates)
    _score, _tie, _order, view, route = max(ranked)
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
    stats_by_shape: dict[str, ShapeContactStats],
    oc16,
) -> InfoRiskDecision | None:
    trace_window = ewr32._trace_window(records, start_frame, contact_frame)
    if trace_window is None:
        return None
    route_actions = route.actions if route is not None else ()
    first, cpl, exact = ewr32.compare_route_to_window(route_actions, trace_window)
    last, csl = ewr32.compare_route_suffix(route_actions, trace_window)
    selected_shape = _shape_literal(selected, oc16) if selected is not None else None
    stats = stats_by_shape.get(selected_shape or "", ShapeContactStats())
    return InfoRiskDecision(
        contact_frame=contact_frame,
        start_frame=start_frame,
        window=contact_frame - start_frame,
        selector=selector,
        desired_effect=signal,
        cause_clause=cause_clause,
        candidates=candidate_count,
        selected_shape=selected_shape,
        selected_contacts=stats.contacts,
        selected_risk=stats.risk_rate,
        selected_remote_rate=stats.remote_rate,
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


def evaluate_info_risk_replay(
    trace: Path | None,
    levels: set[int],
    window: int,
    selector: str,
) -> list[InfoRiskDecision]:
    events, meta, _counts = rcf18._live_step_events(trace, levels)
    supported = ltr31._supported_effects(events)
    effects = tuple(pp27.LearnedTriggerEffect.from_remote_hypothesis(entry.hypothesis) for entry in supported)
    wanted_by_trace = ewr32._wanted_by_trace(events, meta, effects)

    signal_by_frame: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    for event, info in zip(events, meta, strict=True):
        trace_action = ewr32.cdr29._trace_action_for_event(event)
        for effect in effects:
            signal = (effect.object_id, effect.slot, effect.delta)
            if ewr32.cdr29._effect_present(event, signal) and signal in ewr32.cdr29._signals_for_action(trace_action, effects):
                signal_by_frame[info["trace_index"]].add(signal)

    oc16 = rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    decisions: list[InfoRiskDecision] = []

    for seg in oc16.segments_for_levels(segments, levels):
        belief = oc16.AgentBelief()
        stats_by_shape: dict[str, ShapeContactStats] = defaultdict(ShapeContactStats)
        parts_by_frame: dict[int, list] = {}
        actors_by_frame: dict[int, list] = {}
        stats_by_frame: dict[int, dict[str, ShapeContactStats]] = {}
        future_target_by_event: dict[tuple[int, tuple[str, ...], tuple[str, str, str]], dict[int, object]] = {}

        for idx in range(seg.start, seg.end):
            _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
            actor_views = oc16._actor_views(parts, belief.controlled_sigs)
            parts_by_frame[idx] = parts
            actors_by_frame[idx] = actor_views
            stats_by_frame[idx] = {
                shape: ShapeContactStats(s.contacts, s.remote_effects, s.teleport_like)
                for shape, s in stats_by_shape.items()
            }

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
                actors_t = oc16._actor_views(parts_t, belief.controlled_sigs)
                actors_t1 = oc16._actor_views(parts_t1, belief.controlled_sigs)
                _update_shape_stats(
                    stats_by_shape,
                    records=records,
                    frame=idx,
                    parts_t=parts_t,
                    parts_t1=parts_t1,
                    actor_views_t=actors_t,
                    actor_views_t1=actors_t1,
                    supported_signals=signal_by_frame.get(idx, set()),
                    oc16=oc16,
                )
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
                    frame_stats = stats_by_frame.get(start_frame, {})
                    selected, route, candidate_count = _select_probe(
                        records,
                        start_frame,
                        parts_by_frame.get(start_frame, []),
                        actors_by_frame.get(start_frame, []),
                        cause_clause,
                        selector,
                        frame_stats,
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
                        frame_stats,
                        oc16,
                    )
                    if decision is not None:
                        decisions.append(decision)
    return decisions


def _write_artifact(decisions: list[InfoRiskDecision], levels: set[int], selector: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-{selector}-info-risk-decisions.jsonl"
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
                "selected_contacts": d.selected_contacts,
                "selected_remote_rate": d.selected_remote_rate,
                "selected_risk": d.selected_risk,
                "selected_shape": d.selected_shape,
                "selector": d.selector,
                "start_frame": d.start_frame,
                "trace_window_actions": d.trace_window_actions,
                "window": d.window,
            }, sort_keys=True) + "\n")
    return path


def _table(decisions: list[InfoRiskDecision]) -> dict[int, dict[str, float]]:
    by_window: dict[int, list[InfoRiskDecision]] = defaultdict(list)
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
            "mean_contacts": sum(d.selected_contacts for d in found) / len(found) if found else 0.0,
            "mean_risk": sum(d.selected_risk for d in found) / len(found) if found else 0.0,
        }
    return table


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    stats = ShapeContactStats(contacts=2, remote_effects=1, teleport_like=1)
    checks = [
        _check("1 risk rate is contact-normalized", stats.risk_rate == 0.5, f"risk={stats.risk_rate}"),
        _check("2 remote rate is contact-normalized", stats.remote_rate == 0.5, f"remote={stats.remote_rate}"),
    ]
    return all(checks)


def run_live(trace: Path | None, levels: set[int], window: int, selector: str) -> list[InfoRiskDecision]:
    decisions = evaluate_info_risk_replay(trace, levels, window, selector)
    artifact = _write_artifact(decisions, levels, selector)
    print(f"\nLive ls20 info/risk cold-start probe replay (selector={selector}):")
    print(f"  decisions={len(decisions)} window=0..{window}")
    if decisions:
        print(f"    {'w':>2} {'n':>4} {'probe':>6} {'match':>6} {'m|prb':>6} {'last':>6} {'suffixΔ':>8} {'cand':>6} {'seen':>6} {'risk':>6}")
        for w, stats in _table(decisions).items():
            print(
                f"    {w:>2} {int(stats['n']):>4} {stats['probe_rate']:>6.3f} {stats['match_rate']:>6.3f} "
                f"{stats['match_given_probe']:>6.3f} {stats['last_rate']:>6.3f} {stats['suffix']:>8.3f} "
                f"{stats['mean_candidates']:>6.2f} {stats['mean_contacts']:>6.2f} {stats['mean_risk']:>6.3f}"
            )
    print(f"  info_risk_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate info/risk cold-start probe selection")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=6, help="max frames to step back before each trigger event")
    parser.add_argument("--selector", choices=("shape", "nearest", "shape-nearest", "info-risk"), default="info-risk")
    args = parser.parse_args()

    print("Offline checks (info/risk cold-start probe replay):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before live info/risk replay.")
    print("\nInfo/risk probe helpers can score prior contact outcomes.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window, args.selector)


if __name__ == "__main__":
    main()
