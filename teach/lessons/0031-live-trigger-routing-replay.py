"""0031 - Live trigger-routing replay.

Run from the repo root:
    uv run python teach/lessons/0031-live-trigger-routing-replay.py
    uv run python teach/lessons/0031-live-trigger-routing-replay.py --live --levels 1,2,3,4,5,6,7

0030 proved the closed-loop shape in a toy world. This lesson checks the next
live grounding seam without controlling ls20:

    For each caused supported remote effect in the successful trace, recover the
    trigger object from the learned cause clause and route the perceived actor to
    a pre-contact cell whose final action would fire that trigger.

This is still replay-only. It measures whether the trigger affordance can be
grounded spatially from pixels; it does not yet execute the route in the engine.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0031-live-trigger-routing-replay"


def _load_0029():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0029-counterfactual-descriptor-replay.py"
    spec = _ilu.spec_from_file_location("ltr31_cdr29", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cdr29 = _load_0029()
pp27 = cdr29.pp27
rcf18 = cdr29.rcf18


@dataclass(frozen=True)
class RoutePlan:
    path: tuple[int, ...]
    final_action: int
    target_cell: tuple[int, int]

    @property
    def actions(self) -> tuple[int, ...]:
        return self.path + (self.final_action,)

    @property
    def first_action(self) -> int:
        return self.actions[0]


@dataclass(frozen=True)
class RoutingDecision:
    frame: int
    desired_effect: tuple[str, str, str]
    cause_clause: tuple[str, ...]
    route_actions: tuple[int, ...]
    trace_action: int
    route_found: bool
    final_action_match: bool
    first_action_match: bool
    target_candidates: int


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _actor_centroid(actor_views: list) -> tuple[int, int] | None:
    if not actor_views:
        return None
    body_views = [view for view in actor_views if getattr(view, "area", 0) > 1]
    if body_views:
        actor_views = body_views
    return (
        round(sum(view.centroid[0] for view in actor_views) / len(actor_views)),
        round(sum(view.centroid[1] for view in actor_views) / len(actor_views)),
    )


def _parse_action_clause(cause_clause: tuple[str, ...]) -> int | None:
    for fact in cause_clause:
        if fact.startswith("action="):
            return int(fact.split("=", 1)[1])
    return None


def _parse_trigger_shape(cause_clause: tuple[str, ...]) -> str | None:
    for fact in cause_clause:
        if fact.startswith("trigger_shape="):
            return fact
    return None


def _is_free_cell(
    grid: np.ndarray,
    cell: tuple[int, int],
    *,
    bg: int,
    actor_pixels: frozenset[tuple[int, int]],
) -> bool:
    r, c = cell
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    return int(grid[r, c]) == bg or cell in actor_pixels


def route_to_trigger(
    grid: np.ndarray,
    actor_views: list,
    trigger_view,
    action_deltas: dict[int, tuple[int, int]],
    available_actions: list[int],
    *,
    final_action: int | None = None,
) -> RoutePlan | None:
    start = _actor_centroid(actor_views)
    if start is None:
        return None
    bg = _background(grid)
    body_views = [view for view in actor_views if getattr(view, "area", 0) > 1] or actor_views
    actor_pixels = frozenset(pixel for view in body_views for pixel in view.pixels)
    allowed_final = [final_action] if final_action is not None else list(available_actions)

    goals: dict[tuple[int, int], int] = {}
    for action in allowed_final:
        if action not in action_deltas:
            continue
        dr, dc = action_deltas[action]
        for r, c in trigger_view.pixels:
            approach = (int(r - dr), int(c - dc))
            if _is_free_cell(grid, approach, bg=bg, actor_pixels=actor_pixels):
                goals.setdefault(approach, action)

    if start in goals:
        return RoutePlan((), goals[start], start)

    seen = {start}
    q = deque([(start, ())])
    while q:
        cell, path = q.popleft()
        for action in available_actions:
            if action not in action_deltas:
                continue
            dr, dc = action_deltas[action]
            nxt = (cell[0] + int(dr), cell[1] + int(dc))
            if nxt in seen or not _is_free_cell(grid, nxt, bg=bg, actor_pixels=actor_pixels):
                continue
            npath = path + (action,)
            if nxt in goals:
                return RoutePlan(npath, goals[nxt], nxt)
            seen.add(nxt)
            q.append((nxt, npath))
    return None


def _trigger_candidates(parts: list, actor_views: list, trigger_shape_fact: str | None, oc16) -> list:
    if trigger_shape_fact is None:
        return []
    max_trigger_area = 4 * getattr(oc16, "CELL", 5) ** 2
    actor_ids = {id(view) for view in actor_views}
    return [
        view for view in parts
        if id(view) not in actor_ids
        and view.area <= max_trigger_area
        and f"trigger_shape={oc16.ig.shape_literal(view)}" == trigger_shape_fact
    ]


def _best_route_for_cause(
    grid: np.ndarray,
    parts: list,
    actor_views: list,
    cause_clause: tuple[str, ...],
    records_action: int,
    available_actions: list[int],
    oc16,
) -> tuple[RoutePlan | None, int]:
    trigger_shape = _parse_trigger_shape(cause_clause)
    final_action = _parse_action_clause(cause_clause)
    candidates = _trigger_candidates(parts, actor_views, trigger_shape, oc16)
    if not candidates:
        return None, 0

    # Prefer the trigger object that the trace action actually contacted when it
    # is among the matching candidates; otherwise choose the shortest route.
    start = _actor_centroid(actor_views)
    trace_target = None
    if start is not None and records_action in oc16.ACTION_DELTAS:
        dr, dc = oc16.ACTION_DELTAS[records_action]
        trace_target = (start[0] + int(dr), start[1] + int(dc))
    ordered = sorted(
        candidates,
        key=lambda view: trace_target not in view.pixels if trace_target is not None else True,
    )
    routes = [
        route_to_trigger(
            grid,
            actor_views,
            candidate,
            oc16.ACTION_DELTAS,
            available_actions,
            final_action=final_action,
        )
        for candidate in ordered
    ]
    routes = [route for route in routes if route is not None]
    if not routes:
        return None, len(candidates)
    return min(routes, key=lambda route: (len(route.actions), route.final_action != records_action)), len(candidates)


def _supported_effects(events) -> tuple:
    ledger = rcf18.falsify_remote_hypotheses(events, max_depth=2, min_support=2)
    return tuple(entry for entry in ledger if entry.status == "SUPPORTED")


def evaluate_live_routing_replay(trace: Path | None, levels: set[int]) -> list[RoutingDecision]:
    events, meta, _counts = rcf18._live_step_events(trace, levels)
    supported = _supported_effects(events)
    effects = tuple(pp27.LearnedTriggerEffect.from_remote_hypothesis(entry.hypothesis) for entry in supported)
    supported_by_signal = {entry.hypothesis.effect: entry for entry in supported}
    wanted_by_trace: dict[int, list[tuple[tuple[str, str, str], tuple[str, ...]]]] = {}
    for event, info in zip(events, meta, strict=True):
        trace_action = cdr29._trace_action_for_event(event)
        for effect in effects:
            signal = (effect.object_id, effect.slot, effect.delta)
            if not cdr29._effect_present(event, signal):
                continue
            if signal not in cdr29._signals_for_action(trace_action, effects):
                continue
            wanted_by_trace.setdefault(info["trace_index"], []).append((signal, supported_by_signal[signal].hypothesis.cause_clause))

    oc16 = rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    decisions: list[RoutingDecision] = []

    for seg in oc16.segments_for_levels(segments, levels):
        belief = oc16.AgentBelief()
        for idx in range(seg.start, seg.end):
            if idx in wanted_by_trace:
                grid = records[idx].frame
                _bg, parts = oc16.il.views_from_grid(grid, frame=idx)
                actor_views = oc16._actor_views(parts, belief.controlled_sigs)
                for signal, cause_clause in wanted_by_trace[idx]:
                    route, target_candidates = _best_route_for_cause(
                        grid,
                        parts,
                        actor_views,
                        cause_clause,
                        records[idx + 1].action_num,
                        records[idx].available_actions,
                        oc16,
                    )
                    route_actions = route.actions if route is not None else ()
                    decisions.append(RoutingDecision(
                        frame=idx,
                        desired_effect=signal,
                        cause_clause=cause_clause,
                        route_actions=route_actions,
                        trace_action=records[idx + 1].action_num,
                        route_found=route is not None,
                        final_action_match=route is not None and route.final_action == records[idx + 1].action_num,
                        first_action_match=route is not None and route.first_action == records[idx + 1].action_num,
                        target_candidates=target_candidates,
                    ))

            if idx + 1 <= seg.end and records[idx + 1].action_num is not None:
                _bg_t, parts_t = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                _bg_t1, parts_t1 = oc16.il.views_from_grid(records[idx + 1].frame, frame=idx + 1)
                belief.learn_remote_dynamics(parts_t, parts_t1)
                belief.observe_transition(records[idx].frame, records[idx + 1].action_num, records[idx + 1].frame)
                belief.frame_idx += 1
                belief.perceive(records[idx + 1].frame)
    return decisions


def _write_live_artifact(decisions: list[RoutingDecision], levels: set[int]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-routing-decisions.jsonl"
    with path.open("w") as fh:
        for decision in decisions:
            fh.write(json.dumps({
                "cause_clause": decision.cause_clause,
                "desired_effect": decision.desired_effect,
                "final_action_match": decision.final_action_match,
                "first_action_match": decision.first_action_match,
                "frame": decision.frame,
                "route_actions": decision.route_actions,
                "route_found": decision.route_found,
                "target_candidates": decision.target_candidates,
                "trace_action": decision.trace_action,
            }, sort_keys=True) + "\n")
    return path


class _FakeView:
    def __init__(self, pixels, centroid=None, area=None):
        self.pixels = frozenset(pixels)
        self.centroid = centroid if centroid is not None else next(iter(self.pixels))
        self.area = area if area is not None else len(self.pixels)


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    grid = np.zeros((15, 15), dtype=np.int8)
    actor = _FakeView({(10, 5)}, centroid=(10, 5))
    trigger = _FakeView({(5, 10)}, centroid=(5, 10))
    deltas = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
    route = route_to_trigger(grid, [actor], trigger, deltas, [1, 2, 3, 4], final_action=4)
    checks.append(_check(
        "1 routes to a pre-contact cell then fires constrained action",
        route is not None and route.actions == (1, 4),
        f"route={route}",
    ))

    ready = route_to_trigger(grid, [_FakeView({(5, 5)}, centroid=(5, 5))], trigger, deltas, [1, 2, 3, 4], final_action=4)
    checks.append(_check(
        "2 already-adjacent route is just the final trigger action",
        ready is not None and ready.actions == (4,),
        f"route={ready}",
    ))

    blocked = grid.copy()
    blocked[5, 5] = 9
    blocked_route = route_to_trigger(blocked, [_FakeView({(10, 5)}, centroid=(10, 5))], trigger, deltas, [1, 2, 3, 4], final_action=4)
    checks.append(_check(
        "3 blocked approach fails loud",
        blocked_route is None,
        f"route={blocked_route}",
    ))
    return all(checks)


def run_live(trace: Path | None, levels: set[int]) -> list[RoutingDecision]:
    decisions = evaluate_live_routing_replay(trace, levels)
    artifact = _write_live_artifact(decisions, levels)
    counts = Counter((d.route_found, d.final_action_match, d.first_action_match) for d in decisions)
    print("\nLive ls20 trigger-routing replay:")
    print(f"  decisions={len(decisions)} route/final/first_counts={dict(counts)}")
    if decisions:
        route_rate = sum(d.route_found for d in decisions) / len(decisions)
        final_rate = sum(d.final_action_match for d in decisions) / len(decisions)
        first_rate = sum(d.first_action_match for d in decisions) / len(decisions)
        print(f"  route_found_rate={route_rate:.3f} final_action_match_rate={final_rate:.3f} first_action_match_rate={first_rate:.3f}")
        for decision in decisions[:10]:
            print(
                f"  frame={decision.frame} trace=ACTION{decision.trace_action} route={[f'ACTION{a}' for a in decision.route_actions]} "
                f"candidates={decision.target_candidates} final_match={decision.final_action_match} first_match={decision.first_action_match}"
            )
    else:
        print("  no caused supported effects available for routing replay")
    print(f"  routing_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate routing to supported trace triggers")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    args = parser.parse_args()

    print("Offline checks (trigger-routing replay):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using live trigger-routing replay.")
    print("\nTrigger-routing replay can route to pre-contact cells for learned trigger causes.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels)


if __name__ == "__main__":
    main()
