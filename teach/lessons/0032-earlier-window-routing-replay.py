"""0032 - Earlier-window trigger-routing replay.

Run from the repo root:
    uv run python teach/lessons/0032-earlier-window-routing-replay.py
    uv run python teach/lessons/0032-earlier-window-routing-replay.py --live --levels 1,2,3,4,5,6,7
    uv run python teach/lessons/0032-earlier-window-routing-replay.py --live --window 6

0031 routed to the trigger *at the pre-contact frame*, so every route collapsed to
the single firing action (route length 1, trivially matching the trace). That tells
us nothing about whether the trigger affordance can be planned for from a distance.

This lesson moves the planning origin earlier:

    For each caused supported remote effect that fires at frame `idx` (via action
    records[idx+1]), step back w frames to e = idx - w. At e the actor has not yet
    contacted anything. Recover the *future* trigger target -- the specific trigger
    instance the trace will later contact -- by pixel-identity with the trigger that
    is actually contacted at idx (triggers are assumed static within a segment).
    Route the actor at e to that recovered trigger, then ask: does the planned
    route's prefix agree with what the human trace actually did over [e .. idx]?

Hypothesis: a trigger affordance grounded from pixels is plannable several frames
ahead, and a shortest-path route to it reproduces the leading actions of the human
trajectory. Signal: first-action agreement and common-prefix length should stay
high as the window grows; they will decay if (a) the trigger cannot be re-identified
earlier, (b) the human took a non-direct path, or (c) perceived geometry disagrees
with the world. Each failure mode is reported, not smoothed.

Knobs that turn this into a falsifier rather than a single number:
  --recovery-mode {shape,position,track}
      How to re-identify the future trigger at the earlier frame. "shape" is
      0031's frame-local shape literal (churns badly); "position" is spatial
      overlap at the contact-frame location (the trigger also moves); "track"
      follows the trigger backward by cross-frame identity.
  --track-matcher {strict,overlap,continuity,loose} / --track-iou
      "loose" is 0016's matcher, which falls back to the nearest centroid and so
      almost never admits a loss. "strict" gates on real pixel IoU and returns
      nothing when the trigger is lost. An IoU sweep shows recovery is a cliff:
      ~0.1-0.3 at IoU>=0.3, ~0.7-0.9 at IoU=0.1, and 1.0 at IoU<=0.05 (== "any
      overlap"). So the trigger never fully leaves a 6-frame window; it just morphs
      enough that its frame-to-frame self-overlap is tiny (~0.05-0.1 IoU). The
      recoveries below IoU=0.1 are real (route_found and suffix agreement rise with
      them), so default 0.1 under-counts; rigid IoU is simply a weak identity
      signal for these heavily-animating triggers. "overlap" names the useful
      endpoint of that sweep: require any nonzero overlap, but do not require a
      large overlap fraction. "continuity" scores centroid distance + area
      continuity, with overlap as a bonus rather than a gate; it is the first
      tracker here that treats morphing as expected instead of anomalous, though
      on this trace it can follow plausible wrong neighbors at depth.
  --route-target {current,contact}
      Aim the route at the trigger's earlier position ("current") or at the
      contact-frame bump site ("contact", a naive interception heuristic for
      moving triggers -- which produces long detours and is reported, not hidden).

Two comparison alignments are reported per window:
  prefixΔ aligns plan vs trace from the decision point (does early planning pick
      the human's first moves?); suffixΔ aligns them at the bump (does the plan's
      final approach match the human's, even if the human wandered earlier?).
Empirically the plan's *suffix* agrees strongly while its *prefix* does not: the
planner is right about how to finish at the trigger; the human just took an
indirect path to get there.

Current canonical settings for the replay micro-skill are:

    --recovery-mode track --track-matcher overlap --router grid --route-target current

These recover the trigger in every tested earlier window and keep suffix
agreement near 1.0 while still failing when no pixel overlap exists.

Still replay-only: routes are scored against the trace, never executed in ls20.
The controllability signal (which sprite is the avatar) is taken as of the contact
frame and reused for the earlier frames in the same segment -- actor identity is a
stable game property, so this isolates the spatial-routing question from
controllability-learning latency. That reuse is the one piece of mild hindsight here
and is called out so it can be challenged.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0032-earlier-window-routing-replay"


def _load_0031():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0031-live-trigger-routing-replay.py"
    spec = _ilu.spec_from_file_location("ewr32_ltr31", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ltr31 = _load_0031()
cdr29 = ltr31.cdr29
pp27 = ltr31.pp27
rcf18 = ltr31.rcf18


@dataclass(frozen=True)
class EarlierWindowDecision:
    contact_frame: int
    start_frame: int
    window: int
    desired_effect: tuple[str, str, str]
    cause_clause: tuple[str, ...]
    firing_action: int
    candidates: int
    target_recovered: bool
    route_found: bool
    route_actions: tuple[int, ...]
    trace_window_actions: tuple[int, ...]
    first_action_match: bool
    final_action_match: bool
    common_prefix_len: int
    common_suffix_len: int
    last_action_match: bool
    route_matches_window: bool
    route_no_longer_than_window: bool

    @property
    def route_len(self) -> int:
        return len(self.route_actions)

    @property
    def window_len(self) -> int:
        return len(self.trace_window_actions)

    @property
    def common_prefix_frac(self) -> float:
        return self.common_prefix_len / self.route_len if self.route_len else 0.0

    @property
    def common_suffix_frac(self) -> float:
        return self.common_suffix_len / self.route_len if self.route_len else 0.0


# ---------------------------------------------------------------------------
# Pure helpers (engine-free; exercised by offline checks)
# ---------------------------------------------------------------------------
def _common_prefix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _common_suffix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return _common_prefix_len(tuple(reversed(a)), tuple(reversed(b)))


def _match_iou(view, parts: list, min_iou: float):
    """Honest cross-frame identity: best pixel-IoU match, no centroid fallback.

    Returns None when nothing overlaps the tracked view enough. Unlike 0016's
    matcher (which falls back to the nearest centroid and almost never returns
    None on a populated frame), this admits when the trigger has been lost.
    """
    best, best_iou = None, 0.0
    for part in parts:
        inter = len(view.pixels & part.pixels)
        if not inter:
            continue
        union = len(view.pixels | part.pixels)
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best, best_iou = part, iou
    return best if best_iou >= min_iou else None


def _match_overlap(view, parts: list):
    """Cross-frame identity by any nonzero pixel overlap."""
    best, best_overlap = None, 0
    for part in parts:
        overlap = len(view.pixels & part.pixels)
        if overlap > best_overlap:
            best, best_overlap = part, overlap
    return best if best_overlap > 0 else None


def _centroid_distance(a, b) -> float:
    return abs(float(a.centroid[0]) - float(b.centroid[0])) + abs(float(a.centroid[1]) - float(b.centroid[1]))


def _area_delta_frac(a, b) -> float:
    denom = max(float(getattr(a, "area", 0)), float(getattr(b, "area", 0)), 1.0)
    return abs(float(getattr(a, "area", 0)) - float(getattr(b, "area", 0))) / denom


def _pixel_iou(a, b) -> float:
    union = len(a.pixels | b.pixels)
    return len(a.pixels & b.pixels) / union if union else 0.0


def _match_continuity(view, parts: list, cell: int):
    """Cross-frame identity for morphing triggers.

    Rigid IoU under-counts heavily-animating triggers: true frame-to-frame
    self-overlap can be ~0.05. This matcher treats identity as continuity: stay
    spatially near the previous centroid, keep roughly the same scale, and use
    any overlap as a bonus rather than a hard gate.
    """
    max_dist = 3 * cell
    scored = []
    for part in parts:
        dist = _centroid_distance(view, part)
        if dist > max_dist:
            continue
        area_delta = _area_delta_frac(view, part)
        iou = _pixel_iou(view, part)
        score = (dist / max_dist) + (0.5 * area_delta) - (0.25 * iou)
        scored.append((score, part))
    if not scored:
        return None
    scored.sort(key=lambda row: row[0])
    _best_score, best = scored[0]
    return best


def compare_route_to_window(
    route_actions: tuple[int, ...],
    trace_window_actions: tuple[int, ...],
) -> tuple[bool, int, bool]:
    """Compare a planned route against the trace's action window, from the start.

    Returns (first_action_match, common_prefix_len, exact_match). Both sequences
    end at the same firing event, so a route shorter than the window means the
    human took a non-direct path; common_prefix_len measures how far the human
    followed the plan before diverging.
    """
    cpl = _common_prefix_len(route_actions, trace_window_actions)
    first_match = (
        bool(route_actions)
        and bool(trace_window_actions)
        and route_actions[0] == trace_window_actions[0]
    )
    exact = tuple(route_actions) == tuple(trace_window_actions)
    return first_match, cpl, exact


def compare_route_suffix(
    route_actions: tuple[int, ...],
    trace_window_actions: tuple[int, ...],
) -> tuple[bool, int]:
    """Compare aligned at the contact event (the end), not the decision point.

    Both sequences end with the firing action, so a long common suffix means the
    planned approach to the trigger matches the human's final approach even if the
    human wandered earlier. Returns (last_action_match, common_suffix_len).
    """
    csl = _common_suffix_len(route_actions, trace_window_actions)
    last_match = (
        bool(route_actions)
        and bool(trace_window_actions)
        and route_actions[-1] == trace_window_actions[-1]
    )
    return last_match, csl


def recover_future_target(candidates: list, contacted_pixels: frozenset[tuple[int, int]]):
    """Pick the candidate at the earlier frame that is the trace-contacted trigger.

    Identity by maximum pixel overlap with the trigger contacted at the contact
    frame (static-trigger assumption). Returns None if nothing overlaps, which is
    itself the measurement that the trigger could not be re-identified earlier.
    """
    best, best_overlap = None, 0
    for view in candidates:
        overlap = len(view.pixels & contacted_pixels)
        if overlap > best_overlap:
            best, best_overlap = view, overlap
    return best if best_overlap > 0 else None


def _to_cell(coord: tuple[float, float], cell: int) -> tuple[int, int]:
    return (int(round(coord[0] / cell)), int(round(coord[1] / cell)))


def route_to_trigger_grid(
    grid,
    actor_views: list,
    trigger_view,
    action_deltas: dict[int, tuple[int, int]],
    available_actions: list[int],
    *,
    final_action: int | None,
    cell: int,
):
    """CELL-quantized router: the bug-fixed counterpart of ltr31.route_to_trigger.

    The avatar moves exactly one CELL per action, so its reachable positions form
    a single CELL-lattice. The pixel-resolution router requires BFS (stepping by
    CELL) to land *exactly* on `trigger_pixel - delta`, which is impossible when
    the trigger's pixels are not CELL-congruent with the avatar -- the cause of the
    odd/even route_found parity. Here we quantize the avatar, the trigger, and the
    moves to integer CELL coordinates, so a sub-CELL jitter cannot make a reachable
    cell unreachable. Returns a RoutePlan whose actions are real game actions.
    """
    start = ltr31._actor_centroid(actor_views)
    if start is None:
        return None
    bg = ltr31._background(grid)
    body = [v for v in actor_views if getattr(v, "area", 0) > 1] or actor_views
    actor_cells = {_to_cell(p, cell) for v in body for p in v.pixels}
    start_cell = _to_cell(start, cell)
    grid_deltas = {a: (int(round(dr / cell)), int(round(dc / cell))) for a, (dr, dc) in action_deltas.items()}
    trigger_cells = {_to_cell(p, cell) for p in trigger_view.pixels}
    allowed_final = [final_action] if final_action is not None else list(available_actions)

    h, w = grid.shape

    def free(gc: tuple[int, int]) -> bool:
        if gc in actor_cells:
            return True
        pr, pc = gc[0] * cell + cell // 2, gc[1] * cell + cell // 2
        if not (0 <= pr < h and 0 <= pc < w):
            return False
        return int(grid[pr, pc]) == bg

    goals: dict[tuple[int, int], int] = {}
    for action in allowed_final:
        if action not in grid_deltas:
            continue
        gdr, gdc = grid_deltas[action]
        for tc in trigger_cells:
            approach = (tc[0] - gdr, tc[1] - gdc)
            if free(approach):
                goals.setdefault(approach, action)

    if start_cell in goals:
        return ltr31.RoutePlan((), goals[start_cell], start_cell)

    seen = {start_cell}
    q = deque([(start_cell, ())])
    while q:
        gc, path = q.popleft()
        for action in available_actions:
            if action not in grid_deltas:
                continue
            gdr, gdc = grid_deltas[action]
            nxt = (gc[0] + gdr, gc[1] + gdc)
            if nxt in seen or not free(nxt):
                continue
            npath = path + (action,)
            if nxt in goals:
                return ltr31.RoutePlan(npath, goals[nxt], nxt)
            seen.add(nxt)
            q.append((nxt, npath))
    return None


def _earlier_candidates(
    parts: list,
    actor_views: list,
    trigger_shape_fact: str | None,
    recovery_mode: str,
    oc16,
) -> list:
    """Candidate triggers at the earlier frame under a chosen identity model.

    - "shape": same grounding as 0031 -- non-actor views whose frame-local shape
      literal equals the learned trigger_shape. Exposes shape-literal churn.
    - "position": any small non-actor view, identified later purely by spatial
      overlap. Isolates whether the trigger is spatially stable, independent of
      whether its shape literal is stable.
    """
    if recovery_mode == "shape":
        return ltr31._trigger_candidates(parts, actor_views, trigger_shape_fact, oc16)
    if recovery_mode == "position":
        max_area = 4 * getattr(oc16, "CELL", 5) ** 2
        actor_ids = {id(v) for v in actor_views}
        return [v for v in parts if id(v) not in actor_ids and v.area <= max_area]
    raise ValueError(f"unknown recovery_mode: {recovery_mode!r}")


def _contacted_trigger(
    parts: list,
    actor_views: list,
    trigger_shape_fact: str | None,
    firing_action: int,
    oc16,
):
    """The trigger instance the trace contacts at the contact frame (cell the
    firing action steps the actor into), restricted to shape-matching candidates."""
    candidates = ltr31._trigger_candidates(parts, actor_views, trigger_shape_fact, oc16)
    if not candidates:
        return None
    start = ltr31._actor_centroid(actor_views)
    if start is None or firing_action not in oc16.ACTION_DELTAS:
        return None
    dr, dc = oc16.ACTION_DELTAS[firing_action]
    contact = (start[0] + int(dr), start[1] + int(dc))
    for view in candidates:
        if contact in view.pixels:
            return view
    # Fall back to nearest candidate centroid to the contact cell.
    return min(
        candidates,
        key=lambda v: abs(v.centroid[0] - contact[0]) + abs(v.centroid[1] - contact[1]),
    )


def _trace_window(records, start_frame: int, contact_frame: int) -> tuple[int, ...] | None:
    """Actions records[start_frame+1 .. contact_frame+1] (the firing action last)."""
    actions: list[int] = []
    for k in range(start_frame + 1, contact_frame + 2):
        action = records[k].action_num
        if action is None:
            return None
        actions.append(int(action))
    return tuple(actions)


# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------
def _wanted_by_trace(events, meta, effects) -> dict[int, list[tuple[tuple[str, str, str], tuple[str, ...]]]]:
    supported_by_signal = {
        (e.object_id, e.slot, e.delta): e for e in effects
    }
    wanted: dict[int, list[tuple[tuple[str, str, str], tuple[str, ...]]]] = {}
    for event, info in zip(events, meta, strict=True):
        trace_action = cdr29._trace_action_for_event(event)
        for effect in effects:
            signal = (effect.object_id, effect.slot, effect.delta)
            if not cdr29._effect_present(event, signal):
                continue
            if signal not in cdr29._signals_for_action(trace_action, effects):
                continue
            wanted.setdefault(info["trace_index"], []).append((signal, effect.cause_clause))
    return wanted


def _backward_chain(
    records,
    contact_frame: int,
    window: int,
    contacted,
    seg_start: int,
    oc16,
    matcher: str = "strict",
    track_iou: float = 0.1,
) -> dict[int, object]:
    """Follow the trace-contacted trigger backward by cross-frame identity.

    matcher="strict" uses an honest IoU gate; matcher="overlap" admits any
    nonzero pixel overlap; matcher="continuity" uses centroid/area continuity for
    morphing triggers; matcher="loose" uses 0016's centroid-fallback matcher and
    almost never admits a loss. The chain stops where identity is lost.
    """
    if matcher == "strict":
        def match_fn(current, parts):
            return _match_iou(current, parts, track_iou)
    elif matcher == "overlap":
        match_fn = _match_overlap
    elif matcher == "continuity":
        def match_fn(current, parts):
            return _match_continuity(current, parts, oc16.CELL)
    elif matcher == "loose":
        match_fn = oc16._match_part_view
    else:
        raise ValueError(f"unknown matcher: {matcher!r}")

    chain: dict[int, object] = {contact_frame: contacted}
    current = contacted
    for f in range(contact_frame, contact_frame - window, -1):
        e = f - 1
        if e < seg_start:
            break
        _bg, parts_prev = oc16.il.views_from_grid(records[e].frame, frame=e)
        prev = match_fn(current, parts_prev)
        if prev is None:
            break
        chain[e] = prev
        current = prev
    return chain


def _recover_at(
    parts_e: list,
    actor_views_e: list,
    contacted,
    trigger_shape: str | None,
    recovery_mode: str,
    chain: dict[int, object] | None,
    start_frame: int,
    oc16,
) -> tuple[object | None, int]:
    if recovery_mode == "track":
        recovered = (chain or {}).get(start_frame)
        return recovered, (1 if recovered is not None else 0)
    candidates_e = _earlier_candidates(parts_e, actor_views_e, trigger_shape, recovery_mode, oc16)
    return recover_future_target(candidates_e, contacted.pixels), len(candidates_e)


def _window_decision(
    records,
    contact_frame: int,
    start_frame: int,
    signal: tuple[str, str, str],
    cause_clause: tuple[str, ...],
    parts_e: list,
    actor_views_e: list,
    recovered,
    contacted,
    candidates_count: int,
    oc16,
    route_target: str = "current",
    router: str = "grid",
) -> EarlierWindowDecision | None:
    trace_window = _trace_window(records, start_frame, contact_frame)
    if trace_window is None:
        return None
    firing_action = int(records[contact_frame + 1].action_num)
    final_action = ltr31._parse_action_clause(cause_clause)
    if final_action is None:
        final_action = firing_action

    # "current": aim at the trigger where it sits at the earlier frame (needs the
    # recovered instance). "contact": aim at where the bump actually happens (the
    # contact-frame trigger cells) -- an interception heuristic for moving triggers
    # that does not depend on re-identifying the trigger earlier.
    if route_target == "contact":
        target_view = contacted
    elif route_target == "current":
        target_view = recovered
    else:
        raise ValueError(f"unknown route_target: {route_target!r}")

    route = None
    if target_view is not None:
        if router == "grid":
            route = route_to_trigger_grid(
                records[start_frame].frame,
                actor_views_e,
                target_view,
                oc16.ACTION_DELTAS,
                records[start_frame].available_actions,
                final_action=final_action,
                cell=oc16.CELL,
            )
        elif router == "pixel":
            route = ltr31.route_to_trigger(
                records[start_frame].frame,
                actor_views_e,
                target_view,
                oc16.ACTION_DELTAS,
                records[start_frame].available_actions,
                final_action=final_action,
            )
        else:
            raise ValueError(f"unknown router: {router!r}")

    route_actions = route.actions if route is not None else ()
    first_match, cpl, exact = compare_route_to_window(route_actions, trace_window)
    last_match, csl = compare_route_suffix(route_actions, trace_window)
    return EarlierWindowDecision(
        contact_frame=contact_frame,
        start_frame=start_frame,
        window=contact_frame - start_frame,
        desired_effect=signal,
        cause_clause=cause_clause,
        firing_action=firing_action,
        candidates=candidates_count,
        target_recovered=recovered is not None,
        route_found=route is not None,
        route_actions=route_actions,
        trace_window_actions=trace_window,
        first_action_match=first_match,
        final_action_match=route is not None and route.final_action == firing_action,
        common_prefix_len=cpl,
        common_suffix_len=csl,
        last_action_match=last_match,
        route_matches_window=exact,
        route_no_longer_than_window=route is not None and len(route_actions) <= len(trace_window),
    )


def evaluate_earlier_window_routing(
    trace: Path | None,
    levels: set[int],
    window: int,
    recovery_mode: str = "shape",
    route_target: str = "current",
    matcher: str = "strict",
    track_iou: float = 0.1,
    router: str = "grid",
) -> list[EarlierWindowDecision]:
    events, meta, _counts = rcf18._live_step_events(trace, levels)
    supported = ltr31._supported_effects(events)
    effects = tuple(
        pp27.LearnedTriggerEffect.from_remote_hypothesis(entry.hypothesis) for entry in supported
    )
    wanted_by_trace = _wanted_by_trace(events, meta, effects)

    oc16 = rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    decisions: list[EarlierWindowDecision] = []

    for seg in oc16.segments_for_levels(segments, levels):
        belief = oc16.AgentBelief()
        for idx in range(seg.start, seg.end):
            if idx in wanted_by_trace:
                grid_c = records[idx].frame
                _bg_c, parts_c = oc16.il.views_from_grid(grid_c, frame=idx)
                actor_views_c = oc16._actor_views(parts_c, belief.controlled_sigs)
                firing_action = records[idx + 1].action_num
                for signal, cause_clause in wanted_by_trace[idx]:
                    trigger_shape = ltr31._parse_trigger_shape(cause_clause)
                    contacted = _contacted_trigger(
                        parts_c, actor_views_c, trigger_shape, firing_action, oc16
                    )
                    if contacted is None:
                        continue
                    chain = (
                        _backward_chain(records, idx, window, contacted, seg.start, oc16, matcher, track_iou)
                        if recovery_mode == "track"
                        else None
                    )
                    for w in range(0, window + 1):
                        start_frame = idx - w
                        if start_frame < seg.start:
                            break
                        _bg_e, parts_e = oc16.il.views_from_grid(records[start_frame].frame, frame=start_frame)
                        actor_views_e = oc16._actor_views(parts_e, belief.controlled_sigs)
                        recovered, candidates_count = _recover_at(
                            parts_e,
                            actor_views_e,
                            contacted,
                            trigger_shape,
                            recovery_mode,
                            chain,
                            start_frame,
                            oc16,
                        )
                        decision = _window_decision(
                            records,
                            idx,
                            start_frame,
                            signal,
                            cause_clause,
                            parts_e,
                            actor_views_e,
                            recovered,
                            contacted,
                            candidates_count,
                            oc16,
                            route_target,
                            router,
                        )
                        if decision is None:
                            break
                        decisions.append(decision)

            if idx + 1 <= seg.end and records[idx + 1].action_num is not None:
                _bg_t, parts_t = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                _bg_t1, parts_t1 = oc16.il.views_from_grid(records[idx + 1].frame, frame=idx + 1)
                belief.learn_remote_dynamics(parts_t, parts_t1)
                belief.observe_transition(records[idx].frame, records[idx + 1].action_num, records[idx + 1].frame)
                belief.frame_idx += 1
                belief.perceive(records[idx + 1].frame)
    return decisions


def _write_artifact(
    decisions: list[EarlierWindowDecision], levels: set[int], recovery_mode: str, route_target: str, router: str
) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-{recovery_mode}-to-{route_target}-{router}-earlier-window-decisions.jsonl"
    with path.open("w") as fh:
        for d in decisions:
            fh.write(json.dumps({
                "cause_clause": d.cause_clause,
                "common_prefix_len": d.common_prefix_len,
                "common_suffix_len": d.common_suffix_len,
                "contact_frame": d.contact_frame,
                "desired_effect": d.desired_effect,
                "final_action_match": d.final_action_match,
                "firing_action": d.firing_action,
                "first_action_match": d.first_action_match,
                "last_action_match": d.last_action_match,
                "route_actions": d.route_actions,
                "route_found": d.route_found,
                "route_matches_window": d.route_matches_window,
                "route_no_longer_than_window": d.route_no_longer_than_window,
                "start_frame": d.start_frame,
                "target_recovered": d.target_recovered,
                "trace_window_actions": d.trace_window_actions,
                "window": d.window,
            }, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Offline checks
# ---------------------------------------------------------------------------
def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    grid = np.zeros((15, 15), dtype=np.int8)
    deltas = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
    actor = ltr31._FakeView({(10, 5)}, centroid=(10, 5))
    trigger = ltr31._FakeView({(0, 5)}, centroid=(0, 5))
    route = ltr31.route_to_trigger(grid, [actor], trigger, deltas, [1, 2, 3, 4], final_action=1)
    checks.append(_check(
        "1 routes to a far trigger with a multi-step plan ending in the firing action",
        route is not None and route.actions == (1, 1) and route.final_action == 1,
        f"route={route}",
    ))

    first, cpl, exact = compare_route_to_window((1, 1), (1, 1))
    checks.append(_check(
        "2 plan matching the trace window scores exact agreement",
        first and cpl == 2 and exact,
        f"first={first} cpl={cpl} exact={exact}",
    ))

    first_w, cpl_w, exact_w = compare_route_to_window((1, 1), (1, 4, 1))
    checks.append(_check(
        "3 human wander keeps first-action agreement but breaks exact match",
        first_w and cpl_w == 1 and not exact_w,
        f"first={first_w} cpl={cpl_w} exact={exact_w}",
    ))

    first_b, cpl_b, exact_b = compare_route_to_window((2, 1), (1, 1))
    checks.append(_check(
        "4 wrong first move is detected as disagreement at the decision point",
        (not first_b) and cpl_b == 0 and not exact_b,
        f"first={first_b} cpl={cpl_b} exact={exact_b}",
    ))

    contacted = ltr31._FakeView({(0, 5), (0, 6)}, centroid=(0, 5))
    decoy = ltr31._FakeView({(0, 12)}, centroid=(0, 12))
    earlier_same = ltr31._FakeView({(0, 5), (0, 6)}, centroid=(0, 5))
    recovered = recover_future_target([decoy, earlier_same], contacted.pixels)
    checks.append(_check(
        "5 recovers the future trigger instance by pixel identity, not the decoy",
        recovered is earlier_same,
        f"recovered_centroid={None if recovered is None else recovered.centroid}",
    ))

    none_overlap = recover_future_target([decoy], contacted.pixels)
    checks.append(_check(
        "6 reports failure when the trigger cannot be re-identified earlier",
        none_overlap is None,
        f"recovered={none_overlap}",
    ))

    fake_oc16 = SimpleNamespace(CELL=5)
    big = ltr31._FakeView({(r, c) for r in range(11) for c in range(11)}, centroid=(5, 5), area=121)
    small_a = ltr31._FakeView({(0, 5)}, centroid=(0, 5), area=1)
    small_b = ltr31._FakeView({(0, 6)}, centroid=(0, 6), area=1)
    pos_cands = _earlier_candidates([big, small_a, small_b], [small_a], None, "position", fake_oc16)
    checks.append(_check(
        "7 position mode keeps small non-actor parts and drops the actor and oversized regions",
        small_b in pos_cands and small_a not in pos_cands and big not in pos_cands,
        f"candidates={[v.centroid for v in pos_cands]}",
    ))

    chain = {7: small_b}
    rec_hit, n_hit = _recover_at([], [], contacted, "trigger_shape=x", "track", chain, 7, fake_oc16)
    rec_miss, n_miss = _recover_at([big], [], contacted, "trigger_shape=x", "track", chain, 5, fake_oc16)
    checks.append(_check(
        "8 track mode reads the identity chain and ignores frame-local candidates",
        rec_hit is small_b and n_hit == 1 and rec_miss is None and n_miss == 0,
        f"hit={rec_hit is small_b} miss={rec_miss}",
    ))

    moved_9 = ltr31._FakeView({(0, 6)}, centroid=(0, 6), area=1)
    parts_map = {9: [moved_9], 8: []}
    track_oc16 = SimpleNamespace(
        il=SimpleNamespace(views_from_grid=lambda grid, frame: (0, parts_map.get(grid, []))),
        _match_part_view=lambda view, parts: parts[0] if parts else None,
    )
    fake_records = {f: SimpleNamespace(frame=f) for f in (8, 9, 10)}
    contacted_10 = ltr31._FakeView({(0, 7)}, centroid=(0, 7), area=1)
    built = _backward_chain(fake_records, 10, 3, contacted_10, 0, track_oc16, matcher="loose")
    checks.append(_check(
        "9 backward chain (loose matcher) follows identity then stops where the trigger ceases to exist",
        set(built) == {10, 9} and built[9] is moved_9,
        f"chain_frames={sorted(built)}",
    ))

    # Suffix alignment: a wandering human and a direct plan still share the final
    # approach to the trigger, so the common suffix can be long even when the
    # prefix diverges immediately.
    last_s, csl_s = compare_route_suffix((4, 1, 1), (2, 4, 1, 1))
    first_s, cpl_s, _ = compare_route_to_window((4, 1, 1), (2, 4, 1, 1))
    checks.append(_check(
        "10 suffix alignment rewards a matching final approach the prefix misses",
        last_s and csl_s == 3 and not first_s and cpl_s == 0,
        f"last={last_s} csl={csl_s} first={first_s} cpl={cpl_s}",
    ))

    v = ltr31._FakeView({(0, 0), (0, 1)}, centroid=(0, 0))
    overlap = ltr31._FakeView({(0, 1), (0, 2)}, centroid=(0, 1))
    nonoverlap = ltr31._FakeView({(5, 5)}, centroid=(5, 5))
    hit = _match_iou(v, [overlap, nonoverlap], 0.1)
    miss = _match_iou(v, [nonoverlap], 0.1)
    checks.append(_check(
        "11 strict IoU matcher takes the overlapping object and admits loss otherwise",
        hit is overlap and miss is None,
        f"hit={hit is overlap} miss={miss}",
    ))

    # The routing parity bug: a trigger whose pixel is not CELL-congruent with the
    # avatar lattice is unreachable for the pixel BFS (steps of CELL land on the
    # wrong residue) but reachable for the CELL-quantized router.
    big_grid = np.zeros((30, 35), dtype=np.int8)
    actor_blk = ltr31._FakeView(
        {(r, c) for r in range(10, 15) for c in range(10, 15)}, centroid=(12, 12), area=25
    )
    misaligned = ltr31._FakeView({(12, 23)}, centroid=(12, 23), area=1)
    pixel_route = ltr31.route_to_trigger(
        big_grid, [actor_blk], misaligned, deltas, [1, 2, 3, 4], final_action=4
    )
    grid_route = route_to_trigger_grid(
        big_grid, [actor_blk], misaligned, deltas, [1, 2, 3, 4], final_action=4, cell=5
    )
    checks.append(_check(
        "12 grid router reaches a CELL-misaligned trigger that the pixel router cannot",
        pixel_route is None and grid_route is not None and grid_route.actions[-1] == 4,
        f"pixel={pixel_route} grid={None if grid_route is None else grid_route.actions}",
    ))

    morph_prev = ltr31._FakeView({(10, 10), (10, 11), (11, 10), (11, 11)}, centroid=(10.5, 10.5), area=4)
    morph_next = ltr31._FakeView({(12, 12), (12, 13), (13, 12), (13, 13)}, centroid=(12.5, 12.5), area=4)
    far_decoy = ltr31._FakeView({(25, 25), (25, 26), (26, 25), (26, 26)}, centroid=(25.5, 25.5), area=4)
    iou_miss = _match_iou(morph_prev, [morph_next], 0.05)
    continuity_hit = _match_continuity(morph_prev, [far_decoy, morph_next], cell=5)
    checks.append(_check(
        "13 continuity matcher recovers a nearby morph with zero pixel overlap",
        iou_miss is None and continuity_hit is morph_next,
        f"iou={iou_miss} continuity_hit={continuity_hit is morph_next}",
    ))

    continuity_miss = _match_continuity(morph_prev, [far_decoy], cell=5)
    checks.append(_check(
        "14 continuity matcher rejects far same-size decoys",
        continuity_miss is None,
        f"continuity_miss={continuity_miss}",
    ))

    tiny_overlap = ltr31._FakeView({(0, 1), (9, 9)}, centroid=(5, 5), area=2)
    overlap_hit = _match_overlap(v, [nonoverlap, tiny_overlap])
    overlap_miss = _match_overlap(v, [nonoverlap])
    checks.append(_check(
        "15 overlap matcher accepts tiny real overlap and rejects zero overlap",
        overlap_hit is tiny_overlap and overlap_miss is None,
        f"hit={overlap_hit is tiny_overlap} miss={overlap_miss}",
    ))
    return all(checks)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _window_table(decisions: list[EarlierWindowDecision]) -> dict[int, dict[str, float]]:
    by_window: dict[int, list[EarlierWindowDecision]] = defaultdict(list)
    for d in decisions:
        by_window[d.window].append(d)
    table: dict[int, dict[str, float]] = {}
    for w in sorted(by_window):
        rows = by_window[w]
        n = len(rows)
        routed = [d for d in rows if d.route_found]
        table[w] = {
            "n": n,
            "target_recovered_rate": sum(d.target_recovered for d in rows) / n,
            "route_found_rate": sum(d.route_found for d in rows) / n,
            "first_action_match_rate": sum(d.first_action_match for d in rows) / n,
            "exact_match_rate": sum(d.route_matches_window for d in rows) / n,
            "last_action_match_rate": sum(d.last_action_match for d in rows) / n,
            "mean_common_prefix_frac": (
                sum(d.common_prefix_frac for d in routed) / len(routed) if routed else 0.0
            ),
            "mean_common_suffix_frac": (
                sum(d.common_suffix_frac for d in routed) / len(routed) if routed else 0.0
            ),
            "route_no_longer_than_window_rate": (
                sum(d.route_no_longer_than_window for d in routed) / len(routed) if routed else 0.0
            ),
        }
    return table


def run_live(
    trace: Path | None,
    levels: set[int],
    window: int,
    recovery_mode: str,
    route_target: str,
    matcher: str,
    track_iou: float,
    router: str,
) -> list[EarlierWindowDecision]:
    decisions = evaluate_earlier_window_routing(
        trace, levels, window, recovery_mode, route_target, matcher, track_iou, router
    )
    artifact = _write_artifact(decisions, levels, recovery_mode, route_target, router)
    matcher_note = f" matcher={matcher}(iou>={track_iou})" if recovery_mode == "track" else ""
    print(
        f"\nLive ls20 earlier-window trigger-routing replay "
        f"(recovery={recovery_mode} route_to={route_target} router={router}{matcher_note}):"
    )
    print(f"  decisions={len(decisions)} window=0..{window}")
    if not decisions:
        print("  no caused supported effects available for earlier-window routing replay")
        print(f"  routing_artifact={artifact}")
        return decisions

    table = _window_table(decisions)
    print("  per-window agreement (w=0 reproduces 0031's pre-contact baseline):")
    print("    prefixΔ aligns plan vs trace from the decision point; suffixΔ aligns them at the bump.")
    print(
        f"    {'w':>2} {'n':>4} {'recover':>8} {'route':>6} {'first':>6} "
        f"{'last':>6} {'exact':>6} {'prefixΔ':>8} {'suffixΔ':>8}"
    )
    for w, stats in table.items():
        print(
            f"    {w:>2} {int(stats['n']):>4} {stats['target_recovered_rate']:>8.3f} "
            f"{stats['route_found_rate']:>6.3f} {stats['first_action_match_rate']:>6.3f} "
            f"{stats['last_action_match_rate']:>6.3f} {stats['exact_match_rate']:>6.3f} "
            f"{stats['mean_common_prefix_frac']:>8.3f} {stats['mean_common_suffix_frac']:>8.3f}"
        )

    print("\n  sample earlier-window decisions (w>=1):")
    shown = [d for d in decisions if d.window >= 1][:10]
    for d in shown:
        print(
            f"    contact={d.contact_frame} start={d.start_frame} w={d.window} "
            f"route={[f'A{a}' for a in d.route_actions]} trace={[f'A{a}' for a in d.trace_window_actions]} "
            f"recovered={d.target_recovered} first={d.first_action_match} cpl={d.common_prefix_len} "
            f"csl={d.common_suffix_len} exact={d.route_matches_window}"
        )
    print(f"  routing_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate earlier-window routing on the recorded trace")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=6, help="max frames to step back before each trigger event")
    parser.add_argument(
        "--recovery-mode",
        choices=("shape", "position", "track"),
        default="track",
        help="re-identify the future trigger earlier: 0031 shape-literal, spatial overlap, or backward identity track",
    )
    parser.add_argument(
        "--route-target",
        choices=("current", "contact"),
        default="current",
        help="aim the route at the trigger's earlier position (current) or the contact-frame bump site (contact)",
    )
    parser.add_argument(
        "--track-matcher",
        choices=("strict", "overlap", "continuity", "loose"),
        default="overlap",
        help="track-mode identity: strict IoU gate, any-overlap, centroid/area continuity, or 0016's loose centroid-fallback matcher",
    )
    parser.add_argument("--track-iou", type=float, default=0.1, help="min IoU for the strict backward tracker")
    parser.add_argument(
        "--router",
        choices=("grid", "pixel"),
        default="grid",
        help="grid = CELL-quantized BFS (bug-fixed); pixel = 0031's pixel-resolution BFS (reproduces the residue bug)",
    )
    args = parser.parse_args()

    print("Offline checks (earlier-window trigger-routing replay):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using earlier-window trigger-routing replay.")
    print("\nEarlier-window replay can plan trigger routes from a distance and score the plan prefix against the trace.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(
            args.trace,
            levels,
            args.window,
            args.recovery_mode,
            args.route_target,
            args.track_matcher,
            args.track_iou,
            args.router,
        )


if __name__ == "__main__":
    main()
