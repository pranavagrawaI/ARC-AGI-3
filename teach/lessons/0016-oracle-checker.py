"""0016 — Offline oracle-checker: measure agent predictions against a human trace.

Run from the repo root:
    uv run python teach/lessons/0016-oracle-checker.py
    uv run python teach/lessons/0016-oracle-checker.py --levels 1,2

The human trace drives STEPPING only (reproduce state sequence). The agent learns
dynamics from observed transitions but never reads trace reasoning, goals, or
policy. Divergences are reported, never minimized.

Composed from verified lessons 0008 (controllability), 0009 (role falsifier),
0010 (lattice perception), 0011 (identity), 0013 (pipeline), 0015 (goal predicates).
No import of ls20.py or environment win-condition source.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import numpy as np

from arc_agi.rendering import COLOR_MAP, hex_to_rgb

# ---------------------------------------------------------------------------
# Lesson bricks (importlib + sys.modules — Python 3.14 dataclass quirk)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    path = _REPO / rel
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gi = _load("oc16_goal", "teach/lessons/0015-goal-induction.py")
ig = gi.ig
oc = ig.load_pipeline()["oc"]
il = ig.load_pipeline()["il"]
id_mod = ig.load_pipeline()["id"]
df = _load("oc16_df", "teach/lessons/0009-dynamics-falsifier.py")
ad = _load("oc16_ad", "teach/lessons/0016-autonomous-dynamics.py")

CELL = ig.CELL
ACTION_DELTAS = ig.ACTION_DELTAS
ACTION_MAP = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4}
RGB = {v: hex_to_rgb(c) for v, c in COLOR_MAP.items()}
DEFAULT_TRACE = (
    _REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
ARTIFACT_DIR = _REPO / "teach/lessons/.artifacts/0016-oracle-checker"


# ---------------------------------------------------------------------------
# Step 0 — trace loader (checker-only; strips reasoning)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TraceRecord:
    index: int
    frame: np.ndarray
    action_id: str
    action_num: int | None
    levels_completed: int
    win_levels: int
    state: str
    full_reset: bool
    available_actions: list[int]
    game_id: str


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    level: int
    length: int


def _grid_from_raw(raw) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.int8)
    return arr[-1] if arr.ndim == 3 else arr


def load_trace(path: Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    with path.open() as fh:
        for idx, line in enumerate(fh):
            payload = json.loads(line)
            data = payload["data"]
            ai = dict(data["action_input"])
            reasoning = ai.pop("reasoning", None)
            if reasoning is not None:
                print(f"WARNING: record {idx} has non-null reasoning — dropped")
            action_id = str(ai.get("id", ""))
            action_num = ACTION_MAP.get(action_id)
            records.append(
                TraceRecord(
                    index=idx,
                    frame=_grid_from_raw(data["frame"]),
                    action_id=action_id,
                    action_num=action_num,
                    levels_completed=int(data["levels_completed"]),
                    win_levels=int(data.get("win_levels", 7)),
                    state=str(data["state"]),
                    full_reset=bool(data.get("full_reset", False)),
                    available_actions=list(data.get("available_actions", [])),
                    game_id=str(data["game_id"]),
                )
            )
    return records


def hygiene_step0(records: list[TraceRecord]) -> tuple[str, str, list[Segment]]:
    game_ids = {r.game_id for r in records}
    assert len(game_ids) == 1, f"game_id not constant: {game_ids}"
    game_id = next(iter(game_ids))
    print(f"game_id: {game_id}")

    base_actions = tuple(records[0].available_actions)
    action_changes: list[tuple[int, tuple[int, ...]]] = []
    for r in records:
        key = tuple(r.available_actions)
        if key != base_actions:
            action_changes.append((r.index, key))
    if action_changes:
        print(f"available_actions changed at indices: {action_changes}")
    else:
        print("available_actions: constant across trace")

    produced_matches = 0
    produced_total = 0
    from_frame_matches = 0
    from_frame_total = 0
    evidence: list[str] = []

    def _avatar_shift(fr0: np.ndarray, fr1: np.ndarray):
        matches = oc.match_objects(oc.segment_objects(fr0), oc.segment_objects(fr1))
        best, best_mag = None, 0.0
        for _sig, disp in matches:
            mag = abs(disp[0]) + abs(disp[1])
            if mag > best_mag:
                best_mag = mag
                best = disp
        return best

    for t in range(2, min(len(records) - 1, 120)):
        if "RESET" in records[t].action_id:
            continue
        act = records[t].action_num
        if act is None:
            continue
        for t0, t1 in ((t - 1, t), (t, t + 1)):
            if records[t0].full_reset or records[t1].full_reset:
                continue
            if records[t0].levels_completed != records[t1].levels_completed:
                continue
            shift = _avatar_shift(records[t0].frame, records[t1].frame)
            if shift is None:
                continue
            exp = ACTION_DELTAS[act]
            ok = abs(shift[0] - exp[0]) < 2 and abs(shift[1] - exp[1]) < 2
            if t0 == t - 1:
                produced_total += 1
                produced_matches += int(ok)
                if len(evidence) < 4:
                    evidence.append(
                        f"idx {t}: action_input[{t}]={records[t].action_id} "
                        f"shift {records[t0].index}->{t1}={shift} expected={exp} ok={ok}"
                    )
            else:
                from_frame_total += 1
                from_frame_matches += int(ok)

    convention = (
        "action_input[t] PRODUCED frame[t]"
        if produced_matches >= from_frame_matches
        else "action_input[t] is FROM frame[t] (produces frame[t+1])"
    )
    print(
        f"inferred alignment: {convention}\n"
        f"  evidence: produced-frame {produced_matches}/{produced_total}, "
        f"from-frame {from_frame_matches}/{from_frame_total}"
    )
    for line in evidence:
        print(f"    {line}")

    segments: list[Segment] = []
    seg_start = 0
    for i, r in enumerate(records):
        split = False
        if r.full_reset:
            split = True
        if r.state in ("WIN", "GAME_OVER", "RESET"):
            split = True
        if i > 0 and r.levels_completed > records[i - 1].levels_completed:
            split = True
        if split and i > seg_start:
            segments.append(
                Segment(
                    seg_start,
                    i - 1,
                    records[seg_start].levels_completed,
                    i - seg_start,
                )
            )
            seg_start = i
    if seg_start < len(records):
        segments.append(
            Segment(
                seg_start,
                len(records) - 1,
                records[seg_start].levels_completed,
                len(records) - seg_start,
            )
        )
    for seg in segments:
        print(
            f"segment [{seg.start},{seg.end}] level={seg.level} length={seg.length}"
        )
    return game_id, convention, segments


# ---------------------------------------------------------------------------
# Agent belief — learns dynamics only; never reads trace policy/goal
# ---------------------------------------------------------------------------
@dataclass
class AgentBelief:
    controllability: oc.ControllabilityLearner = field(
        default_factory=oc.ControllabilityLearner
    )
    falsifier: df.DynamicsFalsifier | None = None
    controlled_sigs: set[tuple[int, int]] = field(default_factory=set)
    controlled_sizes: set[int] = field(default_factory=set)
    grouper: il.CompositeObjectGrouper = field(default_factory=il.CompositeObjectGrouper)
    tracker: id_mod.ObjectTracker = field(default_factory=id_mod.ObjectTracker)
    previous_views: list = field(default_factory=list)
    transition_counts: Counter = field(default_factory=Counter)
    rotate_ledger: ad.AutonomousRotateLedger = field(
        default_factory=ad.AutonomousRotateLedger
    )
    frame_idx: int = 0

    def _refresh_controlled(self) -> None:
        ctrl = self.controllability.controlled()
        self.controlled_sigs = set(ctrl.keys())
        self.controlled_sizes = {sig[1] for sig in ctrl}
        if self.falsifier is None and self.controlled_sizes:
            self.falsifier = df.DynamicsFalsifier(
                controlled_sizes=self.controlled_sizes, deltas=ACTION_DELTAS, cell=CELL
            )
        elif self.falsifier is not None:
            self.falsifier.sizes = self.controlled_sizes

    def perceive(self, grid: np.ndarray) -> list:
        _bg, parts = il.views_from_grid(grid, frame=self.frame_idx)
        if self.previous_views:
            self.grouper.observe(self.previous_views, parts)
        views = self.grouper.group(grid, parts, frame=self.frame_idx)
        self.tracker.update(views)
        self.previous_views = views
        return views

    def observe_transition(
        self, prev_grid: np.ndarray, action: int, nxt_grid: np.ndarray
    ) -> None:
        self.controllability.observe(prev_grid, action, nxt_grid)
        self._refresh_controlled()
        if self.falsifier is not None:
            self.falsifier.observe(prev_grid, action, nxt_grid)
        key = self._transition_key(prev_grid, action, nxt_grid)
        self.transition_counts[key] += 1

    def _transition_key(
        self, prev_grid: np.ndarray, action: int, nxt_grid: np.ndarray
    ) -> str:
        vals, counts = np.unique(prev_grid, return_counts=True)
        bg = int(vals[counts.argmax()])
        actor_views = [v for v in self.previous_views if ig.is_actor_view(v, self.controlled_sigs)]
        if not actor_views:
            return f"unknown:{action}"
        actor = actor_views[0]
        dr, dc = ACTION_DELTAS.get(action, (0, 0))
        tr = round(actor.centroid[0] + dr)
        tc = round(actor.centroid[1] + dc)
        cls = df.component_at(prev_grid, tr, tc, bg)
        cls_key = f"c{cls[0]}" if cls else "floor"
        return f"{cls_key}:a{action}"

    def reset_episode(self) -> None:
        self.grouper = il.CompositeObjectGrouper()
        self.tracker = id_mod.ObjectTracker()
        self.previous_views = []
        if self.falsifier is not None:
            self.falsifier.acentroid = None
        # rotate_ledger intentionally persists: remote dynamics are a property of
        # the game, not of one episode. Resetting it would re-starve the ledger.

    def learn_remote_dynamics(self, parts_t: list, parts_t1: list) -> None:
        """Feed every non-actor sprite that changed appearance to the rotate ledger.

        Operates on raw same-colour parts (not composites) to match 0016's
        crop-level dynamics design and stay independent of the grouper.
        """
        for pv in parts_t:
            if ig.is_actor_view(pv, self.controlled_sigs):
                continue
            nv = _match_part_view(pv, parts_t1)
            if nv is None:
                continue
            if (
                pv.internal_pattern_lattice["raw"]
                != nv.internal_pattern_lattice["raw"]
            ):
                self.rotate_ledger.observe(pv, nv)

    def dynamics_explains(self, view_prev, view_next) -> bool:
        """True iff the ledger PREDICTS this exact next orientation (predict-before-observe)."""
        if view_prev is None or view_next is None:
            return False
        key = ad.RotateKey(
            view_prev.shape_lattice["rotation_invariant"], view_prev.area
        )
        pred = self.rotate_ledger.predict_raw(
            key, view_prev.internal_pattern_lattice["raw"]
        )
        return pred is not None and pred == view_next.internal_pattern_lattice["raw"]


# ---------------------------------------------------------------------------
# Step 1 — composed frame predictor (controllability + role falsifier)
# ---------------------------------------------------------------------------
def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _actor_views(views, controlled_sigs: set[tuple[int, int]]):
    return [v for v in views if ig.is_actor_view(v, controlled_sigs)]


def _role_survivors(belief: AgentBelief, target_color: int | None) -> set[str] | None:
    if belief.falsifier is None or target_color is None:
        return None
    return belief.falsifier._verdict(target_color)


def _predicted_motion(
    belief: AgentBelief, action: int, target_color: int | None
) -> tuple[int, int] | None:
    ctrl = belief.controllability.controlled()
    if not ctrl:
        return None
    motion = None
    for sig, per_action in ctrl.items():
        if action in per_action:
            dr, dc = per_action[action]
            if abs(dr) + abs(dc) >= CELL * 0.5:
                motion = (int(round(dr)), int(round(dc)))
                break
    if motion is None:
        return None
    surv = _role_survivors(belief, target_color)
    if surv is not None and surv == {"BLOCKER"}:
        return None
    return motion


def _shift_pixels(
    grid: np.ndarray, pixels: frozenset[tuple[int, int]], dr: int, dc: int, bg: int
) -> np.ndarray:
    out = grid.copy()
    colors = {pos: int(grid[pos]) for pos in pixels}
    for r, c in pixels:
        out[r, c] = bg
    h, w = grid.shape
    for (r, c), color in colors.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = color
    return out


def _apply_changer(
    grid: np.ndarray,
    actor_pixels: frozenset[tuple[int, int]],
    belief: AgentBelief,
    target_color: int,
) -> np.ndarray:
    if belief.falsifier is None:
        return grid
    transitions = belief.falsifier.transitions.get(target_color, [])
    if not transitions:
        return grid
    tally: Counter = Counter()
    for prev_sigs, nxt_sigs in transitions:
        if prev_sigs and nxt_sigs:
            tally[nxt_sigs[0][0]] += 1
    if not tally:
        return grid
    new_color = tally.most_common(1)[0][0]
    out = grid.copy()
    for r, c in actor_pixels:
        if int(out[r, c]) != _background(grid):
            out[r, c] = new_color
    return out


def predict_frame(
    grid: np.ndarray, action: int, views: list, belief: AgentBelief
) -> np.ndarray:
    """Copy-forward baseline + controllability displacement + learned role effects."""
    predicted = grid.copy()
    bg = _background(grid)
    actors = _actor_views(views, belief.controlled_sigs)
    if not actors:
        return predicted

    actor_pixels: set[tuple[int, int]] = set()
    for av in actors:
        actor_pixels.update(av.pixels)
    actor_pixels_f = frozenset(actor_pixels)

    centroid = (
        sum(v.centroid[0] for v in actors) / len(actors),
        sum(v.centroid[1] for v in actors) / len(actors),
    )
    dr_a, dc_a = ACTION_DELTAS.get(action, (0, 0))
    tr, tc = round(centroid[0] + dr_a), round(centroid[1] + dc_a)
    cls = df.component_at(grid, tr, tc, bg)
    target_color = cls[0] if cls else None

    motion = _predicted_motion(belief, action, target_color)
    if motion is not None:
        predicted = _shift_pixels(predicted, actor_pixels_f, motion[0], motion[1], bg)

    surv = _role_survivors(belief, target_color)
    if target_color is not None and surv and "CHANGER" in surv:
        predicted = _apply_changer(predicted, actor_pixels_f, belief, target_color)
    return predicted


# ---------------------------------------------------------------------------
# Diff clusters + divergence typing (evidence only — no auto-label)
# ---------------------------------------------------------------------------
@dataclass
class DiffCluster:
    rmin: int
    cmin: int
    rmax: int
    cmax: int
    cells: int

    def as_dict(self) -> dict:
        return {
            "bbox": [self.rmin, self.cmin, self.rmax, self.cmax],
            "cells": self.cells,
        }


def diff_clusters(mask: np.ndarray) -> list[DiffCluster]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    clusters: list[DiffCluster] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = [(r, c)]
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        q.append((ny, nx))
                        cells.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            clusters.append(
                DiffCluster(min(rs), min(cs), max(rs), max(cs), len(cells))
            )
    return clusters


def _view_at(views, r: int, c: int):
    cell = (r, c)
    for v in views:
        if cell in v.pixels:
            return v
    return None


def _match_part_view(view, parts: list, min_iou: float = 0.15):
    """Identity across frames by pixel IoU, then centroid proximity (≤2 cells)."""
    if view is None:
        return None
    best, best_iou = None, 0.0
    for p in parts:
        inter = len(view.pixels & p.pixels)
        union = len(view.pixels | p.pixels)
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best_iou, best = iou, p
    if best is not None and best_iou >= min_iou:
        return best
    ar, ac = view.centroid
    near = [
        p
        for p in parts
        if abs(p.centroid[0] - ar) <= 2 * CELL and abs(p.centroid[1] - ac) <= 2 * CELL
    ]
    if not near:
        return None
    return min(near, key=lambda p: abs(p.centroid[0] - ar) + abs(p.centroid[1] - ac))


def _change_kind(
    grid_t: np.ndarray, grid_t1: np.ndarray, cluster: DiffCluster, actor_views
) -> str:
    r0, c0, r1, c1 = cluster.rmin, cluster.cmin, cluster.rmax, cluster.cmax
    sub_t = grid_t[r0 : r1 + 1, c0 : c1 + 1]
    sub_t1 = grid_t1[r0 : r1 + 1, c0 : c1 + 1]
    actor_cells = set()
    for av in actor_views:
        actor_cells.update(av.pixels)
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    on_actor = (cr, cc) in actor_cells
    if on_actor:
        if np.array_equal(sub_t, sub_t1):
            return "actor_displacement"
        return "actor_attribute_change"
    return "remote_attribute_change"


def grammar_has_slot(kind: str) -> bool:
    return kind in ("actor_displacement", "actor_attribute_change")


def type_divergence_signals(
    cluster: DiffCluster,
    views_t: list,
    grid_t: np.ndarray,
    grid_t1: np.ndarray,
    belief: AgentBelief,
    action: int,
    parts_t: list | None = None,
    parts_t1: list | None = None,
) -> dict:
    cr, cc = (cluster.rmin + cluster.rmax) // 2, (cluster.cmin + cluster.cmax) // 2
    view = _view_at(views_t, cr, cc)
    perception_ok = view is not None
    if view is not None:
        perception_ok = (cr, cc) in view.pixels

    actor_views = _actor_views(views_t, belief.controlled_sigs)
    kind = _change_kind(grid_t, grid_t1, cluster, actor_views)
    grammar_ok = grammar_has_slot(kind)

    # Does the learned remote-dynamics model PREDICT this change? (held-out:
    # the ledger is consulted before observing this step's transition.)
    dynamics_explained = False
    if kind == "remote_attribute_change" and parts_t is not None and parts_t1 is not None:
        part_prev = _view_at(parts_t, cr, cc)
        part_next = _match_part_view(part_prev, parts_t1)
        dynamics_explained = belief.dynamics_explains(part_prev, part_next)

    key = belief._transition_key(grid_t, action, grid_t1)
    prior = belief.transition_counts.get(key, 0)

    return {
        "perception_ok": perception_ok,
        "perception_note": "segment covers region" if perception_ok else "no object view at region",
        "grammar_ok": grammar_ok,
        "grammar_note": f"kind={kind}, slot={'yes' if grammar_ok else 'no (grammar-gap candidate)'}",
        "dynamics_explained": dynamics_explained,
        "prior_instances": prior,
        "underfit_note": "few prior instances" if prior < 3 else "sufficient prior instances",
        "kind": kind,
    }


def classify_candidate(signals: dict) -> str:
    if not signals["perception_ok"]:
        return "perception"
    if not signals["grammar_ok"]:
        return "explained_dynamics" if signals.get("dynamics_explained") else "grammar"
    if signals["prior_instances"] < 3:
        return "underfit"
    return "other"


# ---------------------------------------------------------------------------
# Step 5 — reward-frame predicate vector (increment frame)
# ---------------------------------------------------------------------------
def reward_predicate_vector(
    grid: np.ndarray, views: list, belief: AgentBelief
) -> dict:
    actors = _actor_views(views, belief.controlled_sigs)
    primary = actors[0] if actors else None
    rows: list[dict] = []
    if primary is None:
        return {"avatar_position": None, "pairs": [], "candidate_predicates": []}

    avatar_pos = (round(primary.centroid[0]), round(primary.centroid[1]))
    for target in views:
        if ig.is_actor_view(target, belief.controlled_sigs):
            continue
        lenses = gi.match_lenses(primary, target)
        eq_raw = {
            slot: il.equal(slot, primary, target, "raw")
            for slot in gi.GOAL_SLOTS
        }
        facts = gi.goal_facts_for_pair(primary, target, include_match=True)
        rows.append(
            {
                "target_shape": ig.shape_literal(target),
                "on_target": gi.on_target(primary, target),
                "match_lenses": lenses,
                "equality_raw": eq_raw,
                "match_facts": sorted(gi.match_facts(primary, target)),
                "role": _role_label(belief, target),
            }
        )
    return {
        "avatar_position": avatar_pos,
        "pairs": rows,
        "candidate_predicates": sorted(
            {
                fact
                for row in rows
                for fact in row["match_facts"]
                if row["on_target"]
            }
        ),
    }


def _role_label(belief: AgentBelief, view) -> str:
    if belief.falsifier is None:
        return "undecided"
    colors = {c for c, _ in view.color_lattice["raw"]}
    labels = []
    for color in colors:
        surv = belief.falsifier._verdict(color)
        if surv is None:
            labels.append(f"{color}:undecided")
        elif not surv:
            labels.append(f"{color}:grammar_gap")
        else:
            labels.append(f"{color}:{','.join(sorted(surv))}")
    return "; ".join(labels) if labels else "undecided"


# ---------------------------------------------------------------------------
# Step 1 — instrument self-test
# ---------------------------------------------------------------------------
def run_self_test() -> None:
    """2-frame exact case: instrument fidelity, not controllability learning depth."""
    g0 = np.zeros((64, 64), dtype=np.int8)
    g0[20:22, 20:22] = 7
    g1 = np.zeros((64, 64), dtype=np.int8)
    g1[20:22, 25:27] = 7  # avatar only at destination — no ghost at source

    belief = AgentBelief()
    for _ in range(2):
        belief.observe_transition(g0, 4, g1)
    views = belief.perceive(g0)
    predicted = predict_frame(g0, 4, views, belief)
    diff = predicted != g1
    assert not diff.any(), (
        f"self-test failed: {int(diff.sum())} diff cells — fix predictor before trace"
    )
    print("Step 1 self-test: PASS (zero diff on known-exact displacement)")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def _grid_svg(
    grid: np.ndarray,
    *,
    title: str,
    highlight: np.ndarray | None = None,
    cell_px: int = 6,
) -> str:
    h, w = grid.shape
    rects = []
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            rgb = RGB.get(v, (40, 40, 40))
            fill = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            stroke = ""
            if highlight is not None and highlight[r, c]:
                stroke = ' stroke="#ff2d55" stroke-width="1.2"'
            rects.append(
                f'<rect x="{c * cell_px}" y="{r * cell_px}" '
                f'width="{cell_px}" height="{cell_px}" fill="{fill}"{stroke}/>'
            )
    svg_w, svg_h = w * cell_px, h * cell_px
    return (
        f'<div class="grid-wrap"><div class="grid-title">{escape(title)}</div>'
        f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">'
        + "".join(rects)
        + "</svg></div>"
    )


def _overlay_svg(grid: np.ndarray, views: list, belief: AgentBelief, cell_px: int = 6) -> str:
    h, w = grid.shape
    base = _grid_svg(grid, title="belief overlay (on actual frame)", cell_px=cell_px)
    overlays = []
    actors = _actor_views(views, belief.controlled_sigs)
    actor_ids = {id(v) for v in actors}
    for i, view in enumerate(views):
        r0, c0, r1, c1 = view.bbox
        x, y = c0 * cell_px, r0 * cell_px
        bw, bh = (c1 - c0 + 1) * cell_px, (r1 - r0 + 1) * cell_px
        is_actor = id(view) in actor_ids
        color = "#2563eb" if is_actor else "#b45309"
        label = ig.shape_literal(view)
        if is_actor:
            label = f"ACTOR {label}"
        role = _role_label(belief, view)
        overlays.append(
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="none" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
        overlays.append(
            f'<text x="{x+2}" y="{max(10,y+10)}" font-size="9" fill="{color}">'
            f"{escape(label[:18])}</text>"
        )
        overlays.append(
            f'<text x="{x+2}" y="{min(h*cell_px-2,y+bh-2)}" font-size="8" fill="#555">'
            f"{escape(role[:24])}</text>"
        )
    if actors:
        primary = actors[0]
        for target in views:
            if ig.is_actor_view(target, belief.controlled_sigs):
                continue
            lenses = gi.match_lenses(primary, target)
            if lenses:
                tr, tc = target.centroid
                overlays.append(
                    f'<text x="{tc*cell_px}" y="{tr*cell_px}" font-size="7" fill="#15803d">'
                    f"eq:{','.join(f'{k}@{v[:4]}' for k,v in lenses.items())}</text>"
                )
    insert_at = base.rfind("</svg>")
    return base[:insert_at] + "".join(overlays) + base[insert_at:]


def write_html_report(
    path: Path,
    *,
    game_id: str,
    convention: str,
    steps: list[dict],
    summary: dict,
    reward_frames: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    css = """
    body{font-family:Georgia,serif;max-width:1200px;margin:0 auto;padding:2rem;color:#1a1d23}
    h1,h2{font-family:Helvetica,Arial,sans-serif}
    .grids{display:flex;flex-wrap:wrap;gap:.6rem}
    .grid-wrap{border:1px solid #d7dde4;border-radius:8px;padding:.4rem;background:#f6f8fa}
    .grid-title{font-size:.75rem;color:#5c6470;margin-bottom:.2rem}
    table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
    th,td{border:1px solid #d7dde4;padding:.4rem .6rem;vertical-align:top}
    th{background:#f6f8fa}
    .step{border-top:1px solid #d7dde4;padding:1.2rem 0}
    .boundary{background:#fff1f2;border:2px solid #9f1239;border-radius:10px;padding:1rem;margin:1.5rem 0}
    .signals{font-size:.85rem;color:#334155}
    code{font-family:monospace;background:#eef2f7;padding:.1em .3em;border-radius:3px}
    """
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>0016 Oracle Checker Report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>0016 — Oracle Checker Report</h1>",
        f"<p>game_id: <code>{escape(game_id)}</code> · alignment: <code>{escape(convention)}</code></p>",
        "<h2>Summary dashboard</h2>",
        "<table><tr><th>Level</th><th>perception</th><th>grammar</th>"
        "<th>explained_dynamics</th><th>underfit</th><th>other</th><th>steps</th></tr>",
    ]
    for lvl, counts in sorted(summary["by_level"].items()):
        parts.append(
            f"<tr><td>{lvl}</td><td>{counts.get('perception',0)}</td>"
            f"<td>{counts.get('grammar',0)}</td>"
            f"<td>{counts.get('explained_dynamics',0)}</td>"
            f"<td>{counts.get('underfit',0)}</td>"
            f"<td>{counts.get('other',0)}</td><td>{counts.get('steps',0)}</td></tr>"
        )
    parts.append("</table>")
    parts.append("<h3>Reward-frame predicate vectors</h3><table>")
    parts.append(
        "<tr><th>Level completed</th><th>Frame idx</th><th>Avatar pos</th>"
        "<th>On-target pairs</th><th>Match facts (on-target)</th></tr>"
    )
    for rf in reward_frames:
        parts.append(
            f"<tr><td>{rf['level_completed']}</td><td>{rf['index']}</td>"
            f"<td>{rf['vector']['avatar_position']}</td>"
            f"<td>{sum(1 for p in rf['vector']['pairs'] if p['on_target'])}</td>"
            f"<td><code>{escape(str(rf['vector']['candidate_predicates']))}</code></td></tr>"
        )
    parts.append("</table>")

    if summary.get("level_boundary_highlight"):
        lb = summary["level_boundary_highlight"]
        parts.append(
            f'<div class="boundary"><h2>Level 1 → Level 2 boundary</h2>'
            f"<p>Increment at frame {lb['increment_index']}. "
            f"Level 1 divergence candidates: perception={lb['level1']['perception']}, "
            f"grammar={lb['level1']['grammar']}, underfit={lb['level1']['underfit']}. "
            f"Level 2 (first segment): perception={lb['level2']['perception']}, "
            f"grammar={lb['level2']['grammar']}, underfit={lb['level2']['underfit']}.</p>"
            f"<p><strong>Headline:</strong> {escape(lb['headline'])}</p></div>"
        )

    parts.append("<h2>Per-step detail</h2>")
    for step in steps:
        parts.append(f'<div class="step"><h3>Step {step["index"]} · level {step["human_level"]} · {escape(step["action"])} · |diff|={step["diff_cells"]}</h3>')
        parts.append('<div class="grids">')
        parts.append(_grid_svg(step["predicted"], title="predicted"))
        parts.append(_grid_svg(step["actual"], title="actual"))
        parts.append(
            _grid_svg(
                step["actual"],
                title="diff mask",
                highlight=np.array(step["diff_mask"], dtype=bool),
            )
        )
        parts.append(_overlay_svg(np.array(step["actual"]), step["views"], step["belief_snapshot"]))
        parts.append("</div>")
        if step["clusters"]:
            parts.append("<p>Diff clusters: ")
            parts.append(escape(str([c.as_dict() for c in step["clusters"]])))
            parts.append("</p>")
        for i, sig in enumerate(step["signals"]):
            parts.append(
                f'<p class="signals">Cluster {i}: perception={sig["perception_ok"]} ({sig["perception_note"]}); '
                f'grammar={sig["grammar_ok"]} ({sig["grammar_note"]}); '
                f'underfit prior={sig["prior_instances"]} ({sig["underfit_note"]})</p>'
            )
        parts.append("</div>")

    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main checker loop
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    index: int
    human_level: int
    action: str
    predicted: np.ndarray
    actual: np.ndarray
    diff_mask: np.ndarray
    diff_cells: int
    clusters: list[DiffCluster]
    signals: list[dict]
    views: list
    belief_snapshot: AgentBelief


def human_level(levels_completed: int) -> int:
    return levels_completed + 1


def segments_for_levels(segments: list[Segment], levels: set[int]) -> list[Segment]:
    return [s for s in segments if human_level(s.level) in levels]


def run_checker(
    records: list[TraceRecord],
    segments: list[Segment],
    *,
    levels: set[int],
) -> tuple[list[StepResult], list[dict], dict]:
    belief = AgentBelief()
    step_results: list[StepResult] = []
    reward_frames: list[dict] = []
    type_counts: dict[int, Counter] = defaultdict(Counter)

    for seg in segments_for_levels(segments, levels):
        belief.reset_episode()
        # Reward on the increment frame (segment start after a level-up), not the next frame.
        if (
            seg.start > 0
            and records[seg.start].levels_completed
            > records[seg.start - 1].levels_completed
        ):
            inc = records[seg.start]
            views_inc = belief.perceive(inc.frame)
            reward_frames.append(
                {
                    "index": inc.index,
                    "level_completed": inc.levels_completed,
                    "vector": reward_predicate_vector(inc.frame, views_inc, belief),
                }
            )
        for t in range(seg.start, seg.end):
            if t + 1 > seg.end:
                break
            nxt = records[t + 1]
            if nxt.action_num is None or "RESET" in nxt.action_id:
                continue
            if records[t].levels_completed != nxt.levels_completed:
                continue

            grid_t = records[t].frame
            grid_t1 = nxt.frame
            action = nxt.action_num
            views_t = belief.perceive(grid_t)

            predicted = predict_frame(grid_t, action, views_t, belief)
            diff_mask = predicted != grid_t1
            clusters = diff_clusters(diff_mask)
            _bg_t, parts_t = il.views_from_grid(grid_t, frame=t)
            _bg_t1, parts_t1 = il.views_from_grid(grid_t1, frame=t + 1)
            signals = [
                type_divergence_signals(
                    cl, views_t, grid_t, grid_t1, belief, action, parts_t, parts_t1
                )
                for cl in clusters
            ]
            for sig in signals:
                type_counts[human_level(records[t].levels_completed)][
                    classify_candidate(sig)
                ] += 1
            type_counts[human_level(records[t].levels_completed)]["steps"] += 1

            step_results.append(
                StepResult(
                    index=t,
                    human_level=human_level(records[t].levels_completed),
                    action=nxt.action_id,
                    predicted=predicted,
                    actual=grid_t1,
                    diff_mask=diff_mask,
                    diff_cells=int(diff_mask.sum()),
                    clusters=clusters,
                    signals=signals,
                    views=views_t,
                    belief_snapshot=belief,
                )
            )

            belief.learn_remote_dynamics(parts_t, parts_t1)
            belief.observe_transition(grid_t, action, grid_t1)
            belief.frame_idx += 1
            belief.perceive(grid_t1)

    summary = {"by_level": {k: dict(v) for k, v in type_counts.items()}}
    return step_results, reward_frames, summary


def headline_comparison(summary: dict, increment_index: int | None) -> dict | None:
    by = summary.get("by_level", {})
    l1 = by.get(1, {})
    l2 = by.get(2, {})
    if not l1 and not l2:
        return None
    g1, g2 = l1.get("grammar", 0), l2.get("grammar", 0)
    u1, u2 = l1.get("underfit", 0), l2.get("underfit", 0)
    p1, p2 = l1.get("perception", 0), l2.get("perception", 0)
    if g2 > g1 + u2:
        headline = (
            "Level-2 divergences cluster as grammar-gaps more than level-1 — "
            "the dynamics model from level 1 does not cover level-2 remote changes."
        )
    elif u2 > u1:
        headline = (
            "Level-2 divergences are mostly underfit/perception — "
            "level-1 dynamics largely transfer; gaps are data starvation not schema."
        )
    else:
        headline = (
            "Mixed divergence profile at level 2 — inspect per-cluster signals in the report."
        )
    return {
        "increment_index": increment_index,
        "level1": {"perception": p1, "grammar": g1, "underfit": u1},
        "level2": {"perception": p2, "grammar": g2, "underfit": u2},
        "headline": headline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="0016 offline oracle-checker")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--levels", default="1,2", help="human level numbers e.g. 1,2")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "report.html",
        help="HTML report path (throwaway artifact)",
    )
    args = parser.parse_args()

    print("=== Step 0 — loader hygiene & alignment ===")
    records = load_trace(args.trace)
    game_id, convention, segments = hygiene_step0(records)

    print("\n=== Step 1 — instrument self-test ===")
    run_self_test()

    levels = {int(x) for x in args.levels.split(",") if x.strip()}
    print(f"\n=== Steps 2–5 — checker on human levels {sorted(levels)} ===")
    steps, reward_frames, summary = run_checker(records, segments, levels=levels)

    increment_idx = next(
        (r.index for r in records if r.levels_completed == 1 and records[r.index - 1].levels_completed == 0),
        None,
    )
    boundary = headline_comparison(summary, increment_idx)
    if boundary:
        summary["level_boundary_highlight"] = boundary

    print("\n=== Typed-divergence counts ===")
    for lvl in sorted(summary["by_level"]):
        c = summary["by_level"][lvl]
        print(
            f"  level {lvl}: perception={c.get('perception',0)} "
            f"grammar={c.get('grammar',0)} "
            f"explained_dynamics={c.get('explained_dynamics',0)} "
            f"underfit={c.get('underfit',0)} "
            f"other={c.get('other',0)} steps={c.get('steps',0)}"
        )
    if boundary:
        print(f"\n=== Level 1 → 2 headline ===")
        print(f"  {boundary['headline']}")

    print(f"\n=== Step 6 — HTML report ===")
    write_html_report(
        args.output,
        game_id=game_id,
        convention=convention,
        steps=[
            {
                "index": s.index,
                "human_level": s.human_level,
                "action": s.action,
                "predicted": s.predicted,
                "actual": s.actual,
                "diff_mask": s.diff_mask,
                "diff_cells": s.diff_cells,
                "clusters": s.clusters,
                "signals": s.signals,
                "views": s.views,
                "belief_snapshot": s.belief_snapshot,
            }
            for s in steps
        ],
        summary=summary,
        reward_frames=reward_frames,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
