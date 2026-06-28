"""0016 depth queries — discriminators answerable from the human trace.

Run from repo root:
    uv run python teach/lessons/0016-depth-queries.py
    uv run python teach/lessons/0016-depth-queries.py --only 4,5

Uses the oracle-checker trace walk only for stepping; does not tune the agent.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CELL = 5  # synced with oc16 after load

_REPO = Path(__file__).resolve().parents[2]
_spec = _ilu.spec_from_file_location(
    "oc16", _REPO / "teach/lessons/0016-oracle-checker.py"
)
oc16 = _ilu.module_from_spec(_spec)
_sys.modules[_spec.name] = oc16
_spec.loader.exec_module(oc16)

CELL = oc16.CELL
gi = oc16.gi
il = oc16.il
ig = oc16.ig
LATTICE_NODES = il.LATTICE_NODES
GOAL_SLOTS = gi.GOAL_SLOTS
CHANGE_SLOTS = ("color", "shape", "internal_pattern", "orientation")


def _perceive(grid: np.ndarray, belief: oc16.AgentBelief, frame_idx: int) -> list:
    _bg, parts = il.views_from_grid(grid, frame=frame_idx)
    if belief.previous_views:
        belief.grouper.observe(belief.previous_views, parts)
    views = belief.grouper.group(grid, parts, frame=frame_idx)
    belief.tracker.update(views)
    belief.previous_views = views
    return views


def _walk_belief_to(records, end_idx: int) -> oc16.AgentBelief:
    """Replay trace transitions only to build controllability/falsifier state."""
    belief = oc16.AgentBelief()
    belief.reset_episode()
    seg_for = {i: s for s in load_segments(records) for i in range(s.start, s.end + 1)}

    for t in range(1, end_idx + 1):
        if t - 1 not in seg_for or t not in seg_for:
            continue
        if seg_for[t - 1].start != seg_for[t].start:
            belief.reset_episode()
        prev, curr = records[t - 1], records[t]
        if curr.action_num is None or "RESET" in curr.action_id:
            continue
        if prev.levels_completed != curr.levels_completed:
            continue
        belief.observe_transition(prev.frame, curr.action_num, curr.frame)
        belief.frame_idx = t
        _perceive(curr.frame, belief, t)
    return belief


def _match_view_t1(view_t, views_t1: list):
    best, best_score = None, 0.0
    for v in views_t1:
        inter = len(view_t.pixels & v.pixels)
        union = len(view_t.pixels | v.pixels)
        iou = inter / union if union else 0.0
        if iou > best_score:
            best_score = iou
            best = v
    if best_score >= 0.15:
        return best
    # centroid fallback within 2 cells
    ar, ac = view_t.centroid
    near = [
        v
        for v in views_t1
        if abs(v.centroid[0] - ar) <= 10 and abs(v.centroid[1] - ac) <= 10
    ]
    if not near:
        return None
    return min(
        near,
        key=lambda v: abs(v.centroid[0] - ar) + abs(v.centroid[1] - ac),
    )


def _lattice_change_sig(view_t, view_t1) -> tuple[tuple[str, str], ...]:
    changes: list[tuple[str, str]] = []
    for slot in CHANGE_SLOTS:
        lat_a = view_t.lattice(slot)
        lat_b = view_t1.lattice(slot)
        for lens in lat_a.keys() & lat_b.keys():
            if lat_a[lens] != lat_b[lens]:
                changes.append((slot, lens))
    return tuple(sorted(set(changes)))


def _remote_change_type(
    view_t, view_t1, grid_t: np.ndarray, grid_t1: np.ndarray, cluster: oc16.DiffCluster
) -> str:
    """Coarse autonomous-change TYPE (not instance count)."""
    if view_t is None:
        return "orphan_diff:no_view_t"
    v1 = view_t1 or _match_view_t1(view_t, _views_from_parts(grid_t1))
    if v1 is None:
        return f"orphan_diff:{ig.shape_literal(view_t)}:vanished"
    sig = _lattice_change_sig(view_t, v1)
    if not sig:
        # diff in region but lattices identical — pixel/layout shift or composite seam
        area_d = v1.area - view_t.area
        if abs(area_d) > 2:
            return f"area_morph:{ig.shape_literal(view_t)}:d{area_d:+d}"
        return f"pixel_shift:{ig.shape_literal(view_t)}"
    # compress to primary change letter
    slots = {s for s, _ in sig}
    lenses = {ln for _, ln in sig}
    if slots == {"color"}:
        return f"color_cycle:{ig.shape_literal(view_t)}"
    if "shape" in slots and "rotation_invariant" in lenses:
        return f"rotate:{ig.shape_literal(view_t)}"
    if "shape" in slots:
        return f"reshape:{ig.shape_literal(view_t)}"
    if "internal_pattern" in slots:
        return f"pattern:{ig.shape_literal(view_t)}:{','.join(sorted(lenses))}"
    if "orientation" in slots:
        return f"orient:{ig.shape_literal(view_t)}"
    return f"multi:{ig.shape_literal(view_t)}:{sig}"


def _views_from_parts(grid: np.ndarray) -> list:
    _bg, parts = il.views_from_grid(grid, frame=0)
    grouper = il.CompositeObjectGrouper()
    return grouper.group(grid, parts, frame=0)


def load_segments(records) -> list:
    """Segment table without re-printing hygiene."""
    segments: list = []
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
                oc16.Segment(
                    seg_start,
                    i - 1,
                    records[seg_start].levels_completed,
                    i - seg_start,
                )
            )
            seg_start = i
    if seg_start < len(records):
        segments.append(
            oc16.Segment(
                seg_start,
                len(records) - 1,
                records[seg_start].levels_completed,
                len(records) - seg_start,
            )
        )
    return segments


def _centroid_disp(view_t, view_t1) -> tuple[float, float]:
    return (
        view_t1.centroid[0] - view_t.centroid[0],
        view_t1.centroid[1] - view_t.centroid[1],
    )


def _views_adjacent(a, b) -> bool:
    if a.pixels & b.pixels:
        return True
    for r, c in a.pixels:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in b.pixels:
                return True
    return False


def _union_centroid(pixels: set) -> tuple[float, float]:
    if not pixels:
        return (0.0, 0.0)
    rs = [p[0] for p in pixels]
    cs = [p[1] for p in pixels]
    return (sum(rs) / len(rs), sum(cs) / len(cs))


def classify_pixel_shift(
    view_t,
    v1,
    views_t: list,
    views_t1: list,
) -> str:
    """Rigid translate vs composite segmenter seam vs ambiguous."""
    if view_t is None or v1 is None:
        return "ambiguous"
    if _lattice_change_sig(view_t, v1):
        return "not_pixel_shift"
    disp = _centroid_disp(view_t, v1)
    mag = abs(disp[0]) + abs(disp[1])

    for other in views_t:
        if other is view_t:
            continue
        if not _views_adjacent(view_t, other):
            continue
        o1 = _match_view_t1(other, views_t1)
        if o1 is None:
            continue
        if _lattice_change_sig(other, o1):
            continue
        odisp = _centroid_disp(other, o1)
        omag = abs(odisp[0]) + abs(odisp[1])
        if mag < 2 or omag < 2:
            continue
        # Equal-and-opposite partial shifts?
        if abs(disp[0] + odisp[0]) < 3 and abs(disp[1] + odisp[1]) < 3:
            union_t = set(view_t.pixels) | set(other.pixels)
            union_t1 = set(v1.pixels) | set(o1.pixels)
            udisp = (
                _union_centroid(union_t1)[0] - _union_centroid(union_t)[0],
                _union_centroid(union_t1)[1] - _union_centroid(union_t)[1],
            )
            if abs(udisp[0]) + abs(udisp[1]) < CELL * 0.6:
                return "segmenter_seam"
    if mag >= CELL * 0.45:
        return "rigid_translate"
    if mag < 2:
        return "lattice_stable_no_motion"
    return "ambiguous"


def _increment_indices(records) -> list[int]:
    out = [
        i
        for i in range(1, len(records))
        if records[i].levels_completed > records[i - 1].levels_completed
    ]
    return out


def _object_transition(view_t, view_t1) -> str:
    if view_t is None and view_t1 is not None:
        return "appeared"
    if view_t is not None and view_t1 is None:
        return "disappeared"
    if view_t is None:
        return "unknown"
    sig = _lattice_change_sig(view_t, view_t1)
    if not sig:
        dr, dc = _centroid_disp(view_t, view_t1)
        if abs(dr) + abs(dc) >= CELL * 0.45:
            return "moved_lattice_stable"
        return "unchanged"
    slots = {s for s, _ in sig}
    if slots == {"color"}:
        return "color_changed"
    if "shape" in slots:
        return "shape_changed"
    return f"changed:{sig}"


def query4_pixel_shift_split(records) -> None:
    print("\n" + "=" * 72)
    print("QUERY 4 — pixel_shift: rigid dynamic vs composite segmenter seam")
    print("=" * 72)

    counts: Counter = Counter()
    by_level: dict[int, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)

    belief = oc16.AgentBelief()
    segments = load_segments(records)

    for seg in segments:
        if oc16.human_level(seg.level) > 7:
            continue
        belief.reset_episode()
        lvl = oc16.human_level(seg.level)
        for t in range(seg.start, seg.end):
            if t + 1 > seg.end:
                break
            nxt = records[t + 1]
            if nxt.action_num is None or "RESET" in nxt.action_id:
                continue
            if records[t].levels_completed != nxt.levels_completed:
                continue
            grid_t, grid_t1 = records[t].frame, nxt.frame
            views_t = _perceive(grid_t, belief, t)
            _bg, parts1 = il.views_from_grid(grid_t1, frame=t + 1)
            if belief.previous_views:
                belief.grouper.observe(belief.previous_views, parts1)
            views_t1 = belief.grouper.group(grid_t1, parts1, frame=t + 1)

            actor_views = oc16._actor_views(views_t, belief.controlled_sigs)
            for cluster in oc16.diff_clusters(
                oc16.predict_frame(grid_t, nxt.action_num, views_t, belief) != grid_t1
            ):
                kind = oc16._change_kind(grid_t, grid_t1, cluster, actor_views)
                if kind != "remote_attribute_change":
                    continue
                cr, cc = (cluster.rmin + cluster.rmax) // 2, (cluster.cmin + cluster.cmax) // 2
                view_t = oc16._view_at(views_t, cr, cc)
                v1 = _match_view_t1(view_t, views_t1) if view_t else None
                ctype = _remote_change_type(view_t, v1, grid_t, grid_t1, cluster)
                if not ctype.startswith("pixel_shift:"):
                    continue
                verdict = classify_pixel_shift(view_t, v1, views_t, views_t1)
                counts[verdict] += 1
                by_level[lvl][verdict] += 1
                if len(examples[verdict]) < 3:
                    examples[verdict].append(
                        (t, lvl, ctype, disp if view_t and v1 else None)
                        if (disp := _centroid_disp(view_t, v1) if view_t and v1 else None)
                        else (t, lvl, ctype, None)
                    )

            belief.observe_transition(grid_t, nxt.action_num, grid_t1)
            belief.frame_idx = t + 1
            belief.previous_views = views_t1

    total = sum(counts.values())
    print(f"\npixel_shift instances: {total}")
    for verdict, n in counts.most_common():
        pct = 100 * n / max(1, total)
        print(f"  {verdict:<28} {n:>5}  ({pct:.1f}%)")
    print("\nBy level (rigid vs seam):")
    for lvl in range(1, 8):
        c = by_level[lvl]
        if not c:
            continue
        rigid = c.get("rigid_translate", 0)
        seam = c.get("segmenter_seam", 0)
        print(f"  L{lvl}: rigid={rigid} seam={seam} other={sum(c.values())-rigid-seam}")
    print("\nExamples:")
    for verdict, exs in examples.items():
        for ex in exs:
            print(f"  {verdict}: step {ex[0]} L{ex[1]} {ex[2]} disp={ex[3]}")


def query5_win_change_scan(records) -> None:
    print("\n" + "=" * 72)
    print("QUERY 5 — Win instant CHANGE scan (t-1 → t), all objects")
    print("=" * 72)

    for inc in _increment_indices(records):
        lvl = records[inc].levels_completed
        g0, g1 = records[inc - 1].frame, records[inc].frame
        diff_cells = int(np.sum(g0 != g1))
        print(
            f"\n--- Level {lvl} @ frames {inc-1}→{inc} "
            f"(action={records[inc].action_id}) |diff|={diff_cells} ---"
        )

        _bg0, parts0 = il.views_from_grid(g0, frame=inc - 1)
        _bg1, parts1 = il.views_from_grid(g1, frame=inc)
        # Parts-level inventory (no composite merge — catches HUD swatches)
        matched_t1: set[int] = set()
        transitions: Counter = Counter()
        changed_objects: list[str] = []

        for v0 in parts0:
            best, best_iou = None, 0.0
            for j, v1 in enumerate(parts1):
                inter = len(v0.pixels & v1.pixels)
                union = len(v0.pixels | v1.pixels)
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best = iou, (j, v1)
            if best is None or best_iou < 0.15:
                tr = "disappeared"
                v1 = None
            else:
                matched_t1.add(best[0])
                v1 = best[1]
                tr = _object_transition(v0, v1)
            transitions[tr] += 1
            if tr not in ("unchanged", "moved_lattice_stable"):
                changed_objects.append(
                    f"{ig.shape_literal(v0)}@{v0.centroid[0]:.0f},{v0.centroid[1]:.0f} → {tr}"
                )

        for j, v1 in enumerate(parts1):
            if j not in matched_t1:
                transitions["appeared"] += 1
                changed_objects.append(
                    f"{ig.shape_literal(v1)}@{v1.centroid[0]:.0f},{v1.centroid[1]:.0f} → appeared"
                )

        print(f"  Object transitions: {dict(transitions)}")
        print("  Changed / appeared / disappeared:")
        for line in changed_objects[:14]:
            print(f"    {line}")
        if len(changed_objects) > 14:
            print(f"    ... +{len(changed_objects)-14} more")

        # Actor vs ALL parts (not target-filtered) — any equality at win instant?
        belief = _walk_belief_to(records, inc)
        belief.reset_episode()
        views = _perceive(g1, belief, inc)
        actors = oc16._actor_views(views, belief.controlled_sigs)
        eq_hits: list[str] = []
        if actors:
            for ai, actor in enumerate(actors):
                for v1 in parts1:
                    for slot in GOAL_SLOTS:
                        lens = il.cheapest_equal(slot, actor, v1)
                        if lens:
                            eq_hits.append(
                                f"actor[{ai}] vs part {ig.shape_literal(v1)} "
                                f"{slot}@{lens} on_t={gi.on_target(actor,v1)}"
                            )
        print(f"  Actor×ALL-parts cheapest_equal hits: {len(eq_hits)}")
        for h in eq_hits[:8]:
            print(f"    {h}")

        # Win-unique: changes at this step vs median diff on prior level segment
        seg = next(s for s in load_segments(records) if s.start <= inc - 1 <= s.end)
        prior_diffs = []
        for t in range(max(seg.start, inc - 20), inc - 1):
            if t + 1 >= inc:
                break
            if records[t].levels_completed != records[t + 1].levels_completed:
                continue
            prior_diffs.append(int(np.sum(records[t].frame != records[t + 1].frame)))
        med = sorted(prior_diffs)[len(prior_diffs) // 2] if prior_diffs else 0
        print(
            f"  |diff| at win={diff_cells} vs median prior step on level={med} "
            f"({'WIN-SPIKE' if diff_cells > med * 2 else 'typical'})"
        )


def query1_autonomous_change_types(records, steps) -> None:
    print("\n" + "=" * 72)
    print("QUERY 1 — Count TYPES of autonomous change, not instances")
    print("=" * 72)

    by_level_types: dict[int, Counter] = defaultdict(Counter)
    by_level_instances: dict[int, Counter] = defaultdict(Counter)
    global_types: Counter = Counter()

    for s in steps:
        grid_t = s.actual  # step stores actual as t+1; need t from records
        # StepResult.index is t; frames are records[t] -> records[t+1]
        # Re-derive from stored views: use belief walk indices
        rec_t = records[s.index]
        rec_t1 = records[s.index + 1]
        grid_t = rec_t.frame
        grid_t1 = rec_t1.frame
        views_t1 = _perceive(grid_t1, oc16.AgentBelief(), s.index + 1)

        for cluster, sig in zip(s.clusters, s.signals):
            if sig["kind"] != "remote_attribute_change":
                continue
            cr, cc = (cluster.rmin + cluster.rmax) // 2, (cluster.cmin + cluster.cmax) // 2
            view_t = oc16._view_at(s.views, cr, cc)
            v1 = _match_view_t1(view_t, views_t1) if view_t else None
            ctype = _remote_change_type(view_t, v1, grid_t, grid_t1, cluster)
            lvl = s.human_level
            by_level_types[lvl][ctype] += 1
            by_level_instances[lvl][ctype] += 1
            global_types[ctype] += 1

    # Re-run with proper views_t1 from same step context
    by_level_types = defaultdict(Counter)
    global_types = Counter()
    type_first_level: dict[str, int] = {}

    belief = oc16.AgentBelief()
    segments = load_segments(records)
    active_segments = [s for s in segments if oc16.human_level(s.level) <= 7]

    for seg in active_segments:
        belief.reset_episode()
        for t in range(seg.start, seg.end):
            if t + 1 > seg.end:
                break
            nxt = records[t + 1]
            if nxt.action_num is None or "RESET" in nxt.action_id:
                continue
            if records[t].levels_completed != nxt.levels_completed:
                continue
            grid_t, grid_t1 = records[t].frame, nxt.frame
            views_t = _perceive(grid_t, belief, t)
            predicted = oc16.predict_frame(grid_t, nxt.action_num, views_t, belief)
            diff = predicted != grid_t1
            clusters = oc16.diff_clusters(diff)
            _bg, parts1 = il.views_from_grid(grid_t1, frame=t + 1)
            if belief.previous_views:
                belief.grouper.observe(belief.previous_views, parts1)
            views_t1 = belief.grouper.group(grid_t1, parts1, frame=t + 1)

            actor_views = oc16._actor_views(views_t, belief.controlled_sigs)
            for cluster in clusters:
                cr, cc = (cluster.rmin + cluster.rmax) // 2, (cluster.cmin + cluster.cmax) // 2
                kind = oc16._change_kind(grid_t, grid_t1, cluster, actor_views)
                if kind != "remote_attribute_change":
                    continue
                view_t = oc16._view_at(views_t, cr, cc)
                v1 = _match_view_t1(view_t, views_t1) if view_t else None
                ctype = _remote_change_type(view_t, v1, grid_t, grid_t1, cluster)
                lvl = oc16.human_level(records[t].levels_completed)
                by_level_types[lvl][ctype] += 1
                global_types[ctype] += 1
                type_first_level.setdefault(ctype, lvl)

            belief.observe_transition(grid_t, nxt.action_num, grid_t1)
            belief.frame_idx = t + 1
            belief.previous_views = views_t1
            belief.tracker.update(views_t1)

    print(f"\nDistinct autonomous-change TYPES (full trace): {len(global_types)}")
    print(f"Total remote-change INSTANCES: {sum(global_types.values())}\n")
    print(f"{'Type':<55} {'instances':>9}  first_lvl")
    for ctype, count in global_types.most_common():
        print(f"  {ctype:<53} {count:>9}  L{type_first_level.get(ctype, '?')}")

    print("\nDistinct types per level (cardinality — scopes 0016 alphabet):")
    print(f"  {'Lvl':>3}  {'types':>5}  {'instances':>9}  {'inst/type':>9}")
    for lvl in range(1, 8):
        types_c = by_level_types[lvl]
        inst = sum(types_c.values())
        ntypes = len(types_c)
        ratio = inst / max(1, ntypes)
        print(f"  {lvl:>3}  {ntypes:>5}  {inst:>9}  {ratio:>9.1f}")

    print("\nPhase-change check (new types entering at L5/L6):")
    types_l4 = set(by_level_types[4].keys())
    types_l5 = set(by_level_types[5].keys())
    types_l6 = set(by_level_types[6].keys())
    new_l5 = types_l5 - types_l4
    new_l6 = types_l6 - types_l5
    print(f"  types new at L5 (not seen L4): {len(new_l5)}")
    for t in sorted(new_l5)[:12]:
        print(f"    + {t} ({by_level_types[5][t]} inst)")
    print(f"  types new at L6 (not seen L5): {len(new_l6)}")
    for t in sorted(new_l6)[:12]:
        print(f"    + {t} ({by_level_types[6][t]} inst)")


def query2_win_frame_equality(records) -> None:
    print("\n" + "=" * 72)
    print("QUERY 2 — Win-frame equality matrix (discriminate a/b/c)")
    print("=" * 72)

    increment_indices = []
    for i in range(1, len(records)):
        if records[i].levels_completed > records[i - 1].levels_completed:
            increment_indices.append(i)
    # include terminal WIN if distinct
    if records[-1].state == "WIN":
        increment_indices.append(len(records) - 1)

    print(f"Increment / WIN frames: {increment_indices}\n")

    for inc_idx in increment_indices:
        lvl_done = records[inc_idx].levels_completed
        print(f"--- Level {lvl_done} complete @ frame {inc_idx} "
              f"(action={records[inc_idx].action_id}, state={records[inc_idx].state}) ---")

        frame_candidates = {
            "t-1": inc_idx - 1,
            "t (increment)": inc_idx,
            "t+1": inc_idx + 1 if inc_idx + 1 < len(records) else None,
        }

        best_any = None
        for label, fidx in frame_candidates.items():
            if fidx is None or fidx < 0 or fidx >= len(records):
                print(f"  [{label}] — n/a")
                continue
            belief = _walk_belief_to(records, fidx)
            belief.reset_episode()
            grid = records[fidx].frame
            views = _perceive(grid, belief, fidx)
            actors = oc16._actor_views(views, belief.controlled_sigs)
            if not actors:
                print(f"  [{label}] frame {fidx}: NO ACTOR")
                continue
            primary = actors[0]
            matches: list[dict] = []
            for target in views:
                if ig.is_actor_view(target, belief.controlled_sigs):
                    continue
                row = {
                    "target": ig.shape_literal(target),
                    "on_target": gi.on_target(primary, target),
                    "cheapest": {},
                    "any_lens": [],
                }
                for slot in GOAL_SLOTS:
                    lens = il.cheapest_equal(slot, primary, target)
                    if lens:
                        row["cheapest"][slot] = lens
                    for under in LATTICE_NODES:
                        if il.equal(slot, primary, target, under):
                            row["any_lens"].append(f"{slot}@{under}")
                if row["cheapest"] or row["on_target"]:
                    matches.append(row)

            n_match = len(matches)
            n_on = sum(1 for m in matches if m["on_target"])
            n_cheap = sum(1 for m in matches if m["cheapest"])
            print(
                f"  [{label}] frame {fidx}: pairs_with_any_equality={n_match} "
                f"on_target={n_on} cheapest_equal={n_cheap}"
            )
            for m in matches[:5]:
                print(
                    f"      {m['target']} on_target={m['on_target']} "
                    f"cheapest={m['cheapest']} lenses={m['any_lens'][:4]}"
                )
            if n_match > 0 and (best_any is None or n_match > best_any[0]):
                best_any = (n_match, label, fidx)

        if best_any:
            print(f"  → Best frame for equality hits: {best_any[1]} (frame {best_any[2]}, "
                  f"{best_any[0]} pairs)")
        else:
            print("  → NO pair equal under ANY lens at t-1, t, or t+1")

        # Full pairwise scan at increment frame only
        belief = _walk_belief_to(records, inc_idx)
        belief.reset_episode()
        views = _perceive(records[inc_idx].frame, belief, inc_idx)
        actors = oc16._actor_views(views, belief.controlled_sigs)
        if actors:
            primary = actors[0]
            total_pairs = 0
            any_eq_pairs = 0
            for target in views:
                if ig.is_actor_view(target, belief.controlled_sigs):
                    continue
                total_pairs += 1
                found = False
                for slot in GOAL_SLOTS:
                    for under in LATTICE_NODES:
                        if il.equal(slot, primary, target, under):
                            found = True
                            break
                    if found:
                        break
                if found:
                    any_eq_pairs += 1
            print(
                f"  Full matrix @ increment: {any_eq_pairs}/{total_pairs} targets "
                f"have actor==target under some slot/lens"
            )
        print()


def query3_l7_perception_split(records, steps) -> None:
    print("\n" + "=" * 72)
    print("QUERY 3 — L7 perception misses: static vs grammar-propagation")
    print("=" * 72)

    l7_steps = [s for s in steps if s.human_level == 7]
    static_count = 0
    dynamic_count = 0
    grammar_overlap = 0
    total_perc = 0

    for s in l7_steps:
        t = s.index
        if t < 2:
            continue
        grid_tm1 = records[t - 1].frame
        grid_t = records[t].frame
        grid_t1 = s.actual

        for cluster, sig in zip(s.clusters, s.signals):
            if oc16.classify_candidate(sig) != "perception":
                continue
            total_perc += 1
            r0, c0, r1, c1 = cluster.rmin, cluster.cmin, cluster.rmax, cluster.cmax
            # Static: region identical at t-1 and t (before this action's unpredicted change)
            stable_before = np.array_equal(
                grid_tm1[r0 : r1 + 1, c0 : c1 + 1],
                grid_t[r0 : r1 + 1, c0 : c1 + 1],
            )
            # Actual change magnitude this step in bbox
            actual_delta = int(
                np.sum(grid_t[r0 : r1 + 1, c0 : c1 + 1] != grid_t1[r0 : r1 + 1, c0 : c1 + 1])
            )
            # Was this cluster also a grammar-gap?
            is_grammar = sig["kind"] == "remote_attribute_change" or not sig["grammar_ok"]

            if stable_before and actual_delta < 20:
                static_count += 1
                bucket = "STATIC (real perception drift)"
            else:
                dynamic_count += 1
                bucket = "DYNAMIC (likely grammar-gap wearing perception clothes)"
            if is_grammar:
                grammar_overlap += 1
            if total_perc <= 8:
                print(
                    f"  step {t} cluster@{r0},{c0} stable_before={stable_before} "
                    f"actual_delta={actual_delta} grammar_kind={sig['kind']} → {bucket}"
                )

    print(f"\nL7 perception-miss clusters: {total_perc}")
    print(f"  STATIC (region stable before step, small delta):  {static_count}")
    print(f"  DYNAMIC (region was changing or large delta):     {dynamic_count}")
    print(f"  Overlap with grammar-gap kind:                    {grammar_overlap}")
    if total_perc:
        print(
            f"  → {100*dynamic_count/total_perc:.0f}% plausibly grammar-propagation, "
            f"{100*static_count/total_perc:.0f}% independent perception"
        )


def _part_eq_with_actor(actor, part) -> dict:
    out = {}
    for slot in GOAL_SLOTS:
        lens = il.cheapest_equal(slot, actor, part)
        if lens:
            out[slot] = lens
    return out


def _match_part(view, parts: list, min_iou: float = 0.15):
    """Identity by pixel overlap then centroid proximity."""
    best, best_iou = None, 0.0
    for p in parts:
        inter = len(view.pixels & p.pixels)
        union = len(view.pixels | p.pixels)
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best_iou, best = iou, p
    if best is not None and best_iou >= min_iou:
        return best, best_iou
    ar, ac = view.centroid
    near = [
        p
        for p in parts
        if abs(p.centroid[0] - ar) <= 6 and abs(p.centroid[1] - ac) <= 6
    ]
    if near:
        return min(near, key=lambda p: abs(p.centroid[0] - ar) + abs(p.centroid[1] - ac)), 0.0
    return None, 0.0


def query6_l7_conditional_delta(records) -> None:
    """The naked win predicate: L7 reward fires WITHOUT a layout reset.

    Hold eq_hits constant across 404→405; find the RARE flip given equality
    already held. That flip is the win predicate or its proximate trigger.
    """
    print("\n" + "=" * 72)
    print("QUERY 6 — L7 conditional delta (404→405), equality already holding")
    print("=" * 72)

    # Confirm L7 win is the only reset-free win.
    incs = _increment_indices(records)
    print("\nWin |diff| (reset-confound check):")
    for inc in incs:
        d = int(np.sum(records[inc - 1].frame != records[inc].frame))
        tag = "NAKED (no layout reset)" if d < 400 else "reset-confounded"
        print(f"  L{records[inc].levels_completed} @ {inc}: |diff|={d:>5}  {tag}")

    a_idx, b_idx = 404, 405
    g0, g1 = records[a_idx].frame, records[b_idx].frame
    print(f"\nReading {a_idx}→{b_idx}: |diff|={int(np.sum(g0!=g1))} cells, "
          f"action={records[b_idx].action_id}, state={records[b_idx].state}")

    belief0 = _walk_belief_to(records, a_idx)
    belief0.reset_episode()
    views0 = _perceive(g0, belief0, a_idx)
    actors0 = oc16._actor_views(views0, belief0.controlled_sigs)
    _bg0, parts0 = il.views_from_grid(g0, frame=a_idx)
    _bg1, parts1 = il.views_from_grid(g1, frame=b_idx)

    belief1 = _walk_belief_to(records, b_idx)
    belief1.reset_episode()
    views1 = _perceive(g1, belief1, b_idx)
    actors1 = oc16._actor_views(views1, belief1.controlled_sigs)

    print(f"\nActors: {len(actors0)} @404, {len(actors1)} @405")

    # --- eq_hits at 404 vs 405, per actor, with on_target tracked ---
    print("\nActor↔part equality + on_target, 404 vs 405:")
    for ai, actor0 in enumerate(actors0):
        actor1 = actors1[ai] if ai < len(actors1) else None
        hits0 = []
        for p in parts0:
            eq = _part_eq_with_actor(actor0, p)
            if eq:
                hits0.append((p, eq, gi.on_target(actor0, p)))
        hits1 = []
        if actor1 is not None:
            for p in parts1:
                eq = _part_eq_with_actor(actor1, p)
                if eq:
                    hits1.append((p, eq, gi.on_target(actor1, p)))
        on0 = sum(1 for _, _, o in hits0 if o)
        on1 = sum(1 for _, _, o in hits1 if o)
        print(
            f"  actor[{ai}] @404: eq_hits={len(hits0)} on_target={on0} "
            f"pos={tuple(round(x) for x in actor0.centroid)}"
        )
        print(
            f"  actor[{ai}] @405: eq_hits={len(hits1) if actor1 else 'n/a'} "
            f"on_target={on1} "
            f"pos={tuple(round(x) for x in actor1.centroid) if actor1 else None}"
        )
        # on_target flip with equality holding?
        if on1 > on0:
            print(f"    → on_target FLIPPED up ({on0}→{on1}) with equality held "
                  f"= candidate (eq ∧ on_target)")

    # --- The rare flip: object transitions in the 284-cell delta ---
    print("\nObject-level delta 404→405 (the rare flip given equality held):")
    matched1: set[int] = set()
    flips: list[str] = []
    for v0 in parts0:
        p1, iou = _match_part(v0, parts1)
        if p1 is None:
            flips.append(f"{ig.shape_literal(v0)}@{v0.centroid[0]:.0f},{v0.centroid[1]:.0f} → DISAPPEARED")
            continue
        matched1.add(id(p1))
        tr = _object_transition(v0, p1)
        if tr not in ("unchanged",):
            flips.append(
                f"{ig.shape_literal(v0)}@{v0.centroid[0]:.0f},{v0.centroid[1]:.0f} → {tr}"
            )
    for p1 in parts1:
        if id(p1) not in matched1:
            flips.append(f"{ig.shape_literal(p1)}@{p1.centroid[0]:.0f},{p1.centroid[1]:.0f} → APPEARED")
    for line in flips:
        print(f"    {line}")
    print(f"  total flipped objects: {len(flips)}")

    # --- State-toggle check: same shape+position, pixels changed (the 16-bucket) ---
    print("\nState-toggle (recolor-in-place) flips at win — the 16-case bucket:")
    toggles = []
    for v0 in parts0:
        p1, iou = _match_part(v0, parts1)
        if p1 is None or iou < 0.5:
            continue
        dr, dc = _centroid_disp(v0, p1)
        if abs(dr) + abs(dc) >= 2:
            continue
        sig = _lattice_change_sig(v0, p1)
        raw_changed = (
            v0.internal_pattern_lattice["raw"] != p1.internal_pattern_lattice["raw"]
        )
        if raw_changed:
            slots = {s for s, _ in sig} if sig else set()
            toggles.append(
                f"{ig.shape_literal(v0)}@{v0.centroid[0]:.0f},{v0.centroid[1]:.0f} "
                f"slots={slots or 'pixel-only'} "
                f"on_target_actor={any(gi.on_target(a, v0) for a in actors0)}"
            )
    if toggles:
        for line in toggles:
            print(f"    {line}")
        print(f"  → {len(toggles)} in-place state changes AT the win frame "
              f"(dynamics-residue and goal-predicate may be the same phenomenon)")
    else:
        print("    none — win flip is not in-place recolor")


def query7_lattice_stable_locations(records) -> None:
    """Where do the 16 lattice_stable_no_motion (diff-without-motion) cases land?

    Check proximity to the L7 win frame and whether they co-occur with the actor.
    """
    print("\n" + "=" * 72)
    print("QUERY 7 — lattice_stable_no_motion bucket: locate the 16")
    print("=" * 72)

    belief = oc16.AgentBelief()
    segments = load_segments(records)
    cases: list[tuple] = []

    for seg in segments:
        if oc16.human_level(seg.level) > 7:
            continue
        belief.reset_episode()
        lvl = oc16.human_level(seg.level)
        for t in range(seg.start, seg.end):
            if t + 1 > seg.end:
                break
            nxt = records[t + 1]
            if nxt.action_num is None or "RESET" in nxt.action_id:
                continue
            if records[t].levels_completed != nxt.levels_completed:
                continue
            grid_t, grid_t1 = records[t].frame, nxt.frame
            views_t = _perceive(grid_t, belief, t)
            _bg, parts1 = il.views_from_grid(grid_t1, frame=t + 1)
            if belief.previous_views:
                belief.grouper.observe(belief.previous_views, parts1)
            views_t1 = belief.grouper.group(grid_t1, parts1, frame=t + 1)
            actor_views = oc16._actor_views(views_t, belief.controlled_sigs)

            for cluster in oc16.diff_clusters(
                oc16.predict_frame(grid_t, nxt.action_num, views_t, belief) != grid_t1
            ):
                kind = oc16._change_kind(grid_t, grid_t1, cluster, actor_views)
                if kind != "remote_attribute_change":
                    continue
                cr, cc = (cluster.rmin + cluster.rmax) // 2, (cluster.cmin + cluster.cmax) // 2
                view_t = oc16._view_at(views_t, cr, cc)
                v1 = _match_view_t1(view_t, views_t1) if view_t else None
                ctype = _remote_change_type(view_t, v1, grid_t, grid_t1, cluster)
                if not ctype.startswith("pixel_shift:"):
                    continue
                if classify_pixel_shift(view_t, v1, views_t, views_t1) != "lattice_stable_no_motion":
                    continue
                near_actor = (
                    min(
                        (abs(a.centroid[0] - cr) + abs(a.centroid[1] - cc) for a in actor_views),
                        default=999,
                    )
                )
                cases.append((t + 1, lvl, ctype, (cr, cc), round(near_actor, 1)))

            belief.observe_transition(grid_t, nxt.action_num, grid_t1)
            belief.frame_idx = t + 1
            belief.previous_views = views_t1

    print(f"\n{len(cases)} lattice_stable_no_motion cases:")
    print(f"  {'frame':>6} {'lvl':>3} {'loc':>12} {'manh_to_actor':>13}  type")
    for fr, lvl, ctype, loc, dist in cases:
        flag = "  ←L7-WIN-ADJACENT" if abs(fr - 405) <= 2 else ""
        print(f"  {fr:>6} {lvl:>3} {str(loc):>12} {dist:>13}  {ctype}{flag}")

    l7_near_win = [c for c in cases if abs(c[0] - 405) <= 2]
    print(f"\n  L7 win-adjacent (frame 403-405): {len(l7_near_win)}")
    if l7_near_win:
        print("  → dynamics residue and goal predicate may be ONE discovery.")
    else:
        print("  → the 16 do NOT coincide with the L7 win; distinct phenomena.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="1,2,3,4,5",
        help="comma-separated query numbers to run (default: all)",
    )
    args = parser.parse_args()
    run = {int(x) for x in args.only.split(",") if x.strip()}

    records = oc16.load_trace(oc16.DEFAULT_TRACE)
    segments = load_segments(records)
    steps: list = []
    if run & {1, 3}:
        print("Replaying trace for step cache (all levels)...")
        steps, _, _ = oc16.run_checker(
            records, segments, levels={1, 2, 3, 4, 5, 6, 7}
        )

    if 1 in run:
        query1_autonomous_change_types(records, steps)
    if 2 in run:
        query2_win_frame_equality(records)
    if 3 in run:
        query3_l7_perception_split(records, steps)
    if 4 in run:
        query4_pixel_shift_split(records)
    if 5 in run:
        query5_win_change_scan(records)
    if 6 in run:
        query6_l7_conditional_delta(records)
    if 7 in run:
        query7_lattice_stable_locations(records)


if __name__ == "__main__":
    main()
