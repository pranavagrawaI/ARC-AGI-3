"""0013 — Choose the experiment that kills the most hypotheses per action.

Run from the repo root:
    uv run python teach/lessons/0013-info-gain-policy.py

Audit the perception → event → clause pipeline on live ls20:
    uv run python teach/lessons/0013-info-gain-policy.py --live

Lesson 0012 learns a law from events. This lesson chooses WHICH event to collect
next: the probe that splits the surviving clause hypotheses most sharply, divided
by action cost. NOT pixel surprise, NOT curiosity — explicit hypotheses killed.

GIVEN: an explicit version space of clause hypotheses, a probe menu, agency gating,
       and a maximin kill metric (LR 0011/0013).
LEARNED: probe ordering for this game — which discriminating experiment to run.

The companion also defines the minimal event-record contract between perception
(0010/0011) and reasoning (0012/0013):

    record_interaction_event(facts, actor_changed, status) -> Event

Live ls20 probe selection waits for trustworthy events from the full pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

# Reuse lesson 0012's verified clause machinery.
_spec = _ilu.spec_from_file_location(
    "cs_0012", "teach/lessons/0012-depth-gated-clause-search.py"
)
cs = _ilu.module_from_spec(_spec)
_sys.modules[_spec.name] = cs
_spec.loader.exec_module(cs)

Clause = cs.Clause
Event = cs.Event
trusted = cs.trusted
learn_clause = cs.learn_clause
UNTRUSTED_STATUSES = cs.UNTRUSTED_STATUSES

ACTOR_CHANGE_TRANSITIONS = frozenset({"CHANGED_COLOR", "CHANGED_SHAPE"})


@dataclass(frozen=True)
class Probe:
    """A deliberate experiment the avatar can attempt."""

    facts: frozenset[str]
    action_cost: int = 1
    agency_grounded: bool = True
    label: str = ""


def record_interaction_event(
    facts: frozenset[str],
    *,
    actor_changed: bool,
    status: str = "OBSERVED",
) -> Event:
    """Minimal pipeline bridge: perception facts + actor body change -> Event."""
    return Event(facts=facts, effect=actor_changed, status=status)


def clause_predicts_effect(clause: Clause, facts: frozenset[str]) -> bool:
    """Whether this clause predicts effect=True for an event with these facts."""
    return set(clause.literals) <= facts


def hypothesis_vocabulary(
    events: list[Event],
    probes: list[Probe] | None = None,
) -> list[str]:
    """Literals in play: trusted observations plus probes we could run next."""
    fact_sets = [event.facts for event in trusted(events)]
    if probes:
        fact_sets.extend(probe.facts for probe in probes)
    if not fact_sets:
        return []
    return sorted(set().union(*fact_sets))


def consistent_clauses(
    events: list[Event],
    max_depth: int = 2,
    probes: list[Probe] | None = None,
) -> list[Clause]:
    """Version space: every depth-bounded clause with zero errors on trusted data."""
    clean = trusted(events)
    vocabulary = hypothesis_vocabulary(events, probes)
    if not vocabulary:
        return []
    survivors: list[Clause] = []
    cap = min(max_depth, len(vocabulary))
    for depth in range(0, cap + 1):
        for parts in combinations(vocabulary, depth):
            clause = Clause(tuple(parts))
            if not clean or clause.errors(clean) == 0:
                survivors.append(clause)
    return survivors


def split_counts(clauses: list[Clause], facts: frozenset[str]) -> tuple[int, int]:
    """How many clauses predict effect=True vs False for these facts."""
    predict_true = sum(1 for clause in clauses if clause_predicts_effect(clause, facts))
    return predict_true, len(clauses) - predict_true


def guaranteed_kills(clauses: list[Clause], facts: frozenset[str]) -> int:
    """Maximin kills: whichever outcome arrives, at least this many clauses die."""
    predict_true, predict_false = split_counts(clauses, facts)
    if predict_true == 0 or predict_false == 0:
        return 0
    return min(predict_true, predict_false)


def score_probe(probe: Probe, clauses: list[Clause]) -> float:
    """Hypotheses-killed-per-action under the maximin split (LR 0013)."""
    if not probe.agency_grounded or len(clauses) <= 1:
        return 0.0
    kills = guaranteed_kills(clauses, probe.facts)
    if kills == 0:
        return 0.0
    return kills / max(1, probe.action_cost)


def pick_probe(probes: list[Probe], clauses: list[Clause]) -> Probe | None:
    """Return the highest-scoring agency-grounded probe; lexical tie-break."""
    if len(clauses) <= 1:
        return None
    ranked = sorted(
        probes,
        key=lambda probe: (
            score_probe(probe, clauses),
            probe.label,
            tuple(sorted(probe.facts)),
        ),
        reverse=True,
    )
    best_score = score_probe(ranked[0], clauses)
    if best_score <= 0:
        return None
    return ranked[0]


def version_space(
    events: list[Event],
    probes: list[Probe],
    *,
    max_depth: int = 2,
) -> list[Clause]:
    """Surviving clauses given observations and the probe menu alphabet."""
    return consistent_clauses(events, max_depth=max_depth, probes=probes)


def apply_probe(
    probe: Probe,
    *,
    effect: bool,
    status: str = "OBSERVED",
) -> Event:
    return record_interaction_event(probe.facts, actor_changed=effect, status=status)


def simulate_effect(truth: Clause, facts: frozenset[str]) -> bool:
    """Oracle for synthetic loops: does the hidden law fire on these facts?"""
    return clause_predicts_effect(truth, facts)


def learning_loop(
    probes: list[Probe],
    truth: Clause,
    *,
    seed_events: list[Event] | None = None,
    max_depth: int = 2,
    max_steps: int = 10,
) -> tuple[list[Event], int]:
    """Greedy info-gain loop until the version space is a singleton."""
    events = list(seed_events or [])
    steps = 0
    while steps < max_steps:
        survivors = consistent_clauses(events, max_depth=max_depth, probes=probes)
        if len(survivors) <= 1:
            break
        probe = pick_probe(probes, survivors)
        if probe is None:
            break
        effect = simulate_effect(truth, probe.facts)
        events.append(apply_probe(probe, effect=effect))
        steps += 1
    return events, steps


def fixed_probe_loop(
    probes: list[Probe],
    truth: Clause,
    *,
    pick: Probe,
    seed_events: list[Event] | None = None,
    max_depth: int = 2,
    max_steps: int = 10,
) -> tuple[list[Event], int]:
    """Repeat one fixed probe choice — a deliberately weak baseline."""
    events = list(seed_events or [])
    steps = 0
    while steps < max_steps:
        survivors = consistent_clauses(events, max_depth=max_depth, probes=probes)
        if len(survivors) <= 1:
            break
        effect = simulate_effect(truth, pick.facts)
        events.append(apply_probe(pick, effect=effect))
        steps += 1
    return events, steps


def _probe_menu() -> list[Probe]:
    return [
        Probe(
            frozenset({"contact", "target_shape=cross"}),
            label="contact_cross",
        ),
        Probe(
            frozenset({"contact", "target_shape=square"}),
            label="contact_square",
        ),
        Probe(
            frozenset({"near", "target_shape=cross"}),
            label="near_cross",
        ),
        Probe(
            frozenset({"aligned", "target_shape=square"}),
            label="aligned_square",
        ),
        Probe(
            frozenset({"contact", "target_shape=cross", "actor_color=blue"}),
            action_cost=2,
            label="costly_cross",
        ),
        Probe(
            frozenset({"contact", "target_shape=cross"}),
            agency_grounded=False,
            label="background_noise",
        ),
    ]


def _truth_clause() -> Clause:
    return Clause(("contact", "target_shape=cross"))


def _seed_events() -> list[Event]:
    return [
        record_interaction_event(
            frozenset({"contact", "target_shape=square", "actor_color=blue"}),
            actor_changed=False,
        ),
    ]


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    probes = _probe_menu()
    truth = _truth_clause()
    seed = _seed_events()
    survivors = version_space(seed, probes, max_depth=2)
    checks: list[bool] = []

    split_probe = probes[0]
    noise_probe = probes[5]
    checks.append(_check(
        "1 discriminating probe beats unanimous noise",
        score_probe(split_probe, survivors) > score_probe(noise_probe, survivors),
        f"split={score_probe(split_probe, survivors):.2f}, "
        f"noise={score_probe(noise_probe, survivors):.2f}",
    ))

    cheap = probes[0]
    costly = probes[4]
    cheap_score = score_probe(cheap, survivors)
    costly_score = score_probe(costly, survivors)
    checks.append(_check(
        "2 cheaper probe wins equal split",
        cheap_score >= costly_score and pick_probe([costly, cheap], survivors) == cheap,
        f"cheap={cheap_score:.2f}, costly={costly_score:.2f}, "
        f"picked={pick_probe([costly, cheap], survivors).label}",
    ))

    picked = pick_probe(probes, survivors)
    checks.append(_check(
        "3 picks a splitting agency-grounded probe",
        picked is not None and picked.agency_grounded and guaranteed_kills(survivors, picked.facts) > 0,
        f"picked={picked.label if picked else None}, "
        f"guaranteed_kills={guaranteed_kills(survivors, picked.facts) if picked else 0}",
    ))

    event = record_interaction_event(
        frozenset({"contact", "target_shape=cross"}),
        actor_changed=True,
        status="UNCERTAIN",
    )
    checks.append(_check(
        "4 event bridge maps actor change to effect",
        event.effect is True and event.status == "UNCERTAIN",
        f"effect={event.effect}, status={event.status}",
    ))

    learned_events, greedy_steps = learning_loop(
        probes,
        truth,
        seed_events=seed,
        max_depth=2,
        max_steps=8,
    )
    greedy_result = learn_clause(learned_events, max_depth=2)
    costly = probes[4]
    _, fixed_steps = fixed_probe_loop(
        probes,
        truth,
        pick=costly,
        seed_events=seed,
        max_depth=2,
        max_steps=8,
    )
    checks.append(_check(
        "5 greedy loop learns the hidden law in fewer steps than a fixed weak probe",
        greedy_result.clause == truth and greedy_steps < fixed_steps,
        f"greedy_steps={greedy_steps}, fixed_costly_steps={fixed_steps}, "
        f"clause={greedy_result.clause}",
    ))

    post_greedy = version_space(learned_events, probes, max_depth=2)
    checks.append(_check(
        "6 version space collapses to the minimal law",
        len(post_greedy) == 1 and post_greedy[0] == truth,
        f"survivors={len(post_greedy)}, clause={post_greedy[0] if post_greedy else None}",
    ))

    return all(checks)


CELL = 5
ACTION_DELTAS = {1: (-CELL, 0), 2: (CELL, 0), 3: (0, -CELL), 4: (0, CELL)}


def _load_pipeline():
    """Load verified bricks 0008, 0010, 0011 by path (house style)."""
    modules = {}
    for name, path in (
        ("oc", "teach/lessons/0008-object-controllability.py"),
        ("il", "teach/lessons/0010-invariance-lattice-perception.py"),
        ("id", "teach/lessons/0011-object-identity-occlusion.py"),
    ):
        spec = _ilu.spec_from_file_location(f"pipe_{name}", path)
        mod = _ilu.module_from_spec(spec)
        _sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        modules[name] = mod
    return modules


def _tag(prefix: str, value) -> str:
    return f"{prefix}_{hash(value) & 0xFFFF:04x}"


def _shape_literal(view) -> str:
    return _tag("target_shape", view.shape_lattice["rotation_invariant"])


def _palette_literal(view, prefix: str) -> str:
    palette = view.color_lattice.get(
        "color_invariant", view.color_lattice["raw"]
    )
    return _tag(prefix, palette)


def _manh(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pixels_touch(a, b) -> bool:
    if a.pixels & b.pixels:
        return True
    for r, c in a.pixels:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in b.pixels:
                return True
    return False


def _spatial_relation(actor, target) -> str | None:
    if _pixels_touch(actor, target):
        return "contact"
    if _manh(actor.centroid, target.centroid) <= 2 * CELL:
        return "near"
    ar, ac = actor.centroid
    tr, tc = target.centroid
    if abs(ar - tr) <= CELL / 2 or abs(ac - tc) <= CELL / 2:
        return "aligned"
    return None


def _is_actor_view(view, controlled_sigs: set[tuple[int, int]]) -> bool:
    if not controlled_sigs:
        return False
    if len(view.color_lattice["raw"]) == 1:
        colour, count = view.color_lattice["raw"][0]
        return (colour, view.area) in controlled_sigs or (colour, count) in controlled_sigs
    for colour, count in view.color_lattice["raw"]:
        if (colour, count) in controlled_sigs or (colour, view.area) in controlled_sigs:
            return True
    return False


def _component_at(grid, row: int, col: int, background: int) -> tuple[int, int] | None:
    height, width = grid.shape
    if not (0 <= row < height and 0 <= col < width):
        return None
    colour = int(grid[row, col])
    if colour == background:
        return None
    return colour, 1


def _view_covering_cell(views, row: int, col: int):
    cell = (row, col)
    for view in views:
        if cell in view.pixels:
            return view
    return None


def _facts_from_interaction(
    *,
    relation: str,
    actor_view,
    target_view,
    action: int,
) -> frozenset[str]:
    facts = {
        relation,
        _shape_literal(target_view),
        f"action={action}",
        "target_visible",
        _palette_literal(actor_view, "actor_palette"),
    }
    if len(actor_view.color_lattice["raw"]) == 1:
        facts.add(_palette_literal(actor_view, "actor_color"))
    return frozenset(facts)


def _epistemic_status(*statuses: str) -> str:
    if "LOST" in statuses:
        return "LOST"
    if "UNCERTAIN" in statuses:
        return "UNCERTAIN"
    return "OBSERVED"


def _tracked_by_view(tracked) -> dict[int, object]:
    return {
        id(item.object_view): item
        for item in tracked
        if item.object_view is not None and item.track_id is not None
    }


def _warmup_controllability(oc, env, game_frame, steps: int) -> set[tuple[int, int]]:
    import numpy as np
    from arcengine import GameAction

    learner = oc.ControllabilityLearner()
    prev = np.asarray(game_frame.frame[-1], dtype=np.int8)
    for _ in range(steps):
        action = game_frame.available_actions[0] if len(game_frame.available_actions) == 1 else __import__(
            "random"
        ).choice(game_frame.available_actions)
        game_frame = env.step(GameAction.from_id(action))
        if game_frame.state.value in ("GAME_OVER", "WIN") or not game_frame.frame:
            game_frame = env.reset()
            prev = np.asarray(game_frame.frame[-1], dtype=np.int8)
            continue
        nxt = np.asarray(game_frame.frame[-1], dtype=np.int8)
        learner.observe(prev, action, nxt)
        prev = nxt
    return set(learner.controlled().keys())


def _path_target_for_action(
    previous,
    prev_grid,
    actor_views,
    action: int,
    background: int,
):
    actor_centroid = (
        sum(view.centroid[0] for view in actor_views) / len(actor_views),
        sum(view.centroid[1] for view in actor_views) / len(actor_views),
    )
    dr, dc = ACTION_DELTAS.get(action, (0, 0))
    target_cell = (round(actor_centroid[0] + dr), round(actor_centroid[1] + dc))
    path_target = _view_covering_cell(previous, *target_cell)
    if path_target is None:
        cls = _component_at(prev_grid, target_cell[0], target_cell[1], background)
        if cls is not None:
            colour, _size = cls
            path_target = next(
                (
                    view
                    for view in previous
                    if any(c == colour for c, _ in view.color_lattice["raw"])
                    and target_cell in view.pixels
                ),
                None,
            )
    return path_target, actor_centroid, target_cell


def _actor_body_changed(
    actor_views,
    current_views,
    controlled_sigs: set[tuple[int, int]],
    actor_tracks,
) -> bool:
    if any(t.transition in ACTOR_CHANGE_TRANSITIONS for t in actor_tracks):
        return True
    if not actor_views:
        return False
    curr_actor_views = [
        view for view in current_views if _is_actor_view(view, controlled_sigs)
    ]
    prev_patterns = {view.internal_pattern_lattice["raw"] for view in actor_views}
    curr_patterns = {view.internal_pattern_lattice["raw"] for view in curr_actor_views}
    prev_palettes = {view.color_lattice["raw"] for view in actor_views}
    curr_palettes = {view.color_lattice["raw"] for view in curr_actor_views}
    return prev_patterns != curr_patterns or prev_palettes != curr_palettes


# Pipeline surface for lesson 0014
load_pipeline = _load_pipeline
warmup_controllability = _warmup_controllability
shape_literal = _shape_literal
palette_literal = _palette_literal
manh = _manh
is_actor_view = _is_actor_view
component_at = _component_at
view_covering_cell = _view_covering_cell
facts_from_interaction = _facts_from_interaction
epistemic_status = _epistemic_status
tracked_by_view = _tracked_by_view
path_target_for_action = _path_target_for_action
actor_body_changed = _actor_body_changed


def audit_ls20_pipeline(
    *,
    warmup_steps: int = 80,
    steps: int = 200,
    seed: int = 17,
    use_composites: bool = True,
) -> dict:
    """Run one live pass: perceive, track, emit events, try clause search."""
    import random

    import arc_agi
    import numpy as np
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    oc, il, id_mod = _load_pipeline().values()
    ObjectTracker = id_mod.ObjectTracker
    UNCERTAIN = id_mod.UNCERTAIN
    LOST = id_mod.LOST

    rng = random.Random(seed)
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ls20")
    game_frame = env.reset()

    controlled_sigs = _warmup_controllability(oc, env, game_frame, warmup_steps)
    game_frame = env.reset()

    grid = np.asarray(game_frame.frame[-1])
    _, previous_parts = il.views_from_grid(grid, frame=0)
    grouper = il.CompositeObjectGrouper()
    previous = (
        grouper.group(grid, previous_parts, frame=0)
        if use_composites
        else previous_parts
    )
    tracker = ObjectTracker()
    previous_tracked = _tracked_by_view(tracker.update(previous))

    events: list[Event] = []
    event_examples: list[str] = []
    relation_counts: Counter = Counter()
    status_counts: Counter = Counter()
    effect_counts: Counter = Counter()
    actor_change_counts: Counter = Counter()
    skipped_no_actor = 0
    skipped_teleport = 0
    resets = 0

    for step in range(1, steps + 1):
        action = rng.choice(game_frame.available_actions)
        prev_grid = np.asarray(game_frame.frame[-1], dtype=np.int8)
        vals, counts = np.unique(prev_grid, return_counts=True)
        background = int(vals[counts.argmax()])

        actor_views = [v for v in previous if _is_actor_view(v, controlled_sigs)]
        if not actor_views:
            skipped_no_actor += 1
            game_frame = env.step(GameAction.from_id(action))
            if game_frame.state.value in ("GAME_OVER", "WIN") or not game_frame.frame:
                resets += 1
                game_frame = env.reset()
                grid = np.asarray(game_frame.frame[-1])
                _, previous_parts = il.views_from_grid(grid, frame=step)
                previous = (
                    grouper.group(grid, previous_parts, frame=step)
                    if use_composites
                    else previous_parts
                )
                tracker = ObjectTracker()
                previous_tracked = _tracked_by_view(tracker.update(previous))
            continue

        actor_centroid = (
            sum(v.centroid[0] for v in actor_views) / len(actor_views),
            sum(v.centroid[1] for v in actor_views) / len(actor_views),
        )
        path_target, actor_centroid, _target_cell = _path_target_for_action(
            previous, prev_grid, actor_views, action, background
        )

        game_frame = env.step(GameAction.from_id(action))
        if game_frame.state.value in ("GAME_OVER", "WIN") or not game_frame.frame:
            resets += 1
            game_frame = env.reset()
            grid = np.asarray(game_frame.frame[-1])
            _, previous_parts = il.views_from_grid(grid, frame=step)
            previous = (
                grouper.group(grid, previous_parts, frame=step)
                if use_composites
                else previous_parts
            )
            tracker = ObjectTracker()
            previous_tracked = _tracked_by_view(tracker.update(previous))
            continue

        grid = np.asarray(game_frame.frame[-1])
        _, current_parts = il.views_from_grid(grid, frame=step)
        if use_composites:
            grouper.observe(previous, current_parts)
            current = grouper.group(grid, current_parts, frame=step)
        else:
            current = current_parts
        tracked = tracker.update(current, action=action)
        current_tracked = _tracked_by_view(tracked)

        actor_tracks = [
            t
            for t in tracked
            if t.object_view is not None and _is_actor_view(t.object_view, controlled_sigs)
        ]
        actor_changed = _actor_body_changed(
            actor_views, current, controlled_sigs, actor_tracks
        )
        actor_change_counts["changed" if actor_changed else "unchanged"] += 1

        moved = any(t.transition == id_mod.MOVED for t in actor_tracks)
        if action in ACTION_DELTAS and moved:
            new_centroid = (
                sum(t.object_view.centroid[0] for t in actor_tracks if t.object_view)
                / max(1, len(actor_tracks)),
                sum(t.object_view.centroid[1] for t in actor_tracks if t.object_view)
                / max(1, len(actor_tracks)),
            )
            jump = _manh(actor_centroid, new_centroid)
            if jump > CELL * 1.6:
                skipped_teleport += 1
                previous = current
                previous_tracked = current_tracked
                continue

        interactions: list[tuple[str, object, object]] = []
        primary_actor = actor_views[0]
        if path_target is not None and not _is_actor_view(path_target, controlled_sigs):
            interactions.append(("contact", primary_actor, path_target))
        elif moved:
            for actor_view in actor_views:
                for target_view in current:
                    if _is_actor_view(target_view, controlled_sigs):
                        continue
                    if _spatial_relation(actor_view, target_view) == "contact":
                        interactions.append(("contact", actor_view, target_view))
                        break
                if interactions:
                    break

        if not interactions:
            previous = current
            previous_tracked = current_tracked
            continue

        relation, actor_view, target_view = interactions[0]
        actor_item = current_tracked.get(id(actor_view))
        target_item = current_tracked.get(id(target_view))
        status = _epistemic_status(
            *(t.transition for t in (actor_item, target_item) if t is not None)
        )
        facts = _facts_from_interaction(
            relation=relation,
            actor_view=actor_view,
            target_view=target_view,
            action=action,
        )
        event = record_interaction_event(
            facts,
            actor_changed=actor_changed,
            status=status,
        )
        events.append(event)
        relation_counts[relation] += 1
        status_counts[status] += 1
        effect_counts["effect" if event.effect else "no_effect"] += 1
        if len(event_examples) < 8:
            event_examples.append(
                f"{status} {relation} effect={event.effect} facts={sorted(facts)}"
            )

        previous = current
        previous_tracked = current_tracked

    clean = trusted(events)
    clause_result = None
    clause_error = None
    if clean:
        if any(event.effect for event in clean) and any(not event.effect for event in clean):
            try:
                clause_result = learn_clause(clean, max_depth=2)
            except ValueError as exc:
                clause_error = str(exc)
        else:
            clause_error = "need both positive and negative effect events"

    probes = [
        Probe(frozenset({"contact", f"target_shape_{shape}"}), label=f"contact_{shape}")
        for shape in sorted(
            {
                fact.removeprefix("target_shape_")
                for event in clean
                for fact in event.facts
                if fact.startswith("target_shape_")
            }
        )
    ]
    survivors = version_space(clean, probes, max_depth=2) if clean else []
    next_probe = pick_probe(probes, survivors) if probes and survivors else None
    unique_shapes = sorted(
        {
            fact
            for event in clean
            for fact in event.facts
            if fact.startswith("target_shape_")
        }
    )

    return {
        "warmup_steps": warmup_steps,
        "steps": steps,
        "seed": seed,
        "use_composites": use_composites,
        "controlled_signatures": sorted(controlled_sigs),
        "resets": resets,
        "skipped_no_actor": skipped_no_actor,
        "skipped_teleport": skipped_teleport,
        "events_total": len(events),
        "events_trusted": len(clean),
        "events_filtered": len(events) - len(clean),
        "relation_counts": relation_counts,
        "status_counts": status_counts,
        "effect_counts": effect_counts,
        "actor_change_counts": actor_change_counts,
        "event_examples": event_examples,
        "clause_result": clause_result,
        "clause_error": clause_error,
        "version_space_size": len(survivors),
        "unique_target_shapes": len(unique_shapes),
        "next_probe": next_probe.label if next_probe else None,
    }


def _print_live_pipeline_audit(report: dict) -> None:
    print(
        f"\nLive ls20 pipeline audit "
        f"({report['steps']} actions after {report['warmup_steps']}-step warmup, "
        f"seed={report['seed']}, composites={report['use_composites']}):"
    )
    print(f"  controlled signatures from 0008: {report['controlled_signatures']}")
    print(
        f"  events: {report['events_total']} total, "
        f"{report['events_trusted']} trusted, {report['events_filtered']} filtered"
    )
    print(f"  relations: {dict(report['relation_counts'])}")
    print(f"  statuses: {dict(report['status_counts'])}")
    print(f"  effects: {dict(report['effect_counts'])}")
    print(f"  actor body changes: {dict(report['actor_change_counts'])}")
    print(
        f"  skipped: no_actor={report['skipped_no_actor']}, "
        f"teleport={report['skipped_teleport']}, resets={report['resets']}"
    )
    if report["event_examples"]:
        print("  sample events:")
        for example in report["event_examples"]:
            print(f"    {example}")
    if report["clause_result"] is not None:
        result = report["clause_result"]
        print(
            f"  learn_clause: {result.status} {result.clause} "
            f"(trusted={result.trusted_events}, filtered={result.filtered_events})"
        )
    elif report["clause_error"]:
        print(f"  learn_clause: skipped — {report['clause_error']}")
    print(
        f"  version_space={report['version_space_size']} "
        f"({report['unique_target_shapes']} distinct target shapes), "
        f"next_probe={report['next_probe']}"
    )
    print("\nAudit verdict:")
    if report["events_trusted"] == 0:
        print("  FAIL — pipeline emitted no trusted events.")
    elif report["status_counts"].get("UNCERTAIN", 0) > report["events_trusted"] * 0.5:
        print("  WARN — majority of events are UNCERTAIN; clause search inputs are noisy.")
    elif report["events_trusted"] < 10:
        print("  WARN — very few trusted events; need more contact-rich exploration.")
    else:
        print("  PASS (narrow) — pipeline emits trusted interaction events on live frames.")
    if report["effect_counts"].get("effect", 0) == 0:
        print("  WARN — no actor-attribute-change events; CHANGER rules are invisible.")
    if report["clause_result"] and report["clause_result"].status == "GRAMMAR_GAP":
        print(
            "  EXPECTED — live contacts are contradictory at depth 2; "
            "need repeated probes on the same target shapes or cleaner CHANGER signal."
        )
    if report["version_space_size"] == 0 and report["events_trusted"] > 0:
        print(
            "  NOTE — empty version space means NO clause fits all trusted events yet, "
            "not that search failed."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="audit perception → event → clause pipeline on live ls20",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--primitive",
        action="store_true",
        help="use primitive same-colour components instead of learned composites",
    )
    args = parser.parse_args()

    print("Offline checks (synthetic probe selection + learning loop):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using the info-gain policy.")

    if args.live:
        _print_live_pipeline_audit(
            audit_ls20_pipeline(
                warmup_steps=args.warmup,
                steps=args.steps,
                seed=args.seed,
                use_composites=not args.primitive,
            )
        )
        raise SystemExit(0)

    probes = _probe_menu()
    seed = _seed_events()
    survivors = version_space(seed, probes, max_depth=2)
    probe = pick_probe(probes, survivors)
    print("\nAfter one trusted negative contact (square, no effect):")
    print(f"  version_space={len(survivors)} clauses")
    print(f"  best_probe={probe.label if probe else None} "
          f"(score={score_probe(probe, survivors) if probe else 0:.2f})")

    events, steps = learning_loop(probes, _truth_clause(), seed_events=seed, max_depth=2)
    result = learn_clause(events, max_depth=2)
    print(
        f"\nGreedy loop finished in {steps} probe(s).\n"
        f"{result.status}: {result.clause}\n"
        "Next: wire record_interaction_event to live 0010/0011 output, then stack "
        "0008→0013 into one agent loop."
    )
