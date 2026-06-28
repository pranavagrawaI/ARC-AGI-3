"""0014 — The scientist agent: perceive, discriminate, act.

Run offline checks from the repo root:
    uv run python teach/lessons/0014-agent-loop.py

Run the full stack on live ls20:
    uv run python teach/lessons/0014-agent-loop.py --live

Stacks 0008 controllability → 0010 lattice → 0011 identity → 0013 info-gain
probe choice → 0012 clause search. Unlike the 0013 audit's random walk, this
agent *moves toward* the probe that best splits the surviving clause hypotheses.

GIVEN: the pipeline bricks from lessons 0008–0013.
LEARNED: action policy for this game — which direction earns the next experiment.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import random
import sys as _sys
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

# Lessons 0012 + 0013 verified machinery.
for name, path in (
    ("cs_0012", "teach/lessons/0012-depth-gated-clause-search.py"),
    ("ig_0013", "teach/lessons/0013-info-gain-policy.py"),
):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

cs = _sys.modules["cs_0012"]
ig = _sys.modules["ig_0013"]

Event = cs.Event
Clause = cs.Clause
trusted = cs.trusted
learn_clause = cs.learn_clause
Probe = ig.Probe
pick_probe = ig.pick_probe
score_probe = ig.score_probe
version_space = ig.version_space
record_interaction_event = ig.record_interaction_event
ACTOR_CHANGE_TRANSITIONS = ig.ACTOR_CHANGE_TRANSITIONS
CELL = ig.CELL
ACTION_DELTAS = ig.ACTION_DELTAS


@dataclass
class AgentState:
    """Mutable world state for one ls20 episode."""

    controlled_sigs: set[tuple[int, int]]
    events: list[Event] = field(default_factory=list)
    frame_idx: int = 0
    probe_picks: int = 0
    navigate_picks: int = 0
    contact_events: int = 0


def build_probe_menu(
    events: list[Event],
    views,
    controlled_sigs: set[tuple[int, int]],
) -> list[Probe]:
    """Candidate experiments: shapes seen before plus shapes visible now."""
    probes: dict[frozenset[str], Probe] = {}
    for event in trusted(events):
        shape_facts = [fact for fact in event.facts if fact.startswith("target_shape_")]
        if not shape_facts:
            continue
        facts = frozenset({"contact", shape_facts[0]})
        probes[facts] = Probe(facts, label=f"revisit_{shape_facts[0]}")
    for view in views:
        if ig.is_actor_view(view, controlled_sigs):
            continue
        shape = ig.shape_literal(view)
        facts = frozenset({"contact", shape})
        probes[facts] = Probe(facts, label=f"visible_{shape}")
    return list(probes.values())


def simulated_contact_probe(
    *,
    actor_view,
    path_target,
    action: int,
    controlled_sigs: set[tuple[int, int]],
) -> Probe | None:
    if path_target is None or ig.is_actor_view(path_target, controlled_sigs):
        return None
    facts = ig.facts_from_interaction(
        relation="contact",
        actor_view=actor_view,
        target_view=path_target,
        action=action,
    )
    return Probe(facts, label="simulated")


def _shape_fact(probe: Probe) -> str | None:
    for fact in probe.facts:
        if fact.startswith("target_shape_"):
            return fact
    return None


def _views_by_shape(views, controlled_sigs: set[tuple[int, int]]) -> dict[str, list]:
    out: dict[str, list] = {}
    for view in views:
        if ig.is_actor_view(view, controlled_sigs):
            continue
        out.setdefault(ig.shape_literal(view), []).append(view)
    return out


def _fallback_target_view(
    views,
    events: list[Event],
    controlled_sigs: set[tuple[int, int]],
):
    """When no probe splits, revisit the least-sampled visible shape."""
    shape_counts: Counter = Counter()
    for event in trusted(events):
        for fact in event.facts:
            if fact.startswith("target_shape_"):
                shape_counts[fact] += 1
    by_shape = _views_by_shape(views, controlled_sigs)
    if not by_shape:
        return None
    shape = min(by_shape, key=lambda s: shape_counts.get(s, 0))
    return by_shape[shape][0]


def _navigation_bonus(
    actor_centroid: tuple[float, float],
    target_view,
    action: int,
) -> float:
    dr, dc = ACTION_DELTAS.get(action, (0, 0))
    new_pos = (actor_centroid[0] + dr, actor_centroid[1] + dc)
    before = ig.manh(actor_centroid, target_view.centroid)
    after = ig.manh(new_pos, target_view.centroid)
    return before - after


def choose_action(
    *,
    state: AgentState,
    previous,
    prev_grid,
    available_actions: list[int],
    background: int,
    rng: random.Random,
) -> tuple[int, str]:
    """Pick the action that best realizes the top info-gain probe."""
    actor_views = [v for v in previous if ig.is_actor_view(v, state.controlled_sigs)]
    if not actor_views:
        return rng.choice(available_actions), "random_no_actor"

    primary_actor = actor_views[0]
    probes = build_probe_menu(state.events, previous, state.controlled_sigs)
    survivors = version_space(state.events, probes, max_depth=2)
    target_probe = pick_probe(probes, survivors) if len(survivors) > 1 else None

    navigate_view = None
    if target_probe is not None:
        shape = _shape_fact(target_probe)
        if shape:
            candidates = _views_by_shape(previous, state.controlled_sigs).get(shape, [])
            if candidates:
                navigate_view = min(
                    candidates,
                    key=lambda view: ig.manh(primary_actor.centroid, view.centroid),
                )
    if navigate_view is None:
        navigate_view = _fallback_target_view(
            previous, state.events, state.controlled_sigs
        )

    best_action = None
    best_score = float("-inf")
    reason = "random_tiebreak"
    for action in available_actions:
        path_target, actor_centroid, _cell = ig.path_target_for_action(
            previous, prev_grid, actor_views, action, background
        )
        sim = simulated_contact_probe(
            actor_view=primary_actor,
            path_target=path_target,
            action=action,
            controlled_sigs=state.controlled_sigs,
        )
        score = 0.0
        if sim is not None and not ig.is_actor_view(path_target, state.controlled_sigs):
            score = score_probe(sim, survivors) if len(survivors) > 1 else 1.0
            if target_probe is not None:
                if sim.facts == target_probe.facts:
                    score += 1000.0
                elif target_probe.facts <= sim.facts:
                    score += 100.0
        if navigate_view is not None:
            score += _navigation_bonus(actor_centroid, navigate_view, action)
        if score > best_score:
            best_score = score
            best_action = action
            if sim is not None and score >= 100.0:
                reason = "probe_match"
            elif navigate_view is not None and score > 0:
                reason = "navigate"
            else:
                reason = "score"

    if best_action is None:
        return rng.choice(available_actions), "random_fallback"
    return best_action, reason


def _reset_perception(il, grid, frame_idx: int, use_composites: bool):
    _, previous_parts = il.views_from_grid(grid, frame=frame_idx)
    grouper = il.CompositeObjectGrouper()
    previous = (
        grouper.group(grid, previous_parts, frame=frame_idx)
        if use_composites
        else previous_parts
    )
    tracker = ig.load_pipeline()["id"].ObjectTracker()
    return grouper, tracker, previous


def run_ls20_agent(
    *,
    warmup_steps: int = 80,
    steps: int = 400,
    seed: int = 17,
    use_composites: bool = True,
) -> dict:
    import arc_agi
    import numpy as np
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    oc, il, id_mod = ig.load_pipeline().values()
    MOVED = id_mod.MOVED

    rng = random.Random(seed)
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ls20")
    game_frame = env.reset()

    controlled_sigs = ig.warmup_controllability(oc, env, game_frame, warmup_steps)
    game_frame = env.reset()

    grid = np.asarray(game_frame.frame[-1])
    grouper, tracker, previous = _reset_perception(il, grid, 0, use_composites)
    state = AgentState(controlled_sigs=controlled_sigs)

    reason_counts: Counter = Counter()
    skipped_teleport = 0
    skipped_no_actor = 0
    resets = 0
    effect_counts: Counter = Counter()
    status_counts: Counter = Counter()
    shape_touch_counts: Counter = Counter()

    for _step in range(1, steps + 1):
        available = list(game_frame.available_actions)
        prev_grid = np.asarray(game_frame.frame[-1], dtype=np.int8)
        vals, counts = np.unique(prev_grid, return_counts=True)
        background = int(vals[counts.argmax()])

        actor_views = [
            view for view in previous if ig.is_actor_view(view, state.controlled_sigs)
        ]
        if not actor_views:
            skipped_no_actor += 1
            action = rng.choice(available)
            reason = "random_no_actor"
        else:
            action, reason = choose_action(
                state=state,
                previous=previous,
                prev_grid=prev_grid,
                available_actions=available,
                background=background,
                rng=rng,
            )
        reason_counts[reason] += 1
        if reason == "probe_match":
            state.probe_picks += 1
        elif reason == "navigate":
            state.navigate_picks += 1

        path_target, actor_centroid, _cell = ig.path_target_for_action(
            previous, prev_grid, actor_views, action, background
        ) if actor_views else (None, (0.0, 0.0), (0, 0))

        game_frame = env.step(GameAction.from_id(action))
        state.frame_idx += 1

        if game_frame.state.value in ("GAME_OVER", "WIN") or not game_frame.frame:
            resets += 1
            game_frame = env.reset()
            grid = np.asarray(game_frame.frame[-1])
            grouper, tracker, previous = _reset_perception(
                il, grid, state.frame_idx, use_composites
            )
            continue

        grid = np.asarray(game_frame.frame[-1])
        _, current_parts = il.views_from_grid(grid, frame=state.frame_idx)
        if use_composites:
            grouper.observe(previous, current_parts)
            current = grouper.group(grid, current_parts, frame=state.frame_idx)
        else:
            current = current_parts
        tracked = tracker.update(current, action=action)
        current_tracked = ig.tracked_by_view(tracked)

        if not actor_views:
            previous = current
            continue

        actor_tracks = [
            t
            for t in tracked
            if t.object_view is not None
            and ig.is_actor_view(t.object_view, state.controlled_sigs)
        ]
        moved = any(t.transition == MOVED for t in actor_tracks)
        if moved and action in ACTION_DELTAS:
            new_centroid = (
                sum(t.object_view.centroid[0] for t in actor_tracks if t.object_view)
                / max(1, len(actor_tracks)),
                sum(t.object_view.centroid[1] for t in actor_tracks if t.object_view)
                / max(1, len(actor_tracks)),
            )
            if ig.manh(actor_centroid, new_centroid) > CELL * 1.6:
                skipped_teleport += 1
                previous = current
                continue

        if path_target is None or ig.is_actor_view(path_target, state.controlled_sigs):
            previous = current
            continue

        actor_changed = ig.actor_body_changed(
            actor_views, current, state.controlled_sigs, actor_tracks
        )
        actor_item = current_tracked.get(id(actor_views[0]))
        target_item = current_tracked.get(id(path_target))
        status = ig.epistemic_status(
            *(t.transition for t in (actor_item, target_item) if t is not None)
        )
        facts = ig.facts_from_interaction(
            relation="contact",
            actor_view=actor_views[0],
            target_view=path_target,
            action=action,
        )
        event = record_interaction_event(
            facts,
            actor_changed=actor_changed,
            status=status,
        )
        state.events.append(event)
        state.contact_events += 1
        effect_counts["effect" if event.effect else "no_effect"] += 1
        status_counts[status] += 1
        for fact in facts:
            if fact.startswith("target_shape_"):
                shape_touch_counts[fact] += 1

        previous = current

    clean = trusted(state.events)
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

    probes = build_probe_menu(clean, previous, state.controlled_sigs)
    survivors = version_space(clean, probes, max_depth=2) if clean else []
    unique_shapes = len(shape_touch_counts)
    revisit_ratio = (
        sum(1 for count in shape_touch_counts.values() if count > 1) / unique_shapes
        if unique_shapes
        else 0.0
    )

    return {
        "warmup_steps": warmup_steps,
        "steps": steps,
        "seed": seed,
        "controlled_signatures": sorted(state.controlled_sigs),
        "events_total": len(state.events),
        "events_trusted": len(clean),
        "unique_target_shapes": unique_shapes,
        "revisit_ratio": revisit_ratio,
        "top_shapes": shape_touch_counts.most_common(5),
        "effect_counts": effect_counts,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "probe_picks": state.probe_picks,
        "navigate_picks": state.navigate_picks,
        "skipped_teleport": skipped_teleport,
        "skipped_no_actor": skipped_no_actor,
        "resets": resets,
        "clause_result": clause_result,
        "clause_error": clause_error,
        "version_space_size": len(survivors),
        "singleton_law": survivors[0] if len(survivors) == 1 else None,
    }


def _mock_view(shape_key: str, centroid: tuple[float, float], area: int = 10):
    return SimpleNamespace(
        shape_lattice={
            "rotation_invariant": shape_key,
            "raw": shape_key,
        },
        color_lattice={"raw": ((3, area),), "color_invariant": (area,)},
        internal_pattern_lattice={"raw": shape_key},
        centroid=centroid,
        area=area,
        pixels=frozenset(),
    )


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks: list[bool] = []

    menu = build_probe_menu(
        [],
        [
            _mock_view("cross", (30.0, 35.0)),
            _mock_view("square", (30.0, 45.0)),
        ],
        {(9, 15)},
    )
    checks.append(_check(
        "1 probe menu includes visible shapes",
        len(menu) == 2,
        f"probes={len(menu)}",
    ))

    events = [
        record_interaction_event(
            frozenset({"contact", "target_shape_aaaa", "action=1"}),
            actor_changed=False,
        ),
        record_interaction_event(
            frozenset({"contact", "target_shape_bbbb", "action=2"}),
            actor_changed=True,
        ),
    ]
    probes = build_probe_menu(events, [], {(9, 15)})
    survivors = version_space(events, probes, max_depth=2)
    checks.append(_check(
        "2 version space non-empty on mixed effects",
        len(survivors) > 0,
        f"survivors={len(survivors)}",
    ))

    state = AgentState(controlled_sigs={(9, 15)})
    state.events = events
    actor = _mock_view("actor", (30.0, 30.0), area=15)
    actor.color_lattice = {"raw": ((9, 15),), "color_invariant": (15,)}
    target = _mock_view("cross", (30.0, 40.0))
    square = _mock_view("square", (30.0, 50.0))
    previous = [actor, target, square]
    import numpy as np

    prev_grid = np.zeros((64, 64), dtype=np.int8)
    prev_grid[30, 35] = 5
    action, reason = choose_action(
        state=state,
        previous=previous,
        prev_grid=prev_grid,
        available_actions=[1, 2, 3, 4],
        background=0,
        rng=random.Random(0),
    )
    checks.append(_check(
        "3 choose_action prefers discriminating direction",
        action in (2, 4) and reason in ("probe_match", "navigate", "score"),
        f"action={action}, reason={reason}",
    ))

    sim = simulated_contact_probe(
        actor_view=actor,
        path_target=target,
        action=2,
        controlled_sigs={(9, 15)},
    )
    checks.append(_check(
        "4 simulated probe carries contact + shape",
        sim is not None and "contact" in sim.facts,
        f"facts={sorted(sim.facts) if sim else None}",
    ))

    return all(checks)


def _print_agent_report(report: dict) -> None:
    print(
        f"\nScientist agent on ls20 ({report['steps']} steps, "
        f"warmup={report['warmup_steps']}, seed={report['seed']}):"
    )
    print(f"  controlled signatures: {report['controlled_signatures']}")
    print(
        f"  events: {report['events_trusted']} trusted contacts "
        f"({report['unique_target_shapes']} shapes, "
        f"revisit_ratio={report['revisit_ratio']:.2f})"
    )
    print(f"  top shapes: {report['top_shapes']}")
    print(f"  effects: {dict(report['effect_counts'])}")
    print(f"  action reasons: {dict(report['reason_counts'])}")
    print(
        f"  probe_picks={report['probe_picks']}, "
        f"navigate_picks={report['navigate_picks']}"
    )
    if report["clause_result"] is not None:
        result = report["clause_result"]
        print(f"  learn_clause: {result.status} {result.clause}")
    elif report["clause_error"]:
        print(f"  learn_clause: skipped — {report['clause_error']}")
    if report["singleton_law"] is not None:
        print(f"  version_space singleton: {report['singleton_law']}")
    else:
        print(f"  version_space size: {report['version_space_size']}")
    print("\nAudit verdict:")
    if report["revisit_ratio"] > 0.3:
        print("  PASS — agent revisits shapes instead of one-off random contacts.")
    else:
        print("  WARN — low shape revisit rate; navigation may need tuning.")
    if report["probe_picks"] > 0:
        print("  PASS — info-gain probe matching drove at least one action.")
    else:
        print("  WARN — no direct probe matches; mostly navigation fallback.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run scientist agent on ls20")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("Offline checks (action selection + probe simulation):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before running the agent.")

    if args.live:
        _print_agent_report(
            run_ls20_agent(
                warmup_steps=args.warmup,
                steps=args.steps,
                seed=args.seed,
            )
        )
        raise SystemExit(0)

    print(
        "\nAll checks pass. Run with --live to stack the pipeline into one "
        "info-gain-guided agent on ls20."
    )
