"""0015 — Goal induction: aim the clause searcher at the reward channel.

Run offline checks from the repo root:
    uv run python teach/lessons/0015-goal-induction.py

Run the live goal probe on ls20:
    uv run python teach/lessons/0015-goal-induction.py --live --steps 400 --warmup 120

Lesson 0014 learns DYNAMICS: "what does contact do to the actor's body?".
That is a rule without a purpose. This lesson learns the GOAL with the SAME
machinery — clause search from 0012 — by relabelling the consequent from
"actor body changed" to "the success channel advanced this step".

GIVEN (Core Knowledge, never a game role):
  - a scalar success channel exists. On ls20 the only one the agent may read is
    `levels_completed` (an int that only rises) plus terminal WIN/GAME_OVER.
    Agentness says: there is a success signal; it does NOT say what triggers it.
  - the closed alphabet (0010 slots + lenses, 0011 identity) and clause search.

LEARNED:
  - which clause over the alphabet predicts a reward increment;
  - that ls20's goal is a *match* between the actor and a candidate object,
    which absolute symbols cannot express — so the searcher returns GRAMMAR_GAP.
    That proven gap is the corpse (LR_0016) that earns exactly ONE new letter:
    a generic relational `match_<slot>` built from 0010's `equal`/`cheapest_equal`.
    It is game-agnostic (any actor vs any object); the agent must still LEARN
    which match predicts success. Reading ls20.py would still not help.

Honest boundary (the live result predicts it): with no planner, the agent almost
never satisfies the win condition by luck, so the reward channel never fires and
`learn_goal` returns NO_GOAL_EVIDENCE. That fail-loud names the next brick:
reward-seeking exploration / planning over the induced model.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter
from dataclasses import dataclass


def _load(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# Verified bricks. cs = clause search (0012), ig = pipeline + facts (0013),
# il = invariance lattice (0010, the only sanctioned object comparison).
cs = _load("goal_cs_0012", "teach/lessons/0012-depth-gated-clause-search.py")
ig = _load("goal_ig_0013", "teach/lessons/0013-info-gain-policy.py")
il = ig.load_pipeline()["il"]

Event = cs.Event
Clause = cs.Clause
trusted = cs.trusted
learn_clause = cs.learn_clause
CELL = ig.CELL
ACTION_DELTAS = ig.ACTION_DELTAS

# Which slots a relational match letter may compare. These are GIVEN attribute
# slots from 0010, not ls20 concepts. We do NOT include position: "match where
# you stand" is the spatial `on_target` letter below, kept separate on purpose.
GOAL_SLOTS = ("shape", "color", "internal_pattern")


# ---------------------------------------------------------------------------
# The earned relational letters (the corpse in check 3 buys them — LR_0016).
# ---------------------------------------------------------------------------
def match_lenses(actor_view, target_view, slots=GOAL_SLOTS) -> dict[str, str]:
    """For each slot, the CHEAPEST lens under which actor == target, or absent.

    `cheapest_equal` encodes the invariance prior: assert the least-forgiving
    lens that holds (raw before rotation), never over-claim invariance.
    """
    out: dict[str, str] = {}
    for slot in slots:
        lens = il.cheapest_equal(slot, actor_view, target_view)
        if lens is not None:
            out[slot] = lens
    return out


def match_facts(actor_view, target_view, slots=GOAL_SLOTS) -> frozenset[str]:
    """Generic existence letters: `match_<slot>` if equal under SOME lens.

    A clause literal must recur verbatim across events to be learnable, so the
    vocabulary letter is lens-agnostic; the chosen lens is reported separately
    by `match_lenses` for diagnostics.
    """
    return frozenset(f"match_{slot}" for slot in match_lenses(actor_view, target_view, slots))


def on_target(actor_view, target_view) -> bool:
    """Spatial letter: the actor occupies the candidate's cell.

    Pixel overlap is the clean signal; centroid coincidence within half a cell
    is the occlusion-tolerant fallback (standing on a tile hides it — the known
    ls20 seam from LR-0017/0018)."""
    if actor_view.pixels & target_view.pixels:
        return True
    return ig.manh(actor_view.centroid, target_view.centroid) <= CELL / 2


def _actor_shape_literal(view) -> str:
    return f"actor_shape_{hash(view.shape_lattice['rotation_invariant']) & 0xFFFF:04x}"


def goal_facts_for_pair(actor_view, target_view, *, include_match: bool) -> frozenset[str]:
    """The state snapshot at a candidate goal contact, in the closed alphabet.

    With `include_match=False` only absolute symbols are emitted (what the agent
    had before this lesson). With it, the earned `match_<slot>` letters join."""
    facts = {ig.shape_literal(target_view), _actor_shape_literal(actor_view)}
    if on_target(actor_view, target_view):
        facts.add("on_target")
    if include_match:
        facts |= match_facts(actor_view, target_view)
    return frozenset(facts)


# ---------------------------------------------------------------------------
# Goal learner: clause search, relabelled, with two honest refusals.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoalResult:
    status: str  # LEARNED | GRAMMAR_GAP | NO_GOAL_EVIDENCE
    clause: object | None
    positives: int
    negatives: int
    detail: str

    def __str__(self) -> str:
        if self.status == "LEARNED":
            return f"LEARNED goal: {self.clause}"
        return f"{self.status}: {self.detail}"


def learn_goal(events: list[Event], max_depth: int = 3) -> GoalResult:
    """Goal = a clause whose consequent is `reward_up`. Same searcher as 0012.

    Refuses two ways instead of fabricating a goal:
      - NO_GOAL_EVIDENCE: no rewarded step seen yet (the sparse-reward reality),
        or every step looks rewarded (nothing to discriminate against).
      - GRAMMAR_GAP: rewards exist and differ, but no clause in the alphabet
        separates them — the alphabet is too weak (a slot-gap to investigate).
    """
    clean = trusted(events)
    positives = sum(1 for e in clean if e.effect)
    negatives = sum(1 for e in clean if not e.effect)
    if positives == 0:
        return GoalResult("NO_GOAL_EVIDENCE", None, positives, negatives,
                          "the success channel never advanced — nothing to explain")
    if negatives == 0:
        return GoalResult("NO_GOAL_EVIDENCE", None, positives, negatives,
                          "every step looks rewarded — no contrast to learn from")
    result = learn_clause(clean, max_depth=max_depth)
    return GoalResult(result.status, result.clause, positives, negatives,
                      f"explored {result.explored_candidates} clauses to depth {result.depth_reports[-1].depth}")


# ---------------------------------------------------------------------------
# Offline checks — synthetic, deterministic, no engine.
# ---------------------------------------------------------------------------
def _view(rows: list[list[int]], origin=(0, 0)):
    """A real 0010 ObjectView from a tiny crop; -1 marks non-object cells."""
    import numpy as np

    return il.ObjectView.from_crop(np.array(rows), origin=origin)


CROSS = [[-1, 1, -1], [1, 1, 1], [-1, 1, -1]]
CROSS_RED = [[-1, 2, -1], [2, 2, 2], [-1, 2, -1]]  # cross shape, different colour
# Two shapes with the SAME cell count (4) but different arrangement: colour-count
# cannot separate them, so check 4 is forced to learn the SHAPE match.
SQUARE = [[1, 1], [1, 1]]
LINE = [[1, 1, 1, 1]]


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks: list[bool] = []

    # 1 — Goal induction IS relabelled clause search. Reward fires when the
    # actor stands on a target AND carries a particular colour; absolute literals
    # express this, so the searcher recovers it unchanged.
    expressible = [
        Event(frozenset({"on_target", "actor_color_blue", "target_shape_a"}), True),
        Event(frozenset({"on_target", "actor_color_blue", "target_shape_b"}), True),
        Event(frozenset({"on_target", "actor_color_red", "target_shape_a"}), False),
        Event(frozenset({"actor_color_blue", "target_shape_a"}), False),  # not on target
    ]
    g1 = learn_goal(expressible, max_depth=2)
    checks.append(_check(
        "1 goal = clause search, relabelled to reward",
        g1.status == "LEARNED"
        and set(g1.clause.literals) == {"on_target", "actor_color_blue"},
        f"{g1} | clause={g1.clause}",
    ))

    # 2 — Sparse-reward refusal. No rewarded step => no fabricated goal.
    starved = [
        Event(frozenset({"on_target", "actor_shape_x"}), False),
        Event(frozenset({"on_target", "actor_shape_y"}), False),
    ]
    g2 = learn_goal(starved, max_depth=2)
    checks.append(_check(
        "2 sparse reward => NO_GOAL_EVIDENCE, not a guess",
        g2.status == "NO_GOAL_EVIDENCE" and g2.clause is None and g2.positives == 0,
        f"{g2}",
    ))

    # 3 — THE CORPSE. ls20's goal is a MATCH: reward iff actor.shape == target.shape
    # AND the actor stands on it. Built from real ObjectViews, but encoded with
    # ABSOLUTE symbols only (what the agent had before this lesson). No single
    # conjunction covers a cross-match AND a square-match without also firing on
    # the mismatched negatives => the alphabet is provably too weak.
    pairs = [
        (_view(SQUARE, (10, 10)), _view(SQUARE, (10, 10)), True),  # square on square -> reward
        (_view(LINE, (20, 20)), _view(LINE, (20, 20)), True),      # line on line -> reward
        (_view(SQUARE, (30, 30)), _view(LINE, (30, 30)), False),   # mismatch, on -> no
        (_view(LINE, (35, 35)), _view(SQUARE, (35, 35)), False),   # mismatch, on -> no
        (_view(SQUARE, (55, 55)), _view(SQUARE, (5, 5)), False),   # match, but standing elsewhere -> no
    ]
    abs_events = [
        Event(goal_facts_for_pair(a, t, include_match=False), reward)
        for a, t, reward in pairs
    ]
    g3 = learn_goal(abs_events, max_depth=3)
    checks.append(_check(
        "3 absolute symbols cannot say 'match' => GRAMMAR_GAP",
        g3.status == "GRAMMAR_GAP" and g3.positives == 2 and g3.negatives == 3,
        f"{g3} (pos={g3.positives}, neg={g3.negatives})",
    ))

    # 4 — The earned letter. Re-encode the SAME pairs with `match_<slot>` and the
    # goal becomes expressible. The relational letter was bought by check 3's gap.
    rel_events = [
        Event(goal_facts_for_pair(a, t, include_match=True), reward)
        for a, t, reward in pairs
    ]
    g4 = learn_goal(rel_events, max_depth=3)
    learned = g4.status == "LEARNED" and g4.clause is not None
    has_match = learned and any(lit.startswith("match_") for lit in g4.clause.literals)
    on_t = learned and "on_target" in g4.clause.literals
    checks.append(_check(
        "4 the earned match letter recovers the goal",
        learned and has_match and on_t,
        f"{g4} | clause={g4.clause if learned else None}",
    ))

    # 5 — Lens discipline. `cheapest_equal` climbs the lattice only as far as it
    # must: raw when raw holds; rotation only when an asymmetric shape is turned;
    # colour-relabel (counts, not hue) for a recoloured twin. Never over-claim.
    import numpy as np

    ell = [[1, -1], [1, -1], [1, 1]]  # asymmetric: rotation actually changes raw
    ell_rot = np.rot90(np.array(ell)).tolist()
    raw_lens = match_lenses(_view(ell), _view(ell)).get("shape")
    rot_lens = match_lenses(_view(ell), _view(ell_rot)).get("shape")
    colour_lens = match_lenses(_view(CROSS), _view(CROSS_RED)).get("color")
    checks.append(_check(
        "5 match respects the cheapest invariance lens",
        raw_lens == "raw"
        and rot_lens == "rotation_invariant"
        and colour_lens == "color_invariant",
        f"raw_shape={raw_lens}, rotated_shape={rot_lens}, recoloured_color={colour_lens}",
    ))

    return all(checks)


# ---------------------------------------------------------------------------
# Live goal probe on ls20.
# ---------------------------------------------------------------------------
def _navigate_action(actor_centroid, target_view, available, rng, epsilon=0.1):
    """Simple revisit-navigation: step toward the target object (or explore).

    Deliberately simpler than 0014's info-gain policy — this lesson studies the
    goal channel, not movement. Reusing 0014's policy here is future work."""
    if target_view is None or rng.random() < epsilon:
        return rng.choice(available)
    best_action, best_gain = rng.choice(available), float("-inf")
    for action in available:
        dr, dc = ACTION_DELTAS.get(action, (0, 0))
        new_pos = (actor_centroid[0] + dr, actor_centroid[1] + dc)
        gain = ig.manh(actor_centroid, target_view.centroid) - ig.manh(new_pos, target_view.centroid)
        if gain > best_gain:
            best_gain, best_action = gain, action
    return best_action


def run_goal_probe(*, warmup_steps=120, steps=400, seed=17, use_composites=True) -> dict:
    import random

    import arc_agi
    import numpy as np
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    oc, il_mod, id_mod = ig.load_pipeline().values()
    ObjectTracker = id_mod.ObjectTracker

    rng = random.Random(seed)
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ls20")
    game_frame = env.reset()
    controlled_sigs = ig.warmup_controllability(oc, env, game_frame, warmup_steps)
    game_frame = env.reset()

    grid = np.asarray(game_frame.frame[-1])
    _, parts = il_mod.views_from_grid(grid, frame=0)
    grouper = il_mod.CompositeObjectGrouper()
    previous = grouper.group(grid, parts, frame=0) if use_composites else parts
    tracker = ObjectTracker()
    tracker.update(previous)

    goal_events: list[Event] = []
    contact_counts: Counter = Counter()
    match_fire = Counter()
    reward_steps = 0
    death_steps = 0
    resets = 0
    prev_levels = int(game_frame.levels_completed)

    def reperceive(frame_idx):
        nonlocal grid, previous, tracker
        grid = np.asarray(game_frame.frame[-1])
        _, p = il_mod.views_from_grid(grid, frame=frame_idx)
        previous = grouper.group(grid, p, frame=frame_idx) if use_composites else p
        tracker = ObjectTracker()
        tracker.update(previous)

    for step in range(1, steps + 1):
        available = list(game_frame.available_actions)
        prev_grid = np.asarray(game_frame.frame[-1], dtype=np.int8)
        vals, counts = np.unique(prev_grid, return_counts=True)
        background = int(vals[counts.argmax()])

        actor_views = [v for v in previous if ig.is_actor_view(v, controlled_sigs)]
        if not actor_views:
            game_frame = env.step(GameAction.from_id(rng.choice(available)))
            if game_frame.state.value in ("GAME_OVER", "WIN") or not game_frame.frame:
                resets += 1
                game_frame = env.reset()
                prev_levels = int(game_frame.levels_completed)
                reperceive(step)
            continue

        primary = actor_views[0]
        actor_centroid = (
            sum(v.centroid[0] for v in actor_views) / len(actor_views),
            sum(v.centroid[1] for v in actor_views) / len(actor_views),
        )
        # Navigate toward the least-contacted visible non-actor object.
        targets = [v for v in previous if not ig.is_actor_view(v, controlled_sigs)]
        nav_target = (
            min(targets, key=lambda v: contact_counts[ig.shape_literal(v)]) if targets else None
        )
        action = _navigate_action(actor_centroid, nav_target, available, rng)

        path_target, actor_centroid, _cell = ig.path_target_for_action(
            previous, prev_grid, actor_views, action, background
        )

        game_frame = env.step(GameAction.from_id(action))
        # Read the reward channel on EVERY frame, including the terminal one, and
        # BEFORE any reset. levels_completed ticks on the WIN frame itself; the old
        # code reset+continued first and discarded exactly that frame (the bug).
        curr_levels = int(game_frame.levels_completed)
        state = game_frame.state.value
        terminal = state in ("WIN", "GAME_OVER") or not game_frame.frame
        reward_up = curr_levels > prev_levels
        if reward_up:
            reward_steps += 1
        if state == "GAME_OVER":
            death_steps += 1

        # Log the goal-candidate contact (pair captured pre-move) with the label now
        # correct on terminal frames too. GAME_OVER is wired as a free negative bit:
        # it validates the PLUMBING (events log, labels attach, learn_goal runs on
        # real frames), NOT that we can catch a win — death is proximal, a win distal.
        if path_target is not None and not ig.is_actor_view(path_target, controlled_sigs):
            shape = ig.shape_literal(path_target)
            contact_counts[shape] += 1
            lenses = match_lenses(primary, path_target)
            for slot in lenses:
                match_fire[slot] += 1
            facts = set(goal_facts_for_pair(primary, path_target, include_match=True))
            facts.add("on_target")  # the actor moved onto the tile this step
            if state == "GAME_OVER":
                facts.add("led_to_death")
            goal_events.append(Event(frozenset(facts), reward_up))

        if terminal:
            resets += 1
            game_frame = env.reset()
            prev_levels = int(game_frame.levels_completed)
            reperceive(step)
            continue
        prev_levels = curr_levels
        reperceive(step)

    goal_result = learn_goal(goal_events, max_depth=3) if goal_events else GoalResult(
        "NO_GOAL_EVIDENCE", None, 0, 0, "no goal-candidate contacts logged"
    )
    return {
        "warmup_steps": warmup_steps,
        "steps": steps,
        "seed": seed,
        "controlled_signatures": sorted(controlled_sigs),
        "goal_events": len(goal_events),
        "reward_steps": reward_steps,
        "death_steps": death_steps,
        "match_fire": dict(match_fire),
        "distinct_shapes_touched": len(contact_counts),
        "resets": resets,
        "goal_result": goal_result,
    }


def _print_report(report: dict) -> None:
    print(f"\nGoal probe on ls20 ({report['steps']} steps, warmup={report['warmup_steps']}, "
          f"seed={report['seed']}):")
    print(f"  controlled signatures: {report['controlled_signatures']}")
    print(f"  goal-candidate events: {report['goal_events']} "
          f"({report['distinct_shapes_touched']} distinct shapes touched)")
    print(f"  reward (level-advance) steps: {report['reward_steps']}")
    print(f"  death (GAME_OVER) steps: {report['death_steps']}  [free negative — plumbing test]")
    print(f"  match letter fired (slot -> steps): {report['match_fire']}")
    print(f"  resets: {report['resets']}")
    print(f"  learn_goal: {report['goal_result']}")
    print("\nAudit verdict:")
    print("  PASS (epistemics) — learn_goal REFUSES (NO_GOAL_EVIDENCE) instead of")
    print("  fabricating a win condition. Honest silence beats a confident wrong rule.")
    if report["match_fire"]:
        print("  the earned match letter fires on real frames "
              "(actor body equals some tile under a lens).")
    else:
        print("  FINDING (a second corpse) — the match letter never fires: the avatar")
        print("  is a 2-colour composite and tiles are simple, so actor-vs-tile is the")
        print("  wrong comparison. ls20's goal compares a CARRIED attribute to a target,")
        print("  not the avatar's raw body. The alphabet needs a learned answer to")
        print("  'WHICH objects get compared', not just 'under which lens'.")
    if report["reward_steps"] == 0:
        print("  FAIL-LOUD (as predicted) — the success channel never advanced, so the")
        print("  goal cannot be induced yet. Next brick: reward-seeking planning over")
        print("  the induced dynamics (deliberately MATCH, then step on target).")
    else:
        print(f"  reward fired {report['reward_steps']}x — feed these into learn_goal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run the goal probe on ls20")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("Offline checks (goal = reward-labelled clause search):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before running the goal probe.")

    if args.live:
        _print_report(run_goal_probe(warmup_steps=args.warmup, steps=args.steps, seed=args.seed))
        raise SystemExit(0)

    print("\nAll checks pass. Run with --live to log reward-conditioned events on ls20.")
