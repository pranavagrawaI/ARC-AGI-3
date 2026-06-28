"""0017 — Remote causal attribution.

Run from the repo root:
    uv run python teach/lessons/0017-remote-causal-attribution.py

Lessons 0012–0016 can refuse when the old actor-centered grammar cannot express
remote effects. This lesson adds one reusable alphabet letter:

    contact(actor, trigger) -> change(remote_object, slot, delta)

It does not plan and it does not know ls20 roles. It only asks whether local
interaction facts predict global object diffs better than copy-forward.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util as _ilu
import sys as _sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

UNTRUSTED_STATUSES = frozenset({"UNCERTAIN", "LOST"})
CHANGE_SLOTS = ("color", "shape", "internal_pattern", "orientation")


@dataclass(frozen=True)
class Diff:
    """Observed non-actor state change.

    object_id=None means perception saw a changed region but could not attach it
    to a stable object identity. That must stay fail-loud: no synthetic patient.
    """

    object_id: str | None
    slot: str
    before: str
    after: str
    status: str = "OBSERVED"

    @property
    def delta(self) -> str:
        return f"{self.before}->{self.after}"

    @property
    def signature(self) -> tuple[str, str, str]:
        if self.object_id is None:
            raise ValueError("orphan diffs have no stable effect signature")
        return (self.object_id, self.slot, self.delta)


@dataclass(frozen=True)
class RemoteEvent:
    action: str
    local_facts: frozenset[str]
    global_diffs: tuple[Diff, ...]
    reward_delta: int = 0
    status: str = "OBSERVED"


@dataclass(frozen=True)
class RemoteRule:
    cause_clause: tuple[str, ...]
    effect: tuple[str, str, str]
    confidence: float
    support: int
    counterexamples: int

    def __str__(self) -> str:
        body = " AND ".join(self.cause_clause) if self.cause_clause else "TRUE"
        obj, slot, delta = self.effect
        return f"IF {body} THEN change({obj}, {slot}, {delta})"


@dataclass(frozen=True)
class EffectReport:
    effect: tuple[str, str, str]
    status: str  # LEARNED | UNDERFIT
    best_clause: tuple[str, ...]
    support: int
    counterexamples: int


@dataclass(frozen=True)
class AttributionResult:
    status: str  # LEARNED | UNDERFIT | ORPHAN_REMOTE_DIFF | NO_REMOTE_DIFFS
    rules: tuple[RemoteRule, ...]
    effect_reports: tuple[EffectReport, ...]
    orphan_diffs: tuple[Diff, ...]
    trusted_events: int
    filtered_events: int


def _trusted(events: list[RemoteEvent]) -> list[RemoteEvent]:
    return [event for event in events if event.status not in UNTRUSTED_STATUSES]


def _candidate_clauses(vocabulary: list[str], max_depth: int) -> list[tuple[str, ...]]:
    clauses: list[tuple[str, ...]] = []
    for depth in range(1, min(max_depth, len(vocabulary)) + 1):
        clauses.extend(tuple(parts) for parts in combinations(vocabulary, depth))
    return clauses


def _clause_is_observed(events: list[RemoteEvent], clause: tuple[str, ...]) -> bool:
    return any(_clause_predicts(event, clause) for event in events)


def _effect_present(event: RemoteEvent, effect: tuple[str, str, str]) -> bool:
    return any(
        diff.object_id is not None
        and diff.status not in UNTRUSTED_STATUSES
        and diff.signature == effect
        for diff in event.global_diffs
    )


def _clause_predicts(event: RemoteEvent, clause: tuple[str, ...]) -> bool:
    return set(clause) <= event.local_facts


def _score_clause(
    events: list[RemoteEvent], effect: tuple[str, str, str], clause: tuple[str, ...]
) -> tuple[int, int, int]:
    support = 0
    counterexamples = 0
    missed = 0
    for event in events:
        predicted = _clause_predicts(event, clause)
        present = _effect_present(event, effect)
        if predicted and present:
            support += 1
        elif predicted and not present:
            counterexamples += 1
        elif present and not predicted:
            missed += 1
    return support, counterexamples, missed


def _learn_effect(
    events: list[RemoteEvent], effect: tuple[str, str, str], max_depth: int, min_support: int
) -> EffectReport:
    vocabulary = sorted(set().union(*(event.local_facts for event in events)))
    clauses = _candidate_clauses(vocabulary, max_depth)
    if not clauses:
        return EffectReport(effect, "UNDERFIT", (), 0, 0)

    ranked = []
    for clause in clauses:
        if not _clause_is_observed(events, clause):
            continue
        support, counterexamples, missed = _score_clause(events, effect, clause)
        ranked.append((counterexamples + missed, counterexamples, missed, -support, len(clause), clause))
    if not ranked:
        return EffectReport(effect, "UNDERFIT", (), 0, 0)
    _errors, counterexamples, missed, neg_support, _depth, best = min(ranked)
    support = -neg_support

    if support >= min_support and counterexamples == 0 and missed == 0:
        return EffectReport(effect, "LEARNED", best, support, counterexamples)
    return EffectReport(effect, "UNDERFIT", best, support, counterexamples + missed)


def learn_remote_rules(
    events: list[RemoteEvent], max_depth: int = 3, min_support: int = 2
) -> AttributionResult:
    """Learn local-contact clauses that predict remote diffs.

    A remote rule is accepted only with positive support, negative contrast, and
    no counterexamples. This is intentionally stricter than scoring: it preserves
    the fail-loud posture until the causal alphabet earns a stable rule.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if min_support < 1:
        raise ValueError("min_support must be >= 1")

    clean = _trusted(events)
    orphan_diffs = tuple(
        diff
        for event in clean
        for diff in event.global_diffs
        if diff.object_id is None or diff.status in UNTRUSTED_STATUSES
    )
    effects = sorted(
        {
            diff.signature
            for event in clean
            for diff in event.global_diffs
            if diff.object_id is not None and diff.status not in UNTRUSTED_STATUSES
        }
    )

    if not effects:
        status = "ORPHAN_REMOTE_DIFF" if orphan_diffs else "NO_REMOTE_DIFFS"
        return AttributionResult(status, (), (), orphan_diffs, len(clean), len(events) - len(clean))

    reports = tuple(
        _learn_effect(clean, effect, max_depth=max_depth, min_support=min_support)
        for effect in effects
    )
    rules = tuple(
        RemoteRule(
            report.best_clause,
            report.effect,
            confidence=1.0,
            support=report.support,
            counterexamples=report.counterexamples,
        )
        for report in reports
        if report.status == "LEARNED"
    )

    if orphan_diffs and not rules:
        status = "ORPHAN_REMOTE_DIFF"
    elif len(rules) == len(reports):
        status = "LEARNED"
    elif rules:
        status = "UNDERFIT"
    else:
        status = "UNDERFIT"
    return AttributionResult(status, rules, reports, orphan_diffs, len(clean), len(events) - len(clean))


def _load_oc16():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0016-oracle-checker.py"
    spec = _ilu.spec_from_file_location("rca17_oc16", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pixels_touch(a, b) -> bool:
    if a.pixels & b.pixels:
        return True
    b_pixels = b.pixels
    for r, c in a.pixels:
        if (
            (r + 1, c) in b_pixels
            or (r - 1, c) in b_pixels
            or (r, c + 1) in b_pixels
            or (r, c - 1) in b_pixels
        ):
            return True
    return False


def _view_state(view) -> str:
    raw = ";".join(
        f"{slot}={view.lattice(slot).get('raw')}"
        for slot in CHANGE_SLOTS
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _live_delta(view_t, view_t1, slots: tuple[str, ...]) -> tuple[str, str, str]:
    """Return one compact live diff slot plus abstract delta.

    Exact before/after patterns are too specific for cyclic remote controls. If
    orientation changes, prefer its modulo delta; otherwise use a stable digest.
    """

    if "orientation" in slots:
        before = view_t.orientation_lattice.get("raw")
        after = view_t1.orientation_lattice.get("raw")
        if isinstance(before, int) and isinstance(after, int):
            return "orientation", "any", f"rot{(after - before) % 4:+d}mod4"
        return "orientation", str(before), str(after)
    if "internal_pattern" in slots:
        return "internal_pattern", "state", f"digest:{_view_state(view_t)}->{_view_state(view_t1)}"
    if "shape" in slots:
        return "shape", "state", f"digest:{_view_state(view_t)}->{_view_state(view_t1)}"
    if "color" in slots:
        return "color", str(view_t.color_lattice.get("raw")), str(view_t1.color_lattice.get("raw"))
    return "pixel_region", "state", f"digest:{_view_state(view_t)}->{_view_state(view_t1)}"


def _object_key(view, oc16) -> str:
    r, c = view.centroid
    # Live attribution needs a patient identity that survives appearance change.
    # Coarse location is a generic fallback: it says "same remote region", not
    # "bottom-left workspace" or any other game-specific role name.
    cell = getattr(oc16, "CELL", 5)
    return f"remote_region@{round(r / cell) * cell},{round(c / cell) * cell}"


def _changed_slots(view_t, view_t1) -> tuple[str, ...]:
    changed = []
    for slot in CHANGE_SLOTS:
        if view_t.lattice(slot).get("raw") != view_t1.lattice(slot).get("raw"):
            changed.append(slot)
    return tuple(changed)


def _view_covering_cell(views: list, row: int, col: int):
    cell = (row, col)
    for view in views:
        if cell in view.pixels:
            return view
    return None


def _live_local_facts(parts_t: list, controlled_sigs: set[tuple[int, int]], action: int, oc16) -> frozenset[str]:
    actors = oc16._actor_views(parts_t, controlled_sigs)
    facts = {f"action={action}"}
    max_trigger_area = 4 * getattr(oc16, "CELL", 5) ** 2
    if not actors:
        facts.add("actor_missing")
        return frozenset(facts)

    dr, dc = oc16.ACTION_DELTAS.get(action, (0, 0))
    ar = sum(actor.centroid[0] for actor in actors) / len(actors)
    ac = sum(actor.centroid[1] for actor in actors) / len(actors)
    target = _view_covering_cell(parts_t, round(ar + dr), round(ac + dc))
    if (
        target is not None
        and not oc16.ig.is_actor_view(target, controlled_sigs)
        and target.area <= max_trigger_area
    ):
        facts.add("contact")
        facts.add(f"trigger_shape={oc16.ig.shape_literal(target)}")
    if "contact" not in facts:
        facts.add("no_contact")
    return frozenset(facts)


def live_remote_events(trace: Path | None = None, levels: set[int] | None = None) -> tuple[list[RemoteEvent], Counter]:
    """Convert the recorded ls20 trace into 0017 attribution events.

    The trace supplies only actions and frames. Local trigger facts are read from
    the post-action frame; remote effects are unmatched prediction residue from
    the old actor-centered model, attached to stable object identity when possible.
    """

    oc16 = _load_oc16()
    trace_path = trace or oc16.DEFAULT_TRACE
    records = oc16.load_trace(trace_path)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    active_levels = levels or {1, 2, 3, 4, 5, 6, 7}
    belief = oc16.AgentBelief()
    events: list[RemoteEvent] = []
    counts: Counter = Counter()

    for seg in oc16.segments_for_levels(segments, active_levels):
        belief.reset_episode()
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
            predicted = oc16.predict_frame(grid_t, action, views_t, belief)
            actor_views = oc16._actor_views(views_t, belief.controlled_sigs)
            _bg_t, parts_t = oc16.il.views_from_grid(grid_t, frame=t)
            _bg_t1, parts_t1 = oc16.il.views_from_grid(grid_t1, frame=t + 1)

            diffs: list[Diff] = []
            for cluster in oc16.diff_clusters(predicted != grid_t1):
                kind = oc16._change_kind(grid_t, grid_t1, cluster, actor_views)
                if kind != "remote_attribute_change":
                    continue
                cr = (cluster.rmin + cluster.rmax) // 2
                cc = (cluster.cmin + cluster.cmax) // 2
                view_t = oc16._view_at(parts_t, cr, cc)
                view_t1 = oc16._match_part_view(view_t, parts_t1)
                if view_t is None or view_t1 is None:
                    diffs.append(Diff(None, "remote_region", "unmatched", "changed"))
                    counts["orphan_diffs"] += 1
                    continue
                slots = _changed_slots(view_t, view_t1)
                if not slots:
                    slots = ("pixel_region",)
                object_id = _object_key(view_t, oc16)
                slot, before, after = _live_delta(view_t, view_t1, slots)
                diffs.append(Diff(object_id, slot, before, after))
                counts[f"slot:{slot}"] += 1

            facts = _live_local_facts(parts_t, belief.controlled_sigs, action, oc16)
            events.append(RemoteEvent(nxt.action_id, facts, tuple(diffs)))
            counts["events"] += 1
            counts["events_with_remote_diff"] += int(bool(diffs))
            counts["events_with_contact"] += int("contact" in facts)

            belief.learn_remote_dynamics(parts_t, parts_t1)
            belief.observe_transition(grid_t, action, grid_t1)
            belief.frame_idx += 1
            belief.perceive(grid_t1)

    return events, counts


def run_live_trace(trace: Path | None = None, levels: set[int] | None = None) -> AttributionResult:
    events, counts = live_remote_events(trace, levels)
    result = learn_remote_rules(events, max_depth=2, min_support=2)
    print("\nLive ls20 passive trace attribution:")
    print(f"  events={counts['events']} contact_events={counts['events_with_contact']} remote_diff_events={counts['events_with_remote_diff']}")
    if counts["orphan_diffs"]:
        print(f"  orphan_diffs={counts['orphan_diffs']}")
    slot_counts = {k.removeprefix("slot:"): v for k, v in counts.items() if k.startswith("slot:")}
    print(f"  remote_slots={slot_counts}")
    print(f"  status={result.status} rules={len(result.rules)} trusted={result.trusted_events}")
    for rule in result.rules[:12]:
        print(f"  RULE support={rule.support}: {rule}")
    underfit = [r for r in result.effect_reports if r.status == "UNDERFIT"]
    if underfit:
        print(f"  underfit_effects={len(underfit)}")
        for report in underfit[:6]:
            print(f"    UNDERFIT support={report.support} counter={report.counterexamples}: {report.effect}")
    return result


def _event(trigger: str, *diffs: Diff) -> RemoteEvent:
    return RemoteEvent(
        action="move",
        local_facts=frozenset({"contact", f"trigger_id={trigger}", f"trigger_shape={trigger}"}),
        global_diffs=tuple(diffs),
    )


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []

    synthetic = [
        _event("X", Diff("B", "color", "red", "blue")),
        _event("X", Diff("B", "color", "red", "blue")),
        _event("Y", Diff("C", "orientation", "north", "east")),
        _event("Y", Diff("C", "orientation", "north", "east")),
        _event("Z"),
        _event("Z"),
    ]
    result = learn_remote_rules(synthetic, max_depth=2, min_support=2)
    learned = {rule.effect: rule.cause_clause for rule in result.rules}
    checks.append(_check(
        "1 learns stable remote colour rule",
        learned.get(("B", "color", "red->blue")) == ("trigger_id=X",),
        f"rules={[str(rule) for rule in result.rules]}",
    ))
    checks.append(_check(
        "2 learns stable remote rotation rule",
        learned.get(("C", "orientation", "north->east")) == ("trigger_id=Y",),
        f"rules={[str(rule) for rule in result.rules]}",
    ))
    checks.append(_check(
        "3 no rule for inert trigger",
        all("trigger_id=Z" not in rule.cause_clause for rule in result.rules),
        f"rules={[str(rule) for rule in result.rules]}",
    ))

    confounded = [
        _event("X", Diff("B", "color", "red", "blue"), Diff("C", "pattern", "a", "b")),
        _event("X", Diff("B", "color", "red", "blue")),
        _event("Y", Diff("C", "pattern", "a", "b")),
        _event("Y"),
        _event("Z"),
    ]
    conf = learn_remote_rules(confounded, max_depth=2, min_support=2)
    statuses = {report.effect: report.status for report in conf.effect_reports}
    checks.append(_check(
        "4 keeps stable cause under simultaneous confound",
        statuses.get(("B", "color", "red->blue")) == "LEARNED",
        f"reports={conf.effect_reports}",
    ))
    checks.append(_check(
        "5 marks unstable diff underfit",
        statuses.get(("C", "pattern", "a->b")) == "UNDERFIT",
        f"reports={conf.effect_reports}",
    ))

    orphan = learn_remote_rules([
        _event("X", Diff(None, "pixel_region", "hash1", "hash2")),
        _event("X", Diff(None, "pixel_region", "hash1", "hash2")),
        _event("Z"),
    ])
    checks.append(_check(
        "6 orphan remote diff refuses rule",
        orphan.status == "ORPHAN_REMOTE_DIFF" and not orphan.rules and len(orphan.orphan_diffs) == 2,
        f"status={orphan.status}, rules={orphan.rules}, orphans={len(orphan.orphan_diffs)}",
    ))

    return all(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also replay the recorded ls20 trace")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    args = parser.parse_args()

    print("Offline checks (synthetic local-contact / remote-diff traces):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using remote attribution.")

    print(
        "\nRemote causal attribution adds one causal arity: local contact can be "
        "evidence for a remote patient changing. It still refuses orphan identity "
        "and unstable confounds."
    )
    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live_trace(args.trace, levels)


if __name__ == "__main__":
    main()
