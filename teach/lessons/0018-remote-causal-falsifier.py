"""0018 — Remote causal falsifier.

Run from the repo root:
    uv run python teach/lessons/0018-remote-causal-falsifier.py
    uv run python teach/lessons/0018-remote-causal-falsifier.py --live --levels 2

0017 proposes local-contact -> remote-change rules. This lesson downgrades them
when the same remote effect happens without the proposed cause, or when the cause
fires without the effect. That is the difference between correlation and causal
evidence.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _load_0017():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0017-remote-causal-attribution.py"
    spec = _ilu.spec_from_file_location("rcf18_rca17", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rca17 = _load_0017()
Diff = rca17.Diff
RemoteEvent = rca17.RemoteEvent
RemoteRule = rca17.RemoteRule

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0018-remote-causal-falsifier"


@dataclass(frozen=True)
class RemoteHypothesis:
    cause_clause: tuple[str, ...]
    effect: tuple[str, str, str]

    def cause_fires(self, event: RemoteEvent) -> bool:
        return set(self.cause_clause) <= event.local_facts

    def effect_fires(self, event: RemoteEvent) -> bool:
        return rca17._effect_present(event, self.effect)

    def __str__(self) -> str:
        body = " AND ".join(self.cause_clause) if self.cause_clause else "TRUE"
        obj, slot, delta = self.effect
        return f"IF {body} THEN change({obj}, {slot}, {delta})"


@dataclass(frozen=True)
class LedgerEntry:
    hypothesis: RemoteHypothesis
    support: int
    trigger_failures: int
    spontaneous_effects: int
    quiet_negatives: int
    status: str  # SUPPORTED | AUTONOMOUS_CANDIDATE | REFUTED | CONTESTED
    confidence: float

    @property
    def cause_total(self) -> int:
        return self.support + self.trigger_failures

    @property
    def noncause_total(self) -> int:
        return self.spontaneous_effects + self.quiet_negatives


@dataclass(frozen=True)
class StepThought:
    step_index: int
    human_level: int
    action: str
    local_facts: tuple[str, ...]
    effects: tuple[tuple[str, str, str], ...]
    orphan_diffs: int
    fired: tuple[LedgerEntry, ...]
    contradicted: tuple[LedgerEntry, ...]
    spontaneous: tuple[LedgerEntry, ...]


def classify_counts(
    support: int,
    trigger_failures: int,
    spontaneous_effects: int,
    quiet_negatives: int,
    *,
    min_support: int = 2,
) -> tuple[str, float]:
    cause_total = support + trigger_failures
    noncause_total = spontaneous_effects + quiet_negatives
    p_effect_given_cause = support / cause_total if cause_total else 0.0
    p_effect_given_not = spontaneous_effects / noncause_total if noncause_total else 0.0
    contrast = max(0.0, p_effect_given_cause - p_effect_given_not)

    if trigger_failures >= min_support and trigger_failures >= support:
        return "REFUTED", contrast
    if spontaneous_effects >= min_support and (
        p_effect_given_not >= 0.5 or spontaneous_effects >= support
    ):
        return "AUTONOMOUS_CANDIDATE", contrast
    if support >= min_support and p_effect_given_cause >= 0.8 and p_effect_given_not <= 0.2:
        return "SUPPORTED", contrast
    return "CONTESTED", contrast


def score_hypothesis(
    events: list[RemoteEvent], hypothesis: RemoteHypothesis, *, min_support: int = 2
) -> LedgerEntry:
    support = trigger_failures = spontaneous_effects = quiet_negatives = 0
    for event in rca17._trusted(events):
        cause = hypothesis.cause_fires(event)
        effect = hypothesis.effect_fires(event)
        if cause and effect:
            support += 1
        elif cause and not effect:
            trigger_failures += 1
        elif not cause and effect:
            spontaneous_effects += 1
        else:
            quiet_negatives += 1
    status, confidence = classify_counts(
        support,
        trigger_failures,
        spontaneous_effects,
        quiet_negatives,
        min_support=min_support,
    )
    return LedgerEntry(
        hypothesis,
        support,
        trigger_failures,
        spontaneous_effects,
        quiet_negatives,
        status,
        confidence,
    )


def propose_hypotheses(
    events: list[RemoteEvent], *, max_depth: int = 2, min_support: int = 2
) -> tuple[RemoteHypothesis, ...]:
    attribution = rca17.learn_remote_rules(events, max_depth=max_depth, min_support=min_support)
    hypotheses = {
        RemoteHypothesis(rule.cause_clause, rule.effect)
        for rule in attribution.rules
    }
    for report in attribution.effect_reports:
        if report.best_clause and report.support >= min_support:
            hypotheses.add(RemoteHypothesis(report.best_clause, report.effect))
    return tuple(sorted(hypotheses, key=lambda h: (h.effect, h.cause_clause)))


def falsify_remote_hypotheses(
    events: list[RemoteEvent], *, max_depth: int = 2, min_support: int = 2
) -> tuple[LedgerEntry, ...]:
    return tuple(
        score_hypothesis(events, hypothesis, min_support=min_support)
        for hypothesis in propose_hypotheses(events, max_depth=max_depth, min_support=min_support)
    )


def cumulative_step_thoughts(
    events: list[RemoteEvent], hypotheses: tuple[RemoteHypothesis, ...]
) -> tuple[StepThought, ...]:
    thoughts: list[StepThought] = []
    history: list[RemoteEvent] = []
    for i, event in enumerate(events):
        history.append(event)
        entries = tuple(score_hypothesis(history, h) for h in hypotheses)
        effects = tuple(
            diff.signature
            for diff in event.global_diffs
            if diff.object_id is not None and diff.status not in rca17.UNTRUSTED_STATUSES
        )
        orphan_diffs = sum(
            1
            for diff in event.global_diffs
            if diff.object_id is None or diff.status in rca17.UNTRUSTED_STATUSES
        )
        fired = tuple(e for e in entries if e.hypothesis.cause_fires(event) and e.hypothesis.effect_fires(event))
        contradicted = tuple(e for e in entries if e.hypothesis.cause_fires(event) and not e.hypothesis.effect_fires(event))
        spontaneous = tuple(e for e in entries if not e.hypothesis.cause_fires(event) and e.hypothesis.effect_fires(event))
        thoughts.append(
            StepThought(
                step_index=i,
                human_level=-1,
                action=event.action,
                local_facts=tuple(sorted(event.local_facts)),
                effects=effects,
                orphan_diffs=orphan_diffs,
                fired=fired,
                contradicted=contradicted,
                spontaneous=spontaneous,
            )
        )
    return tuple(thoughts)


def _event(trigger: str, *diffs: Diff) -> RemoteEvent:
    return RemoteEvent(
        action="move",
        local_facts=frozenset({"contact", f"trigger_shape={trigger}"}),
        global_diffs=tuple(diffs),
    )


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    true_trigger = [
        _event("X", Diff("B", "orientation", "any", "rot+1")),
        _event("X", Diff("B", "orientation", "any", "rot+1")),
        _event("Y"),
        _event("Y"),
        _event("Z"),
    ]
    ledger = falsify_remote_hypotheses(true_trigger)
    checks.append(_check(
        "1 true remote trigger supported",
        len(ledger) == 1 and ledger[0].status == "SUPPORTED",
        f"ledger={ledger}",
    ))

    autonomous = [
        _event("X", Diff("B", "orientation", "any", "rot+1")),
        _event("X", Diff("B", "orientation", "any", "rot+1")),
        _event("Y", Diff("B", "orientation", "any", "rot+1")),
        _event("Z", Diff("B", "orientation", "any", "rot+1")),
    ]
    auto_h = RemoteHypothesis(("trigger_shape=X",), ("B", "orientation", "any->rot+1"))
    auto_entry = score_hypothesis(autonomous, auto_h)
    checks.append(_check(
        "2 autonomous cycler downgraded",
        auto_entry.status == "AUTONOMOUS_CANDIDATE",
        f"status={auto_entry.status}, spontaneous={auto_entry.spontaneous_effects}",
    ))

    refuted = [
        _event("X", Diff("B", "orientation", "any", "rot+1")),
        _event("X"),
        _event("X"),
        _event("Y"),
    ]
    ref_entry = score_hypothesis(refuted, auto_h)
    checks.append(_check(
        "3 trigger failures refute",
        ref_entry.status == "REFUTED",
        f"status={ref_entry.status}, failures={ref_entry.trigger_failures}, support={ref_entry.support}",
    ))

    accidental_train = [
        _event("X", Diff("B", "orientation", "any", "rot+1")),
        _event("X", Diff("B", "orientation", "any", "rot+1")),
        _event("Y"),
    ]
    accidental_heldout = accidental_train + [
        _event("Y", Diff("B", "orientation", "any", "rot+1")),
        _event("Z", Diff("B", "orientation", "any", "rot+1")),
    ]
    trained = score_hypothesis(accidental_train, auto_h)
    heldout = score_hypothesis(accidental_heldout, auto_h)
    checks.append(_check(
        "4 held-out spontaneous effects downgrade correlation",
        trained.status == "SUPPORTED" and heldout.status == "AUTONOMOUS_CANDIDATE",
        f"trained={trained.status}, heldout={heldout.status}",
    ))
    return all(checks)


def _live_step_events(trace: Path | None, levels: set[int]) -> tuple[list[RemoteEvent], list[dict], Counter]:
    """0017 live replay plus step metadata for inspection artifacts."""

    oc16 = rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    belief = oc16.AgentBelief()
    events: list[RemoteEvent] = []
    meta: list[dict] = []
    counts: Counter = Counter()

    for seg in oc16.segments_for_levels(segments, levels):
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
                slots = rca17._changed_slots(view_t, view_t1) or ("pixel_region",)
                slot, before, after = rca17._live_delta(view_t, view_t1, slots)
                diffs.append(Diff(rca17._object_key(view_t, oc16), slot, before, after))
                counts[f"slot:{slot}"] += 1

            facts = rca17._live_local_facts(parts_t, belief.controlled_sigs, action, oc16)
            event = RemoteEvent(nxt.action_id, facts, tuple(diffs))
            events.append(event)
            meta.append({
                "trace_index": t,
                "produced_frame": t + 1,
                "human_level": oc16.human_level(records[t].levels_completed),
                "action": nxt.action_id,
            })
            counts["events"] += 1
            counts["events_with_contact"] += int("contact" in facts)
            counts["events_with_remote_diff"] += int(bool(diffs))

            belief.learn_remote_dynamics(parts_t, parts_t1)
            belief.observe_transition(grid_t, action, grid_t1)
            belief.frame_idx += 1
            belief.perceive(grid_t1)

    return events, meta, counts


def _summarize_entries(entries: tuple[LedgerEntry, ...], limit: int = 3) -> str:
    parts = []
    for entry in entries[:limit]:
        parts.append(
            f"{entry.status}:s{entry.support}/f{entry.trigger_failures}/sp{entry.spontaneous_effects} "
            f"{entry.hypothesis}"
        )
    if len(entries) > limit:
        parts.append(f"+{len(entries) - limit} more")
    return " | ".join(parts)


def _write_live_artifact(
    thoughts: list[StepThought], meta: list[dict], levels: set[int]
) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-step-thoughts.tsv"
    with path.open("w") as fh:
        fh.write(
            "trace_index\tproduced_frame\thuman_level\taction\tlocal_facts\t"
            "effects\torphan_diffs\tfired\tcontradicted\tspontaneous\n"
        )
        for thought, info in zip(thoughts, meta, strict=True):
            fh.write(
                f"{info['trace_index']}\t{info['produced_frame']}\t{info['human_level']}\t{info['action']}\t"
                f"{json.dumps(thought.local_facts)}\t{json.dumps(thought.effects)}\t{thought.orphan_diffs}\t"
                f"{json.dumps(_summarize_entries(thought.fired))}\t"
                f"{json.dumps(_summarize_entries(thought.contradicted))}\t"
                f"{json.dumps(_summarize_entries(thought.spontaneous))}\n"
            )
    return path


def run_live(trace: Path | None, levels: set[int]) -> tuple[LedgerEntry, ...]:
    events, meta, counts = _live_step_events(trace, levels)
    ledger = falsify_remote_hypotheses(events, max_depth=2, min_support=2)
    hypotheses = tuple(entry.hypothesis for entry in ledger)
    raw_thoughts = cumulative_step_thoughts(events, hypotheses)
    thoughts = [
        StepThought(
            step_index=t.step_index,
            human_level=info["human_level"],
            action=t.action,
            local_facts=t.local_facts,
            effects=t.effects,
            orphan_diffs=t.orphan_diffs,
            fired=t.fired,
            contradicted=t.contradicted,
            spontaneous=t.spontaneous,
        )
        for t, info in zip(raw_thoughts, meta, strict=True)
    ]
    artifact = _write_live_artifact(thoughts, meta, levels)
    status_counts = Counter(entry.status for entry in ledger)

    print("\nLive ls20 remote causal falsifier:")
    print(
        f"  events={counts['events']} contact_events={counts['events_with_contact']} "
        f"remote_diff_events={counts['events_with_remote_diff']} orphan_diffs={counts['orphan_diffs']}"
    )
    slot_counts = {k.removeprefix("slot:"): v for k, v in counts.items() if k.startswith("slot:")}
    print(f"  remote_slots={slot_counts}")
    print(f"  hypotheses={len(ledger)} statuses={dict(status_counts)}")
    for entry in sorted(ledger, key=lambda e: (e.status != "SUPPORTED", -e.support, -e.confidence))[:12]:
        print(
            f"  {entry.status:<20} support={entry.support:<3} fail={entry.trigger_failures:<3} "
            f"spont={entry.spontaneous_effects:<3} quiet={entry.quiet_negatives:<3} "
            f"conf={entry.confidence:.2f} {entry.hypothesis}"
        )
    print(f"  step_thoughts={artifact}")
    interesting = [t for t in thoughts if t.fired or t.contradicted or t.spontaneous or "contact" in t.local_facts]
    print("\n  First step-thought rows:")
    for thought, info in list(zip(thoughts, meta, strict=True))[:3]:
        print(
            f"    trace={info['trace_index']} L{info['human_level']} {info['action']} "
            f"facts={thought.local_facts} effects={len(thought.effects)} orphans={thought.orphan_diffs}"
        )
    if interesting:
        print("\n  First interesting model updates:")
        for thought in interesting[:6]:
            print(
                f"    step={thought.step_index} L{thought.human_level} {thought.action} "
                f"facts={thought.local_facts} effects={len(thought.effects)} "
                f"fired={len(thought.fired)} contradicted={len(thought.contradicted)} spontaneous={len(thought.spontaneous)}"
            )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="replay recorded ls20 trace and write step-thought TSV")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="2", help="comma-separated human levels for live replay")
    args = parser.parse_args()

    print("Offline checks (synthetic causal falsification traces):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using remote causal falsification.")
    print("\nRemote causal falsification can now downgrade correlation to autonomous/refuted.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels)


if __name__ == "__main__":
    main()
