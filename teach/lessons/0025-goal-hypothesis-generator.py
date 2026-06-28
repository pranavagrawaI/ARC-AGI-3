"""0025 — Goal hypothesis generator.

Run from the repo root:
    uv run python teach/lessons/0025-goal-hypothesis-generator.py
    uv run python teach/lessons/0025-goal-hypothesis-generator.py --live --levels 1,2,3,4,5,6,7

0024 verified a user-supplied hypothesis: shape + color + orientation. This
lesson makes the agent generate that conjunction from scripted-play evidence by
searching a small closed alphabet of attribute-match slots.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0025-goal-hypothesis-generator"
SLOTS = ("shape", "color", "orientation")


def _load_0024():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0024-attribute-match-goal-hypothesis.py"
    spec = _ilu.spec_from_file_location("gh25_am24", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


am24 = _load_0024()
ri19 = am24.ri19
sal20 = am24.sal20


@dataclass(frozen=True)
class GoalEvidence:
    facts: frozenset[str]
    reward_next: bool
    level: int
    frame: int
    pair: str


@dataclass(frozen=True)
class GoalHypothesis:
    slots: tuple[str, ...]
    positives_covered: int
    positives_total: int
    false_positives: int
    negatives_total: int
    status: str  # GENERATED | GENERATED_DESCRIPTOR | FALSIFIED | UNDERFIT
    score: float

    def predicts(self, event: GoalEvidence) -> bool:
        return set(self.slots) <= event.facts

    def __str__(self) -> str:
        return " AND ".join(f"{slot}_match" for slot in self.slots) if self.slots else "TRUE"


def _facts(*slots: str) -> frozenset[str]:
    return frozenset(slots)


def learn_goal_hypotheses(events: list[GoalEvidence]) -> tuple[GoalHypothesis, ...]:
    positives = [e for e in events if e.reward_next]
    negatives = [e for e in events if not e.reward_next]
    if not positives:
        return ()
    hypotheses: list[GoalHypothesis] = []
    positive_common = tuple(slot for slot in SLOTS if all(slot in e.facts for e in positives))
    for depth in range(1, len(SLOTS) + 1):
        for slots in combinations(SLOTS, depth):
            pred_pos = sum(set(slots) <= e.facts for e in positives)
            false_pos = sum(set(slots) <= e.facts for e in negatives)
            coverage = pred_pos / len(positives)
            false_rate = false_pos / max(1, len(negatives))
            exact = pred_pos == len(positives) and false_pos == 0
            status = "GENERATED" if exact else "FALSIFIED"
            if slots == positive_common and pred_pos == len(positives):
                # Attribute match is a state predicate, not a complete reward
                # predictor. It may hold for several frames before terminal entry.
                status = "GENERATED_DESCRIPTOR"
            # Exactness dominates; among exact clauses, prefer fewer slots.
            score = coverage - false_rate - 0.03 * len(slots)
            hypotheses.append(GoalHypothesis(
                slots=slots,
                positives_covered=pred_pos,
                positives_total=len(positives),
                false_positives=false_pos,
                negatives_total=len(negatives),
                status=status,
                score=score,
            ))
    priority = {"GENERATED": 0, "GENERATED_DESCRIPTOR": 1, "FALSIFIED": 2, "UNDERFIT": 3}
    return tuple(sorted(hypotheses, key=lambda h: (priority.get(h.status, 9), -h.score, len(h.slots), h.slots)))


def best_goal_hypothesis(events: list[GoalEvidence]) -> GoalHypothesis | None:
    hypotheses = learn_goal_hypotheses(events)
    return hypotheses[0] if hypotheses else None


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    events = [
        GoalEvidence(_facts("shape", "color"), False, 1, 1, "near_miss"),
        GoalEvidence(_facts("shape", "orientation"), False, 1, 2, "near_miss"),
        GoalEvidence(_facts("color", "orientation"), False, 1, 3, "near_miss"),
        GoalEvidence(_facts("shape", "color", "orientation"), True, 1, 4, "goal"),
        GoalEvidence(_facts("shape", "color", "orientation"), True, 2, 5, "goal"),
    ]
    best = best_goal_hypothesis(events)
    checks.append(_check(
        "1 generates full conjunction from near misses",
        best is not None and best.slots == ("shape", "color", "orientation")
        and best.status in {"GENERATED", "GENERATED_DESCRIPTOR"},
        f"best={best}",
    ))

    weak = learn_goal_hypotheses([
        GoalEvidence(_facts("shape"), True, 1, 1, "goal"),
        GoalEvidence(_facts("shape"), False, 1, 2, "static_false_positive"),
    ])
    checks.append(_check(
        "2 does not treat always-true shape-only as reward predictor",
        not any(h.status == "GENERATED" and h.slots == ("shape",) for h in weak),
        f"generated={[str(h) for h in weak if h.status == 'GENERATED']}",
    ))
    return all(checks)


def _view_label(view, oc16) -> str:
    coarse = ri19.rcf18.rca17._object_key(view, oc16)
    return sal20._view_label(view, oc16, coarse)


def _events_from_live(trace: Path | None, levels: set[int], window: int) -> list[GoalEvidence]:
    oc16 = ri19.rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    events: list[GoalEvidence] = []

    for seg in oc16.segments_for_levels(segments, levels):
        level = oc16.human_level(seg.level)
        for inc in am24.ras23._reward_indices(records, seg):
            start = max(seg.start, inc - window)
            for idx in range(start, inc):
                reward_next = idx == inc - 1
                _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                for i, a in enumerate(parts):
                    ma = sal20._mask_from_view(a)
                    if not am24.ras23._valid_scan_object(a, ma):
                        continue
                    ca = am24.ras23._complexity(ma)
                    if not ca["nontrivial"]:
                        continue
                    for b in parts[i + 1:]:
                        mb = sal20._mask_from_view(b)
                        if not am24.ras23._valid_scan_object(b, mb):
                            continue
                        cb = am24.ras23._complexity(mb)
                        if not cb["nontrivial"]:
                            continue
                        match = am24.attribute_match(a, b)
                        if not match.shape_match:
                            continue
                        facts = set()
                        if match.shape_match:
                            facts.add("shape")
                        if match.color_match:
                            facts.add("color")
                        if match.orientation_match:
                            facts.add("orientation")
                        pair = f"{_view_label(a, oc16)} <-> {_view_label(b, oc16)}"
                        events.append(GoalEvidence(frozenset(facts), reward_next, level, idx, pair))
    return events


def run_live(trace: Path | None, levels: set[int], window: int) -> tuple[list[GoalEvidence], tuple[GoalHypothesis, ...]]:
    events = _events_from_live(trace, levels, window)
    hypotheses = learn_goal_hypotheses(events)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    event_path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-goal-events.jsonl"
    hyp_path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-goal-hypotheses.jsonl"
    with event_path.open("w") as fh:
        for e in events:
            fh.write(json.dumps({
                "level": e.level,
                "frame": e.frame,
                "pair": e.pair,
                "facts": sorted(e.facts),
                "reward_next": e.reward_next,
            }, sort_keys=True) + "\n")
    with hyp_path.open("w") as fh:
        for h in hypotheses:
            fh.write(json.dumps({
                "hypothesis": str(h),
                "slots": h.slots,
                "status": h.status,
                "score": h.score,
                "positives_covered": h.positives_covered,
                "positives_total": h.positives_total,
                "false_positives": h.false_positives,
                "negatives_total": h.negatives_total,
            }, sort_keys=True) + "\n")

    print("\nLive ls20 goal hypothesis generation:")
    print(f"  events={len(events)} positives={sum(e.reward_next for e in events)} negatives={sum(not e.reward_next for e in events)}")
    for h in hypotheses[:10]:
        print(
            f"  {h.status:<10} score={h.score:.3f} pos={h.positives_covered}/{h.positives_total} "
            f"fp={h.false_positives}/{h.negatives_total} IF {h}"
        )
    if hypotheses and hypotheses[0].status == "GENERATED":
        print(f"  best_generated={hypotheses[0]}")
    elif hypotheses and hypotheses[0].status == "GENERATED_DESCRIPTOR":
        print(f"  best_attribute_descriptor={hypotheses[0]}")
        print("  terminal predicate still required for a full reward predictor")
    else:
        print("  no exact generated hypothesis")
    print(f"  event_artifact={event_path}")
    print(f"  hypothesis_artifact={hyp_path}")
    return events, hypotheses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="generate goal hypotheses from recorded scripted play")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=4, help="frames before reward to use as evidence")
    args = parser.parse_args()

    print("Offline checks (goal hypothesis generation):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using goal hypothesis generation.")
    print("\nGoal hypothesis generator can compose attribute slots instead of receiving the conjunction.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window)


if __name__ == "__main__":
    main()
