"""0026 — Terminal predicate discovery.

Run from the repo root:
    uv run python teach/lessons/0026-terminal-predicate-discovery.py
    uv run python teach/lessons/0026-terminal-predicate-discovery.py --live --levels 1,2,3,4,5,6,7

0025 generated the matched-state descriptor:

    shape_match AND color_match AND orientation_match

That descriptor is not enough: it can hold before reward. This lesson searches
the local action predicate that turns the matched state into level completion.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0026-terminal-predicate-discovery"
TERMINAL_FACTS = (
    "next_overlap_smaller",
    "next_touch_smaller",
    "next_overlap_larger",
    "next_touch_larger",
    "before_touch_smaller",
    "before_touch_larger",
)


def _load_0025():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0025-goal-hypothesis-generator.py"
    spec = _ilu.spec_from_file_location("tp26_gh25", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gh25 = _load_0025()
am24 = gh25.am24
sal20 = gh25.sal20
ri19 = gh25.ri19


@dataclass(frozen=True)
class TerminalEvidence:
    facts: frozenset[str]
    reward_next: bool
    level: int
    frame: int
    action: str
    pair: str


@dataclass(frozen=True)
class TerminalHypothesis:
    slots: tuple[str, ...]
    positives_covered: int
    positives_total: int
    false_positives: int
    negatives_total: int
    status: str  # GENERATED | FALSIFIED | UNDERFIT
    score: float

    def predicts(self, event: TerminalEvidence) -> bool:
        return set(self.slots) <= event.facts

    def __str__(self) -> str:
        return " AND ".join(self.slots) if self.slots else "TRUE"


def _touches_pixels(pixels: frozenset[tuple[int, int]], view) -> bool:
    if pixels & view.pixels:
        return True
    target = view.pixels
    return any(
        (r + 1, c) in target
        or (r - 1, c) in target
        or (r, c + 1) in target
        or (r, c - 1) in target
        for r, c in pixels
    )


def _shift_pixels(pixels: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in pixels)


def terminal_facts(actors: list, small, large, dr: int, dc: int) -> frozenset[str]:
    facts: set[str] = set()
    for actor in actors:
        moved = _shift_pixels(actor.pixels, dr, dc)
        if moved & small.pixels:
            facts.add("next_overlap_smaller")
        if moved & large.pixels:
            facts.add("next_overlap_larger")
        if _touches_pixels(moved, small):
            facts.add("next_touch_smaller")
        if _touches_pixels(moved, large):
            facts.add("next_touch_larger")
        if _touches_pixels(actor.pixels, small):
            facts.add("before_touch_smaller")
        if _touches_pixels(actor.pixels, large):
            facts.add("before_touch_larger")
    return frozenset(facts)


def learn_terminal_hypotheses(events: list[TerminalEvidence]) -> tuple[TerminalHypothesis, ...]:
    positives = [event for event in events if event.reward_next]
    negatives = [event for event in events if not event.reward_next]
    if not positives:
        return ()
    observed = sorted(set().union(*(event.facts for event in events)))
    hypotheses: list[TerminalHypothesis] = []
    for depth in range(1, min(3, len(observed)) + 1):
        for slots in combinations(observed, depth):
            pred_pos = sum(set(slots) <= event.facts for event in positives)
            false_pos = sum(set(slots) <= event.facts for event in negatives)
            coverage = pred_pos / len(positives)
            false_rate = false_pos / max(1, len(negatives))
            exact = pred_pos == len(positives) and false_pos == 0
            status = "GENERATED" if exact else "FALSIFIED"
            score = coverage - false_rate - 0.02 * len(slots)
            hypotheses.append(TerminalHypothesis(
                slots=slots,
                positives_covered=pred_pos,
                positives_total=len(positives),
                false_positives=false_pos,
                negatives_total=len(negatives),
                status=status,
                score=score,
            ))
    return tuple(sorted(hypotheses, key=lambda h: (h.status != "GENERATED", -h.score, len(h.slots), h.slots)))


def best_terminal_hypothesis(events: list[TerminalEvidence]) -> TerminalHypothesis | None:
    hypotheses = learn_terminal_hypotheses(events)
    return hypotheses[0] if hypotheses else None


def _facts(*slots: str) -> frozenset[str]:
    return frozenset(slots)


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    events = [
        TerminalEvidence(_facts(), False, 1, 1, "ACTION4", "matched_pair"),
        TerminalEvidence(_facts("before_touch_smaller"), False, 1, 2, "ACTION1", "matched_pair"),
        TerminalEvidence(_facts("next_overlap_smaller", "next_touch_smaller"), True, 1, 3, "ACTION1", "matched_pair"),
        TerminalEvidence(_facts("next_overlap_smaller", "next_touch_smaller"), True, 2, 4, "ACTION2", "matched_pair"),
    ]
    best = best_terminal_hypothesis(events)
    checks.append(_check(
        "1 generates action-entry predicate from matched-state negatives",
        best is not None and best.status == "GENERATED" and best.slots == ("next_overlap_smaller",),
        f"best={best}",
    ))

    weak = learn_terminal_hypotheses([
        TerminalEvidence(_facts("next_touch_smaller"), True, 1, 1, "ACTION1", "matched_pair"),
        TerminalEvidence(_facts("next_touch_smaller"), False, 1, 2, "ACTION1", "matched_pair"),
    ])
    checks.append(_check(
        "2 rejects terminal fact with false positives",
        not any(h.status == "GENERATED" for h in weak),
        f"generated={[str(h) for h in weak if h.status == 'GENERATED']}",
    ))

    grid_actor = np.zeros((8, 8), dtype=bool)
    grid_actor[3:5, 3:5] = True
    grid_small = np.zeros((8, 8), dtype=bool)
    grid_small[2, 3:5] = True
    actor = am24._FakeView(grid_actor, 12, 0)
    small = am24._FakeView(grid_small, 9, 0)
    grid_large = np.zeros((8, 8), dtype=bool)
    grid_large[6:8, 6:8] = True
    large = am24._FakeView(grid_large, 9, 0)
    facts = terminal_facts([actor], small, large, -1, 0)
    checks.append(_check(
        "3 shifted actor detects overlap with smaller matched object",
        "next_overlap_smaller" in facts and "next_touch_smaller" in facts and "next_overlap_larger" not in facts,
        f"facts={sorted(facts)}",
    ))
    return all(checks)


def _view_label(view, oc16) -> str:
    coarse = ri19.rcf18.rca17._object_key(view, oc16)
    return sal20._view_label(view, oc16, coarse)


def _attribute_match_pairs(parts: list, oc16) -> list[tuple[object, object]]:
    pairs = []
    for i, a in enumerate(parts):
        ma = sal20._mask_from_view(a)
        if not am24.ras23._valid_scan_object(a, ma) or not am24.ras23._complexity(ma)["nontrivial"]:
            continue
        for b in parts[i + 1:]:
            mb = sal20._mask_from_view(b)
            if not am24.ras23._valid_scan_object(b, mb) or not am24.ras23._complexity(mb)["nontrivial"]:
                continue
            if am24.attribute_match(a, b).status == "ATTRIBUTE_MATCH":
                pairs.append((a, b))
    return pairs


def _events_from_live(trace: Path | None, levels: set[int], window: int) -> list[TerminalEvidence]:
    oc16 = ri19.rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    events: list[TerminalEvidence] = []

    for seg in oc16.segments_for_levels(segments, levels):
        level = oc16.human_level(seg.level)
        belief = oc16.AgentBelief()
        reward_indices = set(am24.ras23._reward_indices(records, seg))
        if not reward_indices:
            continue
        reward_windows = {idx for inc in reward_indices for idx in range(max(seg.start, inc - window), inc)}
        for idx in range(seg.start, seg.end + 1):
            _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
            if idx in reward_windows and idx + 1 < len(records) and records[idx + 1].action_num is not None:
                actors = oc16._actor_views(parts, belief.controlled_sigs)
                dr, dc = oc16.ACTION_DELTAS[records[idx + 1].action_num]
                for left, right in _attribute_match_pairs(parts, oc16):
                    small, large = (left, right) if left.area <= right.area else (right, left)
                    facts = terminal_facts(actors, small, large, int(dr), int(dc))
                    pair = f"{_view_label(small, oc16)} <-> {_view_label(large, oc16)}"
                    events.append(TerminalEvidence(
                        facts=facts,
                        reward_next=(idx + 1) in reward_indices,
                        level=level,
                        frame=idx,
                        action=records[idx + 1].action_id,
                        pair=pair,
                    ))
            if idx + 1 <= seg.end and records[idx + 1].action_num is not None:
                _bg1, parts1 = oc16.il.views_from_grid(records[idx + 1].frame, frame=idx + 1)
                belief.learn_remote_dynamics(parts, parts1)
                belief.observe_transition(records[idx].frame, records[idx + 1].action_num, records[idx + 1].frame)
                belief.frame_idx += 1
                belief.perceive(records[idx + 1].frame)
    return events


def run_live(trace: Path | None, levels: set[int], window: int) -> tuple[list[TerminalEvidence], tuple[TerminalHypothesis, ...]]:
    events = _events_from_live(trace, levels, window)
    hypotheses = learn_terminal_hypotheses(events)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    event_path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-terminal-events.jsonl"
    hyp_path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-terminal-hypotheses.jsonl"
    with event_path.open("w") as fh:
        for event in events:
            fh.write(json.dumps({
                "action": event.action,
                "facts": sorted(event.facts),
                "frame": event.frame,
                "level": event.level,
                "pair": event.pair,
                "reward_next": event.reward_next,
            }, sort_keys=True) + "\n")
    with hyp_path.open("w") as fh:
        for hyp in hypotheses:
            fh.write(json.dumps({
                "false_positives": hyp.false_positives,
                "hypothesis": str(hyp),
                "negatives_total": hyp.negatives_total,
                "positives_covered": hyp.positives_covered,
                "positives_total": hyp.positives_total,
                "score": hyp.score,
                "slots": hyp.slots,
                "status": hyp.status,
            }, sort_keys=True) + "\n")

    print("\nLive ls20 terminal predicate discovery:")
    print(f"  events={len(events)} positives={sum(e.reward_next for e in events)} negatives={sum(not e.reward_next for e in events)}")
    print(f"  fact_counts={dict(Counter(fact for event in events for fact in event.facts))}")
    for hyp in hypotheses[:10]:
        print(
            f"  {hyp.status:<10} score={hyp.score:.3f} pos={hyp.positives_covered}/{hyp.positives_total} "
            f"fp={hyp.false_positives}/{hyp.negatives_total} IF {hyp}"
        )
    if hypotheses and hypotheses[0].status == "GENERATED":
        print(f"  best_terminal_predicate={hypotheses[0]}")
        print(f"  full_goal = attribute_descriptor AND {hypotheses[0]}")
    else:
        print("  no exact terminal predicate generated")
    print(f"  event_artifact={event_path}")
    print(f"  hypothesis_artifact={hyp_path}")
    return events, hypotheses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="discover terminal predicate from recorded scripted play")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=4, help="frames before reward to use as evidence")
    args = parser.parse_args()

    print("Offline checks (terminal predicate discovery):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using terminal predicate discovery.")
    print("\nTerminal predicate discovery can separate matched-state from reward-producing action.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window)


if __name__ == "__main__":
    main()
