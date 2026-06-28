"""0022 — Goal falsifier over template candidates.

Run from the repo root:
    uv run python teach/lessons/0022-goal-falsifier.py
    uv run python teach/lessons/0022-goal-falsifier.py --live --levels 2,3

0021 proposes workspace/template pairs. This lesson tests the narrow goal claim
that a workspace-template match is necessary before level completion. It does not
yet claim sufficiency; terminal/chamber interaction remains a separate predicate.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0022-goal-falsifier"


def _load_0021():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0021-template-candidate-filter.py"
    spec = _ilu.spec_from_file_location("gf22_tcf21", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tcf21 = _load_0021()
sal20 = tcf21.sal20
ri19 = tcf21.ri19


@dataclass(frozen=True)
class GoalStep:
    match: bool
    terminal: bool
    reward: bool


@dataclass(frozen=True)
class GoalFalsifierResult:
    status: str  # GOAL_SUPPORTED | GOAL_FALSIFIED | INSUFFICIENT_REWARD_EVENTS | TERMINAL_UNKNOWN | AMBIGUOUS_NO_CANDIDATE
    reward_events: int
    reward_with_match: int
    reward_without_match: int
    match_without_reward: int
    terminal_with_match_reward: int
    evidence: dict[str, object]


def falsify_goal_prerequisite(steps: list[GoalStep]) -> GoalFalsifierResult:
    rewards = [s for s in steps if s.reward]
    if not rewards:
        return GoalFalsifierResult(
            "INSUFFICIENT_REWARD_EVENTS", 0, 0, 0, 0, 0,
            {"note": "no level-completion/reward event observed"},
        )
    reward_with_match = sum(s.match for s in rewards)
    reward_without_match = len(rewards) - reward_with_match
    match_without_reward = sum(s.match and not s.reward for s in steps)
    terminal_with_match_reward = sum(s.terminal and s.match and s.reward for s in steps)
    evidence = {
        "reward_events": len(rewards),
        "reward_with_match": reward_with_match,
        "reward_without_match": reward_without_match,
        "match_without_reward": match_without_reward,
        "terminal_observed": any(s.terminal for s in steps),
        "terminal_with_match_reward": terminal_with_match_reward,
    }
    if reward_without_match:
        return GoalFalsifierResult(
            "GOAL_FALSIFIED", len(rewards), reward_with_match, reward_without_match,
            match_without_reward, terminal_with_match_reward, evidence,
        )
    if not any(s.terminal for s in steps):
        return GoalFalsifierResult(
            "TERMINAL_UNKNOWN", len(rewards), reward_with_match, reward_without_match,
            match_without_reward, terminal_with_match_reward, evidence,
        )
    return GoalFalsifierResult(
        "GOAL_SUPPORTED", len(rewards), reward_with_match, reward_without_match,
        match_without_reward, terminal_with_match_reward, evidence,
    )


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    supported = falsify_goal_prerequisite([
        GoalStep(match=False, terminal=False, reward=False),
        GoalStep(match=True, terminal=False, reward=False),
        GoalStep(match=True, terminal=True, reward=True),
    ])
    checks.append(_check(
        "1 match+terminal reward supports prerequisite",
        supported.status == "GOAL_SUPPORTED",
        f"status={supported.status}, evidence={supported.evidence}",
    ))

    falsified = falsify_goal_prerequisite([
        GoalStep(match=False, terminal=True, reward=True),
        GoalStep(match=True, terminal=False, reward=False),
    ])
    checks.append(_check(
        "2 reward without match falsifies prerequisite",
        falsified.status == "GOAL_FALSIFIED",
        f"status={falsified.status}, evidence={falsified.evidence}",
    ))

    no_reward = falsify_goal_prerequisite([
        GoalStep(match=True, terminal=False, reward=False),
        GoalStep(match=False, terminal=False, reward=False),
    ])
    checks.append(_check(
        "3 no reward refuses",
        no_reward.status == "INSUFFICIENT_REWARD_EVENTS",
        f"status={no_reward.status}",
    ))

    terminal_unknown = falsify_goal_prerequisite([
        GoalStep(match=True, terminal=False, reward=True),
    ])
    checks.append(_check(
        "4 reward with match but no terminal signal stays incomplete",
        terminal_unknown.status == "TERMINAL_UNKNOWN",
        f"status={terminal_unknown.status}",
    ))
    return all(checks)


def _parse_centroid(label: str) -> tuple[int, int] | None:
    marker = "|centroid="
    if marker not in label:
        return None
    try:
        raw = label.split(marker, 1)[1].split("|", 1)[0]
        r, c = raw.split(",")
        return int(r), int(c)
    except ValueError:
        return None


def _candidate_live_rows(trace: Path | None, level: int) -> list[tcf21.TemplateCandidate]:
    return [
        c for c in tcf21._level_live_candidates(trace, level)
        if c.status == "TEMPLATE_CANDIDATE"
    ]


def _view_for_region(parts: list, object_id: str):
    loc = sal20._parse_region(object_id)
    if loc is None:
        return None
    return sal20._view_near(parts, *loc)


def _view_for_template(parts: list, template_id: str):
    loc = _parse_centroid(template_id)
    if loc is None:
        return None
    return sal20._view_near(parts, *loc)


def _candidate_match_on_frame(parts: list, candidate: tcf21.TemplateCandidate) -> tuple[bool, dict[str, object]]:
    workspace = _view_for_region(parts, candidate.workspace_id)
    template = _view_for_template(parts, candidate.template_id)
    if workspace is None or template is None:
        return False, {"status": "PERCEPTION_MISSING"}
    match = sal20.scale_congruence(sal20._mask_from_view(workspace), sal20._mask_from_view(template))
    return match.status == "MATCH", {
        "status": match.status,
        "score": round(match.score, 4),
        "transform": None if match.transform is None else match.transform.__dict__,
        "evidence": match.evidence,
    }


def run_live(trace: Path | None, levels: set[int], window: int = 3) -> list[dict[str, object]]:
    oc16 = ri19.rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    candidates_by_level = {level: _candidate_live_rows(trace, level) for level in levels}
    rows: list[dict[str, object]] = []

    for seg in oc16.segments_for_levels(segments, levels):
        level = oc16.human_level(seg.level)
        candidates = candidates_by_level.get(level, [])
        if not candidates:
            rows.append({
                "level": level,
                "candidate": None,
                "status": "AMBIGUOUS_NO_CANDIDATE",
                "reward_events": 0,
                "detail": "no TEMPLATE_CANDIDATE from 0021",
            })
            continue

        reward_indices = [
            i for i in range(seg.start + 1, min(seg.end + 2, len(records)))
            if records[i].levels_completed > records[i - 1].levels_completed
        ]
        for candidate in candidates:
            steps: list[GoalStep] = []
            inspections = []
            for reward_idx in reward_indices:
                start = max(seg.start, reward_idx - window)
                pre_reward_matches = []
                for idx in range(start, reward_idx):
                    _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                    matched, match_detail = _candidate_match_on_frame(parts, candidate)
                    pre_reward_matches.append(matched)
                    steps.append(GoalStep(match=matched, terminal=False, reward=False))
                    inspections.append({
                        "frame": idx,
                        "reward": False,
                        "matched": matched,
                        "match_detail": match_detail,
                    })
                reward_matched = any(pre_reward_matches)
                steps.append(GoalStep(match=reward_matched, terminal=True, reward=True))
                inspections.append({
                    "frame": reward_idx,
                    "reward": True,
                    "matched": reward_matched,
                    "match_detail": {
                        "status": "PRE_REWARD_WINDOW_MATCH" if reward_matched else "NO_PRE_REWARD_WINDOW_MATCH",
                        "window": [start, reward_idx - 1],
                    },
                })
            result = falsify_goal_prerequisite(steps)
            rows.append({
                "level": level,
                "candidate": f"{candidate.workspace_id} -> {candidate.template_id}",
                "status": result.status,
                "reward_events": result.reward_events,
                "reward_with_match": result.reward_with_match,
                "reward_without_match": result.reward_without_match,
                "match_without_reward": result.match_without_reward,
                "evidence": result.evidence,
                "inspections": inspections,
            })

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-goal-falsifier.jsonl"
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print("\nLive ls20 goal falsifier:")
    print(f"  levels={sorted(levels)} rows={len(rows)} statuses={dict(Counter(row['status'] for row in rows))}")
    for row in rows:
        print(
            f"  L{row['level']} {row['status']}: candidate={row.get('candidate')} "
            f"reward={row.get('reward_with_match', 0)}/{row.get('reward_events', 0)} "
            f"without_match={row.get('reward_without_match', 0)}"
        )
        inspections = row.get("inspections") or []
        for item in inspections[:5]:
            print(
                f"    frame={item['frame']} reward={item['reward']} matched={item['matched']} "
                f"detail={item['match_detail'].get('status')} score={item['match_detail'].get('score')}"
            )
    print(f"  goal_artifact={path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="test template candidates against recorded reward increments")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="2,3", help="comma-separated human levels for live replay")
    parser.add_argument("--window", type=int, default=3, help="frames before reward to inspect")
    args = parser.parse_args()

    print("Offline checks (goal prerequisite falsifier):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using goal falsification.")
    print("\nGoal falsifier can now test template-match necessity, not planner sufficiency.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, window=args.window)


if __name__ == "__main__":
    main()
