"""0023 — Reward-adjacent predicate scan.

Run from the repo root:
    uv run python teach/lessons/0023-reward-adjacent-scan.py
    uv run python teach/lessons/0023-reward-adjacent-scan.py --live --levels 1,2,3,4,5,6,7

0022 falsified the current template candidates. This lesson goes back to the
reward boundary and scans all object pairs in the pre-reward window to discover
which match predicates were actually true near completion.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0023-reward-adjacent-scan"


def _load_0022():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0022-goal-falsifier.py"
    spec = _ilu.spec_from_file_location("ras23_gf22", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gf22 = _load_0022()
sal20 = gf22.sal20
ri19 = gf22.ri19


@dataclass(frozen=True)
class RewardAdjacentMatch:
    level: int
    frame: int
    left: str
    right: str
    status: str
    score: float
    evidence: dict[str, object]
    left_complexity: dict[str, object]
    right_complexity: dict[str, object]


def _view_label(view, oc16) -> str:
    coarse = ri19.rcf18.rca17._object_key(view, oc16)
    return sal20._view_label(view, oc16, coarse)


def _complexity(mask: np.ndarray) -> dict[str, object]:
    return gf22.tcf21.shape_complexity(mask)


def _valid_scan_object(view, mask: np.ndarray) -> bool:
    c = _complexity(mask)
    # Exclude giant room/floor strips and trivial noise. Keep some simple objects
    # in artifacts only if they are part of a strict match, but do not rank them.
    if c["area"] <= 0 or c["area"] > 120:
        return False
    return True


def scan_frame_pairs(parts: list, oc16, *, level: int, frame: int) -> list[RewardAdjacentMatch]:
    rows: list[RewardAdjacentMatch] = []
    items = []
    for view in parts:
        mask = sal20._mask_from_view(view)
        if _valid_scan_object(view, mask):
            items.append((view, mask, _view_label(view, oc16), _complexity(mask)))

    for i, (a, ma, la, ca) in enumerate(items):
        for b, mb, lb, cb in items[i + 1:]:
            match = sal20.scale_congruence(ma, mb)
            if match.status == "REJECT" and match.score < 0.85:
                continue
            rows.append(RewardAdjacentMatch(
                level=level,
                frame=frame,
                left=la,
                right=lb,
                status=match.status,
                score=match.score,
                evidence=match.evidence,
                left_complexity=ca,
                right_complexity=cb,
            ))
    return rows


def rank_match(row: RewardAdjacentMatch) -> float:
    if row.status != "MATCH":
        return 0.0
    left_nt = bool(row.left_complexity.get("nontrivial"))
    right_nt = bool(row.right_complexity.get("nontrivial"))
    complexity_bonus = 1.0 if left_nt and right_nt else 0.25
    area_balance = min(row.left_complexity["area"], row.right_complexity["area"]) / max(
        1, max(row.left_complexity["area"], row.right_complexity["area"])
    )
    return row.score * complexity_bonus * (0.5 + 0.5 * area_balance)


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    shape = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int)
    big = np.kron(shape, np.ones((2, 2), dtype=int))
    match = sal20.scale_congruence(shape, big)
    ca, cb = _complexity(shape), _complexity(big)
    row = RewardAdjacentMatch(1, 10, "workspace", "template", match.status, match.score, match.evidence, ca, cb)
    checks.append(_check(
        "1 non-trivial reward-adjacent match ranks high",
        row.status == "MATCH" and rank_match(row) > 0.5,
        f"status={row.status}, rank={rank_match(row):.3f}",
    ))

    dot = np.ones((1, 1), dtype=int)
    square = np.ones((2, 2), dtype=int)
    trivial = sal20.scale_congruence(dot, square)
    tr = RewardAdjacentMatch(1, 10, "dot", "square", trivial.status, trivial.score, trivial.evidence, _complexity(dot), _complexity(square))
    checks.append(_check(
        "2 trivial match ranks low",
        tr.status == "MATCH" and rank_match(tr) < 0.2,
        f"status={tr.status}, rank={rank_match(tr):.3f}",
    ))

    mismatch = sal20.scale_congruence(shape, np.ones((4, 4), dtype=int))
    checks.append(_check(
        "3 topology mismatch is not a ranked match",
        mismatch.status == "REJECT",
        f"status={mismatch.status}, evidence={mismatch.evidence}",
    ))
    return all(checks)


def _reward_indices(records, seg) -> list[int]:
    return [
        i for i in range(seg.start + 1, min(seg.end + 2, len(records)))
        if records[i].levels_completed > records[i - 1].levels_completed
    ]


def run_live(trace: Path | None, levels: set[int], window: int) -> list[RewardAdjacentMatch]:
    oc16 = ri19.rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    rows: list[RewardAdjacentMatch] = []
    reward_rows: list[dict[str, object]] = []

    for seg in oc16.segments_for_levels(segments, levels):
        level = oc16.human_level(seg.level)
        incs = _reward_indices(records, seg)
        if not incs:
            reward_rows.append({"level": level, "status": "NO_REWARD_INCREMENT", "segment": [seg.start, seg.end]})
            continue
        for inc in incs:
            start = max(seg.start, inc - window)
            for idx in range(start, inc):
                _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                rows.extend(scan_frame_pairs(parts, oc16, level=level, frame=idx))
            reward_rows.append({"level": level, "reward_frame": inc, "pre_window": [start, inc - 1]})

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-reward-adjacent-matches.tsv"
    with path.open("w") as fh:
        fh.write("level\tframe\tstatus\tscore\trank\tleft\tright\tevidence\tleft_complexity\tright_complexity\n")
        for row in sorted(rows, key=lambda r: (r.level, r.frame, -rank_match(r), r.status)):
            fh.write(
                f"{row.level}\t{row.frame}\t{row.status}\t{row.score:.3f}\t{rank_match(row):.3f}\t"
                f"{row.left}\t{row.right}\t{json.dumps(row.evidence, sort_keys=True)}\t"
                f"{json.dumps(row.left_complexity, sort_keys=True)}\t{json.dumps(row.right_complexity, sort_keys=True)}\n"
            )

    reward_path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-reward-windows.json"
    with reward_path.open("w") as fh:
        json.dump(reward_rows, fh, indent=2, sort_keys=True)

    print("\nLive ls20 reward-adjacent scan:")
    print(f"  levels={sorted(levels)} reward_windows={reward_rows}")
    print(f"  candidate_pairs={len(rows)} statuses={dict(Counter(r.status for r in rows))}")
    by_level = defaultdict(list)
    for row in rows:
        by_level[row.level].append(row)
    for level in sorted(levels):
        ranked = sorted(by_level[level], key=lambda r: (-rank_match(r), r.status, r.left, r.right))
        print(f"  L{level}: pairs={len(ranked)} status={dict(Counter(r.status for r in ranked))}")
        for row in ranked[:8]:
            print(
                f"    frame={row.frame} {row.status:<9} rank={rank_match(row):.3f} score={row.score:.2f} "
                f"{row.left} <-> {row.right}"
            )
    print(f"  match_artifact={path}")
    print(f"  reward_windows={reward_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="scan all object pairs near recorded reward increments")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels for live replay")
    parser.add_argument("--window", type=int, default=4, help="frames before reward to scan")
    args = parser.parse_args()

    print("Offline checks (reward-adjacent predicate scan):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using reward-adjacent scan.")
    print("\nReward-adjacent scan can now search beyond rejected template candidates.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window)


if __name__ == "__main__":
    main()
