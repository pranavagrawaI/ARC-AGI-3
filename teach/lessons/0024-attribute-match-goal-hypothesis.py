"""0024 — Attribute match goal hypothesis.

Run from the repo root:
    uv run python teach/lessons/0024-attribute-match-goal-hypothesis.py
    uv run python teach/lessons/0024-attribute-match-goal-hypothesis.py --live --levels 1,2,3,4,5,6,7

The goal hypothesis is stricter than shape congruence:

    shape + color + orientation match, then terminal interaction can pass.

This lesson tests the attribute-match half against reward-adjacent frames.
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

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0024-attribute-match-goal-hypothesis"


def _load_0023():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0023-reward-adjacent-scan.py"
    spec = _ilu.spec_from_file_location("am24_ras23", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ras23 = _load_0023()
sal20 = ras23.sal20
ri19 = ras23.ri19


@dataclass(frozen=True)
class AttributeMatch:
    status: str  # ATTRIBUTE_MATCH | SHAPE_ONLY | REJECT | AMBIGUOUS
    shape_match: bool
    color_match: bool
    orientation_match: bool
    score: float
    evidence: dict[str, object]


def _dominant_colors(view) -> tuple[int, ...]:
    raw = tuple(view.color_lattice.get("raw", ()))
    return tuple(color for color, _count in raw)


def _orientation(view) -> object:
    return view.orientation_lattice.get("raw")


def attribute_match(a, b) -> AttributeMatch:
    ma = sal20._mask_from_view(a)
    mb = sal20._mask_from_view(b)
    shape = sal20.scale_congruence(ma, mb)
    shape_ok = shape.status == "MATCH"
    color_a = _dominant_colors(a)
    color_b = _dominant_colors(b)
    color_ok = color_a == color_b
    orient_a = _orientation(a)
    orient_b = _orientation(b)
    orient_ok = orient_a == orient_b
    evidence = {
        "shape_status": shape.status,
        "shape_score": round(shape.score, 4),
        "shape_transform": None if shape.transform is None else shape.transform.__dict__,
        "shape_evidence": shape.evidence,
        "color_a": color_a,
        "color_b": color_b,
        "orientation_a": orient_a,
        "orientation_b": orient_b,
    }
    if not shape_ok:
        return AttributeMatch("REJECT", False, color_ok, orient_ok, 0.0, evidence)
    score = (1.0 + float(color_ok) + float(orient_ok)) / 3.0
    if color_ok and orient_ok:
        return AttributeMatch("ATTRIBUTE_MATCH", True, True, True, score, evidence)
    return AttributeMatch("SHAPE_ONLY", True, color_ok, orient_ok, score, evidence)


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


class _FakeView:
    def __init__(self, mask, color: int, orientation: int):
        self.mask = np.asarray(mask, dtype=bool)
        self.bbox = (0, 0, self.mask.shape[0] - 1, self.mask.shape[1] - 1)
        self.pixels = frozenset((int(r), int(c)) for r, c in zip(*np.where(self.mask)))
        self.color_lattice = {"raw": ((color, int(self.mask.sum())),)}
        self.orientation_lattice = {"raw": orientation}


def _run_checks() -> bool:
    checks = []
    small = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int)
    large = np.kron(small, np.ones((2, 2), dtype=int))
    good = attribute_match(_FakeView(small, 9, 2), _FakeView(large, 9, 2))
    checks.append(_check(
        "1 shape+color+orientation accepted",
        good.status == "ATTRIBUTE_MATCH",
        f"status={good.status}, evidence={good.evidence}",
    ))
    color_bad = attribute_match(_FakeView(small, 9, 2), _FakeView(large, 10, 2))
    checks.append(_check(
        "2 shape-only color mismatch rejected as goal match",
        color_bad.status == "SHAPE_ONLY" and not color_bad.color_match,
        f"status={color_bad.status}, color={color_bad.color_match}",
    ))
    orient_bad = attribute_match(_FakeView(small, 9, 2), _FakeView(large, 9, 3))
    checks.append(_check(
        "3 shape-only orientation mismatch rejected as goal match",
        orient_bad.status == "SHAPE_ONLY" and not orient_bad.orientation_match,
        f"status={orient_bad.status}, orientation={orient_bad.orientation_match}",
    ))
    return all(checks)


def _view_label(view, oc16) -> str:
    coarse = ri19.rcf18.rca17._object_key(view, oc16)
    return sal20._view_label(view, oc16, coarse)


def _reward_indices(records, seg) -> list[int]:
    return ras23._reward_indices(records, seg)


def scan_reward_attribute_matches(trace: Path | None, levels: set[int], window: int) -> list[dict[str, object]]:
    oc16 = ri19.rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    rows: list[dict[str, object]] = []

    for seg in oc16.segments_for_levels(segments, levels):
        level = oc16.human_level(seg.level)
        for inc in _reward_indices(records, seg):
            start = max(seg.start, inc - window)
            for idx in range(start, inc):
                _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                for i, a in enumerate(parts):
                    ma = sal20._mask_from_view(a)
                    if not ras23._valid_scan_object(a, ma):
                        continue
                    ca = ras23._complexity(ma)
                    if not ca["nontrivial"]:
                        continue
                    for b in parts[i + 1:]:
                        mb = sal20._mask_from_view(b)
                        if not ras23._valid_scan_object(b, mb):
                            continue
                        cb = ras23._complexity(mb)
                        if not cb["nontrivial"]:
                            continue
                        result = attribute_match(a, b)
                        if result.status == "REJECT":
                            continue
                        rows.append({
                            "level": level,
                            "frame": idx,
                            "reward_frame": inc,
                            "left": _view_label(a, oc16),
                            "right": _view_label(b, oc16),
                            "status": result.status,
                            "score": result.score,
                            "shape_match": result.shape_match,
                            "color_match": result.color_match,
                            "orientation_match": result.orientation_match,
                            "evidence": result.evidence,
                            "left_complexity": ca,
                            "right_complexity": cb,
                        })
    return rows


def run_live(trace: Path | None, levels: set[int], window: int) -> list[dict[str, object]]:
    rows = scan_reward_attribute_matches(trace, levels, window)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-attribute-matches.jsonl"
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    by_level = defaultdict(list)
    for row in rows:
        by_level[row["level"]].append(row)
    print("\nLive ls20 reward-adjacent attribute-match scan:")
    print(f"  levels={sorted(levels)} rows={len(rows)} statuses={dict(Counter(r['status'] for r in rows))}")
    for level in sorted(levels):
        ranked = sorted(
            by_level[level],
            key=lambda r: (r["status"] != "ATTRIBUTE_MATCH", -r["score"], r["frame"], r["left"]),
        )
        print(f"  L{level}: rows={len(ranked)} statuses={dict(Counter(r['status'] for r in ranked))}")
        for row in ranked[:8]:
            ev = row["evidence"]
            print(
                f"    frame={row['frame']} {row['status']:<15} score={row['score']:.2f} "
                f"shape={row['shape_match']} color={row['color_match']} orient={row['orientation_match']} "
                f"colors={ev['color_a']}->{ev['color_b']} orient={ev['orientation_a']}->{ev['orientation_b']} "
                f"{row['left']} <-> {row['right']}"
            )
    print(f"  attribute_artifact={path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="scan reward-adjacent shape+color+orientation matches")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=4, help="frames before reward to scan")
    args = parser.parse_args()

    print("Offline checks (shape + color + orientation hypothesis):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using attribute-match hypothesis.")
    print("\nAttribute-match hypothesis now separates shape-only from full goal match.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window)


if __name__ == "__main__":
    main()
