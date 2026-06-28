"""0020 — Strict scale-aware equality lens.

Run from the repo root:
    uv run python teach/lessons/0020-scale-aware-equality-lens.py
    uv run python teach/lessons/0020-scale-aware-equality-lens.py --live --levels 2

0019 can identify mutable remote workspace candidates, but it refuses to name a
target/template without trustworthy comparability evidence. This lesson adds the
first such lens: integer-scale structural congruence over binary object masks.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0020-scale-aware-equality-lens"


@dataclass(frozen=True)
class ScaleTransform:
    scale: int
    rotation_degrees: int
    reflected: bool


@dataclass(frozen=True)
class ScaleMatch:
    status: str  # MATCH | REJECT | AMBIGUOUS
    score: float
    transform: ScaleTransform | None
    evidence: dict[str, object]


def _load_0019():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0019-role-induction-from-causal-evidence.py"
    spec = _ilu.spec_from_file_location("sal20_ri19", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _as_mask(arr) -> np.ndarray:
    mask = np.asarray(arr) != 0
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    return _crop(mask)


def _crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].astype(bool)


def _components(mask: np.ndarray, value: bool) -> int:
    if mask.size == 0:
        return 0
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    count = 0
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or bool(mask[r, c]) != value:
                continue
            count += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen or bool(mask[nr, nc]) != value:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return count


def _hole_count(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    padded = np.pad(mask, 1, constant_values=True)
    h, w = padded.shape
    seen: set[tuple[int, int]] = set()
    holes = 0
    for r in range(h):
        for c in range(w):
            if padded[r, c] or (r, c) in seen:
                continue
            touches_border = False
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                touches_border = touches_border or rr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if padded[nr, nc] or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _transforms(mask: np.ndarray, *, allow_rotation: bool, allow_reflection: bool):
    rotations = (0, 1, 2, 3) if allow_rotation else (0,)
    reflections = (False, True) if allow_reflection else (False,)
    seen: set[bytes] = set()
    for reflected in reflections:
        base = np.fliplr(mask) if reflected else mask
        for k in rotations:
            transformed = np.rot90(base, k)
            key = transformed.tobytes() + str(transformed.shape).encode()
            if key in seen:
                continue
            seen.add(key)
            yield transformed, ScaleTransform(scale=1, rotation_degrees=(k * 90) % 360, reflected=reflected)


def _integer_scale_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
    if a.size == 0 or b.size == 0:
        return None
    if a.shape == b.shape:
        return a, b, 1
    # Always expand the smaller-area mask, but require uniform integer scale.
    small, large = (a, b) if a.sum() <= b.sum() else (b, a)
    sh, sw = small.shape
    lh, lw = large.shape
    if sh == 0 or sw == 0 or lh % sh != 0 or lw % sw != 0:
        return None
    kh, kw = lh // sh, lw // sw
    if kh != kw or kh < 1:
        return None
    expanded = np.kron(small, np.ones((kh, kh), dtype=bool))
    return expanded, large, kh


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0 or b.size == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def scale_congruence(
    a,
    b,
    *,
    allow_rotation: bool = True,
    allow_reflection: bool = False,
    match_threshold: float = 0.98,
    ambiguous_threshold: float = 0.72,
) -> ScaleMatch:
    """Compare two binary masks under strict integer-scale congruence.

    MATCH requires component/hole agreement, uniform integer scale, and high IoU.
    Near evidence without a proof is AMBIGUOUS; weak evidence is REJECT.
    """

    ma = _as_mask(a)
    mb = _as_mask(b)
    if ma.size == 0 or mb.size == 0:
        return ScaleMatch("REJECT", 0.0, None, {"non_empty": False})

    topology_a = {"components": _components(ma, True), "holes": _hole_count(ma)}
    best: tuple[float, ScaleTransform | None, dict[str, object]] = (0.0, None, {})

    for tb, transform in _transforms(mb, allow_rotation=allow_rotation, allow_reflection=allow_reflection):
        topology_b = {"components": _components(tb, True), "holes": _hole_count(tb)}
        same_topology = topology_a == topology_b
        pair = _integer_scale_pair(ma, tb)
        integer_scale = pair is not None
        score = 0.0
        scale = None
        if pair is not None:
            expanded, large, scale = pair
            score = _iou(expanded, large)
        else:
            # Coarse fallback only for deciding REJECT vs AMBIGUOUS. It cannot match.
            ah, aw = ma.shape
            bh, bw = tb.shape
            aspect_a = ah / aw if aw else 0.0
            aspect_b = bh / bw if bw else 0.0
            score = max(0.0, 1.0 - min(1.0, abs(aspect_a - aspect_b)))
        evidence = {
            "shape_a": ma.shape,
            "shape_b": tb.shape,
            "component_count_equal": topology_a["components"] == topology_b["components"],
            "hole_count_equal": topology_a["holes"] == topology_b["holes"],
            "same_topology": same_topology,
            "integer_uniform_scale": integer_scale,
            "scale": scale,
            "occupancy_iou": round(score, 4) if integer_scale else None,
            "rotation_degrees": transform.rotation_degrees,
            "reflected": transform.reflected,
        }
        if score > best[0]:
            best = (score, transform, evidence)

    score, transform, evidence = best
    if evidence.get("integer_uniform_scale") and evidence.get("same_topology") and score >= match_threshold:
        assert transform is not None
        transform = ScaleTransform(
            scale=int(evidence["scale"]),
            rotation_degrees=transform.rotation_degrees,
            reflected=transform.reflected,
        )
        return ScaleMatch("MATCH", score, transform, evidence)
    if evidence.get("same_topology") and score >= ambiguous_threshold:
        return ScaleMatch("AMBIGUOUS", score, None, evidence)
    return ScaleMatch("REJECT", score, None, evidence)


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    l_small = np.array([[1, 0], [1, 1]], dtype=int)
    l_big = np.kron(l_small, np.ones((3, 3), dtype=int))
    m1 = scale_congruence(l_small, l_big)
    checks.append(_check(
        "1 exact integer scale proves congruence",
        m1.status == "MATCH" and m1.transform is not None and m1.transform.scale == 3,
        f"status={m1.status}, transform={m1.transform}, evidence={m1.evidence}",
    ))

    rotated = np.rot90(l_big)
    m2 = scale_congruence(l_small, rotated, allow_rotation=True)
    checks.append(_check(
        "2 rotation can be part of the proof",
        m2.status == "MATCH" and m2.transform is not None and m2.transform.rotation_degrees in {90, 270},
        f"status={m2.status}, transform={m2.transform}",
    ))

    false_positive = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
    ], dtype=int)
    m3 = scale_congruence(l_small, false_positive)
    checks.append(_check(
        "3 rejects near-looking non-congruent shape",
        m3.status == "REJECT",
        f"status={m3.status}, score={m3.score:.2f}, evidence={m3.evidence}",
    ))

    non_integer = np.array([[1, 0, 0], [1, 1, 1]], dtype=int)
    m4 = scale_congruence(l_small, non_integer)
    checks.append(_check(
        "4 non-integer scale refuses to prove match",
        m4.status in {"AMBIGUOUS", "REJECT"} and m4.status != "MATCH",
        f"status={m4.status}, evidence={m4.evidence}",
    ))

    donut = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    filled = np.ones((3, 3), dtype=int)
    m5 = scale_congruence(donut, filled)
    checks.append(_check(
        "5 topology mismatch rejects filled-vs-hole",
        m5.status == "REJECT" and not m5.evidence.get("hole_count_equal", True),
        f"status={m5.status}, evidence={m5.evidence}",
    ))
    return all(checks)


def _parse_region(object_id: str) -> tuple[int, int] | None:
    if not object_id.startswith("remote_region@"):
        return None
    try:
        r, c = object_id.removeprefix("remote_region@").split(",")
        return int(r), int(c)
    except ValueError:
        return None


def _view_near(views: list, row: int, col: int):
    if not views:
        return None
    return min(views, key=lambda v: abs(v.centroid[0] - row) + abs(v.centroid[1] - col))


def _mask_from_view(view) -> np.ndarray:
    r0, c0, r1, c1 = view.bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in view.pixels:
        mask[r - r0, c - c0] = True
    return mask


def _view_label(view, oc16, coarse_key: str) -> str:
    r, c = view.centroid
    return (
        f"{coarse_key}|"
        f"shape={oc16.ig.shape_literal(view)}|area={view.area}|centroid={round(r)},{round(c)}"
    )


def _live_views_for_levels(levels: set[int], trace: Path | None):
    ri19 = _load_0019()
    oc16 = ri19.rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    frames = []
    for seg in oc16.segments_for_levels(segments, levels):
        idx = max(seg.start, seg.end - 1)
        _bg, parts = oc16.il.views_from_grid(records[idx].frame, frame=idx)
        frames.append((oc16.human_level(records[idx].levels_completed), idx, parts, oc16))
    return frames


def run_live(trace: Path | None, levels: set[int]) -> None:
    ri19 = _load_0019()
    events, _meta, _counts = ri19.rcf18._live_step_events(trace, levels)
    ledger = ri19.rcf18.falsify_remote_hypotheses(events, max_depth=2, min_support=2)
    roles = ri19.induce_roles(events, ledger)
    workspace_ids = [a.object_id for a in roles.workspace_candidates]
    changed = set(_ for _ in ri19._changed_object_counts(events).keys())
    frames = _live_views_for_levels(levels, trace)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-scale-candidates.tsv"
    rows = []
    with path.open("w") as fh:
        fh.write("level\tframe\tworkspace\tcandidate\tstatus\tscore\ttransform\tevidence\n")
        for level, frame_idx, parts, oc16 in frames:
            for workspace_id in workspace_ids:
                loc = _parse_region(workspace_id)
                if loc is None:
                    continue
                workspace_view = _view_near(parts, *loc)
                if workspace_view is None:
                    continue
                wm = _mask_from_view(workspace_view)
                for candidate in parts:
                    cand_region = ri19.rcf18.rca17._object_key(candidate, oc16)
                    if cand_region == workspace_id or cand_region in changed:
                        continue
                    cand_id = _view_label(candidate, oc16, cand_region)
                    cm = _mask_from_view(candidate)
                    match = scale_congruence(wm, cm)
                    if match.status == "REJECT" and match.score < 0.72:
                        continue
                    transform = None if match.transform is None else match.transform.__dict__
                    row = (level, frame_idx, workspace_id, cand_id, match.status, match.score, transform, match.evidence)
                    rows.append(row)
                    fh.write(
                        f"{level}\t{frame_idx}\t{workspace_id}\t{cand_id}\t{match.status}\t{match.score:.3f}\t"
                        f"{json.dumps(transform, sort_keys=True)}\t{json.dumps(match.evidence, sort_keys=True)}\n"
                    )

    status_counts = Counter(row[4] for row in rows)
    print("\nLive ls20 scale-aware equality inspection:")
    print(f"  workspace_candidates={workspace_ids}")
    print(f"  candidate_pairs={len(rows)} statuses={dict(status_counts)}")
    for row in rows[:12]:
        level, frame_idx, workspace_id, cand_id, status, score, transform, evidence = row
        print(
            f"  L{level} frame={frame_idx} {workspace_id} vs {cand_id}: "
            f"{status} score={score:.2f} transform={transform} evidence={evidence}"
        )
    if not rows:
        print("  no non-rejected stable candidates found by strict integer-scale lens")
    print(f"  scale_artifact={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="inspect scale matches on recorded ls20 trace")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="2", help="comma-separated human levels for live replay")
    args = parser.parse_args()

    print("Offline checks (strict scale-aware equality lens):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using scale-aware equality.")
    print("\nStrict scale-aware equality can prove integer-scale congruence and refuse weak matches.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels)


if __name__ == "__main__":
    main()
