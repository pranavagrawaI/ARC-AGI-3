"""0021 — Template candidate filter.

Run from the repo root:
    uv run python teach/lessons/0021-template-candidate-filter.py
    uv run python teach/lessons/0021-template-candidate-filter.py --live --levels 2

0020 proves shape congruence, but many congruent shapes are semantically cheap
(1x1 blocks, repeated HUD squares, bars). This lesson filters scale matches into
template candidates using role, stability, complexity, and temporal evidence.
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

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0021-template-candidate-filter"


def _load_0020():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0020-scale-aware-equality-lens.py"
    spec = _ilu.spec_from_file_location("tcf21_sal20", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sal20 = _load_0020()
ri19 = sal20._load_0019()


@dataclass(frozen=True)
class TemplateCandidate:
    workspace_id: str
    template_id: str
    status: str  # TEMPLATE_CANDIDATE | TOO_TRIVIAL | UNSTABLE | STATIC_FALSE_POSITIVE | AMBIGUOUS | REJECTED
    score: float
    evidence: dict[str, object]


def shape_complexity(mask: np.ndarray) -> dict[str, object]:
    mask = sal20._as_mask(mask)
    area = int(mask.sum())
    h, w = mask.shape if mask.size else (0, 0)
    filled_rectangle = bool(area == h * w) if area else False
    single_line = bool(h == 1 or w == 1) if area else False
    components = sal20._components(mask, True)
    holes = sal20._hole_count(mask)
    # Conservative floor: simple bars/points are common distractors unless they
    # carry topology or enough spatial extent.
    nontrivial = area >= 5 and not filled_rectangle and not single_line
    if holes > 0 or components > 1:
        nontrivial = area >= 5
    return {
        "area": area,
        "shape": (h, w),
        "filled_rectangle": filled_rectangle,
        "single_line": single_line,
        "components": components,
        "holes": holes,
        "nontrivial": nontrivial,
    }


def filter_template_candidate(
    *,
    workspace_id: str,
    template_id: str,
    workspace_mask,
    template_mask,
    match: sal20.ScaleMatch,
    workspace_supported: bool,
    template_changed_steps: int,
    was_matched_before_control: bool = False,
) -> TemplateCandidate:
    wc = shape_complexity(workspace_mask)
    tc = shape_complexity(template_mask)
    evidence = {
        "match_status": match.status,
        "match_score": round(match.score, 4),
        "match_evidence": match.evidence,
        "workspace_supported": workspace_supported,
        "template_changed_steps": template_changed_steps,
        "workspace_complexity": wc,
        "template_complexity": tc,
        "was_matched_before_control": was_matched_before_control,
    }

    if not workspace_supported:
        return TemplateCandidate(workspace_id, template_id, "REJECTED", 0.0, evidence)
    if match.status == "REJECT":
        return TemplateCandidate(workspace_id, template_id, "REJECTED", 0.0, evidence)
    if template_changed_steps > 0:
        return TemplateCandidate(workspace_id, template_id, "UNSTABLE", 0.0, evidence)
    if was_matched_before_control:
        return TemplateCandidate(workspace_id, template_id, "STATIC_FALSE_POSITIVE", 0.0, evidence)
    if not wc["nontrivial"] or not tc["nontrivial"]:
        return TemplateCandidate(workspace_id, template_id, "TOO_TRIVIAL", match.score * 0.25, evidence)
    if match.status == "MATCH":
        return TemplateCandidate(workspace_id, template_id, "TEMPLATE_CANDIDATE", match.score, evidence)
    return TemplateCandidate(workspace_id, template_id, "AMBIGUOUS", match.score * 0.5, evidence)


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    workspace = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int)
    template = np.kron(workspace, np.ones((2, 2), dtype=int))
    match = sal20.scale_congruence(workspace, template)
    c1 = filter_template_candidate(
        workspace_id="workspace",
        template_id="template",
        workspace_mask=workspace,
        template_mask=template,
        match=match,
        workspace_supported=True,
        template_changed_steps=0,
    )
    checks.append(_check(
        "1 accepts non-trivial stable scale match",
        c1.status == "TEMPLATE_CANDIDATE",
        f"status={c1.status}, evidence={c1.evidence}",
    ))

    dot = np.ones((1, 1), dtype=int)
    square = np.ones((2, 2), dtype=int)
    c2 = filter_template_candidate(
        workspace_id="dot_ws",
        template_id="dot_template",
        workspace_mask=dot,
        template_mask=square,
        match=sal20.scale_congruence(dot, square),
        workspace_supported=True,
        template_changed_steps=0,
    )
    checks.append(_check(
        "2 rejects trivial dot/square match",
        c2.status == "TOO_TRIVIAL",
        f"status={c2.status}",
    ))

    c3 = filter_template_candidate(
        workspace_id="workspace",
        template_id="moving_template",
        workspace_mask=workspace,
        template_mask=template,
        match=match,
        workspace_supported=True,
        template_changed_steps=2,
    )
    checks.append(_check(
        "3 rejects unstable template",
        c3.status == "UNSTABLE",
        f"status={c3.status}",
    ))

    c4 = filter_template_candidate(
        workspace_id="workspace",
        template_id="spawn_match",
        workspace_mask=workspace,
        template_mask=template,
        match=match,
        workspace_supported=True,
        template_changed_steps=0,
        was_matched_before_control=True,
    )
    checks.append(_check(
        "4 rejects static spawn false positive",
        c4.status == "STATIC_FALSE_POSITIVE",
        f"status={c4.status}",
    ))

    c5 = filter_template_candidate(
        workspace_id="underfit_ws",
        template_id="template",
        workspace_mask=workspace,
        template_mask=template,
        match=match,
        workspace_supported=False,
        template_changed_steps=0,
    )
    checks.append(_check(
        "5 rejects unsupported workspace",
        c5.status == "REJECTED",
        f"status={c5.status}",
    ))
    return all(checks)


def _parse_region(object_id: str) -> tuple[int, int] | None:
    return sal20._parse_region(object_id)


def _view_label(view, oc16, coarse_key: str) -> str:
    return sal20._view_label(view, oc16, coarse_key)


def _changed_counts_by_object(events) -> Counter:
    return ri19._changed_object_counts(events)


def _first_supported_step(events, workspace_id: str) -> int | None:
    for i, event in enumerate(events):
        for diff in event.global_diffs:
            if diff.object_id == workspace_id and "contact" in event.local_facts:
                return i
    return None


def _level_live_candidates(trace: Path | None, level: int) -> list[TemplateCandidate]:
    events, _meta, _counts = ri19.rcf18._live_step_events(trace, {level})
    ledger = ri19.rcf18.falsify_remote_hypotheses(events, max_depth=2, min_support=2)
    roles = ri19.induce_roles(events, ledger)
    workspace_ids = [a.object_id for a in roles.workspace_candidates]
    supported = set(workspace_ids)
    changed_counts = _changed_counts_by_object(events)
    frames = sal20._live_views_for_levels({level}, trace)
    candidates: list[TemplateCandidate] = []

    for _human_level, _frame_idx, parts, oc16 in frames:
        for workspace_id in workspace_ids:
            loc = _parse_region(workspace_id)
            if loc is None:
                continue
            workspace_view = sal20._view_near(parts, *loc)
            if workspace_view is None:
                continue
            wm = sal20._mask_from_view(workspace_view)
            first_control = _first_supported_step(events, workspace_id)
            for template_view in parts:
                coarse = ri19.rcf18.rca17._object_key(template_view, oc16)
                if coarse == workspace_id:
                    continue
                tm = sal20._mask_from_view(template_view)
                match = sal20.scale_congruence(wm, tm)
                if match.status == "REJECT" and match.score < 0.72:
                    continue
                candidate = filter_template_candidate(
                    workspace_id=workspace_id,
                    template_id=_view_label(template_view, oc16, coarse),
                    workspace_mask=wm,
                    template_mask=tm,
                    match=match,
                    workspace_supported=workspace_id in supported,
                    template_changed_steps=changed_counts.get(coarse, 0),
                    was_matched_before_control=first_control == 0,
                )
                if candidate.status != "REJECTED":
                    candidates.append(candidate)
    return candidates


def run_live(trace: Path | None, levels: set[int]) -> list[TemplateCandidate]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    all_candidates: list[TemplateCandidate] = []
    level_for: dict[int, list[TemplateCandidate]] = {}
    for level in sorted(levels):
        candidates = _level_live_candidates(trace, level)
        level_for[level] = candidates
        all_candidates.extend(candidates)

    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-template-candidates.tsv"
    with path.open("w") as fh:
        fh.write("level\tworkspace\ttemplate\tstatus\tscore\tevidence\n")
        for level, candidates in level_for.items():
            for c in candidates:
                fh.write(
                    f"{level}\t{c.workspace_id}\t{c.template_id}\t{c.status}\t{c.score:.3f}\t"
                    f"{json.dumps(c.evidence, sort_keys=True)}\n"
                )

    print("\nLive ls20 template-candidate filter:")
    print(f"  levels={sorted(levels)} candidates={len(all_candidates)} statuses={dict(Counter(c.status for c in all_candidates))}")
    for level in sorted(level_for):
        counts = Counter(c.status for c in level_for[level])
        print(f"  L{level}: {dict(counts)}")
        ranked = sorted(level_for[level], key=lambda c: (c.status != "TEMPLATE_CANDIDATE", -c.score, c.template_id))
        for c in ranked[:6]:
            print(
                f"    {c.status:<22} score={c.score:.2f} {c.workspace_id} -> {c.template_id} "
                f"complexity={c.evidence['template_complexity']}"
            )
    print(f"  template_artifact={path}")
    return all_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="inspect template candidates on recorded ls20 trace")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="2", help="comma-separated human levels for live replay")
    args = parser.parse_args()

    print("Offline checks (template candidate filtering):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using template candidate filtering.")
    print("\nTemplate filtering now separates semantic candidates from cheap congruences.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels)


if __name__ == "__main__":
    main()
