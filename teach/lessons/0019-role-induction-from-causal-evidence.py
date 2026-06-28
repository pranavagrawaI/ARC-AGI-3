"""0019 — Role induction from causal evidence.

Run from the repo root:
    uv run python teach/lessons/0019-role-induction-from-causal-evidence.py
    uv run python teach/lessons/0019-role-induction-from-causal-evidence.py --live --levels 2

0018 tells us which remote changes are controlled, autonomous, or refuted. This
lesson turns those mechanism facts into provisional object roles without naming
ls20 objects: workspace candidates, autonomous/background candidates, refuted
noise, and target/template candidates when stable comparability evidence exists.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def _load_0018():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0018-remote-causal-falsifier.py"
    spec = _ilu.spec_from_file_location("ri19_rcf18", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rcf18 = _load_0018()
rca17 = rcf18.rca17
RemoteEvent = rcf18.RemoteEvent
LedgerEntry = rcf18.LedgerEntry

ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0019-role-induction"


@dataclass(frozen=True)
class StableObjectEvidence:
    object_id: str
    observations: int
    changed_steps: int = 0


@dataclass(frozen=True)
class ComparablePair:
    mutable_object: str
    stable_object: str
    score: float
    status: str  # MATCH_CANDIDATE | AMBIGUOUS | REJECTED


@dataclass
class ObjectRoleEvidence:
    object_id: str
    changed_steps: int = 0
    supported_rules: list[LedgerEntry] = field(default_factory=list)
    autonomous_rules: list[LedgerEntry] = field(default_factory=list)
    refuted_rules: list[LedgerEntry] = field(default_factory=list)
    contested_rules: list[LedgerEntry] = field(default_factory=list)
    stable_observations: int = 0
    comparable_templates: list[ComparablePair] = field(default_factory=list)

    @property
    def supported_support(self) -> int:
        return sum(entry.support for entry in self.supported_rules)

    @property
    def autonomous_spontaneous(self) -> int:
        return sum(entry.spontaneous_effects for entry in self.autonomous_rules)


@dataclass(frozen=True)
class RoleAssignment:
    object_id: str
    role: str  # WORKSPACE_CANDIDATE | AUTONOMOUS_BACKGROUND | TARGET_TEMPLATE_CANDIDATE | REFUTED_NOISE | UNDERFIT_REMOTE_OBJECT
    confidence: float
    evidence: dict[str, object]


@dataclass(frozen=True)
class RoleInductionResult:
    assignments: tuple[RoleAssignment, ...]
    workspace_candidates: tuple[RoleAssignment, ...]
    target_template_candidates: tuple[RoleAssignment, ...]
    autonomous_candidates: tuple[RoleAssignment, ...]
    refuted_noise: tuple[RoleAssignment, ...]
    underfit: tuple[RoleAssignment, ...]


def _effect_object(effect: tuple[str, str, str]) -> str:
    return effect[0]


def _changed_object_counts(events: list[RemoteEvent]) -> Counter:
    counts: Counter = Counter()
    for event in rca17._trusted(events):
        seen_this_step = set()
        for diff in event.global_diffs:
            if diff.object_id is None or diff.status in rca17.UNTRUSTED_STATUSES:
                continue
            seen_this_step.add(diff.object_id)
        counts.update(seen_this_step)
    return counts


def _collect_evidence(
    events: list[RemoteEvent],
    ledger: tuple[LedgerEntry, ...],
    stable_objects: tuple[StableObjectEvidence, ...],
    comparable_pairs: tuple[ComparablePair, ...],
) -> dict[str, ObjectRoleEvidence]:
    by_object: dict[str, ObjectRoleEvidence] = {}

    def get(object_id: str) -> ObjectRoleEvidence:
        if object_id not in by_object:
            by_object[object_id] = ObjectRoleEvidence(object_id)
        return by_object[object_id]

    for object_id, count in _changed_object_counts(events).items():
        get(object_id).changed_steps = count

    for entry in ledger:
        ev = get(_effect_object(entry.hypothesis.effect))
        if entry.status == "SUPPORTED":
            ev.supported_rules.append(entry)
        elif entry.status == "AUTONOMOUS_CANDIDATE":
            ev.autonomous_rules.append(entry)
        elif entry.status == "REFUTED":
            ev.refuted_rules.append(entry)
        else:
            ev.contested_rules.append(entry)

    for stable in stable_objects:
        ev = get(stable.object_id)
        ev.stable_observations = stable.observations
        ev.changed_steps = max(ev.changed_steps, stable.changed_steps)

    for pair in comparable_pairs:
        get(pair.mutable_object).comparable_templates.append(pair)
        get(pair.stable_object).comparable_templates.append(pair)

    return by_object


def _assign_role(ev: ObjectRoleEvidence) -> RoleAssignment:
    evidence = {
        "changed_steps": ev.changed_steps,
        "supported_rules": len(ev.supported_rules),
        "supported_support": ev.supported_support,
        "autonomous_rules": len(ev.autonomous_rules),
        "autonomous_spontaneous": ev.autonomous_spontaneous,
        "refuted_rules": len(ev.refuted_rules),
        "contested_rules": len(ev.contested_rules),
        "stable_observations": ev.stable_observations,
        "comparable_templates": [pair.stable_object for pair in ev.comparable_templates if pair.mutable_object == ev.object_id],
    }

    usable_pairs = [
        pair for pair in ev.comparable_templates
        if pair.stable_object == ev.object_id and pair.status == "MATCH_CANDIDATE"
    ]
    if ev.stable_observations >= 2 and ev.changed_steps == 0 and usable_pairs:
        return RoleAssignment(
            ev.object_id,
            "TARGET_TEMPLATE_CANDIDATE",
            max(pair.score for pair in usable_pairs),
            evidence,
        )

    if ev.supported_rules and ev.supported_support > ev.autonomous_spontaneous:
        confidence = ev.supported_support / max(1, ev.supported_support + ev.autonomous_spontaneous + len(ev.refuted_rules))
        return RoleAssignment(ev.object_id, "WORKSPACE_CANDIDATE", confidence, evidence)

    if ev.autonomous_rules:
        confidence = ev.autonomous_spontaneous / max(1, ev.autonomous_spontaneous + ev.supported_support)
        return RoleAssignment(ev.object_id, "AUTONOMOUS_BACKGROUND", confidence, evidence)

    if ev.refuted_rules and not ev.supported_rules:
        return RoleAssignment(ev.object_id, "REFUTED_NOISE", 1.0, evidence)

    return RoleAssignment(ev.object_id, "UNDERFIT_REMOTE_OBJECT", 0.0, evidence)


def induce_roles(
    events: list[RemoteEvent],
    ledger: tuple[LedgerEntry, ...],
    *,
    stable_objects: tuple[StableObjectEvidence, ...] = (),
    comparable_pairs: tuple[ComparablePair, ...] = (),
) -> RoleInductionResult:
    evidence = _collect_evidence(events, ledger, stable_objects, comparable_pairs)
    assignments = tuple(sorted(
        (_assign_role(ev) for ev in evidence.values()),
        key=lambda a: (a.role, -a.confidence, a.object_id),
    ))
    return RoleInductionResult(
        assignments=assignments,
        workspace_candidates=tuple(a for a in assignments if a.role == "WORKSPACE_CANDIDATE"),
        target_template_candidates=tuple(a for a in assignments if a.role == "TARGET_TEMPLATE_CANDIDATE"),
        autonomous_candidates=tuple(a for a in assignments if a.role == "AUTONOMOUS_BACKGROUND"),
        refuted_noise=tuple(a for a in assignments if a.role == "REFUTED_NOISE"),
        underfit=tuple(a for a in assignments if a.role == "UNDERFIT_REMOTE_OBJECT"),
    )


def _event(trigger: str, *diffs) -> RemoteEvent:
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
    events = [
        _event("X", rcf18.Diff("workspace", "orientation", "any", "rot+1"), rcf18.Diff("cycler", "orientation", "any", "rot+1")),
        _event("X", rcf18.Diff("workspace", "orientation", "any", "rot+1"), rcf18.Diff("cycler", "orientation", "any", "rot+1")),
        _event("Y", rcf18.Diff("cycler", "orientation", "any", "rot+1")),
        _event("Z", rcf18.Diff("cycler", "orientation", "any", "rot+1")),
        _event("Q"),
    ]
    ledger = (
        rcf18.score_hypothesis(events, rcf18.RemoteHypothesis(("trigger_shape=X",), ("workspace", "orientation", "any->rot+1"))),
        rcf18.score_hypothesis(events, rcf18.RemoteHypothesis(("trigger_shape=X",), ("cycler", "orientation", "any->rot+1"))),
        rcf18.score_hypothesis(events, rcf18.RemoteHypothesis(("trigger_shape=Q",), ("noise", "pixel_region", "a->b"))),
    )
    result = induce_roles(
        events,
        ledger,
        stable_objects=(StableObjectEvidence("template", observations=5),),
        comparable_pairs=(ComparablePair("workspace", "template", score=0.87, status="MATCH_CANDIDATE"),),
    )
    roles = {a.object_id: a.role for a in result.assignments}
    checks.append(_check(
        "1 supported controlled change becomes workspace candidate",
        roles.get("workspace") == "WORKSPACE_CANDIDATE",
        f"roles={roles}",
    ))
    checks.append(_check(
        "2 spontaneous remote change becomes autonomous background",
        roles.get("cycler") == "AUTONOMOUS_BACKGROUND",
        f"roles={roles}",
    ))
    checks.append(_check(
        "3 stable comparable object becomes target template candidate",
        roles.get("template") == "TARGET_TEMPLATE_CANDIDATE",
        f"roles={roles}",
    ))
    checks.append(_check(
        "4 no template guessed without comparability",
        not induce_roles(events, ledger).target_template_candidates,
        "target candidates require explicit stable comparability evidence",
    ))
    return all(checks)


def _write_role_artifact(result: RoleInductionResult, levels: set[int]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-roles.tsv"
    with path.open("w") as fh:
        fh.write("object_id\trole\tconfidence\tevidence\n")
        for assignment in result.assignments:
            fh.write(
                f"{assignment.object_id}\t{assignment.role}\t{assignment.confidence:.3f}\t"
                f"{json.dumps(assignment.evidence, sort_keys=True)}\n"
            )
    return path


def run_live(trace: Path | None, levels: set[int]) -> RoleInductionResult:
    events, _meta, counts = rcf18._live_step_events(trace, levels)
    ledger = rcf18.falsify_remote_hypotheses(events, max_depth=2, min_support=2)
    result = induce_roles(events, ledger)
    artifact = _write_role_artifact(result, levels)
    role_counts = Counter(a.role for a in result.assignments)

    print("\nLive ls20 role induction from causal evidence:")
    print(
        f"  events={counts['events']} contact_events={counts['events_with_contact']} "
        f"remote_diff_events={counts['events_with_remote_diff']} orphan_diffs={counts['orphan_diffs']}"
    )
    print(f"  roles={dict(role_counts)}")
    for assignment in sorted(result.assignments, key=lambda a: (a.role != "WORKSPACE_CANDIDATE", -a.confidence, a.object_id))[:14]:
        print(
            f"  {assignment.role:<24} conf={assignment.confidence:.2f} {assignment.object_id} "
            f"evidence={assignment.evidence}"
        )
    if not result.target_template_candidates:
        print("  target_template_candidates=0 (no stable scale/comparability evidence supplied yet)")
    print(f"  role_artifact={artifact}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="replay recorded ls20 trace and write role TSV")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="2", help="comma-separated human levels for live replay")
    args = parser.parse_args()

    print("Offline checks (synthetic role induction traces):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using role induction.")
    print("\nRole induction now consumes controlled/autonomous/refuted causal evidence.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels)


if __name__ == "__main__":
    main()
