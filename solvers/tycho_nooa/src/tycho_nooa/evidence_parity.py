"""Frozen-workspace parity between legacy wmlib and sterile evidence queries."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.evidence import WorkspaceEvidenceCapability
from tycho_nooa.sterile_execution import ExecutionManifest, SterileExecutor

_QUERY_DIGEST_CELL = r"""
import hashlib
import json

def collect(kind, frontier=None):
    records = []
    after = None
    pages = 0
    while True:
        arguments = {'kind': kind, 'limit': 32}
        if frontier is not None:
            arguments['frontier'] = frontier
        if after is not None:
            arguments['after'] = after
        page = evidence.query(**arguments)
        pages += 1
        if frontier is None:
            frontier = page['frontier']
        assert page['frontier'] == frontier
        records.extend(page['records'])
        if not page['truncated']:
            return records, frontier, pages
        after = page['next_after']

def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()

def analysis(frames, transitions):
    frame_profiles = []
    for frame in frames:
        counts = {}
        for row in frame['grid']:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1
        frame_profiles.append([
            frame['level'], frame['turn'], sorted(counts.items())
        ])
    transition_profiles = []
    for transition in transitions:
        changed = []
        for row, (before, after) in enumerate(zip(transition['prev'], transition['next'])):
            for col, (old, new) in enumerate(zip(before, after)):
                if old != new:
                    changed.append([row, col])
        bbox = None
        if changed:
            rows = [cell[0] for cell in changed]
            cols = [cell[1] for cell in changed]
            bbox = [min(rows), min(cols), max(rows), max(cols)]
        transition_profiles.append([
            transition['level'], transition['turn'], len(changed), bbox
        ])
    return {'frames': frame_profiles, 'transitions': transition_profiles}

frames, frontier, frame_pages = collect('frames')
transitions, transition_frontier, transition_pages = collect(
    'transitions', frontier
)
assert transition_frontier == frontier
return_result({
    'frontier': frontier,
    'frames': len(frames),
    'transitions': len(transitions),
    'frame_pages': frame_pages,
    'transition_pages': transition_pages,
    'frames_sha256': digest(frames),
    'transitions_sha256': digest(transitions),
    'analysis_sha256': digest(analysis(frames, transitions)),
})
"""


class EvidenceDigest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frontier: dict[str, int] | None
    frames: int = Field(ge=0)
    transitions: int = Field(ge=0)
    frame_pages: int = Field(ge=1)
    transition_pages: int = Field(ge=1)
    frames_sha256: str
    transitions_sha256: str
    analysis_sha256: str


class WorkspaceParityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    workspace: str
    status: Literal["passed", "failed"]
    checkpoint_sha256: str
    legacy_wmlib_sha256: str
    frontier: dict[str, int] | None
    frames: int = Field(ge=0)
    transitions: int = Field(ge=0)
    frame_pages: int = Field(ge=1)
    transition_pages: int = Field(ge=1)
    capability_calls: int = Field(ge=0)
    frames_sha256: str
    transitions_sha256: str
    analysis_sha256: str
    positive_control_detected: bool
    mismatches: tuple[str, ...]


class ParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_schema: Literal[1] = 1
    contract: str = (
        "legacy workspace wmlib with canonical refs synthesized from level/turn "
        "and sterile evidence.query produce identical ordered frame/transition "
        "values and derived color/diff profiles"
    )
    all_passed: bool
    checkpoints: tuple[WorkspaceParityResult, ...]


async def compare_workspace(
    label: str,
    workspace: Path,
    *,
    workspace_label: str | None = None,
) -> WorkspaceParityResult:
    """Compare one frozen workspace through both evidence interfaces."""
    workspace = workspace.resolve()
    legacy_frames, legacy_transitions, frontier, wmlib_hash = _legacy_evidence(
        workspace
    )
    expected = _digest(
        legacy_frames,
        legacy_transitions,
        frontier=frontier,
        frame_pages=_page_count(len(legacy_frames)),
        transition_pages=_page_count(len(legacy_transitions)),
    )
    capability = WorkspaceEvidenceCapability.from_root(workspace)
    manifest = ExecutionManifest(
        invocation_id=f"evidence-parity-{hashlib.sha256(label.encode()).hexdigest()[:16]}",
        role="actor",
        capabilities=("evidence.query",),
        wall_time_seconds=30,
    )
    async with SterileExecutor(
        manifest,
        return_type=EvidenceDigest,
        capability_handlers={"evidence.query": capability.query},
    ) as executor:
        execution = await executor.run_cell(_QUERY_DIGEST_CELL)
    actual = execution.return_value
    if actual is None:
        raise RuntimeError(
            f"sterile parity cell did not return: {execution.exception!r}"
        )
    mismatches = _mismatches(expected, actual)
    positive_control = _positive_control_detected(
        legacy_frames, legacy_transitions, expected
    )
    if not positive_control:
        mismatches.append("positive control did not change the frame digest")
    return WorkspaceParityResult(
        label=label,
        workspace=workspace_label or str(workspace),
        status="passed" if not mismatches else "failed",
        checkpoint_sha256=_checkpoint_sha256(workspace),
        legacy_wmlib_sha256=wmlib_hash,
        frontier=actual.frontier,
        frames=actual.frames,
        transitions=actual.transitions,
        frame_pages=actual.frame_pages,
        transition_pages=actual.transition_pages,
        capability_calls=len(execution.capability_records),
        frames_sha256=actual.frames_sha256,
        transitions_sha256=actual.transitions_sha256,
        analysis_sha256=actual.analysis_sha256,
        positive_control_detected=positive_control,
        mismatches=tuple(mismatches),
    )


def _legacy_evidence(
    workspace: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int] | None, str]:
    wmlib_path = workspace / "wmlib.py"
    module = _load_module(wmlib_path)
    root = str(workspace)
    frames = [
        {
            "type": "frame",
            "level": int(level),
            "turn": int(turn),
            "ref": f"L{int(level)}:T{int(turn)}",
            "grid": grid.tolist(),
        }
        for (level, turn), grid in sorted(module.frames(root).items())
    ]
    transitions = [
        {
            "type": "transition",
            "level": int(item["level"]),
            "turn": int(item["turn"]),
            "ref": f"L{int(item['level'])}:T{int(item['turn'])}",
            "prev": item["prev"].tolist(),
            "next": item["next"].tolist(),
            "action": item.get("action"),
            "row": item.get("row"),
            "col": item.get("col"),
        }
        for item in module.transitions(root)
    ]
    key = module.current_frame_key(root)
    frontier = None if key is None else {"level": int(key[0]), "turn": int(key[1])}
    return frames, transitions, frontier, _file_sha256(wmlib_path)


def _load_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"frozen workspace has no wmlib.py: {path}")
    name = f"_frozen_wmlib_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen wmlib.py: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(
    frames: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    *,
    frontier: dict[str, int] | None,
    frame_pages: int,
    transition_pages: int,
) -> EvidenceDigest:
    return EvidenceDigest(
        frontier=frontier,
        frames=len(frames),
        transitions=len(transitions),
        frame_pages=frame_pages,
        transition_pages=transition_pages,
        frames_sha256=_json_sha256(frames),
        transitions_sha256=_json_sha256(transitions),
        analysis_sha256=_json_sha256(_analysis(frames, transitions)),
    )


def _analysis(
    frames: list[dict[str, Any]], transitions: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "frames": [_frame_profile(frame) for frame in frames],
        "transitions": [
            _transition_profile(transition) for transition in transitions
        ],
    }


def _frame_profile(frame: dict[str, Any]) -> list[Any]:
    counts: dict[int, int] = {}
    for row in frame["grid"]:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return [frame["level"], frame["turn"], sorted(counts.items())]


def _transition_profile(transition: dict[str, Any]) -> list[Any]:
    changed = [
        (row, col)
        for row, (before, after) in enumerate(
            zip(transition["prev"], transition["next"], strict=True)
        )
        for col, (old, new) in enumerate(zip(before, after, strict=True))
        if old != new
    ]
    bbox = None
    if changed:
        rows, cols = zip(*changed, strict=True)
        bbox = [min(rows), min(cols), max(rows), max(cols)]
    return [transition["level"], transition["turn"], len(changed), bbox]


def _mismatches(expected: EvidenceDigest, actual: EvidenceDigest) -> list[str]:
    fields = (
        "frontier",
        "frames",
        "transitions",
        "frame_pages",
        "transition_pages",
        "frames_sha256",
        "transitions_sha256",
        "analysis_sha256",
    )
    return [
        f"{field}: legacy={getattr(expected, field)!r} sterile={getattr(actual, field)!r}"
        for field in fields
        if getattr(expected, field) != getattr(actual, field)
    ]


def _positive_control_detected(
    frames: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    expected: EvidenceDigest,
) -> bool:
    if not frames or not frames[0]["grid"] or not frames[0]["grid"][0]:
        return False
    mutated = json.loads(json.dumps(frames))
    original = int(mutated[0]["grid"][0][0])
    mutated[0]["grid"][0][0] = (original + 1) % 16
    control = _digest(
        mutated,
        transitions,
        frontier=expected.frontier,
        frame_pages=expected.frame_pages,
        transition_pages=expected.transition_pages,
    )
    return bool(_mismatches(expected, control))


def _checkpoint_sha256(workspace: Path) -> str:
    paths = [workspace / ".current_frame.json", workspace / "wmlib.py"]
    paths.extend(workspace.glob("level_*/turn_*.txt"))
    paths.extend(workspace.glob("level_*/turn_*.json"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(workspace).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _page_count(records: int) -> int:
    return max(1, (records + 31) // 32)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run_manifest(manifest_path: Path, repo_root: Path) -> ParityReport:
    manifest = json.loads(manifest_path.read_text())
    checkpoints = []
    for item in manifest["checkpoints"]:
        checkpoints.append(
            await compare_workspace(
                item["label"],
                repo_root / item["workspace"],
                workspace_label=item["workspace"],
            )
        )
    return ParityReport(
        all_passed=all(item.status == "passed" for item in checkpoints),
        checkpoints=tuple(checkpoints),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = asyncio.run(run_manifest(args.manifest, args.repo_root.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n")
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
