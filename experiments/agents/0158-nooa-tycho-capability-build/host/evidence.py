"""Content-addressed empirical evidence with Tycho-style channel separation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .canonical import (
    append_jsonl,
    content_sha256,
    read_jsonl,
    write_atomic_json,
)


def _cell_deltas(a, b) -> list[tuple[int, int, int, int]]:
    out = []
    for r in range(min(len(a), len(b))):
        for c in range(min(len(a[r]), len(b[r]))):
            if a[r][c] != b[r][c]:
                out.append((r, c, int(a[r][c]), int(b[r][c])))
    return out


def _runs(cols: list[int]) -> str:
    """Collapse a sorted col list into comma-separated ranges: [36,37,38,40] -> '36-38,40'."""
    cs = sorted(cols); out = []; s = p = cs[0]
    for c in cs[1:]:
        if c == p + 1:
            p = c
        else:
            out.append((s, p)); s = p = c
    out.append((s, p))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in out)


def diff_text(prev, cur) -> str:
    """Return Tycho's lossless, compact grouping of exact cell deltas."""
    d = _cell_deltas(prev, cur)
    if not d:
        return "no cells changed"
    rs = [r for r, _, _, _ in d]; cs = [c for _, c, _, _ in d]
    bbox = f"rows {min(rs)}-{max(rs)}, cols {min(cs)}-{max(cs)}"
    by_trans: dict = defaultdict(lambda: defaultdict(list))
    for r, c, o, n in d:
        by_trans[(o, n)][r].append(c)
    lines = []
    for (o, n), rows in sorted(by_trans.items()):
        rowspec = [(r, _runs(rows[r])) for r in sorted(rows)]
        i = 0
        while i < len(rowspec):
            r0, spec = rowspec[i]; j = i
            while j + 1 < len(rowspec) and rowspec[j + 1][1] == spec and rowspec[j + 1][0] == rowspec[j][0] + 1:
                j += 1
            rng = f"rows {r0}-{rowspec[j][0]}" if j > i else f"row {r0}"
            lines.append(f"  {o}->{n}  {rng}, cols {spec}")
            i = j + 1
    return f"{len(d)} cells changed; region [{bbox}]\n" + "\n".join(lines)


def _normalize_grid(grid: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in grid)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("grid must be a non-empty rectangle")
    if any(not 0 <= value <= 15 for row in rows for value in row):
        raise ValueError("public grid values must be in 0..15")
    return rows


class EvidenceStore:
    """Append-only indexes over immutable content-addressed public records."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects = root / "objects"
        self.indexes = root / "indexes"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.indexes.mkdir(parents=True, exist_ok=True)

    def _put(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = {"schema": "0158-evidence-v1", "kind": kind, **body}
        evidence_id = f"{kind}-" + content_sha256(payload)
        record = {"evidence_id": evidence_id, **payload}
        path = self.objects / f"{evidence_id}.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise RuntimeError("content-addressed evidence collision")
        else:
            write_atomic_json(path, record)
            path.chmod(0o444)
            append_jsonl(self.indexes / f"{kind}.jsonl", {"evidence_id": evidence_id})
        return record

    def get(self, evidence_id: str) -> dict[str, Any]:
        path = self.objects / f"{evidence_id}.json"
        if not path.is_file():
            raise KeyError(evidence_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def record_frame(
        self,
        grid: Iterable[Iterable[int]],
        *,
        level: int,
        turn: int,
        attempt_id: str,
        available_actions: Iterable[int],
    ) -> dict[str, Any]:
        rows = _normalize_grid(grid)
        actions = tuple(int(action) for action in available_actions)
        if type(level) is not int or level < 0 or type(turn) is not int or turn < 0:
            raise ValueError("level and turn must be non-negative integers")
        if not attempt_id:
            raise ValueError("attempt_id is required")
        return self._put(
            "frame",
            {
                "level": level,
                "turn": turn,
                "attempt_id": attempt_id,
                "grid": [list(row) for row in rows],
                "available_actions": list(actions),
            },
        )

    def record_transition(
        self,
        before_id: str,
        after_id: str,
        *,
        action: int,
        data: dict[str, Any] | None,
        commitment_id: str,
    ) -> dict[str, Any]:
        before = self.get(before_id)
        after = self.get(after_id)
        if before["kind"] != "frame" or after["kind"] != "frame":
            raise ValueError("ordinary transitions require two decision frames")
        if before["attempt_id"] != after["attempt_id"] or before["level"] != after["level"]:
            raise ValueError("ordinary transitions cannot cross attempt or level boundaries")
        if after["turn"] != before["turn"] + 1:
            raise ValueError("ordinary transitions must link consecutive turns")
        return self._put(
            "transition",
            {
                "before_id": before_id,
                "after_id": after_id,
                "attempt_id": before["attempt_id"],
                "level": before["level"],
                "action": int(action),
                "data": data,
                "commitment_id": commitment_id,
                "diff": diff_text(before["grid"], after["grid"]),
            },
        )

    def record_probe_result(
        self,
        *,
        before_id: str,
        after_id: str,
        grounding_claim: dict[str, Any],
        probe_commitment: dict[str, Any],
        verifier_verdict: dict[str, Any],
        change_accounting: dict[str, Any],
    ) -> dict[str, Any]:
        """Record the model-authored hypothesis and its non-oracular empirical result."""
        before = self.get(before_id)
        after = self.get(after_id)
        if before["kind"] != "frame" or after["kind"] != "frame":
            raise ValueError("probe results require two decision frames")
        if verifier_verdict.get("claim_id") != grounding_claim.get("claim_id"):
            raise ValueError("probe verdict is not bound to the grounding claim")
        if verifier_verdict.get("commitment_id") != probe_commitment.get("commitment_id"):
            raise ValueError("probe verdict is not bound to the commitment")
        return self._put(
            "probe_result",
            {
                "before_id": before_id,
                "after_id": after_id,
                "grounding_claim": grounding_claim,
                "probe_commitment": probe_commitment,
                "verifier_verdict": verifier_verdict,
                "change_accounting": change_accounting,
            },
        )

    def record_event(
        self,
        event_kind: str,
        *,
        level: int,
        attempt_id: str,
        before_id: str | None,
        grids: Iterable[Iterable[Iterable[int]]],
        action: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if event_kind not in {"terminal", "death", "animation"}:
            raise ValueError("event_kind must be terminal, death, or animation")
        normalized = [[list(row) for row in _normalize_grid(grid)] for grid in grids]
        if not normalized:
            raise ValueError("separate event requires at least one observed grid")
        return self._put(
            event_kind,
            {
                "level": level,
                "attempt_id": attempt_id,
                "before_id": before_id,
                "action": action,
                "data": data,
                "grids": normalized,
            },
        )

    def _records(self, kind: str) -> list[dict[str, Any]]:
        return [self.get(item["evidence_id"]) for item in read_jsonl(self.indexes / f"{kind}.jsonl")]

    def frames(self) -> list[dict[str, Any]]:
        return self._records("frame")

    def transitions(self) -> list[dict[str, Any]]:
        return self._records("transition")

    def attempts(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for frame in self.frames():
            attempt = grouped.setdefault(
                frame["attempt_id"],
                {"attempt_id": frame["attempt_id"], "levels": set(), "frame_ids": []},
            )
            attempt["levels"].add(frame["level"])
            attempt["frame_ids"].append(frame["evidence_id"])
        return [
            {**attempt, "levels": sorted(attempt["levels"])}
            for attempt in grouped.values()
        ]

    def probe_results(self) -> list[dict[str, Any]]:
        return self._records("probe_result")

    def events(self) -> dict[str, list[dict[str, Any]]]:
        return {kind: self._records(kind) for kind in ("terminal", "death", "animation")}

    def model_view(self, *, frame_window: int | None = None) -> dict[str, Any]:
        frames = self.frames()
        if frame_window is not None:
            if type(frame_window) is not int or frame_window < 1:
                raise ValueError("frame_window must be a positive integer or null")
            visible_frames = frames[-frame_window:]
        else:
            visible_frames = frames
        frame_summaries = [
            {key: value for key, value in frame.items() if key != "grid"}
            for frame in frames
        ]
        view = {
            "frames": visible_frames,
            "frame_summaries": frame_summaries,
            "transitions": self.transitions(),
            "attempts": self.attempts(),
            "probe_results": self.probe_results(),
            "events": self.events(),
        }
        return {**view, "commitment": content_sha256(view)}
