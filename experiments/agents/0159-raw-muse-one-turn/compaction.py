"""Build and inspect evidence-linked continuation checkpoints.

This module is deliberately independent of NOOA's live event loop.  It is the
smallest offline seam needed to compare candidate compressions against a frozen
run before changing context behavior or paying for another game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "0159-continuation-checkpoint-v1"
MAX_CHECKPOINT_CHARS = 16_000
STREAM_FILES = {
    "transcript": "transcript.jsonl",
    "trajectory": "trajectory.jsonl",
    "provider": "provider-events.jsonl",
}
SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)", re.IGNORECASE
)
KEY_URL = re.compile(r"(/keys/)[A-Za-z0-9_-]+")
NOTE_NAME = re.compile(r"^[a-z0-9_-]+\.md$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redact(value: Any, *, key: str | None = None) -> Any:
    """Create a model-searchable view without copying obvious credentials."""
    if key is not None and SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return KEY_URL.sub(r"\1[REDACTED]", value)
    return value


def _turn(record: dict[str, Any]) -> int | None:
    for key in ("turn", "sequence"):
        value = record.get(key)
        if type(value) is int:
            return value
    return None


@dataclass(frozen=True)
class ArchiveEvent:
    event_id: str
    stream: str
    ordinal: int | str
    turn: int | None
    kind: str
    record: Any

    def searchable_text(self) -> str:
        return json.dumps(self.record, sort_keys=True, default=str).casefold()

    def model_dump(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": self.stream,
            "ordinal": self.ordinal,
            "turn": self.turn,
            "kind": self.kind,
            "record": self.record,
        }


class HistoryArchive:
    """Read-only deterministic lookup over the run's lossless evidence files."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.events: dict[str, ArchiveEvent] = {}
        self.source_files: dict[str, Path] = {}
        self._load_jsonl_streams()
        self._load_knowledge()

    def _load_jsonl_streams(self) -> None:
        for stream, filename in STREAM_FILES.items():
            path = self.run_dir / filename
            if not path.is_file():
                continue
            self.source_files[stream] = path
            for ordinal, line in enumerate(
                path.read_text(encoding="utf-8").splitlines()
            ):
                if not line.strip():
                    continue
                raw = json.loads(line)
                record = _redact(raw)
                event_id = f"{stream}:{ordinal:06d}"
                kind = str(record.get("type") or record.get("event") or "record")
                self.events[event_id] = ArchiveEvent(
                    event_id=event_id,
                    stream=stream,
                    ordinal=ordinal,
                    turn=_turn(record),
                    kind=kind,
                    record=record,
                )

    def _load_knowledge(self) -> None:
        knowledge_dir = self.run_dir / "knowledge"
        if not knowledge_dir.is_dir():
            return
        for path in sorted(knowledge_dir.glob("*.md")):
            event_id = f"knowledge:{path.name}"
            content = _redact(path.read_text(encoding="utf-8"))
            self.source_files[event_id] = path
            self.events[event_id] = ArchiveEvent(
                event_id=event_id,
                stream="knowledge",
                ordinal=path.name,
                turn=None,
                kind="durable_note",
                record={"name": path.name, "content": content, "sha256": _sha256(path)},
            )

    def ids(self) -> set[str]:
        return set(self.events)

    def get(self, event_ids: Iterable[str]) -> list[dict[str, Any]]:
        missing = [event_id for event_id in event_ids if event_id not in self.events]
        if missing:
            raise KeyError(f"unknown archive event ids: {missing}")
        return [self.events[event_id].model_dump() for event_id in event_ids]

    def turns(
        self, start: int, end: int, *, streams: set[str] | None = None
    ) -> list[dict[str, Any]]:
        if start > end:
            raise ValueError("start turn must not exceed end turn")
        return [
            event.model_dump()
            for event in self.events.values()
            if event.turn is not None
            and start <= event.turn <= end
            and (streams is None or event.stream in streams)
        ]

    def find(
        self,
        query: str,
        *,
        streams: set[str] | None = None,
        kinds: set[str] | None = None,
        start_turn: int | None = None,
        end_turn: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        terms = [term for term in query.casefold().split() if term]
        if not terms:
            raise ValueError("query must contain at least one term")
        if limit < 1:
            raise ValueError("limit must be positive")
        matches: list[ArchiveEvent] = []
        for event in self.events.values():
            if streams is not None and event.stream not in streams:
                continue
            if kinds is not None and event.kind not in kinds:
                continue
            if start_turn is not None and (
                event.turn is None or event.turn < start_turn
            ):
                continue
            if end_turn is not None and (event.turn is None or event.turn > end_turn):
                continue
            haystack = event.searchable_text()
            if all(term in haystack for term in terms):
                matches.append(event)
            if len(matches) == limit:
                break
        return [event.model_dump() for event in matches]

    def turn_view(self, turn: int) -> dict[str, Any]:
        """Return decision-shaped evidence for one turn without provider noise."""
        transcript = [
            event.model_dump()
            for event in self.events.values()
            if event.stream == "transcript" and event.turn == turn
        ]
        transitions = [
            event
            for event in self.events.values()
            if event.stream == "trajectory" and event.turn == turn
        ]
        return {
            "schema": "0159-history-turn-view-v1",
            "turn": turn,
            "transcript": transcript,
            "mechanical_transitions": [
                self._mechanical_transition(event) for event in transitions
            ],
            "omitted_streams": ["provider"],
            "truncated": False,
        }

    @staticmethod
    def _value_bbox(grid: list[list[int]], value: int) -> list[int] | None:
        points = [
            (row, column)
            for row, values in enumerate(grid)
            for column, cell in enumerate(values)
            if cell == value
        ]
        if not points:
            return None
        rows = [point[0] for point in points]
        columns = [point[1] for point in points]
        return [min(rows), min(columns), max(rows), max(columns)]

    @classmethod
    def _mechanical_transition(cls, event: ArchiveEvent) -> dict[str, Any]:
        record = event.record
        before = record.get("before")
        after = record.get("after")
        if not isinstance(before, list) or not isinstance(after, list):
            return {
                "event_id": event.event_id,
                "action": record.get("action"),
                "error": "transition has no before/after grids",
            }
        changes = [
            [row, column, before_value, after_value]
            for row, (before_row, after_row) in enumerate(
                zip(before, after, strict=True)
            )
            for column, (before_value, after_value) in enumerate(
                zip(before_row, after_row, strict=True)
            )
            if before_value != after_value
        ]
        values = sorted(
            {cell for row in before for cell in row}
            | {cell for row in after for cell in row}
        )
        motions: dict[str, Any] = {}
        for value in values:
            before_bbox = cls._value_bbox(before, value)
            after_bbox = cls._value_bbox(after, value)
            if before_bbox == after_bbox:
                continue
            delta = None
            if before_bbox is not None and after_bbox is not None:
                before_shape = (
                    before_bbox[2] - before_bbox[0],
                    before_bbox[3] - before_bbox[1],
                )
                after_shape = (
                    after_bbox[2] - after_bbox[0],
                    after_bbox[3] - after_bbox[1],
                )
                if before_shape == after_shape:
                    delta = [
                        after_bbox[0] - before_bbox[0],
                        after_bbox[1] - before_bbox[1],
                    ]
            motions[str(value)] = {
                "before_bbox": before_bbox,
                "after_bbox": after_bbox,
                "bbox_delta_row_col": delta,
            }
        return {
            "event_id": event.event_id,
            "action": record.get("action"),
            "levels_completed": record.get("levels_completed"),
            "state": record.get("state"),
            "changed_cell_count": len(changes),
            "changed_cells": changes[:40],
            "changed_cells_truncated": len(changes) > 40,
            "value_bbox_motions": motions,
        }

    def around(
        self, event_id: str, *, before: int = 1, after: int = 1
    ) -> dict[str, Any]:
        """Return a bounded same-stream causal neighborhood around an event."""
        if event_id not in self.events:
            raise KeyError(f"unknown archive event id: {event_id}")
        if before < 0 or after < 0 or before + after > 12:
            raise ValueError(
                "before and after must be non-negative and total at most 12"
            )
        target = self.events[event_id]
        stream_events = [
            event for event in self.events.values() if event.stream == target.stream
        ]
        index = stream_events.index(target)
        selected = stream_events[max(0, index - before) : index + after + 1]
        items = []
        for event in selected:
            record = event.record
            encoded = json.dumps(record, sort_keys=True, default=str)
            if len(encoded) > 6_000:
                record = {
                    "truncated": True,
                    "available_keys": sorted(record)
                    if isinstance(record, dict)
                    else [],
                    "preview": encoded[:2_000],
                }
            items.append({**event.model_dump(), "record": record})
        return {
            "schema": "0159-history-around-v1",
            "target": event_id,
            "stream": target.stream,
            "items": items,
            "truncated": any(item["record"].get("truncated") for item in items),
        }

    def note(self, name: str) -> dict[str, Any]:
        """Return one exact durable note through its dedicated affordance."""
        if not NOTE_NAME.fullmatch(name):
            raise ValueError("note name must match [a-z0-9_-]+.md")
        event_id = f"knowledge:{name}"
        if event_id not in self.events:
            raise KeyError(f"unknown durable note: {name}")
        return {
            "schema": "0159-history-note-v1",
            **self.events[event_id].model_dump(),
            "truncated": False,
        }

    def provider_episode(self, sequence: int, phase: str) -> dict[str, Any]:
        """Group provider transport events by causal phase rather than keywords."""
        events = [
            event
            for event in self.events.values()
            if event.stream == "provider" and event.turn == sequence
        ]
        if phase == "context_recovery":
            return self._provider_context_recovery(sequence, events)
        if phase == "commitment":
            return self._provider_commitment(sequence, events)
        if phase == "failures":
            failures = [
                self._compact_provider_event(event)
                for event in events
                if event.kind in {"provider_call_failed", "provider_retry_scheduled"}
            ]
            return {
                "schema": "0159-provider-episode-v1",
                "sequence": sequence,
                "phase": phase,
                "failures": failures,
                "truncated": False,
            }
        raise ValueError("phase must be context_recovery, commitment, or failures")

    @staticmethod
    def _compact_provider_event(event: ArchiveEvent) -> dict[str, Any]:
        keys = {
            "event",
            "sequence",
            "message_count",
            "image_blocks",
            "text_chars",
            "encoded_image_chars",
            "error_type",
            "status_code",
            "message",
            "retry_number",
            "analysis_calls_used",
            "max_validation_attempts",
            "tool_choice",
            "tool_names",
            "finish_reason",
            "tool_calls",
            "reasoning",
        }
        return {
            "event_id": event.event_id,
            **{key: value for key, value in event.record.items() if key in keys},
        }

    def _provider_context_recovery(
        self, sequence: int, events: list[ArchiveEvent]
    ) -> dict[str, Any]:
        starts = [event for event in events if event.kind == "provider_call_started"]
        context_events = [
            event
            for event in events
            if "ContextWindowExceededError" in json.dumps(event.record, default=str)
        ]
        initial = starts[0] if starts else None
        last_context_ordinal = max(
            (int(event.ordinal) for event in context_events), default=-1
        )
        recovered = next(
            (event for event in starts if int(event.ordinal) > last_context_ordinal),
            None,
        )
        return {
            "schema": "0159-provider-episode-v1",
            "sequence": sequence,
            "phase": "context_recovery",
            "initial_request": (
                self._compact_provider_event(initial) if initial is not None else None
            ),
            "context_errors": [
                self._compact_provider_event(event) for event in context_events
            ],
            "recovered_request": (
                self._compact_provider_event(recovered)
                if recovered is not None
                else None
            ),
            "truncated": False,
        }

    def _provider_commitment(
        self, sequence: int, events: list[ArchiveEvent]
    ) -> dict[str, Any]:
        start_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.kind == "action_commitment_started"
            ),
            None,
        )
        selected = events[start_index:] if start_index is not None else []
        selected = [
            event
            for event in selected
            if event.kind
            in {
                "action_commitment_started",
                "provider_call_started",
                "provider_call_completed",
                "action_commitment_response",
                "action_commitment_completed",
            }
        ]
        return {
            "schema": "0159-provider-episode-v1",
            "sequence": sequence,
            "phase": "commitment",
            "events": [self._compact_provider_event(event) for event in selected],
            "truncated": False,
        }

    def manifest(self) -> dict[str, Any]:
        files = []
        for stream, path in sorted(self.source_files.items()):
            files.append(
                {
                    "stream": stream,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        return {
            "schema": "0159-compaction-source-manifest-v1",
            "run_dir": str(self.run_dir),
            "event_count": len(self.events),
            "source_files": files,
            "retrieval": {
                "get": "history.get(event_ids)",
                "turns": "history.turns(start, end, streams=None)",
                "find": "history.find(query, filters)",
                "view": "redacted model-searchable projection; source files remain immutable",
            },
        }


def _event_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"event_ids", "evidence_event_ids"}:
                if not isinstance(item, list) or not all(
                    isinstance(ref, str) for ref in item
                ):
                    raise ValueError(f"{key} must be a list of event-id strings")
                refs.update(item)
            else:
                refs.update(_event_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_event_refs(item))
    return refs


def validate_checkpoint(
    candidate: dict[str, Any], archive: HistoryArchive
) -> dict[str, Any]:
    """Validate structure and provenance, never the candidate's game semantics."""
    encoded = json.dumps(candidate, sort_keys=True)
    if len(encoded) > MAX_CHECKPOINT_CHARS:
        raise ValueError(
            f"checkpoint is {len(encoded)} characters; cap is {MAX_CHECKPOINT_CHARS}"
        )
    required = {
        "schema",
        "resume_for",
        "episode_summary",
        "episode_index",
        "beliefs",
        "contradictions",
        "open_questions",
        "current_plan",
        "artifact_manifest",
        "review",
        "next_moves",
        "suggested_capabilities",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    if candidate["schema"] != SCHEMA:
        raise ValueError(f"checkpoint schema must be {SCHEMA!r}")
    if (
        not isinstance(candidate["episode_summary"], str)
        or not candidate["episode_summary"].strip()
    ):
        raise ValueError("episode_summary must be non-empty text")
    list_fields = required - {"schema", "resume_for", "episode_summary"}
    for field in list_fields:
        if not isinstance(candidate[field], list):
            raise TypeError(f"{field} must be a list")
    refs = _event_refs(candidate)
    missing_refs = sorted(refs - archive.ids())
    if missing_refs:
        raise ValueError(f"checkpoint cites unknown archive events: {missing_refs}")
    for item in candidate["artifact_manifest"]:
        if not isinstance(item, dict) or not isinstance(item.get("artifact"), str):
            raise TypeError("each artifact_manifest entry must name an artifact")
        relative = Path(item["artifact"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact path must remain inside the run: {relative}")
        path = (archive.run_dir / relative).resolve()
        if not path.is_relative_to(archive.run_dir) or not path.is_file():
            raise ValueError(f"checkpoint cites missing run artifact: {relative}")
        claimed_hash = item.get("sha256")
        if claimed_hash is not None and claimed_hash != _sha256(path):
            raise ValueError(f"artifact hash mismatch: {relative}")
    return candidate


def _latest_observation(archive: HistoryArchive, resume_turn: int) -> dict[str, Any]:
    observations = [
        event
        for event in archive.events.values()
        if event.stream == "transcript"
        and event.kind == "observation"
        and event.turn == resume_turn
    ]
    if len(observations) != 1:
        raise ValueError(
            f"expected exactly one observation at resume turn {resume_turn}; "
            f"found {len(observations)}"
        )
    record = observations[0].record
    return {
        key: record.get(key)
        for key in (
            "turn",
            "state",
            "levels_completed",
            "full_reset",
            "legal_actions",
            "trajectory_length",
            "trajectory_sha256",
            "image_path",
        )
    }


def render_recovery_packet(
    candidate: dict[str, Any],
    archive: HistoryArchive,
    *,
    resume_turn: int,
    recent_turns: int = 6,
) -> str:
    validate_checkpoint(candidate, archive)
    if recent_turns < 1:
        raise ValueError("recent_turns must be positive")
    recent_start = max(0, resume_turn - recent_turns)
    recent = archive.turns(recent_start, resume_turn - 1, streams={"transcript"})
    mechanical = _latest_observation(archive, resume_turn)
    sections = [
        "# CONTEXT RECOVERY — SAME GAME, SAME AGENT",
        "",
        (
            "This is a context transition, not a game reset and not an action result. "
            "Continue the existing run. Claims in the review are advisory and must be "
            "checked against evidence."
        ),
        "",
        "## Resume purpose",
        str(candidate["resume_for"]),
        "",
        "## Host-owned current state",
        "```json",
        json.dumps(mechanical, indent=2, sort_keys=True),
        "```",
        "",
        "## What happened",
        candidate["episode_summary"],
        "",
        "## What it currently means",
        "```json",
        json.dumps(
            {
                "beliefs": candidate["beliefs"],
                "contradictions": candidate["contradictions"],
                "open_questions": candidate["open_questions"],
                "current_plan": candidate["current_plan"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Episode index",
        "```json",
        json.dumps(candidate["episode_index"], indent=2, sort_keys=True),
        "```",
        "",
        "## Durable artifacts",
        "```json",
        json.dumps(candidate["artifact_manifest"], indent=2, sort_keys=True),
        "```",
        "",
        "## Advisory metacognitive review",
        "```json",
        json.dumps(candidate["review"], indent=2, sort_keys=True),
        "```",
        "",
        "## Next moves",
        "```json",
        json.dumps(candidate["next_moves"], indent=2, sort_keys=True),
        "```",
        "",
        "## Suggested capabilities",
        "```json",
        json.dumps(candidate["suggested_capabilities"], indent=2, sort_keys=True),
        "```",
        "",
        f"## Recent contiguous suffix (turns {recent_start}–{resume_turn - 1})",
        "```json",
        json.dumps(recent, indent=2, sort_keys=True),
        "```",
        "",
        "## Retrieval contract",
        (
            "Use `history.get(event_ids)` for cited originals, "
            "`history.turns(start, end)` for a contiguous range, and "
            "`history.find(query, filters)` for deterministic search. The searchable "
            "view is redacted; immutable source paths and hashes are in "
            "`source-manifest.json`."
        ),
    ]
    return "\n".join(sections) + "\n"


def write_bundle(
    run_dir: Path,
    candidate_path: Path,
    output_dir: Path,
    *,
    resume_turn: int,
    recent_turns: int = 6,
) -> dict[str, Any]:
    archive = HistoryArchive(run_dir)
    candidate = validate_checkpoint(
        json.loads(candidate_path.read_text(encoding="utf-8")), archive
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    packet = render_recovery_packet(
        candidate, archive, resume_turn=resume_turn, recent_turns=recent_turns
    )
    outputs = {
        "source-manifest.json": archive.manifest(),
        "checkpoint.json": candidate,
        "episode-index.json": candidate["episode_index"],
        "review.json": candidate["review"],
    }
    for name, value in outputs.items():
        (output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output_dir / "recovery-packet.md").write_text(packet, encoding="utf-8")
    return {
        "output_dir": str(output_dir.resolve()),
        "event_count": len(archive.events),
        "checkpoint_chars": len(json.dumps(candidate, sort_keys=True)),
        "packet_chars": len(packet),
        "resume_turn": resume_turn,
        "recent_turns": recent_turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-turn", type=int, required=True)
    parser.add_argument("--recent-turns", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            write_bundle(
                args.run_dir,
                args.candidate,
                args.output_dir,
                resume_turn=args.resume_turn,
                recent_turns=args.recent_turns,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
