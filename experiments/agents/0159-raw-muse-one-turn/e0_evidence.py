"""Typed, append-only host evidence for the 0159 Executable Theory Lane.

E0 deliberately contains no game interpretation.  It classifies only public
host facts: actions, reset flags, terminal state, level-count changes, and ARC
animation frame sequences.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

ActionLabel = Literal[
    "RESET", "UP", "DOWN", "LEFT", "RIGHT", "SPACEBAR", "CLICK", "UNDO"
]
Boundary = Literal["none", "explicit_reset", "api_full_reset"]
Outcome = Literal["ongoing", "level_complete", "game_over"]
Grid = tuple[tuple[int, ...], ...]

ACTION_LABELS = frozenset(
    {"RESET", "UP", "DOWN", "LEFT", "RIGHT", "SPACEBAR", "CLICK", "UNDO"}
)


class EvidenceIntegrityError(RuntimeError):
    """The supplied record would make the authoritative evidence diverge."""


@dataclass(frozen=True)
class Action:
    label: ActionLabel
    x: int | None = None
    y: int | None = None

    def __post_init__(self) -> None:
        if self.label not in ACTION_LABELS:
            raise ValueError(f"unsupported action label {self.label!r}")
        if self.label == "CLICK":
            if type(self.x) is not int or type(self.y) is not int:
                raise ValueError("CLICK requires integer x and y coordinates")
            if not 0 <= self.x <= 63 or not 0 <= self.y <= 63:
                raise ValueError("CLICK coordinates must be in [0, 63]")
        elif self.x is not None or self.y is not None:
            raise ValueError("only CLICK accepts coordinates")


@dataclass(frozen=True)
class PublicObservationInput:
    frames: tuple[Grid, ...]
    public_state: str
    levels_completed: int
    full_reset: bool
    legal_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(_normalize_grid(grid) for grid in self.frames)
        if not normalized:
            raise ValueError("a public observation requires at least one frame")
        shape = (len(normalized[0]), len(normalized[0][0]))
        if any((len(grid), len(grid[0])) != shape for grid in normalized):
            raise ValueError("all animation frames must have one shared shape")
        if not isinstance(self.public_state, str) or not self.public_state:
            raise ValueError("public_state must be a non-empty string")
        if type(self.levels_completed) is not int or self.levels_completed < 0:
            raise ValueError("levels_completed must be a non-negative integer")
        if type(self.full_reset) is not bool:
            raise ValueError("full_reset must be a boolean")
        if any(action not in ACTION_LABELS for action in self.legal_actions):
            raise ValueError("legal_actions contains an unsupported action label")
        object.__setattr__(self, "frames", normalized)
        object.__setattr__(self, "legal_actions", tuple(self.legal_actions))

    @property
    def settled_grid(self) -> Grid:
        return self.frames[-1]


@dataclass(frozen=True)
class ObservationOccurrence:
    observation_id: str
    source_event_id: str
    global_turn: int
    grid_sha256: str
    public_state: str
    levels_completed: int
    full_reset: bool
    legal_actions: tuple[str, ...]
    attempt_id: str


@dataclass(frozen=True)
class EvaluationTransition:
    source_event_id: str
    global_turn: int
    action: Action
    before_observation_id: str
    after_observation_id: str
    before_grid_sha256: str
    after_grid_sha256: str
    public_state_after: str
    levels_completed_before: int
    levels_completed_after: int
    full_reset_after: bool
    boundary: Boundary
    outcome: Outcome
    attempt_before: str
    attempt_after: str | None
    animation_event_id: str | None


@dataclass(frozen=True)
class TerminalEvent:
    terminal_event_id: str
    source_event_id: str
    global_turn: int
    observation_id: str
    grid_sha256: str
    outcome: Literal["level_complete", "game_over"]
    attempt_id: str


@dataclass(frozen=True)
class AnimationEventRef:
    animation_event_id: str
    source_event_id: str
    global_turn: int
    frame_sha256: tuple[str, ...]
    settled_frame_index: int
    manifest_sha256: str


@dataclass(frozen=True)
class EvidenceSnapshot:
    snapshot_id: str
    through_host_sequence: int
    occurrence_prefix_sha256: str
    transition_records_sha256: str
    source_event_ids: tuple[str, ...]
    occurrence_records_json: str
    transitions: tuple[EvaluationTransition, ...]
    terminal_events: tuple[TerminalEvent, ...]
    animation_events: tuple[AnimationEventRef, ...]


@dataclass(frozen=True)
class ActionResult:
    observation: ObservationOccurrence
    snapshot: EvidenceSnapshot


@dataclass(frozen=True)
class FrontierInstallResult:
    status: Literal["installed", "idempotent"]
    evidence_snapshot_id: str


def canonical_json(value: Any) -> str:
    """Encode committed values using the documented E0 canonical JSON form."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def verify_snapshot(snapshot: EvidenceSnapshot) -> bool:
    """Verify authoritative-prefix and derived-manifest commitments separately."""

    try:
        records = json.loads(snapshot.occurrence_records_json)
    except json.JSONDecodeError as error:
        raise EvidenceIntegrityError("invalid occurrence-prefix encoding") from error
    if not isinstance(records, list) or not records:
        raise EvidenceIntegrityError("occurrence prefix must be a non-empty list")
    expected_sequences = list(range(len(records)))
    if [record.get("host_sequence") for record in records] != expected_sequences:
        raise EvidenceIntegrityError(
            "occurrence prefix host sequences are not monotonic"
        )
    event_ids = tuple(str(record.get("event_id")) for record in records)
    if event_ids != snapshot.source_event_ids:
        raise EvidenceIntegrityError("source event ids do not match occurrence prefix")
    if snapshot.through_host_sequence != len(records) - 1:
        raise EvidenceIntegrityError(
            "snapshot frontier does not match occurrence prefix"
        )
    if content_sha256(records) != snapshot.occurrence_prefix_sha256:
        raise EvidenceIntegrityError("occurrence-prefix commitment does not verify")

    manifest = {
        "transitions": _plain(snapshot.transitions),
        "terminal_events": _plain(snapshot.terminal_events),
        "animation_events": _plain(snapshot.animation_events),
    }
    if content_sha256(manifest) != snapshot.transition_records_sha256:
        raise EvidenceIntegrityError("derived-manifest commitment does not verify")
    known_events = set(event_ids)
    derived_sources = {
        item.source_event_id
        for items in (
            snapshot.transitions,
            snapshot.terminal_events,
            snapshot.animation_events,
        )
        for item in items
    }
    if not derived_sources <= known_events:
        raise EvidenceIntegrityError("derived record cites an unknown source event")
    records_by_event = {record["event_id"]: record for record in records}
    if any(
        records_by_event[transition.source_event_id].get("type") != "action_result"
        for transition in snapshot.transitions
    ):
        raise EvidenceIntegrityError("transition does not cite an action-result event")
    for animation in snapshot.animation_events:
        animation_manifest = {
            "frame_sha256": list(animation.frame_sha256),
            "settled_frame_index": animation.settled_frame_index,
        }
        if animation.settled_frame_index != len(animation.frame_sha256) - 1:
            raise EvidenceIntegrityError("animation settled-frame index is invalid")
        if content_sha256(animation_manifest) != animation.manifest_sha256:
            raise EvidenceIntegrityError(
                "animation manifest commitment does not verify"
            )
    transition_ids = {item.source_event_id for item in snapshot.transitions}
    if any(
        item.source_event_id not in transition_ids for item in snapshot.terminal_events
    ):
        raise EvidenceIntegrityError("terminal event has no corresponding transition")
    if (
        "snapshot-" + content_sha256(_snapshot_identity(snapshot))
        != snapshot.snapshot_id
    ):
        raise EvidenceIntegrityError("snapshot identity commitment does not verify")
    return True


def load_snapshot(path: Path) -> EvidenceSnapshot:
    """Load and independently verify one persisted E0 snapshot artifact."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceIntegrityError(f"cannot load evidence snapshot {path}") from error
    if not isinstance(value, dict):
        raise EvidenceIntegrityError("evidence snapshot must be a JSON object")
    snapshot = _snapshot_from_plain(value)
    verify_snapshot(snapshot)
    return snapshot


class EvidenceSnapshotFrontier:
    """Install only one append-only sequence of self-verifying E0 snapshots."""

    def __init__(self) -> None:
        self.current: EvidenceSnapshot | None = None

    def install(self, snapshot: EvidenceSnapshot) -> FrontierInstallResult:
        current = self.current
        if current is not None:
            if snapshot.snapshot_id == current.snapshot_id:
                verify_snapshot(snapshot)
                return FrontierInstallResult("idempotent", snapshot.snapshot_id)
            if snapshot.through_host_sequence < current.through_host_sequence:
                raise EvidenceIntegrityError("older evidence frontier is rejected")
            if snapshot.through_host_sequence == current.through_host_sequence:
                verify_snapshot(snapshot)
                raise EvidenceIntegrityError(
                    "divergent snapshot at the installed frontier is rejected"
                )
            prefix_length = len(current.source_event_ids)
            if snapshot.source_event_ids[:prefix_length] != current.source_event_ids:
                raise EvidenceIntegrityError("successor occurrence prefix diverges")
        verify_snapshot(snapshot)
        self.current = snapshot
        return FrontierInstallResult("installed", snapshot.snapshot_id)


class HostEvidenceLedger:
    """One run's append-only host evidence and deterministic E0 projection."""

    def __init__(self, root: Path, *, run_id: str) -> None:
        if not run_id or any(character.isspace() for character in run_id):
            raise ValueError("run_id must be a non-empty identifier without whitespace")
        self.root = root
        self.run_id = run_id
        self.ledger_path = root / "ledger.jsonl"
        self.blobs = root / "blobs"
        self.snapshots = root / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs.mkdir(exist_ok=True)
        self.snapshots.mkdir(exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._transitions: list[EvaluationTransition] = []
        self._terminal_events: list[TerminalEvent] = []
        self._animation_events: list[AnimationEventRef] = []
        self._current: ObservationOccurrence | None = None
        self._active_attempt_id: str | None = None
        self._attempt_count = 0
        self._observation_count = 0
        if self.ledger_path.exists() and self.ledger_path.stat().st_size:
            self._restore()

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    def record_initial_reset(
        self, observation: PublicObservationInput
    ) -> ObservationOccurrence:
        if self._current is not None:
            raise EvidenceIntegrityError("initial reset is already recorded")
        attempt_id = self._allocate_attempt()
        event_id = self._event_id()
        occurrence = self._observation(
            observation,
            event_id=event_id,
            global_turn=0,
            attempt_id=attempt_id,
        )
        self._append_record(
            {
                "schema": "0159-host-evidence-v1",
                "type": "initial_reset",
                "event_id": event_id,
                "observation": _plain(occurrence),
                "frame_sha256": self._store_frames(observation.frames),
            }
        )
        self._current = occurrence
        self._persist_snapshot()
        return occurrence

    def record_host_reset(
        self, observation: PublicObservationInput, *, global_turn: int
    ) -> ObservationOccurrence:
        """Record an actual host reset that is not an action-result boundary."""

        if self._current is None:
            raise EvidenceIntegrityError("initial reset must be recorded first")
        if type(global_turn) is not int or global_turn < 0:
            raise ValueError("global_turn must be a non-negative integer")
        attempt_id = self._allocate_attempt()
        event_id = self._event_id()
        occurrence = self._observation(
            observation,
            event_id=event_id,
            global_turn=global_turn,
            attempt_id=attempt_id,
        )
        self._append_record(
            {
                "schema": "0159-host-evidence-v1",
                "type": "host_reset",
                "event_id": event_id,
                "observation": _plain(occurrence),
                "frame_sha256": self._store_frames(observation.frames),
            }
        )
        self._current = occurrence
        self._persist_snapshot()
        return occurrence

    def record_decision_observation(
        self, observation: PublicObservationInput, *, global_turn: int
    ) -> ObservationOccurrence:
        """Open the next replay segment after a level-complete successor.

        During an ordinary turn the action-result successor is already the
        decision observation, so the call is idempotent.  Only a previously
        closed level-complete segment creates a new occurrence and attempt.
        GAME_OVER requires :meth:`record_host_reset` instead.
        """

        if self._current is None:
            raise EvidenceIntegrityError("initial reset must be recorded first")
        if self._active_attempt_id is not None:
            if (
                self._current.global_turn != global_turn
                or self._current.grid_sha256
                != content_sha256(_plain(observation.settled_grid))
                or self._current.levels_completed != observation.levels_completed
                or self._current.public_state != observation.public_state
                or self._current.full_reset != observation.full_reset
                or self._current.legal_actions != observation.legal_actions
            ):
                raise EvidenceIntegrityError(
                    "decision observation diverges from the recorded successor"
                )
            return self._current
        if self._current.public_state.upper() == "GAME_OVER":
            raise EvidenceIntegrityError("GAME_OVER requires an actual host reset")
        attempt_id = self._allocate_attempt()
        event_id = self._event_id()
        occurrence = self._observation(
            observation,
            event_id=event_id,
            global_turn=global_turn,
            attempt_id=attempt_id,
        )
        self._append_record(
            {
                "schema": "0159-host-evidence-v1",
                "type": "decision_observation",
                "event_id": event_id,
                "observation": _plain(occurrence),
                "frame_sha256": self._store_frames(observation.frames),
            }
        )
        self._current = occurrence
        self._persist_snapshot()
        return occurrence

    def record_action_result(
        self,
        *,
        global_turn: int,
        action: Action,
        successor: PublicObservationInput,
    ) -> ActionResult:
        if self._current is None or self._active_attempt_id is None:
            raise EvidenceIntegrityError("an active decision observation is required")
        if type(global_turn) is not int or global_turn < 0:
            raise ValueError("global_turn must be a non-negative integer")

        before = self._current
        if before.global_turn != global_turn:
            raise EvidenceIntegrityError(
                "action result global turn does not match its decision observation"
            )
        attempt_before = self._active_attempt_id
        boundary = self._classify_boundary(action, successor)
        if successor.levels_completed < before.levels_completed and boundary == "none":
            raise EvidenceIntegrityError(
                "levels_completed decreased without an explicit reset boundary"
            )
        outcome = self._classify_outcome(before, successor)
        if boundary == "none":
            attempt_after = None if outcome == "game_over" else attempt_before
        else:
            attempt_after = self._allocate_attempt()

        event_id = self._event_id()
        observation_attempt = attempt_after or attempt_before
        after = self._observation(
            successor,
            event_id=event_id,
            global_turn=global_turn + 1,
            attempt_id=observation_attempt,
        )
        frame_hashes = self._store_frames(successor.frames)
        animation = self._animation_ref(
            event_id=event_id,
            global_turn=global_turn,
            frame_hashes=frame_hashes,
        )
        transition = EvaluationTransition(
            source_event_id=event_id,
            global_turn=global_turn,
            action=action,
            before_observation_id=before.observation_id,
            after_observation_id=after.observation_id,
            before_grid_sha256=before.grid_sha256,
            after_grid_sha256=after.grid_sha256,
            public_state_after=successor.public_state,
            levels_completed_before=before.levels_completed,
            levels_completed_after=successor.levels_completed,
            full_reset_after=successor.full_reset,
            boundary=boundary,
            outcome=outcome,
            attempt_before=attempt_before,
            attempt_after=attempt_after,
            animation_event_id=(
                animation.animation_event_id if animation is not None else None
            ),
        )
        terminal = self._terminal_event(transition, after)
        record = {
            "schema": "0159-host-evidence-v1",
            "type": "action_result",
            "event_id": event_id,
            "action": _plain(action),
            "before_observation_id": before.observation_id,
            "after_observation": _plain(after),
            "frame_sha256": list(frame_hashes),
        }
        self._append_record(record)
        self._transitions.append(transition)
        if terminal is not None:
            self._terminal_events.append(terminal)
        if animation is not None:
            self._animation_events.append(animation)
        self._current = after
        self._active_attempt_id = (
            None if outcome in {"level_complete", "game_over"} else observation_attempt
        )
        snapshot = self._persist_snapshot()
        return ActionResult(observation=after, snapshot=snapshot)

    def snapshot(self) -> EvidenceSnapshot:
        if not self._records:
            raise EvidenceIntegrityError("cannot snapshot an empty ledger")
        return self._make_snapshot()

    def _event_id(self) -> str:
        return f"{self.run_id}:event:{len(self._records):08d}"

    def _allocate_attempt(self) -> str:
        attempt_id = f"attempt-{self._attempt_count:06d}"
        self._attempt_count += 1
        self._active_attempt_id = attempt_id
        return attempt_id

    def _observation(
        self,
        source: PublicObservationInput,
        *,
        event_id: str,
        global_turn: int,
        attempt_id: str,
    ) -> ObservationOccurrence:
        occurrence = ObservationOccurrence(
            observation_id=f"{self.run_id}:observation:{self._observation_count:08d}",
            source_event_id=event_id,
            global_turn=global_turn,
            grid_sha256=content_sha256(_plain(source.settled_grid)),
            public_state=source.public_state,
            levels_completed=source.levels_completed,
            full_reset=source.full_reset,
            legal_actions=source.legal_actions,
            attempt_id=attempt_id,
        )
        self._observation_count += 1
        return occurrence

    def _classify_boundary(
        self, action: Action, successor: PublicObservationInput
    ) -> Boundary:
        if action.label == "RESET":
            return "explicit_reset"
        if successor.full_reset:
            return "api_full_reset"
        return "none"

    def _classify_outcome(
        self, before: ObservationOccurrence, successor: PublicObservationInput
    ) -> Outcome:
        if successor.levels_completed > before.levels_completed:
            return "level_complete"
        if successor.public_state.upper() == "GAME_OVER":
            return "game_over"
        if successor.public_state.upper() == "WIN":
            raise EvidenceIntegrityError(
                "WIN without a levels_completed increase has no valid E0 outcome"
            )
        return "ongoing"

    def _store_frames(self, frames: tuple[Grid, ...]) -> tuple[str, ...]:
        hashes: list[str] = []
        for grid in frames:
            plain = _plain(grid)
            digest = content_sha256(plain)
            path = self.blobs / f"grid-{digest}.json"
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != plain:
                    raise EvidenceIntegrityError("content-addressed grid collision")
            else:
                _write_atomic_json(path, plain, immutable=True)
            hashes.append(digest)
        return tuple(hashes)

    def _animation_ref(
        self,
        *,
        event_id: str,
        global_turn: int,
        frame_hashes: tuple[str, ...],
    ) -> AnimationEventRef | None:
        if len(frame_hashes) == 1:
            return None
        manifest = {
            "frame_sha256": list(frame_hashes),
            "settled_frame_index": len(frame_hashes) - 1,
        }
        return AnimationEventRef(
            animation_event_id=f"{event_id}:animation",
            source_event_id=event_id,
            global_turn=global_turn,
            frame_sha256=frame_hashes,
            settled_frame_index=len(frame_hashes) - 1,
            manifest_sha256=content_sha256(manifest),
        )

    def _terminal_event(
        self,
        transition: EvaluationTransition,
        observation: ObservationOccurrence,
    ) -> TerminalEvent | None:
        if transition.outcome == "ongoing":
            return None
        return TerminalEvent(
            terminal_event_id=f"{transition.source_event_id}:terminal",
            source_event_id=transition.source_event_id,
            global_turn=transition.global_turn,
            observation_id=observation.observation_id,
            grid_sha256=observation.grid_sha256,
            outcome=transition.outcome,
            attempt_id=transition.attempt_before,
        )

    def _append_record(self, record: dict[str, Any]) -> None:
        expected_sequence = len(self._records)
        committed = {"host_sequence": expected_sequence, **record}
        if committed["event_id"] != self._event_id():
            raise EvidenceIntegrityError("event id does not match host sequence")
        _append_jsonl(self.ledger_path, committed)
        self._records.append(committed)

    def _make_snapshot(self) -> EvidenceSnapshot:
        through = len(self._records) - 1
        event_ids = tuple(record["event_id"] for record in self._records)
        occurrence_records_json = canonical_json(self._records)
        occurrence_digest = content_sha256(self._records)
        manifest = {
            "transitions": _plain(tuple(self._transitions)),
            "terminal_events": _plain(tuple(self._terminal_events)),
            "animation_events": _plain(tuple(self._animation_events)),
        }
        transition_digest = content_sha256(manifest)
        snapshot = EvidenceSnapshot(
            snapshot_id="",
            through_host_sequence=through,
            occurrence_prefix_sha256=occurrence_digest,
            transition_records_sha256=transition_digest,
            source_event_ids=event_ids,
            occurrence_records_json=occurrence_records_json,
            transitions=tuple(self._transitions),
            terminal_events=tuple(self._terminal_events),
            animation_events=tuple(self._animation_events),
        )
        return replace(
            snapshot,
            snapshot_id="snapshot-" + content_sha256(_snapshot_identity(snapshot)),
        )

    def _persist_snapshot(self) -> EvidenceSnapshot:
        snapshot = self._make_snapshot()
        path = self.snapshots / f"{snapshot.snapshot_id}.json"
        if not path.exists():
            _write_atomic_json(path, _plain(snapshot), immutable=True)
        _write_atomic_json(
            self.root / "current-snapshot.json",
            {
                "snapshot_id": snapshot.snapshot_id,
                "through_host_sequence": snapshot.through_host_sequence,
            },
        )
        return snapshot

    def _restore(self) -> None:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.ledger_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EvidenceIntegrityError(
                    f"ledger line {line_number} is not an object"
                )
            records.append(value)
        if not records:
            raise EvidenceIntegrityError("non-empty ledger contains no records")
        for sequence, record in enumerate(records):
            if record.get("host_sequence") != sequence:
                raise EvidenceIntegrityError("persisted host sequence diverges")
            if record.get("event_id") != f"{self.run_id}:event:{sequence:08d}":
                raise EvidenceIntegrityError("persisted event id diverges")

        pointer_path = self.root / "current-snapshot.json"
        if not pointer_path.is_file():
            raise EvidenceIntegrityError("persisted ledger has no current snapshot")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        snapshot_path = self.snapshots / f"{pointer.get('snapshot_id')}.json"
        if not snapshot_path.is_file():
            raise EvidenceIntegrityError("current snapshot artifact is missing")
        snapshot = _snapshot_from_plain(
            json.loads(snapshot_path.read_text(encoding="utf-8"))
        )
        verify_snapshot(snapshot)
        if json.loads(snapshot.occurrence_records_json) != records:
            raise EvidenceIntegrityError("ledger diverges from current snapshot prefix")
        if snapshot.through_host_sequence != len(records) - 1:
            raise EvidenceIntegrityError(
                "current snapshot is not at the ledger frontier"
            )

        self._records = records
        self._transitions = list(snapshot.transitions)
        self._terminal_events = list(snapshot.terminal_events)
        self._animation_events = list(snapshot.animation_events)
        last_record = records[-1]
        plain_observation = last_record.get("observation") or last_record.get(
            "after_observation"
        )
        if not isinstance(plain_observation, dict):
            raise EvidenceIntegrityError("last ledger record has no observation")
        self._current = ObservationOccurrence(**plain_observation)
        last_transition = snapshot.transitions[-1] if snapshot.transitions else None
        if (
            last_transition is not None
            and last_transition.source_event_id == last_record["event_id"]
            and last_transition.outcome in {"level_complete", "game_over"}
        ):
            self._active_attempt_id = None
        else:
            self._active_attempt_id = self._current.attempt_id
        attempt_ids = {
            occurrence_id
            for transition in snapshot.transitions
            for occurrence_id in (
                transition.attempt_before,
                transition.attempt_after,
            )
            if occurrence_id is not None
        }
        observation_ids: set[str] = set()
        for record in records:
            occurrence = record.get("observation") or record.get("after_observation")
            if isinstance(occurrence, dict):
                attempt_id = occurrence.get("attempt_id")
                observation_id = occurrence.get("observation_id")
                if isinstance(attempt_id, str):
                    attempt_ids.add(attempt_id)
                if isinstance(observation_id, str):
                    observation_ids.add(observation_id)
        self._attempt_count = max(_attempt_number(item) for item in attempt_ids) + 1
        self._observation_count = (
            max(_observation_number(item) for item in observation_ids) + 1
        )


def _normalize_grid(grid: Any) -> Grid:
    try:
        rows = tuple(tuple(int(value) for value in row) for row in grid)
    except (TypeError, ValueError) as error:
        raise ValueError("grid must contain integer-valued rows") from error
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("grid must be a non-empty rectangle")
    if any(not 0 <= value <= 15 for row in rows for value in row):
        raise ValueError("public grid values must be in 0..15")
    return rows


def _write_atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = canonical_json(value) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    if immutable:
        path.chmod(0o444)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _snapshot_from_plain(value: dict[str, Any]) -> EvidenceSnapshot:
    try:
        transitions = tuple(
            EvaluationTransition(
                **{
                    **item,
                    "action": Action(**item["action"]),
                }
            )
            for item in value["transitions"]
        )
        terminals = tuple(TerminalEvent(**item) for item in value["terminal_events"])
        animations = tuple(
            AnimationEventRef(
                **{
                    **item,
                    "frame_sha256": tuple(item["frame_sha256"]),
                }
            )
            for item in value["animation_events"]
        )
        return EvidenceSnapshot(
            snapshot_id=value["snapshot_id"],
            through_host_sequence=value["through_host_sequence"],
            occurrence_prefix_sha256=value["occurrence_prefix_sha256"],
            transition_records_sha256=value["transition_records_sha256"],
            source_event_ids=tuple(value["source_event_ids"]),
            occurrence_records_json=value["occurrence_records_json"],
            transitions=transitions,
            terminal_events=terminals,
            animation_events=animations,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceIntegrityError("persisted snapshot schema is invalid") from error


def _attempt_number(attempt_id: str) -> int:
    prefix = "attempt-"
    if not attempt_id.startswith(prefix) or not attempt_id[len(prefix) :].isdigit():
        raise EvidenceIntegrityError(f"invalid attempt id {attempt_id!r}")
    return int(attempt_id[len(prefix) :])


def _observation_number(observation_id: str) -> int:
    marker = ":observation:"
    if marker not in observation_id:
        raise EvidenceIntegrityError(f"invalid observation id {observation_id!r}")
    suffix = observation_id.rsplit(marker, 1)[1]
    if not suffix.isdigit():
        raise EvidenceIntegrityError(f"invalid observation id {observation_id!r}")
    return int(suffix)


def _snapshot_identity(snapshot: EvidenceSnapshot) -> dict[str, Any]:
    return {
        "through_host_sequence": snapshot.through_host_sequence,
        "occurrence_prefix_sha256": snapshot.occurrence_prefix_sha256,
        "transition_records_sha256": snapshot.transition_records_sha256,
        "source_event_ids": list(snapshot.source_event_ids),
    }
