"""Frozen evidence packet handed to the critic.

The packet is built once per trigger firing from the same `tool_runtime_state.json`
Duck reads, plus the pieces of `ToolAgent` state that are not on disk (the carried
world model, the last step summary, the last action result, valid actions). The
critic never receives anything newer than the packet it was given.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from inference.agent.runtime_state import (
    Frame,
    HistoryEntry,
    frame_to_payload,
    history_entry_to_payload,
    load_runtime_state,
)

DEFAULT_HISTORY_LIMIT = 12
DEFAULT_TRANSITION_LIMIT = 25
PACKET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TransitionRecord:
    """One executed action as observable from consecutive runtime-state frames."""

    action: str
    action_num: int
    level: int
    board_changed: bool
    level_completed: bool

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TransitionRecord:
        return cls(
            action=str(payload.get("action", "")),
            action_num=int(payload.get("action_num", 0)),
            level=int(payload.get("level", 1)),
            board_changed=bool(payload.get("board_changed")),
            level_completed=bool(payload.get("level_completed")),
        )


def transitions_from_history(history: Iterable[HistoryEntry]) -> list[TransitionRecord]:
    """Derive per-action transitions from the runtime-state history.

    Mirrors the solver's own definitions: `board_changed` is a grid inequality between
    consecutive frames, and a level completion is a strictly increasing frame level.
    The seeded first entry (empty action) is the pre-action frame and yields no record.
    """
    records: list[TransitionRecord] = []
    previous: Frame | None = None
    for entry in history:
        if previous is not None and entry.action:
            records.append(
                TransitionRecord(
                    action=entry.action,
                    action_num=entry.frame.step,
                    level=entry.frame.level,
                    board_changed=entry.frame.grid != previous.grid,
                    level_completed=entry.frame.level > previous.level,
                )
            )
        previous = entry.frame
    return records


def _listed(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _listed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listed(item) for item in value]
    return value


@dataclass(frozen=True)
class CriticPacket:
    level: int
    action_num: int
    trigger_cause: str
    trigger_reason: str
    current_frame: dict[str, Any] | None
    previous_frame: dict[str, Any] | None
    history: tuple[dict[str, Any], ...]
    history_total: int
    transitions: tuple[dict[str, Any], ...]
    transitions_total: int
    valid_actions: tuple[str, ...]
    last_action_result: dict[str, Any]
    actor_beliefs: dict[str, str]
    step_summary: dict[str, Any] | None
    schema_version: int = PACKET_SCHEMA_VERSION
    packet_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.packet_id:
            object.__setattr__(self, "packet_id", self.content_hash())

    def content_payload(self) -> dict[str, Any]:
        payload = _listed(asdict(self))
        payload.pop("packet_id", None)
        return payload

    def content_hash(self) -> str:
        canonical = json.dumps(self.content_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_payload(self) -> dict[str, Any]:
        payload = self.content_payload()
        payload["packet_id"] = self.packet_id
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, indent=indent)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CriticPacket:
        data = dict(payload)
        packet_id = str(data.pop("packet_id", "") or "")
        current_frame = data.get("current_frame")
        previous_frame = data.get("previous_frame")
        packet = cls(
            level=int(data.get("level", 1)),
            action_num=int(data.get("action_num", 0)),
            trigger_cause=str(data.get("trigger_cause", "")),
            trigger_reason=str(data.get("trigger_reason", "")),
            current_frame=_listed(current_frame) if isinstance(current_frame, dict) else None,
            previous_frame=_listed(previous_frame) if isinstance(previous_frame, dict) else None,
            history=tuple(_listed(item) for item in data.get("history", ()) if isinstance(item, dict)),
            history_total=int(data.get("history_total", 0)),
            transitions=tuple(_listed(item) for item in data.get("transitions", ()) if isinstance(item, dict)),
            transitions_total=int(data.get("transitions_total", 0)),
            valid_actions=tuple(str(item) for item in data.get("valid_actions", ())),
            last_action_result=_listed(data.get("last_action_result") or {}),
            actor_beliefs={str(k): str(v) for k, v in (data.get("actor_beliefs") or {}).items()},
            step_summary=_listed(data.get("step_summary")) if isinstance(data.get("step_summary"), dict) else None,
            schema_version=int(data.get("schema_version", PACKET_SCHEMA_VERSION)),
        )
        if packet_id and packet_id != packet.packet_id:
            raise ValueError(f"packet_id mismatch: {packet_id} != {packet.packet_id}")
        return packet

    @classmethod
    def from_json(cls, text: str) -> CriticPacket:
        return cls.from_payload(json.loads(text))

    def current_frame_obj(self) -> Frame | None:
        return _frame_from_payload(self.current_frame)

    def history_entries(self) -> list[HistoryEntry]:
        entries: list[HistoryEntry] = []
        for item in self.history:
            frame = _frame_from_payload(item.get("frame"))
            if frame is not None:
                entries.append(HistoryEntry(action=str(item.get("action", "")), frame=frame))
        return entries

    def transition_records(self) -> list[TransitionRecord]:
        return [TransitionRecord.from_payload(item) for item in self.transitions]


def _frame_from_payload(payload: Any) -> Frame | None:
    if not isinstance(payload, dict):
        return None
    return Frame(
        grid=tuple(tuple(int(cell) for cell in row) for row in payload.get("grid", [])),
        step=int(payload.get("step", 0)),
        level=int(payload.get("level", 1)),
    )


def build_packet_from_state(
    current_frame: Frame | None,
    history: list[HistoryEntry],
    *,
    summarized_knowledge: Mapping[str, str],
    last_step_summary: Mapping[str, Any] | None,
    last_action_result: Mapping[str, Any] | None,
    valid_actions: Iterable[str],
    trigger_cause: str,
    trigger_reason: str,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    transition_limit: int = DEFAULT_TRANSITION_LIMIT,
) -> CriticPacket:
    """Freeze already-loaded runtime state into a packet (pure; no file access)."""
    transitions = transitions_from_history(history)
    bounded_history = history[-max(1, history_limit):] if history else []
    previous_frame = history[-2].frame if len(history) >= 2 else None
    level = current_frame.level if current_frame is not None else 1
    action_num = current_frame.step if current_frame is not None else 0
    return CriticPacket(
        level=level,
        action_num=action_num,
        trigger_cause=trigger_cause,
        trigger_reason=trigger_reason,
        current_frame=frame_to_payload(current_frame),
        previous_frame=frame_to_payload(previous_frame),
        history=tuple(history_entry_to_payload(entry) for entry in bounded_history),
        history_total=len(history),
        transitions=tuple(record.to_payload() for record in transitions[-max(1, transition_limit):]),
        transitions_total=len(transitions),
        valid_actions=tuple(str(item).strip() for item in valid_actions if str(item).strip()),
        last_action_result=_listed(dict(last_action_result or {})),
        actor_beliefs={str(key): str(value) for key, value in summarized_knowledge.items()},
        step_summary=_listed(dict(last_step_summary)) if last_step_summary else None,
    )


def build_packet(
    state_path: Path,
    *,
    summarized_knowledge: Mapping[str, str],
    last_step_summary: Mapping[str, Any] | None,
    last_action_result: Mapping[str, Any] | None,
    valid_actions: Iterable[str],
    trigger: Any,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    transition_limit: int = DEFAULT_TRANSITION_LIMIT,
) -> CriticPacket:
    """Read `tool_runtime_state.json` via Duck's loader and freeze it into a packet.

    `trigger` is a `TriggerDecision` (or anything with `cause` and `reason` attributes).
    """
    current_frame, history = load_runtime_state(state_path)
    return build_packet_from_state(
        current_frame,
        history,
        summarized_knowledge=summarized_knowledge,
        last_step_summary=last_step_summary,
        last_action_result=last_action_result,
        valid_actions=valid_actions,
        trigger_cause=str(getattr(trigger, "cause", "")),
        trigger_reason=str(getattr(trigger, "reason", "")),
        history_limit=history_limit,
        transition_limit=transition_limit,
    )


__all__ = [
    "CriticPacket",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_TRANSITION_LIMIT",
    "TransitionRecord",
    "build_packet",
    "build_packet_from_state",
    "transitions_from_history",
]
