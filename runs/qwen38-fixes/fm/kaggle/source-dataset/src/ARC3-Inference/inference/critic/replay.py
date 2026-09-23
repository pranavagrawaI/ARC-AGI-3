"""Replay a Duck events file through the trigger policy, packet builder, and sandbox.

    python -m inference.critic.replay <game>_p<pass>_events.jsonl [--stall-n 3 --stall-k 25]

The events file (`example-run/artifacts/*_events.jsonl`) is the viewer log the solver
writes: one `initial` event, then `action` events (one per executed action) each
followed, once the turn ends, by an `analysis` event carrying that `analyze()` call's
transcript. From it we rebuild, after every executed action, the same
`tool_runtime_state.json` shape Duck reads, and Duck's carried world model by running
`_extract_scientist_note` over the transcript's ASSISTANT sections with Duck's own
update-and-wipe rules. At every turn start the trigger policy is asked; on a firing
a packet is frozen and one fixed investigation snippet is run in the sandbox.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import _empty_world_model, _extract_scientist_note, _normalize_valid_actions
from inference.critic.packet import CriticPacket, build_packet_from_state, transitions_from_history
from inference.critic.sandbox import run_investigation_python
from inference.critic.trigger import CAUSES, DuckTriggerPolicy
from inference.utils.grid_utils import ARC_COLOR_CHARS

_SECTION_RE = re.compile(r"^\[([A-Z][A-Za-z :_]+)\]\n", re.MULTILINE)
_VALID_ACTIONS_RE = re.compile(r"^Valid actions right now: (.*)\.$", re.MULTILINE)
_WIPED_KEYS = ("world_model", "goal_model", "action_model", "recent_findings", "open_questions", "current_plan")

INVESTIGATION_SNIPPET = """
nodes = current_frame.segmentation["nodes"]
changed_cells = 0
if previous_frame is not None:
    before = previous_frame.ascii.split("\\n")
    after = current_frame.ascii.split("\\n")
    changed_cells = sum(
        1 for row_a, row_b in zip(before, after) for a, b in zip(row_a, row_b) if a != b
    )
result = {
    "nodes": len(nodes),
    "changed_cells": changed_cells,
    "transitions": len(transitions),
    "last_action": last_action,
    "valid_actions": list(valid_actions),
    "shape": list(current_frame.shape),
}
"""


def decode_board_ascii(text: str) -> tuple[tuple[int, ...], ...]:
    """Letter-coded board (see `ARC_COLOR_CHARS`) back to the integer grid."""
    rows: list[tuple[int, ...]] = []
    for line in str(text or "").split("\n"):
        if not line:
            continue
        rows.append(tuple(max(0, ARC_COLOR_CHARS.find(char)) for char in line))
    return tuple(rows)


def event_grid(event: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    board = event.get("board")
    if isinstance(board, list) and board:
        return tuple(tuple(int(cell) for cell in row) for row in board)
    return decode_board_ascii(str(event.get("board_ascii", "")))


def transcript_sections(transcript: str) -> list[tuple[str, str]]:
    parts = _SECTION_RE.split(transcript)
    return [(parts[index], parts[index + 1]) for index in range(1, len(parts) - 1, 2)]


def valid_actions_from_transcript(transcript: str) -> list[str]:
    match = _VALID_ACTIONS_RE.search(transcript)
    if match is None:
        return []
    return _normalize_valid_actions([item.strip() for item in match.group(1).split(",") if item.strip()])


def compact_action_result(event: dict[str, Any]) -> dict[str, Any]:
    """The subset of `ToolAgent._compact_action_result` recoverable from an action event."""
    return {
        "executed": True,
        "action_num": event.get("action_num"),
        "level": event.get("level"),
        "score": event.get("score"),
        "reward": event.get("reward"),
        "state": event.get("state"),
        "board_changed": bool(event.get("board_changed")),
        "done": bool(event.get("done")),
        "level_completed": bool(event.get("level_completed")),
        "game_over": bool(event.get("game_over")),
        "run_complete": bool(event.get("run_complete")),
        "action_display": event.get("action_display") or event.get("action_name"),
        "executed_actions": [str(event.get("action_display") or event.get("action_name") or "")],
        "batch_index": event.get("batch_index"),
        "batch_size": event.get("batch_size"),
    }


def summarize_turn_actions(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Mirror `ToolAgent._summarize_step_sequence` over one turn's executed actions."""
    if not events:
        return None
    last = events[-1]
    stop_reason = None
    for key in ("run_complete", "game_over", "level_completed"):
        if any(bool(item.get(key)) for item in events):
            stop_reason = key
            break
    return {
        "start_action_num": int(events[0].get("action_num") or 0),
        "end_action_num": int(last.get("action_num") or 0),
        "executed_count": len(events),
        "executed_actions": [str(item.get("action_display") or item.get("action_name") or "") for item in events],
        "level": last.get("level"),
        "level_transition": any(bool(item.get("level_completed")) for item in events),
        "run_complete": any(bool(item.get("run_complete")) for item in events),
        "game_over": any(bool(item.get("game_over")) for item in events),
        "board_changed": any(bool(item.get("board_changed")) for item in events),
        "stop_reason": stop_reason,
    }


@dataclass
class Turn:
    analysis: dict[str, Any]
    pre_turn_actions: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)


def iter_turns(events: list[dict[str, Any]]) -> Iterator[Turn]:
    """Group action events under the analysis event that closes their turn.

    Auto-RESET actions the solver issues after GAME_OVER happen before the next
    `analyze()` call and carry the previous analysis_step; they are `pre_turn_actions`.
    """
    pending: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("type")
        if kind == "action":
            pending.append(event)
        elif kind == "analysis":
            step = event.get("analysis_step")
            turn = Turn(analysis=event)
            for item in pending:
                if item.get("analysis_step") != step:
                    turn.pre_turn_actions.append(item)
                else:
                    turn.actions.append(item)
            pending = []
            yield turn
    if pending:
        yield Turn(analysis={}, pre_turn_actions=[], actions=pending)


@dataclass
class ReplayState:
    """Everything Duck would have in memory plus on disk at a turn start."""

    current_frame: Frame | None = None
    history: list[HistoryEntry] = field(default_factory=list)
    knowledge: dict[str, str] = field(default_factory=_empty_world_model)
    last_step_summary: dict[str, Any] | None = None
    last_action_result: dict[str, Any] | None = None
    critic_request: str = ""
    board_changed_mismatches: int = 0
    ascii_board_mismatches: int = 0

    def apply_action(self, event: dict[str, Any]) -> None:
        grid = event_grid(event)
        if event.get("board") and event.get("board_ascii") and decode_board_ascii(event["board_ascii"]) != grid:
            self.ascii_board_mismatches += 1
        frame = Frame(grid=grid, step=int(event.get("action_num") or 0), level=max(1, int(event.get("level") or 1)))
        previous = self.history[-1].frame if self.history else None
        if previous is not None and (previous.grid != grid) != bool(event.get("board_changed")):
            self.board_changed_mismatches += 1
        self.history.append(HistoryEntry(action=str(event.get("action_display") or event.get("action_name") or ""), frame=frame))
        self.current_frame = frame
        self.last_action_result = compact_action_result(event)

    def apply_assistant(self, content: str) -> None:
        note = _extract_scientist_note(content)
        for key, value in note.items():
            if key == "critic_request":
                if value:
                    self.critic_request = value
                continue
            if value:
                self.knowledge[key] = value

    def apply_step_summary(self, summary: dict[str, Any] | None) -> None:
        if not summary:
            return
        self.last_step_summary = summary
        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            for key in _WIPED_KEYS:
                self.knowledge[key] = ""


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def replay_events(
    events: list[dict[str, Any]],
    *,
    stall_no_change: int,
    stall_no_progress: int,
    run_sandbox: bool = True,
    sandbox_timeout: int = 30,
    dump_dir: Path | None = None,
    source_name: str = "",
) -> dict[str, Any]:
    state = ReplayState()
    policy = DuckTriggerPolicy(stall_no_change=stall_no_change, stall_no_progress=stall_no_progress)
    initial = next((event for event in events if event.get("type") == "initial"), None)
    if initial is not None:
        frame = Frame(grid=event_grid(initial), step=int(initial.get("action_num") or 0), level=max(1, int(initial.get("level") or 1)))
        state.current_frame = frame
        state.history.append(HistoryEntry(action="", frame=frame))

    firings: dict[str, int] = {cause: 0 for cause in CAUSES}
    firing_log: list[dict[str, Any]] = []
    packet_bytes: list[int] = []
    sandbox_seconds: list[float] = []
    sandbox_ok = sandbox_err = 0
    turns = executed = assistant_sections = labeled_sections = critic_requests = 0
    turns_with_beliefs = 0
    levels_completed = 0

    for turn_index, turn in enumerate(iter_turns(events), start=1):
        for event in turn.pre_turn_actions:
            state.apply_action(event)
            executed += 1
        if not turn.analysis:
            break
        turns += 1
        valid_actions = valid_actions_from_transcript(str(turn.analysis.get("transcript", "")))
        if any(state.knowledge.values()):
            turns_with_beliefs += 1

        policy.observe(transitions_from_history(state.history))
        request = state.critic_request
        state.critic_request = ""
        if request:
            critic_requests += 1
        level = state.current_frame.level if state.current_frame is not None else 1
        action_num = state.current_frame.step if state.current_frame is not None else 0
        decision = policy.decide(level=level, action_num=action_num, critic_request=request)
        if decision.fired:
            firings[decision.cause] = firings.get(decision.cause, 0) + 1
            packet = build_packet_from_state(
                state.current_frame,
                state.history,
                summarized_knowledge=state.knowledge,
                last_step_summary=state.last_step_summary,
                last_action_result=state.last_action_result,
                valid_actions=valid_actions,
                trigger_cause=decision.cause,
                trigger_reason=decision.reason,
            )
            entry = _fire(packet, turn_index, turn.analysis, run_sandbox=run_sandbox, sandbox_timeout=sandbox_timeout)
            packet_bytes.append(entry["packet_bytes"])
            if run_sandbox:
                sandbox_seconds.append(entry["sandbox_seconds"])
                if entry["sandbox_error"]:
                    sandbox_err += 1
                else:
                    sandbox_ok += 1
            if dump_dir is not None:
                dump_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{source_name or 'replay'}_turn{turn_index:04d}_{packet.packet_id}"
                (dump_dir / f"{stem}.packet.json").write_text(packet.to_json(indent=2), encoding="utf-8")
                (dump_dir / f"{stem}.sandbox.json").write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")
            firing_log.append(entry)

        for event in turn.actions:
            state.apply_action(event)
            executed += 1
            if event.get("level_completed"):
                levels_completed += 1
        for label, body in transcript_sections(str(turn.analysis.get("transcript", ""))):
            if label != "ASSISTANT":
                continue
            assistant_sections += 1
            before = dict(state.knowledge), state.critic_request
            state.apply_assistant(body)
            if (dict(state.knowledge), state.critic_request) != before:
                labeled_sections += 1
        state.apply_step_summary(summarize_turn_actions(turn.actions))

    def _mean(values: list[float] | list[int]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "file": source_name,
        "turns": turns,
        "executed_actions": executed,
        "levels_completed": levels_completed,
        "final_level": state.current_frame.level if state.current_frame is not None else None,
        "firings": firings,
        "firings_total": sum(firings.values()),
        "packets_built": len(packet_bytes),
        "sandbox_ok": sandbox_ok,
        "sandbox_err": sandbox_err,
        "mean_packet_bytes": _mean(packet_bytes),
        "max_packet_bytes": max(packet_bytes) if packet_bytes else None,
        "mean_sandbox_seconds": _mean(sandbox_seconds),
        "assistant_sections": assistant_sections,
        "assistant_sections_updating_beliefs": labeled_sections,
        "turns_with_nonempty_beliefs": turns_with_beliefs,
        "critic_requests": critic_requests,
        "board_changed_mismatches": state.board_changed_mismatches,
        "ascii_board_mismatches": state.ascii_board_mismatches,
        "policy": {"stall_no_change": stall_no_change, "stall_no_progress": stall_no_progress},
        "firing_log": firing_log,
    }


def _fire(
    packet: CriticPacket,
    turn_index: int,
    analysis: dict[str, Any],
    *,
    run_sandbox: bool,
    sandbox_timeout: int,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "turn": turn_index,
        "analysis_step": analysis.get("analysis_step"),
        "action_num": packet.action_num,
        "level": packet.level,
        "cause": packet.trigger_cause,
        "reason": packet.trigger_reason,
        "packet_id": packet.packet_id,
        "packet_bytes": len(packet.to_json().encode("utf-8")),
        "beliefs_nonempty": sorted(key for key, value in packet.actor_beliefs.items() if value),
        "sandbox_seconds": None,
        "sandbox_error": "",
        "sandbox_result": None,
    }
    if run_sandbox:
        started = time.monotonic()
        outcome = run_investigation_python(packet, INVESTIGATION_SNIPPET, timeout_seconds=sandbox_timeout)
        entry["sandbox_seconds"] = round(time.monotonic() - started, 4)
        entry["sandbox_error"] = str(outcome.get("error", "") or "")
        entry["sandbox_result"] = outcome.get("result")
    return entry


def replay_file(
    path: Path,
    *,
    stall_no_change: int,
    stall_no_progress: int,
    run_sandbox: bool = True,
    sandbox_timeout: int = 30,
    dump_dir: Path | None = None,
) -> dict[str, Any]:
    return replay_events(
        load_events(path),
        stall_no_change=stall_no_change,
        stall_no_progress=stall_no_progress,
        run_sandbox=run_sandbox,
        sandbox_timeout=sandbox_timeout,
        dump_dir=dump_dir,
        source_name=path.name.removesuffix("_events.jsonl"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", nargs="+", type=Path, help="one or more *_events.jsonl files")
    parser.add_argument("--stall-n", type=int, default=3, help="consecutive no-board-change actions before a stall firing")
    parser.add_argument("--stall-k", type=int, default=25, help="actions without level progress before a stall firing")
    parser.add_argument("--dump-dir", type=Path, default=None, help="write each packet and sandbox outcome here")
    parser.add_argument("--no-sandbox", action="store_true", help="build packets but skip the sandbox snippet")
    parser.add_argument("--sandbox-timeout", type=int, default=30)
    parser.add_argument("--no-log", action="store_true", help="omit the per-firing log from the summary")
    args = parser.parse_args(argv)

    summaries = []
    for path in args.events:
        summary = replay_file(
            path,
            stall_no_change=args.stall_n,
            stall_no_progress=args.stall_k,
            run_sandbox=not args.no_sandbox,
            sandbox_timeout=args.sandbox_timeout,
            dump_dir=args.dump_dir,
        )
        if args.no_log:
            summary.pop("firing_log", None)
        summaries.append(summary)
    json.dump(summaries if len(summaries) > 1 else summaries[0], sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
