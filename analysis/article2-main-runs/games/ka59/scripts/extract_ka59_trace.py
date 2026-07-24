#!/usr/bin/env python3
"""Reconstruct the ka59 model-turn and environment-action trace.

Reads the archived run without modifying it. Outputs analysis CSVs keyed by the
objective global step_index and the model API turn recorded in agent.log.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

RUN = Path(
    "/mnt/c/AI/datasets/article2_main_runs/runs/"
    "ewma_v1.2_sv_gpt-5.5_xhigh_run1/ka59/run"
)
OUT = Path(__file__).resolve().parents[1] / "data"
LOG = RUN / "agent.log"
SESSION = RUN / "agent_run/client/session"

ACTION_STEP_RE = re.compile(r"(?m)^step:\s*(\d+)\s*$")
ACTION_OUTPUT_RE = re.compile(r"(?m)^action:\s*(ACTION\d+)\s*$")
ATTEMPT_RE = re.compile(r"level_(\d+)_attempt_(\d+)$")
LOCAL_STEP_RE = re.compile(r"step_(\d+)_metadata\.json$")

LEVEL_HUMAN_BASELINES = {1: 28, 2: 109, 3: 51, 4: 51, 5: 33, 6: 132, 7: 326}
LEVEL_END_STEPS = {1: 44, 2: 93, 3: 128, 4: 189, 5: 214, 6: 301, 7: 397}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def command_mode(command: str, batch_size: int) -> str:
    if batch_size == 1:
        return "direct"
    if "for " in command or "for\n" in command or "for index" in command:
        return "scripted_batch"
    return "batch"


def message_kind(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("wrong", "fals", "failed", "mismatch", "did not", "not simply", "revert")):
        return "contradiction_or_repair"
    if any(term in lowered for term in ("verify", "replay", "check", "compare")):
        return "verification"
    if any(term in lowered for term in ("implement", "edit", "refactor", "model now", "adding", "updating")):
        return "construction"
    if any(term in lowered for term in ("hypothesis", "likely", "appears", "infer", "plausible")):
        return "hypothesis"
    if any(term in lowered for term in ("execute", "taking", "move", "plan", "action")):
        return "action_or_plan"
    return "observation_or_other"


def verification_outcome(status: str, output: str) -> str:
    lowered = output.lower()
    if status == "failed" or "verification failed" in lowered or "traceback" in lowered:
        return "failed"
    if "verification passed" in lowered or re.search(r"levels? \d+\.\.\d+ verified", lowered):
        return "passed"
    if "no completed" in lowered or "we skip" in lowered:
        return "skipped"
    if "too slow" in lowered or "timeout" in lowered:
        return "timeout"
    return "other"


def verification_failure_class(verifier: str, outcome: str, output: str) -> str:
    if outcome in {"passed", "skipped"}:
        return outcome
    lowered = output.lower()
    if outcome == "timeout":
        return "timeout"
    if any(term in lowered for term in ("syntaxerror", "indentationerror", "nameerror", "importerror", "modulenotfounderror")):
        return "construction_failure"
    if verifier == "main_planner":
        return "planner_failure"
    if "status mismatch" in lowered or "predicted status" in lowered:
        return "terminal_status_mismatch"
    if "rendered frame mismatch" in lowered:
        return "observed_frame_mismatch"
    if "traceback" in lowered:
        return "world_model_implementation_failure"
    return "tool_or_unclassified_failure"


def verifier_output_segments(command: str, output: str, verifier: str) -> list[tuple[int | None, str]]:
    """Return one output segment per actual verifier invocation.

    Most verifier commands invoke once. Five level-2 execution wrappers invoke
    the world-model verifier after every charged action, so their output must be
    split rather than counted as one command event.
    """
    if "sed -n" in command:
        return []
    if verifier == "world_model" and "python3 - <<'PY'" in command:
        segments: list[tuple[int | None, str]] = []
        for block in re.split(r"(?m)(?=^== real(?: suffix)? action )", output):
            step_match = ACTION_STEP_RE.search(block)
            if step_match and "verify_world_model.py:" in block:
                segments.append((int(step_match.group(1)), block))
        if not segments:
            raise AssertionError("world-model execution wrapper contained no per-action verifier output")
        return segments
    return [(None, output)]


def parse_log() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    turns: dict[int, dict[str, Any]] = {}
    messages: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    action_events: dict[int, dict[str, Any]] = {}

    turn = 0
    last_agent_message = ""
    last_global_step = 0
    previous_usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0}

    with LOG.open(encoding="utf-8") as handle:
        for log_line, line in enumerate(handle, 1):
            event = json.loads(line)
            event_type = event.get("type")

            if event_type == "turn.started":
                turn += 1
                turns[turn] = {
                    "model_turn": turn,
                    "start_log_line": log_line,
                    "end_log_line": "",
                    "start_global_step": last_global_step,
                    "end_global_step": last_global_step,
                    "agent_messages": 0,
                    "command_executions": 0,
                    "file_change_events": 0,
                    "verification_events": 0,
                    "verification_passed": 0,
                    "verification_failed": 0,
                    "verification_skipped": 0,
                    "action_count": 0,
                    "action_global_steps": [],
                }
                continue

            if event_type == "turn.completed":
                usage = event.get("usage") or {}
                row = turns[turn]
                row["end_log_line"] = log_line
                row["end_global_step"] = last_global_step
                for key in previous_usage:
                    cumulative = int(usage.get(key, 0))
                    row[key + "_cumulative"] = cumulative
                    row[key + "_increment"] = cumulative - previous_usage[key]
                    previous_usage[key] = cumulative
                continue

            item = event.get("item") or {}
            if event_type != "item.completed":
                continue
            item_type = item.get("type")

            if item_type == "agent_message":
                text = item.get("text", "")
                last_agent_message = text
                turns[turn]["agent_messages"] += 1
                messages.append(
                    {
                        "log_line": log_line,
                        "model_turn": turn,
                        "global_step_after_prior_action": last_global_step,
                        "message_kind": message_kind(text),
                        "text": text,
                    }
                )
                continue

            if item_type == "file_change":
                paths = [change.get("path", "").rsplit("/", 1)[-1] for change in item.get("changes", [])]
                turns[turn]["file_change_events"] += 1
                changes.append(
                    {
                        "log_line": log_line,
                        "model_turn": turn,
                        "global_step_after_prior_action": last_global_step,
                        "files": ";".join(paths),
                        "preceding_model_message": last_agent_message,
                    }
                )
                continue

            if item_type != "command_execution":
                continue

            turns[turn]["command_executions"] += 1
            command = item.get("command", "")
            output = item.get("aggregated_output", "")
            status = item.get("status", "")

            is_verification = "verify_world_model.py" in command or "verify_main_planner.py" in command
            if is_verification:
                verifier = "world_model" if "verify_world_model.py" in command else "main_planner"
                segments = verifier_output_segments(command, output, verifier)
                for invocation_index, (trigger_step, segment) in enumerate(segments, 1):
                    outcome = verification_outcome(status, segment)
                    mismatch = re.search(r"replay, step (\d+)", segment)
                    verified_range = re.search(r"levels? (\d+)(?:\.\.(\d+))? verification passed", segment)
                    turns[turn]["verification_events"] += 1
                    turns[turn]["verification_" + outcome] = turns[turn].get("verification_" + outcome, 0) + 1
                    verifications.append(
                        {
                            "log_line": log_line,
                            "model_turn": turn,
                            "global_step_context": trigger_step if trigger_step is not None else last_global_step,
                            "verifier": verifier,
                            "invocation_index_within_command": invocation_index,
                            "invocations_in_command": len(segments),
                            "outcome": outcome,
                            "failure_class": verification_failure_class(verifier, outcome, segment),
                            "exit_status": status,
                            "mismatch_replay_step": mismatch.group(1) if mismatch else "",
                            "verified_from_level": verified_range.group(1) if verified_range else "",
                            "verified_to_level": (verified_range.group(2) or verified_range.group(1)) if verified_range else "",
                            "output_summary": segment.strip().replace("\n", " | ")[:1000],
                            "preceding_model_message": last_agent_message,
                        }
                    )

            steps = [int(value) for value in ACTION_STEP_RE.findall(output)]
            has_action_output = bool(ACTION_OUTPUT_RE.search(output))
            is_real_action_command = steps and has_action_output and "client.py" in command and "move" in command
            if not is_real_action_command:
                continue

            batch_size = len(steps)
            for batch_index, step in enumerate(steps, 1):
                if step in action_events:
                    raise AssertionError(f"duplicate real-action log mapping for global step {step}")
                action_events[step] = {
                    "model_turn": turn,
                    "action_command_log_line": log_line,
                    "execution_mode": command_mode(command, batch_size),
                    "batch_action_index": batch_index,
                    "batch_size": batch_size,
                    "preceding_model_message": last_agent_message,
                }
                turns[turn]["action_count"] += 1
                turns[turn]["action_global_steps"].append(step)
            last_global_step = max(last_global_step, max(steps))
            turns[turn]["end_global_step"] = last_global_step

    for row in turns.values():
        steps = row.pop("action_global_steps")
        row["first_action_global_step"] = min(steps) if steps else ""
        row["last_action_global_step"] = max(steps) if steps else ""
        row["action_step_ranges"] = compress_ranges(steps)
    return list(turns.values()), messages, changes, verifications, action_events


def compress_ranges(values: list[int]) -> str:
    if not values:
        return ""
    ordered = sorted(set(values))
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(ranges)


def read_frame(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frame_diff(previous: Path, current: Path) -> tuple[int, int, int]:
    before = read_frame(previous)
    after = read_frame(current)
    if len(before) != len(after) or any(len(a) != len(b) for a, b in zip(before, after)):
        raise AssertionError(f"incompatible frames: {previous} and {current}")
    playfield = bottom = 0
    for y, (before_row, after_row) in enumerate(zip(before, after)):
        changed = sum(a != b for a, b in zip(before_row, after_row))
        if y == len(before) - 1:
            bottom += changed
        else:
            playfield += changed
    return playfield + bottom, playfield, bottom


def parse_actions(action_events: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    previous_levels_completed = 0

    for metadata_path in sorted(SESSION.glob("level_*_attempt_*/step_*_metadata.json")):
        attempt_match = ATTEMPT_RE.match(metadata_path.parent.name)
        local_match = LOCAL_STEP_RE.match(metadata_path.name)
        if not attempt_match or not local_match:
            continue
        level, attempt = map(int, attempt_match.groups())
        local_step = int(local_match.group(1))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        global_step = int(metadata["step_index"])
        if global_step in seen_steps:
            raise AssertionError(f"duplicate objective step metadata {global_step}")
        seen_steps.add(global_step)

        current_frame = metadata_path.with_name(metadata_path.name.replace("_metadata.json", "_final.txt"))
        previous_frame = (
            metadata_path.parent / "initial_frame.txt"
            if local_step == 1
            else metadata_path.parent / f"step_{local_step - 1:04d}_final.txt"
        )
        changed, playfield_changed, bottom_changed = frame_diff(previous_frame, current_frame)
        intermediate_count = len(list(metadata_path.parent.glob(f"step_{local_step:04d}_intermediate_*.txt")))
        action_input = metadata["action_input"]
        action_data = action_input.get("data") or {}
        action_name = action_input["id"]
        action = action_name
        if action_data:
            action += "(" + ",".join(f"{key}={action_data[key]}" for key in sorted(action_data)) + ")"
        completed_after = int(metadata["levels_completed"])
        level_completed = completed_after > previous_levels_completed
        previous_levels_completed = max(previous_levels_completed, completed_after)

        log_mapping = action_events[global_step]
        row = {
            "global_step": global_step,
            "level": level,
            "attempt": attempt,
            "level_action_index": local_step,
            "action": action,
            "action_name": action_name,
            "action_x": action_data.get("x", ""),
            "action_y": action_data.get("y", ""),
            "state_after": metadata["state"],
            "levels_completed_after": completed_after,
            "level_completed_by_action": level_completed,
            "frame_change_pixels": changed,
            "playfield_change_pixels": playfield_changed,
            "bottom_clock_change_pixels": bottom_changed,
            "intermediate_frame_count": intermediate_count,
            "human_baseline_for_level": LEVEL_HUMAN_BASELINES[level],
            **log_mapping,
        }
        actions.append(row)

    actions.sort(key=lambda row: row["global_step"])
    expected = list(range(1, 398))
    actual = [int(row["global_step"]) for row in actions]
    if actual != expected:
        raise AssertionError(f"objective action coverage mismatch: got {len(actual)} rows")
    if set(action_events) != set(expected):
        raise AssertionError("agent.log action mapping does not cover exactly steps 1..397")
    return actions


def _target_coverage_count(state: dict[str, Any]) -> int:
    def signature(item: dict[str, Any]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
        shape = tuple(sorted(tuple(offset) for offset in item["shape"]))
        return shape, tuple(item["pos"])

    entities = Counter(signature(entity) for entity in state.get("entities", ()))
    covered = 0
    for target in state.get("targets", ()):
        target_signature = signature(target)
        if entities[target_signature] > 0:
            entities[target_signature] -= 1
            covered += 1
    return covered


def _action_intent(global_step: int) -> str:
    exploratory_probes = {1, 4, 6, 13, 17, 26, 27, 28, 39, 40, 190, 198, 215, 340}
    planned_setup_ranges = (
        range(5, 6),
        range(7, 13),
        range(14, 17),
        range(18, 26),
        range(29, 39),
        range(191, 198),
        range(216, 233),
    )
    stale_ranges = (range(147, 166), range(269, 277), range(341, 346))
    corrected_ranges = (
        range(166, 190),
        range(212, 215),
        range(277, 302),
        range(346, 398),
    )
    if global_step in exploratory_probes:
        return "exploratory_probe"
    if any(global_step in interval for interval in planned_setup_ranges):
        return "planned_setup_for_probe"
    if any(global_step in interval for interval in stale_ranges):
        return "planned_but_no_longer_validated"
    if any(global_step in interval for interval in corrected_ranges):
        return "planned_route_after_correction"
    return "planned_route"


EVIDENTIARY_OUTCOMES = {
    3: "counterexample: 5 was shoveable, not a finish tile",
    6: "counterexample: F blocked voluntary movement but not forced movement",
    13: "negative endpoint probe: right boundary did not terminate",
    26: "negative endpoint probe: left boundary did not terminate",
    27: "negative endpoint probe: bottom boundary did not terminate",
    40: "positive target probe: piece entered matching outline",
    42: "positive passability observation: framed piece entered 4 outline border",
    44: "level completion established distinct matching-target coverage",
    48: "visual-only mismatch guard: bottom-row clock disagreed",
    58: "dynamics mismatch guard: long field-slide endpoint disagreed",
    79: "visual-only mismatch guard: whole-side contact rendering disagreed",
    80: "dynamics mismatch guard: vertical shove endpoint disagreed",
    146: "counterexample: shove continued while footprint remained on F",
    190: "counterexample: C/D phase component was not directly selectable",
    194: "positive phase observation: phase wrap displaced active piece",
    198: "positive collision observation: phase piece was shoveable",
    210: "counterexample: target-side C/D component was a second phase emitter",
    211: "terminal counterexample: target slot was at x=16, not x=15",
    220: "positive phase observation: upward wave crossed the F band",
    232: "counterexample: being carried onto F did not permit voluntary F crossing",
    268: "counterexample: phase-wave landing was blocked by hard terrain",
    339: "positive prospective test: wave moved a non-active solid piece",
    340: "counterexample: solid plus piece was not directly selectable",
    397: "game WIN",
}


def _semantic_effect(
    action_name: str,
    active_before: int,
    active_after: int,
    moved_indices: list[int],
    collision_indices: list[int],
    active_displacement: int,
    phase_changed: bool,
    terminal: bool,
) -> str:
    if action_name == "ACTION6":
        effect = "selection" if active_before != active_after else "coordinate_noop"
    else:
        other_moved = any(index != active_before for index in moved_indices)
        if collision_indices and other_moved:
            effect = "shove"
            if active_displacement or len([index for index in moved_indices if index != active_before]) > 1:
                effect += "_and_phase_wave"
        elif other_moved:
            effect = "phase_wave_moved_other_piece"
        elif active_displacement > 1:
            effect = "phase_wave_moved_active_piece"
        elif active_displacement == 1:
            effect = "active_move"
        elif phase_changed:
            effect = "phase_advance_without_position_change"
        else:
            effect = "directional_noop"
    return effect + ("+level_completed" if terminal else "")


def enrich_with_final_model(actions: list[dict[str, Any]]) -> dict[str, int]:
    """Replay all actions through the final archived model without writing there."""
    sys.dont_write_bytecode = True
    model_root = RUN / "agent_run"
    sys.path.insert(0, str(model_root))

    import numpy as np
    from game_status import LEVEL_COMPLETED, RUNNING  # type: ignore[import-not-found]
    from world_model_engine import world_model_engine  # type: ignore[import-not-found]
    from world_model_state_io import initial_state_reconstruction, state_renderer  # type: ignore[import-not-found]

    frame_matches = 0
    comparable_frames = 0
    predicted_terminal_steps = 0

    rows_by_level: dict[int, list[dict[str, Any]]] = {}
    for row in actions:
        rows_by_level.setdefault(int(row["level"]), []).append(row)

    for level, level_rows in sorted(rows_by_level.items()):
        attempt_dir = SESSION / f"level_{level:02d}_attempt_01"
        initial_array = np.array(
            [[int(char, 16) for char in line] for line in read_frame(attempt_dir / "initial_frame.txt")],
            dtype=np.int16,
        )
        state = initial_state_reconstruction(level, initial_array)

        for row in level_rows:
            action: dict[str, Any] = {"name": row["action_name"]}
            if row["action_x"] != "":
                action["x"] = int(row["action_x"])
            if row["action_y"] != "":
                action["y"] = int(row["action_y"])

            entities_before = tuple(state.get("entities", ()))
            positions_before = [tuple(entity["pos"]) for entity in entities_before]
            active_before = int(state.get("active_index", 0))
            phase_before = state.get("phase", "")
            coverage_before = _target_coverage_count(state)

            next_state, predicted_status = world_model_engine(state, action)
            entities_after = tuple(next_state.get("entities", ()))
            positions_after = [tuple(entity["pos"]) for entity in entities_after]
            active_after = int(next_state.get("active_index", 0))
            phase_after = next_state.get("phase", "")
            coverage_after = _target_coverage_count(next_state)
            moved_indices = [
                index
                for index, (before, after) in enumerate(zip(positions_before, positions_after))
                if before != after
            ]
            active_before_pos = positions_before[active_before]
            active_after_pos = positions_after[active_after]
            active_displacement = (
                abs(active_after_pos[0] - active_before_pos[0]) + abs(active_after_pos[1] - active_before_pos[1])
                if active_before == active_after
                else 0
            )
            collision_indices: list[int] = []
            direction_by_action = {
                "ACTION1": (0, -1),
                "ACTION2": (0, 1),
                "ACTION3": (-1, 0),
                "ACTION4": (1, 0),
            }
            if action["name"] in direction_by_action:
                dx, dy = direction_by_action[action["name"]]
                active_entity = entities_before[active_before]
                active_shape = tuple(tuple(offset) for offset in active_entity["shape"])
                candidate_footprint = {
                    (active_before_pos[0] + dx + ox, active_before_pos[1] + dy + oy)
                    for ox, oy in active_shape
                }
                for index, entity in enumerate(entities_before):
                    if index == active_before:
                        continue
                    ex, ey = tuple(entity["pos"])
                    footprint = {(ex + ox, ey + oy) for ox, oy in entity["shape"]}
                    if candidate_footprint & footprint:
                        collision_indices.append(index)
            terminal = predicted_status == LEVEL_COMPLETED
            if terminal:
                predicted_terminal_steps += 1

            if predicted_status == RUNNING:
                comparable_frames += 1
                local_step = int(row["level_action_index"])
                real_frame = np.array(
                    [[int(char, 16) for char in line] for line in read_frame(attempt_dir / f"step_{local_step:04d}_final.txt")],
                    dtype=np.int16,
                )
                frame_match: bool | str = bool(np.array_equal(state_renderer(next_state), real_frame))
                if frame_match:
                    frame_matches += 1
            else:
                frame_match = "terminal_status_only"

            row.update(
                {
                    "action_intent": _action_intent(int(row["global_step"])),
                    "evidentiary_outcome": EVIDENTIARY_OUTCOMES.get(int(row["global_step"]), ""),
                    "final_model_predicted_status": predicted_status,
                    "final_model_frame_matches": frame_match,
                    "semantic_effect": _semantic_effect(
                        str(row["action_name"]),
                        active_before,
                        active_after,
                        moved_indices,
                        collision_indices,
                        active_displacement,
                        phase_before != phase_after,
                        terminal,
                    ),
                    "active_entity_before": active_before,
                    "active_entity_after": active_after,
                    "active_position_before": json.dumps(positions_before[active_before], separators=(",", ":")),
                    "active_position_after": json.dumps(positions_after[active_after], separators=(",", ":")),
                    "moved_entity_indices": json.dumps(moved_indices, separators=(",", ":")),
                    "collision_entity_indices_before": json.dumps(collision_indices, separators=(",", ":")),
                    "active_manhattan_displacement": active_displacement,
                    "all_entity_positions_after": json.dumps(positions_after, separators=(",", ":")),
                    "phase_before": phase_before,
                    "phase_after": phase_after,
                    "target_coverage_before": coverage_before,
                    "target_coverage_after": coverage_after,
                    "target_count": len(next_state.get("targets", ())),
                }
            )
            state = next_state

        if level_rows[-1]["final_model_predicted_status"] != LEVEL_COMPLETED:
            raise AssertionError(f"final model did not predict completion for level {level}")

    if frame_matches != comparable_frames:
        raise AssertionError(f"final-model replay matched {frame_matches}/{comparable_frames} comparable frames")
    if predicted_terminal_steps != len(rows_by_level):
        raise AssertionError(f"expected one predicted terminal action per level, got {predicted_terminal_steps}")
    return {
        "comparable_nonterminal_frames": comparable_frames,
        "matching_nonterminal_frames": frame_matches,
        "predicted_terminal_steps": predicted_terminal_steps,
    }


def main() -> None:
    turns, messages, changes, verifications, action_events = parse_log()
    actions = parse_actions(action_events)
    replay_summary = enrich_with_final_model(actions)

    turn_fields = [
        "model_turn", "start_log_line", "end_log_line", "start_global_step", "end_global_step",
        "agent_messages", "command_executions", "file_change_events", "verification_events",
        "verification_passed", "verification_failed", "verification_skipped", "verification_timeout",
        "action_count", "first_action_global_step", "last_action_global_step", "action_step_ranges",
        "input_tokens_increment", "cached_input_tokens_increment", "output_tokens_increment",
        "reasoning_output_tokens_increment", "input_tokens_cumulative", "cached_input_tokens_cumulative",
        "output_tokens_cumulative", "reasoning_output_tokens_cumulative",
    ]
    message_fields = ["log_line", "model_turn", "global_step_after_prior_action", "message_kind", "text"]
    change_fields = ["log_line", "model_turn", "global_step_after_prior_action", "files", "preceding_model_message"]
    verification_fields = [
        "log_line", "model_turn", "global_step_context", "verifier", "invocation_index_within_command",
        "invocations_in_command", "outcome", "failure_class", "exit_status", "mismatch_replay_step",
        "verified_from_level", "verified_to_level", "output_summary", "preceding_model_message",
    ]
    action_fields = [
        "global_step", "level", "attempt", "level_action_index", "action", "action_name", "action_x", "action_y",
        "state_after", "levels_completed_after", "level_completed_by_action", "frame_change_pixels",
        "playfield_change_pixels", "bottom_clock_change_pixels", "intermediate_frame_count",
        "human_baseline_for_level", "model_turn", "action_command_log_line", "execution_mode",
        "batch_action_index", "batch_size", "preceding_model_message", "final_model_predicted_status",
        "action_intent", "evidentiary_outcome",
        "final_model_frame_matches", "semantic_effect", "active_entity_before", "active_entity_after",
        "active_position_before", "active_position_after", "moved_entity_indices", "all_entity_positions_after",
        "collision_entity_indices_before", "active_manhattan_displacement",
        "phase_before", "phase_after", "target_coverage_before", "target_coverage_after", "target_count",
    ]

    write_csv(OUT / "ka59-turns.csv", turns, turn_fields)
    write_csv(OUT / "ka59-model-messages.csv", messages, message_fields)
    write_csv(OUT / "ka59-file-changes.csv", changes, change_fields)
    write_csv(OUT / "ka59-verification-events.csv", verifications, verification_fields)
    write_csv(OUT / "ka59-actions.csv", actions, action_fields)

    print(
        json.dumps(
            {
                "turns": len(turns),
                "model_messages": len(messages),
                "file_changes": len(changes),
                "verifications": len(verifications),
                "actions": len(actions),
                "action_counts_by_level": dict(Counter(row["level"] for row in actions)),
                "action_counts_by_turn": dict(Counter(row["model_turn"] for row in actions)),
                "final_model_replay": replay_summary,
                "outputs": [
                    str(OUT / "ka59-turns.csv"),
                    str(OUT / "ka59-model-messages.csv"),
                    str(OUT / "ka59-file-changes.csv"),
                    str(OUT / "ka59-verification-events.csv"),
                    str(OUT / "ka59-actions.csv"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
