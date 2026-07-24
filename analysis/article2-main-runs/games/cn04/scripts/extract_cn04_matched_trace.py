#!/usr/bin/env python3
"""Extract matched cn04 trace evidence without modifying archived runs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA = Path("/home/hipra/datasets/article2_main_runs/runs")
ANALYSIS_ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data"
ARMS = {
    "textual": DATA / "twma_v1.2_gpt-5.4_xhigh_run1" / "cn04",
    "executable": DATA / "ewma_v1.2_sv_gpt-5.4_xhigh_run1" / "cn04",
}
ATTEMPT_RE = re.compile(r"level_(\d+)_attempt_(\d+)$")
STEP_RE = re.compile(r"step_(\d+)_metadata\.json$")
OUTPUT_STEP_RE = re.compile(r"(?m)^step:\s*(\d+)\s*$")
OUTPUT_ACTION_RE = re.compile(r"(?m)^action:\s*([^\n]+?)\s*$")


@dataclass
class Attempt:
    arm: str
    level: int
    attempt: int
    path: Path
    start_global: int
    end_global: int
    local_actions: int
    start_levels_completed: int
    end_levels_completed: int
    end_state: str
    reset_before: int

    @property
    def successful(self) -> bool:
        return self.end_levels_completed > self.start_levels_completed


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def attempts_for(arm: str, run_root: Path) -> list[Attempt]:
    session = run_root / "agent_run/client/session"
    raw: list[dict[str, Any]] = []
    for path in session.iterdir():
        if not path.is_dir():
            continue
        match = ATTEMPT_RE.fullmatch(path.name)
        if not match:
            continue
        level, attempt = map(int, match.groups())
        initial = load_json(path / "initial_metadata.json")
        step_paths = sorted(path.glob("step_*_metadata.json"))
        if not step_paths:
            final = initial
            local_actions = 0
        else:
            final = load_json(step_paths[-1])
            step_match = STEP_RE.fullmatch(step_paths[-1].name)
            if step_match is None:
                raise AssertionError(f"unexpected metadata filename: {step_paths[-1].name}")
            local_actions = int(step_match.group(1))
        raw.append(
            {
                "level": level,
                "attempt": attempt,
                "path": path,
                "start_global": int(initial["step_index"]),
                "end_global": int(final["step_index"]),
                "local_actions": local_actions,
                "start_levels_completed": int(initial["levels_completed"]),
                "end_levels_completed": int(final["levels_completed"]),
                "end_state": str(final["state"]),
            }
        )
    raw.sort(key=lambda row: (row["start_global"], row["level"], row["attempt"]))
    attempts: list[Attempt] = []
    previous_end = 0
    for index, row in enumerate(raw):
        reset_before = int(index > 0 and row["start_global"] == previous_end + 1)
        attempts.append(Attempt(arm=arm, reset_before=reset_before, **row))
        previous_end = row["end_global"]
    return attempts


def read_frame(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").splitlines()


def frame_delta(previous: list[str] | None, current: list[str] | None) -> dict[str, Any]:
    if previous is None or current is None:
        return {
            "frame_changed": "",
            "changed_cells": "",
            "background_to_object": "",
            "object_to_background": "",
            "color_count_delta": "",
        }
    if len(previous) != len(current) or any(len(a) != len(b) for a, b in zip(previous, current)):
        raise AssertionError("frame dimensions changed")
    changed = 0
    added = 0
    deleted = 0
    before_counts = Counter("".join(previous))
    after_counts = Counter("".join(current))
    for before_row, after_row in zip(previous, current):
        for before, after in zip(before_row, after_row):
            if before == after:
                continue
            changed += 1
            added += int(before == "9" and after != "9")
            deleted += int(before != "9" and after == "9")
    delta = {
        key: after_counts[key] - before_counts[key]
        for key in sorted(set(before_counts) | set(after_counts))
        if after_counts[key] != before_counts[key]
    }
    return {
        "frame_changed": changed > 0,
        "changed_cells": changed,
        "background_to_object": added,
        "object_to_background": deleted,
        "color_count_delta": json.dumps(delta, sort_keys=True, separators=(",", ":")),
    }


def action_id(action: Any) -> tuple[str, str, str]:
    if isinstance(action, str):
        return action, "", ""
    data = action.get("data") or {}
    return str(action.get("id", "")), str(data.get("x", "")), str(data.get("y", ""))


def parse_log(run_root: Path) -> dict[str, Any]:
    turns: dict[int, dict[str, Any]] = {}
    messages: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    step_provenance: dict[int, dict[str, Any]] = {}
    inspections: list[dict[str, Any]] = []
    turn = 0
    last_message = ""
    previous_usage = Counter()
    last_step = 0

    with (run_root / "agent.log").open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "turn.started":
                turn += 1
                turns[turn] = {
                    "turn": turn,
                    "start_log_line": line_no,
                    "end_log_line": "",
                    "start_global_step": last_step,
                    "end_global_step": last_step,
                    "agent_messages": 0,
                    "command_executions": 0,
                    "file_change_events": 0,
                    "verification_events": 0,
                    "action_count": 0,
                    "action_steps": [],
                }
                continue
            if event_type == "turn.completed":
                usage = event.get("usage") or {}
                row = turns[turn]
                row["end_log_line"] = line_no
                row["end_global_step"] = last_step
                for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
                    cumulative = int(usage.get(key, 0))
                    row[f"{key}_cumulative"] = cumulative
                    row[f"{key}_increment"] = cumulative - int(previous_usage[key])
                    previous_usage[key] = cumulative
                continue
            if event_type == "external_agent_log" and event.get("message") == "iteration inspection":
                inspections.append(
                    {
                        "log_line": line_no,
                        "iteration_id": event.get("iteration_id"),
                        "global_step": event.get("n_steps_total"),
                        "level": event.get("current_level_index"),
                        "current_level_steps": event.get("n_steps_current_level"),
                        "current_attempt_steps": event.get("n_steps_current_attempt"),
                        "game_over_attempts": event.get("n_game_over_attempts_current_level"),
                        "solved": event.get("is_solved"),
                        "game_over": event.get("is_game_over"),
                        "level_completed": event.get("is_level_completed"),
                    }
                )
                continue
            if event_type != "item.completed":
                continue
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "agent_message":
                text = str(item.get("text", ""))
                last_message = text
                turns[turn]["agent_messages"] += 1
                messages.append(
                    {
                        "log_line": line_no,
                        "turn": turn,
                        "global_step_after_prior_action": last_step,
                        "text": text,
                    }
                )
                continue
            if item_type == "file_change":
                turns[turn]["file_change_events"] += 1
                for change in item.get("changes", []):
                    changes.append(
                        {
                            "log_line": line_no,
                            "turn": turn,
                            "global_step_after_prior_action": last_step,
                            "kind": change.get("kind", ""),
                            "path": Path(change.get("path", "")).name,
                            "preceding_agent_message": last_message,
                        }
                    )
                continue
            if item_type != "command_execution":
                continue
            turns[turn]["command_executions"] += 1
            command = str(item.get("command", ""))
            output = str(item.get("aggregated_output", ""))
            status = str(item.get("status", ""))
            exit_code = item.get("exit_code")

            world_model_invocation = re.search(
                r"\bpython3(?:\s+-u)?\s+verify_world_model\.py\b", command
            )
            main_planner_invocation = re.search(
                r"\bpython3(?:\s+-u)?\s+verify_main_planner\.py\b", command
            )
            if world_model_invocation or main_planner_invocation:
                verifier = "world_model" if world_model_invocation else "main_planner"
                lower = output.lower()
                outcome = (
                    "failed"
                    if status == "failed" or "verification failed" in lower or "traceback" in lower
                    else "passed"
                    if "verification passed" in lower or re.search(r"levels? \d+\.\.\d+ .*passed", lower)
                    else "skipped"
                    if "we skip" in lower and "failed" not in lower
                    else "other"
                )
                mismatch = re.search(r"(initial state|step \d+): ([^\n]+)", output)
                verifications.append(
                    {
                        "log_line": line_no,
                        "turn": turn,
                        "global_step_context": last_step,
                        "verifier": verifier,
                        "outcome": outcome,
                        "status": status,
                        "exit_code": exit_code,
                        "mismatch": " | ".join(mismatch.groups()) if mismatch else "",
                        "output_summary": output.strip().replace("\n", " | ")[:1500],
                        "preceding_agent_message": last_message,
                    }
                )
                turns[turn]["verification_events"] += 1

            if " status" in command or "Last step information:" in output:
                continue
            steps = [int(value) for value in OUTPUT_STEP_RE.findall(output)]
            actions = [value.strip() for value in OUTPUT_ACTION_RE.findall(output)]
            states = re.findall(r"(?m)^state:\s*([^\n]+?)\s*$", output)
            levels = re.findall(r"(?m)^level:\s*([^\n]+?)\s*$", output)
            if not steps or len(steps) != len(states) or len(steps) != len(levels):
                continue
            if actions and len(actions) != len(steps):
                actions = []
            for batch_index, step in enumerate(steps, 1):
                logged_action = actions[batch_index - 1] if actions else ""
                existing = step_provenance.get(step)
                candidate = {
                    "turn": turn,
                    "log_line": line_no,
                    "batch_index": batch_index,
                    "batch_size": len(steps),
                    "logged_action": logged_action,
                    "preceding_agent_message": last_message,
                }
                if existing and existing != candidate:
                    raise AssertionError(f"duplicate action provenance for global step {step}")
                step_provenance[step] = candidate
                turns[turn]["action_count"] += 1
                turns[turn]["action_steps"].append(step)
            last_step = max(last_step, max(steps))
            turns[turn]["end_global_step"] = last_step

    for row in turns.values():
        steps = sorted(row.pop("action_steps"))
        row["first_action_step"] = steps[0] if steps else ""
        row["last_action_step"] = steps[-1] if steps else ""
    return {
        "turns": list(turns.values()),
        "messages": messages,
        "changes": changes,
        "verifications": verifications,
        "step_provenance": step_provenance,
        "inspections": inspections,
    }


def checkpoint_rows(arm: str, repo: Path) -> list[dict[str, Any]]:
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--reverse", "--format=%H%x09%aI%x09%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    rows = []
    for entry in log:
        commit, timestamp, subject = entry.split("\t", 2)
        status = subprocess.run(
            ["git", "-C", str(repo), "diff-tree", "--root", "--no-commit-id", "--name-status", "-r", commit],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().replace("\n", ";")
        numstat = subprocess.run(
            ["git", "-C", str(repo), "show", "--format=", "--numstat", commit],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().replace("\n", ";")
        step_match = re.search(r"(?:level_\d+|level \d+)\s+(\d+)", subject)
        rows.append(
            {
                "arm": arm,
                "commit": commit,
                "timestamp": timestamp,
                "subject": subject,
                "advertised_global_step": step_match.group(1) if step_match else "",
                "name_status": status,
                "numstat": numstat,
            }
        )
    return rows


def generated_artifact_metrics(
    repo: Path, arm: str, file_changes: list[dict[str, Any]]
) -> dict[str, int]:
    """Measure only files the model actually changed, not provided scaffold code."""
    touched = {str(row["path"]) for row in file_changes if row["arm"] == arm}
    surviving_lines = sum(
        len((repo / relative).read_text(encoding="utf-8").splitlines())
        for relative in touched
        if (repo / relative).is_file()
    )
    additions = 0
    deletions = 0
    commits = subprocess.run(
        ["git", "-C", str(repo), "log", "--reverse", "--format=%H"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for commit in commits:
        numstat = subprocess.run(
            ["git", "-C", str(repo), "show", "--format=", "--numstat", commit],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in numstat.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3 or parts[2] not in touched:
                continue
            if parts[0].isdigit() and parts[1].isdigit():
                additions += int(parts[0])
                deletions += int(parts[1])
    return {
        "touched_artifacts": len(touched),
        "surviving_artifact_lines": surviving_lines,
        "artifact_line_additions": additions,
        "artifact_line_deletions": deletions,
    }


def action_rows(
    arm: str,
    game_root: Path,
    attempts: list[Attempt],
    provenance: dict[int, dict[str, Any]],
    inspections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    archived_actions = json.loads((game_root.parent / "cn04_actions.json").read_text(encoding="utf-8"))
    terminal_global = max(attempt.end_global for attempt in attempts)
    if len(archived_actions) != terminal_global:
        raise AssertionError(f"{arm}: action list {len(archived_actions)} != terminal global {terminal_global}")
    rows: list[dict[str, Any]] = []
    previous_final_by_attempt: dict[tuple[int, int], list[str] | None] = {}
    previous_terminal_frame: list[str] | None = None
    for global_step, raw_action in enumerate(archived_actions, 1):
        name, click_x, click_y = action_id(raw_action)
        if name == "RESET":
            attempt = next((a for a in attempts if a.start_global == global_step and a.reset_before), None)
            if attempt is None:
                raise AssertionError(f"{arm}: no attempt starts at reset {global_step}")
            metadata_path = attempt.path / "initial_metadata.json"
            frame_path = attempt.path / "initial_frame.txt"
            current_frame = read_frame(frame_path)
            previous_frame = previous_terminal_frame
            local_step: int | str = 0
            metadata = load_json(metadata_path)
        else:
            attempt = next((a for a in attempts if a.start_global < global_step <= a.end_global), None)
            if attempt is None:
                raise AssertionError(f"{arm}: no attempt owns action {global_step}")
            local_step = global_step - attempt.start_global
            metadata_path = attempt.path / f"step_{local_step:04d}_metadata.json"
            frame_path = attempt.path / f"step_{local_step:04d}_final.txt"
            metadata = load_json(metadata_path)
            current_frame = read_frame(frame_path)
            key = (attempt.level, attempt.attempt)
            previous_frame = previous_final_by_attempt.get(key)
            if previous_frame is None:
                previous_frame = read_frame(attempt.path / "initial_frame.txt")
            previous_final_by_attempt[key] = current_frame
            if global_step == attempt.end_global:
                previous_terminal_frame = current_frame
        delta = frame_delta(previous_frame, current_frame)
        prov = provenance.get(global_step, {})
        prior_inspection_step = 0
        owning_inspection: dict[str, Any] = {}
        for inspection in inspections:
            inspection_step = int(inspection.get("global_step") or 0)
            if prior_inspection_step < global_step <= inspection_step:
                owning_inspection = inspection
                break
            prior_inspection_step = inspection_step
        row = {
            "arm": arm,
            "global_step": global_step,
            "level": attempt.level,
            "attempt": attempt.attempt,
            "local_step": local_step,
            "action": name,
            "click_x": click_x,
            "click_y": click_y,
            "metadata_state": metadata.get("state", ""),
            "levels_completed": metadata.get("levels_completed", ""),
            "successful_attempt": attempt.successful,
            "metadata_path": str(metadata_path),
            "frame_path": str(frame_path) if frame_path.exists() else "",
            "frame_sha256": hashlib.sha256("\n".join(current_frame or []).encode()).hexdigest() if current_frame else "",
            "turn": prov.get("turn", ""),
            "action_log_line": prov.get("log_line", ""),
            "batch_index": prov.get("batch_index", ""),
            "batch_size": prov.get("batch_size", ""),
            "preceding_agent_message": prov.get("preceding_agent_message", ""),
            "supervisor_iteration": owning_inspection.get("iteration_id", ""),
            "supervisor_iteration_end_step": owning_inspection.get("global_step", ""),
            **delta,
        }
        rows.append(row)
    return rows


def main() -> None:
    all_attempts: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    all_turns: list[dict[str, Any]] = []
    all_messages: list[dict[str, Any]] = []
    all_changes: list[dict[str, Any]] = []
    all_verifications: list[dict[str, Any]] = []
    all_inspections: list[dict[str, Any]] = []
    all_checkpoints: list[dict[str, Any]] = []

    for arm, game_root in ARMS.items():
        run_root = game_root / "run"
        attempts = attempts_for(arm, run_root)
        parsed = parse_log(run_root)
        actions = action_rows(
            arm,
            game_root,
            attempts,
            parsed["step_provenance"],
            parsed["inspections"],
        )
        scorecard = load_json(game_root.parent / "cn04_scorecard.json")["environments"][0]["runs"][0]
        expected_level_actions = Counter()
        for row in actions:
            expected_level_actions[int(row["level"])] += 1
        score_vector = [expected_level_actions[level] for level in range(1, 7)]
        if score_vector != [int(value) for value in scorecard["level_actions"]]:
            if arm != "textual" or score_vector != [16, 236, 31, 34, 77, 249]:
                raise AssertionError(f"{arm}: reconstructed {score_vector} != scorecard {scorecard['level_actions']}")

        for attempt in attempts:
            all_attempts.append(
                {
                    "arm": arm,
                    "level": attempt.level,
                    "attempt": attempt.attempt,
                    "start_global": attempt.start_global,
                    "end_global": attempt.end_global,
                    "local_actions": attempt.local_actions,
                    "reset_before": attempt.reset_before,
                    "successful": attempt.successful,
                    "end_state": attempt.end_state,
                    "start_levels_completed": attempt.start_levels_completed,
                    "end_levels_completed": attempt.end_levels_completed,
                    "path": str(attempt.path),
                }
            )
        for collection_name, target in (
            ("turns", all_turns),
            ("messages", all_messages),
            ("changes", all_changes),
            ("verifications", all_verifications),
            ("inspections", all_inspections),
        ):
            for row in parsed[collection_name]:
                target.append({"arm": arm, **row})
        all_actions.extend(actions)
        all_checkpoints.extend(checkpoint_rows(arm, run_root / "agent_run"))
        mapped = sum(bool(row["action_log_line"]) for row in actions)
        print(
            f"{arm}: attempts={len(attempts)} actions={len(actions)} "
            f"mapped_to_log={mapped} turns={len(parsed['turns'])} "
            f"changes={len(parsed['changes'])} verifications={len(parsed['verifications'])}"
        )

    write_csv(
        "cn04-attempts.csv",
        all_attempts,
        ["arm", "level", "attempt", "start_global", "end_global", "local_actions", "reset_before", "successful", "end_state", "start_levels_completed", "end_levels_completed", "path"],
    )
    write_csv(
        "cn04-actions.csv",
        all_actions,
        ["arm", "global_step", "level", "attempt", "local_step", "action", "click_x", "click_y", "metadata_state", "levels_completed", "successful_attempt", "frame_changed", "changed_cells", "background_to_object", "object_to_background", "color_count_delta", "metadata_path", "frame_path", "frame_sha256", "turn", "action_log_line", "batch_index", "batch_size", "preceding_agent_message", "supervisor_iteration", "supervisor_iteration_end_step"],
    )
    write_csv(
        "cn04-turns.csv",
        all_turns,
        ["arm", "turn", "start_log_line", "end_log_line", "start_global_step", "end_global_step", "first_action_step", "last_action_step", "action_count", "agent_messages", "command_executions", "file_change_events", "verification_events", "input_tokens_increment", "cached_input_tokens_increment", "output_tokens_increment", "reasoning_output_tokens_increment", "input_tokens_cumulative", "cached_input_tokens_cumulative", "output_tokens_cumulative", "reasoning_output_tokens_cumulative"],
    )
    write_csv(
        "cn04-model-messages.csv",
        all_messages,
        ["arm", "log_line", "turn", "global_step_after_prior_action", "text"],
    )
    write_csv(
        "cn04-file-changes.csv",
        all_changes,
        ["arm", "log_line", "turn", "global_step_after_prior_action", "kind", "path", "preceding_agent_message"],
    )
    write_csv(
        "cn04-verifications.csv",
        all_verifications,
        ["arm", "log_line", "turn", "global_step_context", "verifier", "outcome", "status", "exit_code", "mismatch", "output_summary", "preceding_agent_message"],
    )
    write_csv(
        "cn04-inspections.csv",
        all_inspections,
        ["arm", "log_line", "iteration_id", "global_step", "level", "current_level_steps", "current_attempt_steps", "game_over_attempts", "solved", "game_over", "level_completed"],
    )
    write_csv(
        "cn04-checkpoints.csv",
        all_checkpoints,
        ["arm", "commit", "timestamp", "subject", "advertised_global_step", "name_status", "numstat"],
    )

    outcome_rows = list(
        csv.DictReader(
            (ANALYSIS_ROOT / "shared/data/per_game_outcomes.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    summaries: list[dict[str, Any]] = []
    for arm, game_root in ARMS.items():
        run_actions = [row for row in all_actions if row["arm"] == arm]
        run_attempts = [row for row in all_attempts if row["arm"] == arm]
        run_changes = [row for row in all_changes if row["arm"] == arm]
        run_verifications = [row for row in all_verifications if row["arm"] == arm]
        run_turns = [row for row in all_turns if row["arm"] == arm]
        directory = game_root.parent.name
        scorecard = json.loads(
            (game_root.parent / "cn04_scorecard.json").read_text(encoding="utf-8")
        )["environments"][0]["runs"][0]
        outcome = next(
            row
            for row in outcome_rows
            if row["Directory"] == directory and row["Game"] == "cn04"
        )
        agent_parameters = json.loads(
            (game_root / "run/agent.log").read_text(encoding="utf-8").splitlines()[0]
        )
        last_turn = max(run_turns, key=lambda row: int(row["turn"]))
        action_kinds = Counter(str(row["action"]) for row in run_actions)
        level_actions = Counter(int(row["level"]) for row in run_actions)
        explicit_padding = sum(
            1
            for row in run_actions
            if arm == "executable"
            and int(row["level"]) == 3
            and row["action"] == "ACTION6"
            and (
                (row["click_x"], row["click_y"]) == ("63", "63")
                or (
                    int(row["attempt"]) == 3
                    and (row["click_x"], row["click_y"]) == ("17", "20")
                )
            )
        )
        summaries.append(
            {
                "arm": arm,
                "directory": directory,
                "model": agent_parameters["model"],
                "reasoning_effort": agent_parameters["reasoning_effort"],
                "state": outcome["State"],
                "levels_completed": outcome["LevelsCompleted"],
                "win_levels": outcome["WinLevels"],
                "rhae_percent": outcome["RHAEPercent"],
                "reconstructed_global_actions": len(run_actions),
                "reconstructed_global_level_actions": json.dumps(
                    [level_actions[level] for level in sorted(level_actions)],
                    separators=(",", ":"),
                ),
                "scorecard_actions": scorecard["actions"],
                "scorecard_level_actions": json.dumps(
                    scorecard["level_actions"], separators=(",", ":")
                ),
                "post_terminal_metadata_only_calls": sum(
                    row["action"] != "RESET"
                    and row["metadata_state"] in {"GAME_OVER", "WIN"}
                    and not row["frame_path"]
                    for row in run_actions
                ),
                "attempts": len(run_attempts),
                "game_over_attempts": sum(
                    row["end_state"] == "GAME_OVER" for row in run_attempts
                ),
                "resets": action_kinds["RESET"],
                "click_actions": action_kinds["ACTION6"],
                "unchanged_frame_actions": sum(
                    row["frame_changed"] is False for row in run_actions
                ),
                "explicit_level3_padding_actions": explicit_padding,
                "direct_turn_mapped_actions": sum(bool(row["turn"]) for row in run_actions),
                "supervisor_iteration_mapped_actions": sum(
                    bool(row["supervisor_iteration"]) for row in run_actions
                ),
                "file_change_events": len(run_changes),
                "file_add_events": sum(row["kind"] == "add" for row in run_changes),
                "file_update_events": sum(row["kind"] == "update" for row in run_changes),
                "file_delete_events": sum(row["kind"] == "delete" for row in run_changes),
                "verifier_invocations": len(run_verifications),
                "verifier_exit_zero": sum(
                    row["exit_code"] == 0 for row in run_verifications
                ),
                "planner_skip_unsolved_level3": sum(
                    "No completed level-3 attempt found" in str(row["output_summary"])
                    for row in run_verifications
                ),
                "input_tokens": last_turn["input_tokens_cumulative"],
                "output_tokens": last_turn["output_tokens_cumulative"],
                **generated_artifact_metrics(game_root / "run/agent_run", arm, all_changes),
            }
        )
    write_csv(
        "cn04-summary.csv",
        summaries,
        list(summaries[0]),
    )
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
