from __future__ import annotations

import json
from pathlib import Path

from inference.critic import replay
from inference.critic.packet import CriticPacket
from inference.utils.grid_utils import format_grid_ascii

SIZE = 6


def _board(fill: int, mark: tuple[int, int] | None = None) -> list[list[int]]:
    rows = [[fill] * SIZE for _ in range(SIZE)]
    if mark is not None:
        rows[mark[0]][mark[1]] = (fill + 1) % 16
    return rows


def _event(kind: str, board: list[list[int]], **fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "type": kind,
        "board": board,
        "board_ascii": format_grid_ascii(board),
        "score": 0,
        "state": "NOT_FINISHED",
        "level": 1,
        "run_status": "playing",
    }
    base.update(fields)
    return base


def _action(num: int, step: int, board: list[list[int]], *, changed: bool, level: int = 1, **flags: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "title": f"Action {num}",
        "action_num": num,
        "analysis_step": step,
        "action_name": "ACTION1",
        "action_display": "UP",
        "reward": 0.0,
        "board_changed": changed,
        "done": False,
        "level_completed": False,
        "game_over": False,
        "run_complete": False,
        "batch_index": 1,
        "batch_size": 1,
        "level": level,
    }
    fields.update(flags)
    return _event("action", board, **fields)


def _analysis(step: int, num: int, board: list[list[int]], *, level: int = 1, assistant: str = "", valid: str = "UP, DOWN") -> dict[str, object]:
    transcript = (
        f"--- analysis_step={step} | action={num + 1} | 00:00:00 | tool-agent ---\n"
        "[SYSTEM PROMPT]\nsystem\n\n"
        f"[USER PROMPT]\nCurrent state: step {num + 1}, level {level}.\nValid actions right now: {valid}.\n\n"
        "[MODEL RESPONSE META]\nfinish_reason: tool_calls\n\n"
        "[THINKING]\nWorld model: THIS LINE IS IN THINKING AND MUST BE IGNORED\n\n"
        + (f"[ASSISTANT]\n{assistant}\n\n" if assistant else "")
        + "[TOOL CALL: python]\naction(['UP'])\n\n[TOOL RESULT: python]\n{}\n\n[ANALYZER STATUS]\nstep_executed: True\n\n"
    )
    return _event("analysis", board, title=f"Analysis Step {step}", action_num=num, analysis_step=step, level=level, transcript=transcript)


def synthetic_events() -> list[dict[str, object]]:
    b0 = _board(5)
    b1 = _board(5, mark=(2, 2))  # level 2 board
    b2 = _board(9)
    events = [_event("initial", b0, title="Initial State", action_num=0, analysis_step=None, action_display="RESET", reward=0.0)]
    # turn 1: no action (yielded), assistant leaves a world model and a critic request
    events.append(_analysis(1, 0, b0, assistant="World model: a box\nCritic: is the box the goal?\nCross-level notes: keep me"))
    # turn 2: three actions with no board change
    events.extend(_action(n, 2, b0, changed=False) for n in (1, 2, 3))
    events.append(_analysis(2, 3, b0, assistant="Plan: try down"))
    # turn 3: one action completes the level
    events.append(_action(4, 3, b1, changed=True, level=2, level_completed=True))
    events.append(_analysis(3, 4, b1, level=2, assistant="Recent findings: level done"))
    # turn 4: one action, game over
    events.append(_action(5, 4, b2, changed=True, level=2, game_over=True))
    events.append(_analysis(4, 5, b2, level=2))
    # solver auto-reset before turn 5 (carries the previous analysis_step)
    reset = _action(6, 4, b1, changed=True, level=2)
    reset.update({"action_name": "RESET", "action_display": "RESET"})
    events.append(reset)
    events.append(_analysis(5, 6, b1, level=2))
    return events


def test_turn_grouping_puts_auto_reset_before_the_turn() -> None:
    turns = list(replay.iter_turns(synthetic_events()))
    assert [len(t.actions) for t in turns] == [0, 3, 1, 1, 0]
    assert [len(t.pre_turn_actions) for t in turns] == [0, 0, 0, 0, 1]
    assert turns[4].pre_turn_actions[0]["action_display"] == "RESET"


def test_ascii_decoding_matches_board() -> None:
    board = _board(3, mark=(1, 4))
    assert replay.decode_board_ascii(format_grid_ascii(board)) == tuple(tuple(row) for row in board)


def test_replay_reconstructs_state_beliefs_and_firings(tmp_path: Path) -> None:
    dump_dir = tmp_path / "packets"
    summary = replay.replay_events(
        synthetic_events(), stall_no_change=3, stall_no_progress=25, dump_dir=dump_dir, source_name="synthetic"
    )
    assert summary["turns"] == 5
    assert summary["executed_actions"] == 6
    assert summary["levels_completed"] == 1
    assert summary["board_changed_mismatches"] == 0 and summary["ascii_board_mismatches"] == 0
    assert summary["assistant_sections"] == 3 and summary["assistant_sections_updating_beliefs"] == 3
    assert summary["critic_requests"] == 1
    assert summary["firings"] == {"level_start": 2, "actor_request": 1, "stall_no_change": 1, "stall_no_progress": 0}
    assert summary["sandbox_ok"] == 4 and summary["sandbox_err"] == 0

    log = summary["firing_log"]
    assert [(e["turn"], e["cause"], e["action_num"], e["level"]) for e in log] == [
        (1, "level_start", 0, 1),
        (2, "actor_request", 0, 1),
        (3, "stall_no_change", 3, 1),
        (4, "level_start", 4, 2),
    ]
    assert "is the box the goal?" in log[1]["reason"]
    assert log[1]["sandbox_result"]["nodes"] == 1 and log[3]["sandbox_result"]["nodes"] == 2
    assert log[3]["sandbox_result"]["changed_cells"] == 1

    by_turn = {int(p.stem.split("_turn")[1][:4]): CriticPacket.from_json(p.read_text()) for p in dump_dir.glob("*.packet.json")}
    assert len(by_turn) == 4
    assert by_turn[2].actor_beliefs["world_model"] == "a box"
    assert by_turn[2].actor_beliefs["cross_level_notes"] == "keep me"
    assert "critic_request" not in by_turn[2].actor_beliefs
    assert by_turn[2].valid_actions == ("UP", "DOWN")
    assert by_turn[3].actor_beliefs["current_plan"] == "try down"
    assert by_turn[3].transitions_total == 3 and not any(t["board_changed"] for t in by_turn[3].transitions)
    assert by_turn[3].last_action_result["action_num"] == 3
    # level transition wipes the six per-level keys but keeps cross-level notes
    assert by_turn[4].actor_beliefs["world_model"] == "" and by_turn[4].actor_beliefs["current_plan"] == ""
    assert by_turn[4].actor_beliefs["cross_level_notes"] == "keep me"
    assert by_turn[4].step_summary["level_transition"] is True
    assert by_turn[4].previous_frame["level"] == 1 and by_turn[4].current_frame["level"] == 2
    assert all("THINKING" not in v for v in by_turn[4].actor_beliefs.values())


def test_cli_writes_json_summary(tmp_path: Path, capsys) -> None:
    path = tmp_path / "synthetic_p0_events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in synthetic_events()) + "\n", encoding="utf-8")
    assert replay.main([str(path), "--stall-n", "2", "--stall-k", "25", "--no-sandbox", "--no-log"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["file"] == "synthetic_p0"
    assert summary["policy"] == {"stall_no_change": 2, "stall_no_progress": 25}
    assert summary["firings"]["stall_no_change"] == 1
    assert summary["sandbox_ok"] == 0 and summary["mean_sandbox_seconds"] is None
    assert "firing_log" not in summary
