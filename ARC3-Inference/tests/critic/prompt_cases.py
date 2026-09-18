"""Deterministic `_build_user_prompt` inputs shared by the snapshot generator and the tests.

The snapshots under fixtures/ were produced by running `render_case` against the
unmodified upstream `ToolAgent._build_user_prompt` (duck-harness commit 7652836)
before the critic hook was added. The byte-identity test replays the same inputs
through the patched function and compares against those files.
"""
from __future__ import annotations

from typing import Any

from inference.agent.runtime_state import Frame, HistoryEntry


def _frame(step: int, level: int, fill: int) -> Frame:
    grid = tuple(tuple(fill if (r + c) % 3 else 0 for c in range(8)) for r in range(6))
    return Frame(grid=grid, step=step, level=level)


def prompt_cases() -> dict[str, dict[str, Any]]:
    """Return named keyword-argument sets for `_build_user_prompt`, plus knowledge."""
    empty_history = [HistoryEntry(action="", frame=_frame(0, 1, 5))]
    later_history = [
        HistoryEntry(action="", frame=_frame(0, 1, 5)),
        HistoryEntry(action="UP", frame=_frame(1, 1, 6)),
        HistoryEntry(action="MOUSE(row=3, col=4)", frame=_frame(2, 1, 7)),
        HistoryEntry(action="DOWN", frame=_frame(3, 2, 8)),
    ]
    return {
        "first_turn": {
            "knowledge": {},
            "kwargs": {
                "action_num": 0,
                "valid_actions": ["ACTION1", "ACTION2", "ACTION5"],
                "current_frame": _frame(0, 1, 5),
                "history_entries": empty_history,
                "previous_step_summary": None,
            },
        },
        "mid_run_with_world_model": {
            "knowledge": {
                "world_model": "A red block sits on a charcoal frame.",
                "goal_model": "Move the block into the yellow slot.",
                "action_model": "UP shifts the block one cell.",
                "recent_findings": "DOWN did nothing at the bottom edge.",
                "open_questions": "What does SPACE do?",
                "current_plan": "Try UP twice then SPACE.",
                "cross_level_notes": "Level 1 was solved by three UPs.",
            },
            "kwargs": {
                "action_num": 3,
                "valid_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"],
                "current_frame": _frame(3, 2, 8),
                "history_entries": later_history,
                "previous_step_summary": {
                    "start_action_num": 2,
                    "end_action_num": 3,
                    "executed_count": 2,
                    "executed_actions": ["MOUSE(row=3, col=4)", "DOWN"],
                    "level": 2,
                    "level_transition": True,
                    "run_complete": False,
                    "game_over": False,
                    "board_changed": True,
                    "stop_reason": "level_completed",
                },
            },
        },
        "no_summary_captured": {
            "knowledge": {"world_model": "Only a world model line."},
            "kwargs": {
                "action_num": 7,
                "valid_actions": ["ACTION1"],
                "current_frame": _frame(7, 1, 3),
                "history_entries": later_history,
                "previous_step_summary": None,
            },
        },
    }


def render_case(agent: Any, case: dict[str, Any]) -> str:
    agent._summarized_knowledge.update(case["knowledge"])
    return agent._build_user_prompt(
        case["kwargs"]["action_num"],
        valid_actions=case["kwargs"]["valid_actions"],
        current_frame=case["kwargs"]["current_frame"],
        history_entries=case["kwargs"]["history_entries"],
        previous_step_summary=case["kwargs"]["previous_step_summary"],
    )
