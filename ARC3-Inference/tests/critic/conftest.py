from __future__ import annotations

import pytest

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.critic.packet import CriticPacket, build_packet_from_state


def grid(fill: int, *, size: int = 6, mark: tuple[int, int] | None = None) -> tuple[tuple[int, ...], ...]:
    rows = [[fill] * size for _ in range(size)]
    if mark is not None:
        rows[mark[0]][mark[1]] = (fill + 1) % 16
    return tuple(tuple(row) for row in rows)


def frame(step: int, level: int, fill: int, mark: tuple[int, int] | None = None) -> Frame:
    return Frame(grid=grid(fill, mark=mark), step=step, level=level)


@pytest.fixture
def sample_history() -> list[HistoryEntry]:
    return [
        HistoryEntry(action="", frame=frame(0, 1, 5)),
        HistoryEntry(action="UP", frame=frame(1, 1, 5, mark=(1, 1))),
        HistoryEntry(action="DOWN", frame=frame(2, 1, 5, mark=(1, 1))),  # no change
        HistoryEntry(action="MOUSE(row=2, col=3)", frame=frame(3, 2, 7)),  # level up
        HistoryEntry(action="LEFT", frame=frame(4, 2, 7, mark=(0, 0))),
    ]


@pytest.fixture
def sample_packet(sample_history: list[HistoryEntry]) -> CriticPacket:
    return build_packet_from_state(
        sample_history[-1].frame,
        sample_history,
        summarized_knowledge={"world_model": "a block", "current_plan": "go left", "goal_model": ""},
        last_step_summary={"executed_count": 1, "level_transition": False},
        last_action_result={"executed": True, "action_num": 4, "board_changed": True},
        valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"],
        trigger_cause="level_start",
        trigger_reason="test",
    )
