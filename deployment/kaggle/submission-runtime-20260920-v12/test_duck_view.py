"""Duck letter board, tool clamp, 32k trim, and the open-batch switch."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'chassis'))
from duck_view import (
    LETTER_BOARD_CHARS,
    clamp_tool_text,
    duck_batch_keeps_queue,
    duck_batch_open,
    format_grid_ascii,
    trim_messages,
)


def test_letter_board_uses_duck_alphabet_including_numpy():
    assert format_grid_ascii([[0, 15], [4, 9]]) == "Wp\ncb"
    assert format_grid_ascii([]) == "(empty grid)"
    grid = np.array([[0, 15], [4, 9]])
    assert format_grid_ascii(grid) == "Wp\ncb"
    assert format_grid_ascii(np.zeros((0, 64), dtype=int)) == "(empty grid)"


def test_one_letter_board_survives_the_tool_clamp():
    board = format_grid_ascii(np.zeros((64, 64), dtype=int))
    assert len(board) == LETTER_BOARD_CHARS
    assert clamp_tool_text(board) == board
    clamped = clamp_tool_text(board + "EXTRA")
    assert len(clamped) <= LETTER_BOARD_CHARS
    assert "truncated" in clamped
    assert clamp_tool_text("short") == "short"


def test_trim_drops_a_tool_result_with_its_assistant_call():
    system = {"role": "system", "content": "stay"}
    assistant = {"role": "assistant", "content": "y" * 20_000, "tool_calls": [{"id": "1"}]}
    tool = {"role": "tool", "content": "result"}
    recent = {"role": "user", "content": "keep"}
    original = [system, assistant, tool, recent]
    trimmed = trim_messages(original, budget_tokens=1000)
    assert original[1] is assistant
    assert trimmed[0] == system
    assert trimmed[-1]["content"] == "keep"
    assert all(message.get("role") != "tool" for message in trimmed)


def test_duck_batch_keeps_an_unresolved_queue(monkeypatch):
    monkeypatch.delenv("ARC_DUCK_BATCH", raising=False)
    assert duck_batch_keeps_queue("unresolved") is False
    assert duck_batch_keeps_queue("mismatch") is False
    monkeypatch.setenv("ARC_DUCK_BATCH", "1")
    assert duck_batch_open() is True
    assert duck_batch_keeps_queue("unresolved") is True
    assert duck_batch_keeps_queue("mismatch") is False
