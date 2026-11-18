"""Duck timing views that are not the transcript eviction already in v12.

Letter boards replace a printed numeric grid. Tool stdout is clamped to 1024
tokens (4 chars each). Requests are trimmed to a 32k-token budget by dropping
the oldest messages after the system turn. Open batching is opt-in
(``ARC_DUCK_BATCH=1``): one submit may carry several env actions without the
retrodiction earn-in.
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
TOOL_OUTPUT_TOKENS = 1024
CONTEXT_BUDGET_TOKENS = 32768
CHARS_PER_TOKEN = 4
# One 64x64 letter board is 64*64 + 63 newlines. Duck's 4096-char clamp cuts
# the last row, so the view we tell the model to print would arrive damaged.
LETTER_BOARD_CHARS = 64 * 64 + 63


def format_grid_ascii(grid: Sequence[Sequence[int]]) -> str:
    # A NumPy grid has no boolean value; `if not grid` raises.
    if grid is None:
        return "(empty grid)"
    rows = list(grid)
    if not rows:
        return "(empty grid)"
    lines = []
    for row in rows:
        chars = []
        for value in row:
            chars.append(ARC_COLOR_CHARS[max(0, min(15, int(value)))])
        lines.append("".join(chars))
    return "\n".join(lines)


def clamp_tool_text(text: str, *, tokens: int = TOOL_OUTPUT_TOKENS) -> str:
    limit = max(256, tokens * CHARS_PER_TOKEN, LETTER_BOARD_CHARS)
    if len(text) <= limit:
        return text
    # The marker's own length depends on the omitted count. Fit it inside the limit.
    keep = limit
    omitted = len(text) - keep
    marker = f"\n... [truncated {omitted} chars]"
    while keep > 0 and keep + len(marker) > limit:
        keep -= 1
        omitted = len(text) - keep
        marker = f"\n... [truncated {omitted} chars]"
    return text[:keep] + marker


def duck_batch_open() -> bool:
    return os.environ.get("ARC_DUCK_BATCH", "0") == "1"


def duck_batch_keeps_queue(check_status: str) -> bool:
    """Open batching has no frozen model. Unresolved is not a mismatch.

    The worker otherwise clears every queued action when the step check is
    not 'matched'. That would make ARC_DUCK_BATCH submit a sequence and then
    play only the first action.
    """
    return duck_batch_open() and check_status == "unresolved"


def _role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _estimate_tokens(messages: list[Any]) -> int:
    raw = json.dumps(messages, default=str)
    return max(1, len(raw) // CHARS_PER_TOKEN)


def _drop_oldest_block(history: list[Any]) -> bool:
    """Drop one front message plus the tool results that belonged to it.

    Popping a single message can leave a tool result whose assistant call is
    gone. Providers reject that request. An orphan tool result is dropped too.
    """
    if not history:
        return False
    history.pop(0)
    while history and _role(history[0]) == "tool":
        history.pop(0)
    return True


def trim_messages(messages: list[Any], *, budget_tokens: int = CONTEXT_BUDGET_TOKENS) -> list[Any]:
    """Return a request-sized copy. The caller's list is not modified.

    This is the outbound budget, not a second eviction. The caller must keep
    its own transcript and send this copy.
    """
    if not messages or budget_tokens < 1:
        return list(messages)
    history = list(messages)
    system: list[Any] = []
    if _role(history[0]) == "system":
        system = [history.pop(0)]
    while len(history) > 1 and _estimate_tokens(system + history) > budget_tokens and _drop_oldest_block(history):
        pass
    if history and _estimate_tokens(system + history) > budget_tokens:
        last = history[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            limit = max(0, budget_tokens * CHARS_PER_TOKEN - _estimate_tokens(system) * CHARS_PER_TOKEN)
            if len(last["content"]) > limit:
                last = dict(last)
                last["content"] = clamp_tool_text(last["content"], tokens=max(1, limit // CHARS_PER_TOKEN))
                history[-1] = last
    return system + history
