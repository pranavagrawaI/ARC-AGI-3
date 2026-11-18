"""Deterministic bound on the Actor transcript: whole-turn eviction, no model call.

v11 shipped with NOOA's summarizer disabled (``headless.py``: a background LLM
request must not straddle the game boundary), so nothing bounded the CodeAct
transcript. R9 cd82 measured ~5.5k prompt tokens per game turn (the full grid
in every task prompt plus every cell output kept forever): 14k at turn 0,
73.6k at turn 10, the 262k served ceiling near turn 45. A 200-action game
cannot finish under that growth; on overflow the request fails and the game
dies.

This is the repair Duck's harness uses: keep the last ``keep_turns`` game turns
verbatim and collapse everything older into one host-authored digest. The
digest is rebuilt from ``transitions.jsonl`` on every eviction, so nested
collapses lose nothing and no LLM request is ever scheduled. Continuity is
semantic and lives where it already lived — the Actor's notes, its helper
module, and the host's measured trajectory — not in the raw transcript.

A game turn starts at a ``Task`` event (one ``handle()`` wake). Validation
retries reissue ``Task`` inside the same game turn; they count as turns here,
which only makes eviction slightly more conservative.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from nooa.events import Summary, Task

DEFAULT_KEEP_TURNS = 4
DIGEST_MAX_CHARS = 2_000
RECORD = "transcript_evictions.jsonl"


def digest(transitions: list[dict[str, Any]], *, max_chars: int = DIGEST_MAX_CHARS) -> str:
    """One line per executed action, newest last, bounded by ``max_chars``.

    Built from the host's transition records (action, changed cells, level
    count), never from the evicted prose, so it can be regenerated exactly.
    """
    lines = []
    for row in transitions:
        before, after = row["before"], row["after"]
        a, b = before.get("grid"), after.get("grid")
        aligned = (
            isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
            and all(len(x) == len(y) for x, y in zip(a, b))
        )
        changed = (
            sum(x != y for ra, rb in zip(a, b) for x, y in zip(ra, rb)) if aligned else None
        )
        levels = after.get("levels_completed", 0)
        state = after.get("state", "")
        mark = f" L{levels}" if levels != before.get("levels_completed", 0) else ""
        term = f" {state}" if state and state != "NOT_FINISHED" else ""
        cells = "shape-change" if changed is None else f"{changed} cells"
        lines.append(f"t{row['turn']} s{row['step']} {row['action']} -> {cells}{mark}{term}")
    omitted = 0
    while lines and sum(len(line) + 1 for line in lines) > max_chars:
        lines.pop(0)
        omitted += 1
    head = (
        "Older transcript turns were evicted to bound the context. Nothing measured "
        "is lost: self.trajectory() and self.last_transition() hold every executed "
        "action with exact before/after grids, self.observations() the host's "
        "standing facts, read_notes()/load_helpers() your own record. Executed so far"
    )
    if omitted:
        head += f" ({omitted} earliest omitted here; trajectory() has them)"
    return head + ":\n" + "\n".join(lines) if lines else head + ": none yet."


class TranscriptEviction:
    """Collapse game turns older than the last ``keep_turns`` before each model call."""

    def __init__(
        self,
        event_manager: Any,
        *,
        transitions: Callable[[], list[dict[str, Any]]],
        record_path: Path,
        keep_turns: int = DEFAULT_KEEP_TURNS,
    ) -> None:
        if keep_turns < 1:
            raise ValueError("keep_turns must keep at least the current turn")
        self._events = event_manager
        self._transitions = transitions
        self._record_path = record_path
        self.keep_turns = int(keep_turns)
        self.evictions = 0

    def install(self) -> Callable[[], None]:
        return self._events.on("BeforeTurn", self._before_turn)

    # BeforeTurn fires before every LLM call; collapsing here guarantees the
    # next _build_messages() sees the shortened event list.
    def _before_turn(self, _event: Any) -> None:
        try:
            self.evict()
        except Exception as error:  # noqa: BLE001 - eviction is best-effort, never fatal to a turn
            self._record({"event": "EVICTION_ERROR", "error": f"{type(error).__name__}: {error}"})

    def evict(self) -> str | None:
        tags = list(self._events.keys())
        task_tags = [tag for tag in tags if isinstance(self._events[tag], Task)]
        if len(task_tags) <= self.keep_turns:
            return None
        first_kept = tags.index(task_tags[-self.keep_turns])
        if first_kept == 0:
            return None
        if first_kept == 1 and isinstance(self._events[tags[0]], Summary):
            return None  # already collapsed up to the retained window
        start, end = tags[0], tags[first_kept - 1]
        text = digest(self._transitions())
        summary_tag = self._events.collapse(start, end, text)
        self.evictions += 1
        self._record({
            "event": "EVICTED", "range": summary_tag, "kept_from": tags[first_kept],
            "task_events_total": len(task_tags), "digest_chars": len(text),
        })
        return summary_tag

    def _record(self, row: dict[str, Any]) -> None:
        row = {"at": time.time(), **row}
        with self._record_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
