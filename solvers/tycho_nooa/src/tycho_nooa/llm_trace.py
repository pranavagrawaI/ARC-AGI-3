"""Durable record of the model leg: what each round-trip cost, and what it thought.

The other four traces in a run directory are written by our own code about our own
actions — tool calls, turns, trigger fires, frames. None of them observe the LLM
itself, so questions like "does this model emit reasoning at all" were unanswerable
from a finished run.

NOOA already publishes the answer. `ActorRuntime.generate` adds an `LLMComplete`
event after every round-trip, immediately followed by the `LLMOutput` carrying the
same response's prose. `LLMComplete` is `Role.RUNTIME_EVENT` and added with
`record=False`, so it is never stored and never rendered to the model — but
`EventManager.add` emits to handlers *before* the record check, so a subscriber
sees all of them. That is the whole mechanism: no client wrapper, no NOOA change.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Requesting reasoning changes the model call. Capturing response text only
# changes the durable trace. Keep these controls separate so observation does
# not silently become an intervention.
REQUEST_REASONING_ENV = "TYCHO_NOOA_REQUEST_REASONING"
CAPTURE_RESPONSE_TEXT_ENV = "TYCHO_NOOA_CAPTURE_RESPONSE_TEXT"


def request_reasoning(environment: Mapping[str, str] | None = None) -> bool:
    """Require provider reasoning for every live research run."""
    env = os.environ if environment is None else environment
    value = env.get(REQUEST_REASONING_ENV)
    if value in (None, "1"):
        return True
    if value == "0":
        raise RuntimeError("reasoning must remain enabled")
    raise ValueError(f"{REQUEST_REASONING_ENV} must be 1 when set")


def capture_response_text(environment: Mapping[str, str] | None = None) -> bool:
    """Require durable capture of returned prose and reasoning."""
    env = os.environ if environment is None else environment
    value = env.get(CAPTURE_RESPONSE_TEXT_ENV)
    if value in (None, "1"):
        return True
    if value == "0":
        raise RuntimeError("response capture must remain enabled")
    raise ValueError(f"{CAPTURE_RESPONSE_TEXT_ENV} must be 1 when set")


class LLMTrace:
    """One JSONL line per LLM round-trip, across every agent that shares a client."""

    def __init__(self, path: Path | None) -> None:
        capture_response_text()
        self._path = path
        self._call_index = 0
        self._pending: dict[str, Any] | None = None

    def attach(self, event_manager: Any, *, source: str) -> None:
        """Subscribe to one agent's event stream, labelling its calls `source`.

        The actor, the scribe and each per-fire builder are separate agents over
        one client, so the trace is only complete if every one of them is attached.
        """
        event_manager.on("LLMComplete", lambda event: self._begin(event, source))
        event_manager.on("LLMOutput", self._finish)

    def _begin(self, event: Any, source: str) -> None:
        """Stash the metrics; the prose arrives on the very next event."""
        self._call_index += 1
        record: dict[str, Any] = {
            "call_index": self._call_index,
            "source": source,
            "model": event.model_name,
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "cached_tokens": event.cached_tokens,
            "reasoning_tokens": event.reasoning_tokens,
            "cost_usd": event.cost_usd,
            "tool_calls": [call["function_name"] for call in event.tool_calls],
            "tool_call_ids": [call["tool_call_id"] for call in event.tool_calls],
        }
        record["reasoning"] = event.reasoning_content
        self._pending = record

    def _finish(self, event: Any) -> None:
        """Complete the pending record with the assistant's own prose and write it.

        `LLMComplete` carries the reasoning channel but not `response.content`,
        which NOOA puts on the `LLMOutput` it adds on the next line of the same
        return path (`runtime/actor.py::generate`). Pairing is therefore safe:
        the two always arrive adjacent, in that order, for the same response.
        """
        record = self._pending
        self._pending = None
        if record is None:
            return
        record["content"] = event.content
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")
