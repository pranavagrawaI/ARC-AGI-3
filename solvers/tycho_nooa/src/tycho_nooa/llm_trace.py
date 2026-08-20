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

# Asks the provider for reasoning text and writes prose plus reasoning into the
# trace. Counters are always written; text is not, because it is two orders of
# magnitude larger and because asking OpenRouter for reasoning changes the run
# (retained history included), not only what we observe about it.
CAPTURE_REASONING_ENV = "TYCHO_NOOA_CAPTURE_REASONING"


def capture_reasoning(environment: Mapping[str, str] | None = None) -> bool:
    """Whether this run asks for, and records, the model's reasoning text.

    One reader for both halves: `build_llm` must ask the provider for reasoning,
    and `LLMTrace` must write it; a run where only one of those is true records
    a misleading blank.
    """
    env = os.environ if environment is None else environment
    return env.get(CAPTURE_REASONING_ENV) == "1"


class LLMTrace:
    """One JSONL line per LLM round-trip, across every agent that shares a client."""

    def __init__(self, path: Path | None, *, capture_text: bool = False) -> None:
        self._path = path
        self._capture_text = capture_text
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
        if self._capture_text:
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
        if self._capture_text:
            record["content"] = event.content
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")
