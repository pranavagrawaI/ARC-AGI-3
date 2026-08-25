"""Bounds and repairs applied to the actor's growing conversation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from nooa.events import Error, Task
from nooa.runtime.event_manager import EventManager

from tycho_nooa.history import (
    ELIDED_OBSERVATION,
    OBSERVATION_REF,
    collapse_old_actor_turns,
    replayed_reasoning,
    retain_assistant_turn,
)


@dataclass
class _Response:
    content: str
    reasoning: str | None = None


class _RecordingEventManager:
    def __init__(self) -> None:
        self.removed: list[object] = []

    def remove(self, event_id: object) -> None:
        self.removed.append(event_id)


def test_a_reasoning_only_reply_leaves_no_assistant_message_behind() -> None:
    """The failure that cost a whole ar25 episode.

    muse-glimmer returns every token on the reasoning channel and leaves
    `content` empty on all 30 calls of a run. A guard that kept the `LLMOutput`
    for reasoning alone therefore appended `{"role": "assistant", "content": ""}`
    before each tool-call message; the model read its own blank turns as failed
    tool calls and spent all 25 steps hunting a syntax error that did not exist.
    Reasoning belongs on `ToolCallEvent.reasoning`, which rides the tool-call
    message itself, so it is never a reason to keep a second one.
    """
    events = _RecordingEventManager()

    retain_assistant_turn(events, "llm-output-1", _Response(content="", reasoning="w"))

    assert events.removed == ["llm-output-1"]


def test_real_prose_survives_because_the_tool_call_message_cannot_carry_it() -> None:
    """`ToolCallEvent` hardcodes `"content": None`, so prose has nowhere else to go."""
    events = _RecordingEventManager()

    retain_assistant_turn(events, "llm-output-1", _Response(content="I will go up."))

    assert events.removed == []


def test_reasoning_is_replayed_and_falls_back_to_prose() -> None:
    assert replayed_reasoning(_Response(content="", reasoning="because")) == "because"
    assert replayed_reasoning(_Response(content="prose")) == "prose"
    assert replayed_reasoning(_Response(content="")) is None


def test_old_actor_turns_collapse_to_structural_facts_without_deleting_history() -> (
    None
):
    events = EventManager()
    original_tags = []
    for turn in range(5):
        original_tags.append(
            events.add(
                Task(
                    prompt=f"semantic observation {turn}",
                    metadata={
                        ELIDED_OBSERVATION: f"frame pointer {turn}",
                        OBSERVATION_REF: {"level": 0, "turn": turn},
                    },
                )
            )
        )
        events.add(Error(content=f"tool feedback {turn}"))
        collapse_old_actor_turns(events, keep=2)

    active = events.keys()
    summary = events[active[0]]
    payload = json.loads(summary.summary_text)

    assert payload["schema"] == "production_actor_history_collapse.v1"
    assert payload["collapsed_turns"] == 3
    assert "semantic observation" not in summary.summary_text
    assert events[original_tags[0]].prompt == "semantic observation 0"
    assert len(active) == 5  # one summary plus two exact Task/Error pairs
