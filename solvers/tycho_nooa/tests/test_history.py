"""Bounds and repairs applied to the actor's growing conversation."""

from __future__ import annotations

from dataclasses import dataclass

from tycho_nooa.history import replayed_reasoning, retain_assistant_turn


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
