"""Bounds and repairs applied to the actor's growing conversation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from nooa.events import Error, Task
from nooa.runtime.event_manager import EventManager

from tycho_nooa.history import (
    ELIDED_OBSERVATION,
    OBSERVATION_REF,
    bounded_context_text,
    bounded_observation,
    bounded_tool_arguments,
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


def test_large_context_values_become_bounded_identity_receipts() -> None:
    exact = "é" * 2_000

    visible = bounded_context_text(exact, limit=512, kind="tool result")

    assert len(visible.encode("utf-8")) <= 512
    assert "tool result externalized: 4000 bytes" in visible
    assert "sha256=" in visible
    assert bounded_context_text("small", limit=512, kind="tool result") == "small"


def test_large_tool_arguments_become_a_valid_structural_receipt() -> None:
    arguments = {"code": "x" * 4_000, "timeout_s": 15}

    visible = bounded_tool_arguments(arguments, limit=512, tool_name="run_python")

    encoded = json.dumps(visible, separators=(",", ":")).encode()
    receipt = visible["_provider_projection"]
    assert len(encoded) <= 512
    assert receipt["schema"] == "externalized_tool_arguments.v1"
    assert receipt["key_count"] == 2
    assert receipt["bytes"] > 4_000
    assert arguments["code"] == "x" * 4_000


def test_tool_argument_receipt_does_not_replay_adversarial_key_names() -> None:
    adversarial_key = "k" * 16_000

    visible = bounded_tool_arguments(
        {"code": "print(1)", adversarial_key: True},
        limit=256,
        tool_name="run_python",
    )

    encoded = json.dumps(visible, separators=(",", ":")).encode()
    assert len(encoded) <= 256
    assert adversarial_key.encode() not in encoded
    assert visible["_provider_projection"]["key_count"] == 2


def test_large_observation_falls_back_to_external_frame_pointer() -> None:
    visible = bounded_observation(
        "full-grid:" + "x" * 4_000,
        "frame on disk = level_0/turn_003.json",
        limit=512,
    )

    assert len(visible.encode("utf-8")) <= 512
    assert "frame on disk = level_0/turn_003.json" in visible
    assert "observation payload externalized" in visible


def test_replayed_reasoning_is_bounded_without_mutating_the_response() -> None:
    response = _Response(content="", reasoning="r" * 4_000)

    visible = replayed_reasoning(response, max_bytes=512)

    assert visible is not None
    assert len(visible.encode("utf-8")) <= 512
    assert "assistant reasoning externalized" in visible
    assert response.reasoning == "r" * 4_000


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
