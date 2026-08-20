"""Context discipline over NOOA's event history.

Everything here bounds or repairs what the growing conversation shows the model.
It is separate from the strategies because the same bounds apply to the actor,
the scribe and the builder, and because upstream Tycho keeps its own equivalents
in one place (`agent/context_config.py`, `agent/agent.py`'s history rewriters).
"""

from __future__ import annotations

from typing import Any

# Event metadata key carrying an observation's own breadcrumb form. Present only
# on the per-turn observation tasks, so it is also what identifies them among the
# other tasks the actor strategy adds.
ELIDED_OBSERVATION = "tycho_nooa.elided_observation"


def retain_assistant_turn(event_manager: Any, event_id: Any, response: Any) -> None:
    """Keep the assistant's own message only when it carries prose.

    NOOA's `LLMOutput` is the only event that renders as a plain assistant message
    (`{"role": "assistant", "content": ..., "reasoning": ...}`). The `ToolCallEvent`
    that follows it hardcodes `"content": None`, so removing the `LLMOutput`
    erases the model's own prose from its history the moment it also calls a
    tool. Upstream Tycho keeps the two together on one message (`agent/agent.py`:
    `{"role": "assistant", "content": reply["text"], "tool_calls": ...,
    "raw_reasoning": ...}`); the pair of events is the closest shape NOOA's
    formatters can express.

    Reasoning is deliberately not a reason to keep it. `ToolCallEvent` carries its
    own `reasoning` field, which `OpenAIProviderFormatter` attaches to the
    tool-call message itself — the shape upstream uses — so replaying reasoning
    needs no second message. Keeping one anyway is what broke ar25 on
    `runs/tycho-nooa-llm-trace-ar25-20260819T155638Z`: muse-glimmer returns all of
    its output on the reasoning channel and leaves `content` empty on *every*
    call, so this appended `{"role": "assistant", "content": ""}` before each
    tool-call message. The model read its own blank turns as failed tool calls
    ("that means the assistant\'s previous message was not a valid tool call")
    and spent all 25 steps hunting for a syntax error that was never there,
    committing 0 actions where the same code had committed 20 the run before.
    """
    if response.content:
        return
    event_manager.remove(event_id)


def replayed_reasoning(response: Any) -> str | None:
    """The model's own words, to ride on the tool call rather than beside it.

    `ToolCallEvent.reasoning` is replayed on the tool-call message itself, which
    is where upstream Tycho puts it and the only place it costs no extra message.
    Content is the fallback for providers that return prose instead of a
    reasoning channel; see [[retain_assistant_turn]] for why prose alone decides
    whether the assistant event survives.
    """
    return getattr(response, "reasoning", None) or str(response.content) or None


def evict_old_grids(event_manager: Any, *, keep: int) -> None:
    """Leave the inlined grid on the most recent `keep` observations only.

    Tycho's own bound (`agent.py::_truncate_old_grids`): the grid text is the one
    payload that accumulates once per turn, and a long game overflows the model's
    window without it. Eliding is safe because the frame it carried is still on
    disk, and the breadcrumb that replaces it says where.
    """
    observations = [
        tag
        for tag, event in event_manager.items()
        if ELIDED_OBSERVATION in event.metadata
    ]
    stale = observations[:-keep] if keep > 0 else observations
    for tag in stale:
        event = event_manager[tag]
        elided = event.metadata[ELIDED_OBSERVATION]
        if event.prompt != elided:
            event_manager.update(tag, prompt=elided)
