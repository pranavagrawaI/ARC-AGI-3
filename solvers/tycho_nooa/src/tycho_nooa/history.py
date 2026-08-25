"""Context discipline over NOOA's event history.

Everything here bounds or repairs what the growing conversation shows the model.
It is separate from the strategies because the same bounds apply to the actor,
the scribe and the builder, and because upstream Tycho keeps its own equivalents
in one place (`agent/context_config.py`, `agent/agent.py`'s history rewriters).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

# Event metadata key carrying an observation's own breadcrumb form. Present only
# on the per-turn observation tasks, so it is also what identifies them among the
# other tasks the actor strategy adds.
ELIDED_OBSERVATION = "tycho_nooa.elided_observation"
OBSERVATION_REF = "tycho_nooa.observation_ref"


def collapse_old_actor_turns(event_manager: Any, *, keep: int) -> None:
    """Keep recent cognition exact and replace older turns with host facts.

    NOOA's collapse archives the original events rather than deleting them. The
    model-visible replacement is deliberately non-semantic: turn range, tool
    counts, and a pointer to the durable workspace. It does not infer what the
    old evidence meant or which uncertainty matters now.
    """

    if keep < 1:
        raise ValueError("at least one exact Actor turn must remain visible")
    items = list(event_manager.items())
    observations = [
        (tag, event)
        for tag, event in items
        if OBSERVATION_REF in getattr(event, "metadata", {})
    ]
    if len(observations) <= keep:
        return
    first_kept_tag = observations[-keep][0]
    keys = event_manager.keys()
    cutoff = keys.index(first_kept_tag)
    if cutoff == 0:
        return
    collapsed_items = items[:cutoff]
    summary = _actor_history_summary(collapsed_items)
    event_manager.collapse(
        keys[0],
        keys[cutoff - 1],
        summary_text=json.dumps(summary, sort_keys=True, separators=(",", ":")),
    )


def _actor_history_summary(items: list[tuple[str, Any]]) -> dict[str, Any]:
    prior = _prior_actor_summary(items)
    refs = [
        event.metadata[OBSERVATION_REF]
        for _, event in items
        if OBSERVATION_REF in getattr(event, "metadata", {})
    ]
    tool_names = [
        str(event.name)
        for _, event in items
        if hasattr(event, "name") and event.name is not None
    ]
    counts = Counter(prior.get("tool_counts", {}))
    counts.update(tool_names)
    collapsed_turns = int(prior.get("collapsed_turns", 0)) + len(refs)
    last_ref = refs[-1] if refs else prior.get("last_observation_ref")
    tags = [tag for tag, _ in items]
    return {
        "schema": "production_actor_history_collapse.v1",
        "collapsed_turns": collapsed_turns,
        "last_observation_ref": last_ref,
        "tool_counts": dict(sorted(counts.items())),
        "durable_history": (
            "Exact public frames/actions remain in workspace level_N/turn_NNN.* "
            "and are retrievable with read_file or run_python."
        ),
        "collapsed_active_tags_sha256": "sha256:"
        + hashlib.sha256(json.dumps(tags, separators=(",", ":")).encode()).hexdigest(),
    }


def _prior_actor_summary(items: list[tuple[str, Any]]) -> dict[str, Any]:
    for _, event in items:
        text = getattr(event, "summary_text", None)
        if not isinstance(text, str):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if value.get("schema") == "production_actor_history_collapse.v1":
            return value
    return {}


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
