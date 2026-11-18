"""Actor-only treatments: a pruned context and an optional per-decision call cap.

Selected by TYCHO_NOOA_TREATMENT under TYCHO_MODE=no_world_model. Pair-1 lost on
tokens per level: Actor prompts averaged 63-103K tokens (a third of them replayed
reasoning) and ls20 spent up to 23 calls on one decision. Every other treatment
keeps the earlier unbounded behaviour. Imports stay stdlib-only at module level
because the canonical treatment table imports this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# The tool-step ceiling every earlier arm used; pair-1 never reached it.
DEFAULT_TOOL_CALLS = 25


@dataclass(frozen=True)
class Lean:
    name: str
    # Model calls per decision that may use any tool. One more call follows, offering
    # only take_actions with thinking off, when these are spent without an action.
    tool_calls: int
    # Provider projection window in tokens (the donor-derived default is 128K).
    window: int = 64 * 1024
    # Byte bound for tool results, tool-argument strings and prose of closed decisions.
    closed_text_bytes: int = 1536

    @property
    def capped(self):
        return self.tool_calls < DEFAULT_TOOL_CALLS


TREATMENTS = {
    "lean_prune": Lean("lean_prune", tool_calls=DEFAULT_TOOL_CALLS),
    "lean_cap4": Lean("lean_cap4", tool_calls=4),
}


def lean(environment=None):
    """The selected lean treatment, or None for every other treatment."""
    env = os.environ if environment is None else environment
    choice = TREATMENTS.get(env.get("TYCHO_NOOA_TREATMENT", ""))
    if choice is not None and env.get("TYCHO_MODE") != "no_world_model":
        raise ValueError(f"{choice.name} requires TYCHO_MODE='no_world_model'")
    return choice


def treatment_policies(policy, level_checkpoint):
    """Canonical TreatmentPolicy entries: no Scribe; no other flag changes."""
    return {name: policy(name, level_boundary_policy=level_checkpoint) for name in TREATMENTS}


def guidance(choice):
    """Strategy text announcing the cap; empty when no cap applies."""
    if choice is None or not choice.capped:
        return ""
    return (f"\n\nDecision budget: each decision allows at most {choice.tool_calls} model calls "
            "that may use any tool. If no action is committed by then, one final call offers "
            "only take_actions, with thinking off. Spend a call only on a result that could "
            "change the action you would take now; otherwise act. A batch of up to 4 actions "
            "costs one decision, so commit the steps that need no new judgment together.")


def closed_fields(event, limit):
    """Field updates that shrink one event of a closed decision; empty once minimal."""
    from nooa.context_blocks.events import ToolCallEvent
    from nooa.events import LLMOutput, Task
    from tycho_nooa.history import bounded_context_text

    def bound(value, kind):
        return bounded_context_text(value, limit=limit, kind=kind) if isinstance(value, str) else value

    if isinstance(event, Task):
        return {"images": []} if event.images else {}
    if not isinstance(event, (ToolCallEvent, LLMOutput)):
        return {}
    fields = {"reasoning": None} if event.reasoning else {}
    if isinstance(event, LLMOutput):
        fields["content"] = bound(event.content, "assistant prose")
    else:
        fields["arguments"] = {key: bound(value, "tool argument") for key, value in event.arguments.items()}
        if event.result is not None:
            fields["result"] = event.result.model_copy(
                update={"content": bound(event.result.content, "tool result")})
    return {key: value for key, value in fields.items() if value != getattr(event, key)}


def prune_closed_decisions(event_manager, limit):
    """Shrink everything before the newest observation; return the number of events changed.

    Called once when a decision starts, so only the just-closed decision changes and the
    cached prefix before it survives. Events are updated, never removed: each tool call
    keeps its result, so provider tool-call/result pairs stay valid. Bounding is
    idempotent, and the exact values remain in the canonical traces and workspace.
    """
    from tycho_nooa.history import OBSERVATION_REF

    items = list(event_manager.items())
    observations = [i for i, (_, event) in enumerate(items)
                    if OBSERVATION_REF in getattr(event, "metadata", {})]
    changed = 0
    for tag, event in items[:observations[-1]] if observations else []:
        fields = closed_fields(event, limit)
        if fields:
            event_manager.update(tag, **fields)
            changed += 1
    return changed


def prune_for_decision(event_manager):
    """Strategy hook: prune when a lean treatment is selected, otherwise do nothing."""
    choice = lean()
    return 0 if choice is None else prune_closed_decisions(event_manager, choice.closed_text_bytes)
