"""Event-driven epistemic questions for a pending Actor request.

The critic sees the complete rendered context, but only explicitly extracted
Actor-authored events may be cited. It has no tools or action authority. This
module owns cadence, the question contract, source binding, and injection;
transport and lifecycle accounting remain with ``solver_agent``.
"""
from __future__ import annotations

import json
from typing import Any


ACTOR_TEMPO_CONTEXT = r"""
HOST TEMPO CONTRACT

Rule: Treat a legal reversible environment action as an evidence-producing
computation. Prefer it when its possible successors distinguish live hypotheses
faster than another inference call.
Rationale: Thinking can reorganize existing evidence; an environment probe creates
new evidence. Under the game deadline, a late correct conclusion can be useless.

Rule: Before another inspection, note rewrite, helper build, or model repair, state
what new evidence that computation can produce. If it produces no new evidence and
a reversible probe exists, act instead.
Rationale: Re-reading, restating uncertainty, and refining an untested model do not
reduce environmental uncertainty.

Rule: Answer a host epistemic question by narrowing or withdrawing the cited claim,
identifying the evidence that warrants it, or taking a reversible action that can
answer it. Do not spend a response merely acknowledging the question.
Rationale: A review has value only when it changes the warrant or timing of a real
commitment. The critic is fallible and does not own the decision.

Rule: After observing a successor, repair only the contradicted, decision-relevant
claim, then choose the next discriminating action. Do not turn every correction into
an exhaustive world-model rewrite.
Rationale: The objective is timely orientation and level progress, not a perfect
description of the game.
"""


CRITIC_SYSTEM = r"""You are the Actor's epistemic questioner: an orthogonal,
creative falsification intelligence, not a second game solver. Find poor movement
from observation to interpretation or from interpretation to action. Do not verify
game facts, debug code, infer mechanics, construct helpers, solve the level, or
prescribe a replacement plan. The Actor alone decides and acts.

Ask exactly ONE concise question only when its answer could change the Actor's next
commitment. Prefer the earliest question that could force a useful pivot. Useful
tests include:
- Scope: how many targets, states, actions, or outcomes support the claim's scope?
- Absence: what remains untested that could still produce the alleged effect?
- Warrant (Hitchens): what observation earns retaining this claim? Lack of warrant
  leaves a claim open; it does not prove the claim false.
- Parsimony (Occam): does a simpler account explain all observations? Simplicity may
  select among equally adequate accounts; it may not discard unexplained evidence.
- Alternatives: what materially different account predicts a different next action?
- Falsification: what result would make the Actor abandon or narrow the commitment?
- Objective relevance: what evidence connects a proxy change, prediction match, or
  state change to level completion?
- Plan choice: why is this plan better than the live alternatives under the action
  and wall-clock limits?
- Missing evidence: what available evidence could reverse the present decision?
- Contact: which reversible action would resolve the uncertainty faster than more
  thought?
- Triviality: if the proposed objective would be achieved almost automatically,
  what observed feature makes it conditional? Treat benchmark difficulty only as a
  weak warning; a level, subgoal, or mechanism may really be simple.

Do not ask a generic checklist question, answer your own question, or use later
outcomes. Do not demand exhaustive evidence: retained uncertainty matters only when
it can change the next decision. The environment is action-limited and time-limited;
model inference consumes wall time, while reversible actions can create new evidence.

The user message contains the full context plus an ActorEvidence ledger. Only a
``claim_ref`` copied exactly from that ledger may be cited; system, host, user, tool,
and prior critic text are context but are never Actor claims.

Return JSON only with exactly these three string fields:
{"claim_ref":"actor-event reference or empty",
 "question":"one decision-relevant question ending in ? or empty",
 "reason":"why answering it could change the next commitment or empty"}

Use all-empty fields when no consequential question is justified. With a question,
keep it within 36 words and reason within 28 words."""


FIELDS = {"claim_ref", "question", "reason"}
MISMATCH_STATUSES = frozenset({"mismatch", "error", "timeout", "size_mismatch"})


def review_trigger(*, actor_calls: int, step: int, last_actor_calls: int,
                   last_step: int, levels_completed: int,
                   last_levels_completed: int, mismatch_count: int,
                   last_mismatch_count: int, pending_sequence_length: int = 0,
                   pending_sequence_step: int | None = None) -> str | None:
    """Return the highest-priority review boundary before a new Actor call.

    Baselines are advanced only after a valid critic response. Consequently an
    invalid response leaves the same trigger pending for the next Actor call.
    """
    if actor_calls <= 0:
        return None
    if levels_completed > last_levels_completed:
        return "level_advanced"
    if mismatch_count > last_mismatch_count:
        return "prediction_mismatch"
    if (pending_sequence_length > 1 and pending_sequence_step is not None
            and step > pending_sequence_step):
        return "action_sequence_observed"
    if step - last_step >= 3:
        return "three_actions_observed"
    if actor_calls - last_actor_calls >= 8:
        return "eight_actor_calls"
    return None


def count_mismatches(checks: list[dict[str, Any]]) -> int:
    return sum(row.get("status") in MISMATCH_STATUSES for row in checks)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def actor_evidence(messages: list[Any]) -> list[dict[str, str]]:
    """Extract referenceable Actor-authored text, excluding every other role."""
    evidence: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if _field(message, "role") != "assistant":
            continue
        content = _field(message, "content")
        if isinstance(content, str) and content.strip():
            evidence.append({
                "claim_ref": f"actor-message-{message_index}-content",
                "kind": "assistant_content",
                "text": content,
            })
        for call_index, call in enumerate(_field(message, "tool_calls", []) or []):
            function = _field(call, "function", call)
            arguments = _field(function, "arguments")
            if isinstance(arguments, str) and arguments.strip():
                evidence.append({
                    "claim_ref": f"actor-message-{message_index}-tool-{call_index}",
                    "kind": "assistant_tool_call",
                    "text": arguments,
                })
    return evidence


def critic_messages(actor_messages: list[Any], trigger: str) -> list[dict[str, str]]:
    """Supply full context while making Actor authorship structurally explicit."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": json.dumps({
            "review_trigger": trigger,
            "actor_evidence": actor_evidence(actor_messages),
            "full_context_for_interpretation_only": actor_messages,
            "scope": "Ask one consequential epistemic question; no fault is stipulated.",
        }, ensure_ascii=False, default=str)},
    ]


def _decode_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("critic response must be a JSON object")
    return value


def validate(text: str) -> dict[str, str]:
    value = _decode_object(text)
    if set(value) != FIELDS or not all(isinstance(v, str) for v in value.values()):
        raise ValueError("three string critic fields required")
    populated = [bool(value[field].strip()) for field in FIELDS]
    if any(populated) and not all(populated):
        raise ValueError("critic question requires all three fields")
    if not any(populated):
        return value  # type: ignore[return-value]
    if not value["question"].rstrip().endswith("?"):
        raise ValueError("critic question must end with ?")
    if len(value["question"].split()) > 36 or len(value["reason"].split()) > 28:
        raise ValueError("critic field word budget exceeded")
    return value  # type: ignore[return-value]


def bind_claim_ref(question: dict[str, str], evidence: list[dict[str, str]]) -> dict[str, Any]:
    """Bind a question to exactly one ActorEvidence record."""
    claim_ref = question["claim_ref"]
    if not claim_ref:
        return {"claim_ref": "", "match": None}
    matches = [row for row in evidence if row["claim_ref"] == claim_ref]
    if len(matches) != 1:
        raise ValueError(f"claim_ref has {len(matches)} source matches, expected 1")
    return {"claim_ref": claim_ref, "match": matches[0]}


def actor_injection(review_id: str, question: dict[str, str]) -> dict[str, str]:
    return {"role": "user", "content": (
        f"HOST EPISTEMIC QUESTION {review_id} about {question['claim_ref']}.\n"
        "This question is fallible advice about your reasoning, not game knowledge:\n"
        f"{question['question']}\n"
        f"Why it matters now: {question['reason']}\n"
        "Answer through the next commitment: cite the evidence, narrow or withdraw the "
        "claim, or take a reversible action that can answer the question. Do not merely "
        "agree, build a helper for the critic, or spend a response acknowledging it. If "
        "you reject the question, state the concrete reason briefly and continue."
    )}
