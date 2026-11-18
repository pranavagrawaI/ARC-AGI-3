"""Bounded epistemic review of an Actor request.

The critic sees the Actor's complete rendered message context but receives no
tools and has no authority to act.  Its output is advice only.  This module
owns the review cadence, response contract, source binding and injection text;
transport and lifecycle accounting remain with the solver agent.
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

Rule: Submit a reversible probe by the end of your second consecutive response
without an action, unless you identify a concrete irreversible risk, missing input,
or bounded computation whose result will change that probe. A single action does not
require a complete helper, experiment record, objective theory, or multi-action
prediction.
Rationale: The default cost of one probe is lower than repeated model calls. This is
a deadline implication, not permission for random action.

Rule: After observing a successor, repair only the contradicted, decision-relevant
claim, then choose the next discriminating action. Do not turn every correction into
an exhaustive world-model rewrite.
Rationale: The objective is timely orientation and level progress, not a perfect
description of the game.
"""


CRITIC_SYSTEM = r"""You are the Actor's epistemic critic. You are an orthogonal,
creative falsification intelligence, not a second game solver. Review how the Actor
turns observations into interpretations and commitments. Do not verify game facts,
debug code, solve the level, or propose a detailed plan. The Actor alone can act.

The environment is action-limited and time-limited. Model inference consumes most
wall time; legal reversible actions are usually much cheaper and create evidence
that thought cannot. A late correct theory may be operationally useless. Criticism
must therefore find the earliest justified pivot, not demand certainty.

Apply these rules:
- Do not confuse absence of evidence with evidence of absence.
- Scope every claim to the tested target, state, action, and observed outcome.
- Prefer the simplest account that explains the evidence, while keeping materially
  different alternatives alive until a discriminating observation closes them.
- Do not infer a special case or whole action class from one instance.
- Ask whether another mechanism could produce the same observation.
- Ask whether available evidence covers the claim's scope.
- Separate state change, prediction agreement, and objective progress.
- A matched self-prediction supports only that prediction, not the world's truth or
  the usefulness of the objective proxy.
- Do not demand exhaustive evidence. Identify uncertainty only when it can change
  the next decision.
- First ask whether the Actor is delaying a cheap, legal, reversible probe that
  would create decision-relevant evidence. Analysis that creates no evidence can be
  as harmful as premature commitment.

Use ACT_NOW when a named reversible probe separates live alternatives more cheaply
than further thought. Use THINK_FIRST only when you can name a concrete irreversible
risk, missing input, or bounded computation whose result changes the probe. Otherwise
use CONTINUE. Return the smallest intervention that changes the timing or warrant of
the next commitment.

Return JSON only, with exactly these six string fields:
{"assessment":"concern|no_concern|insufficient_context",
 "tempo_verdict":"ACT_NOW|THINK_FIRST|CONTINUE",
 "actor_quote":"exact excerpt from Actor messages, or empty",
 "inference_gap":"unsupported inferential step, or empty",
 "decision_consequence":"commitment affected and why, or empty",
 "minimal_revision":"smallest warranted change in reasoning, or empty"}

For concern, keep actor_quote within 25 words, inference_gap within 32,
decision_consequence within 28, and minimal_revision within 32. For no_concern,
return CONTINUE and leave the other four fields empty. Do not use knowledge of later
outcomes and do not assume that a fault exists."""


FIELDS = {
    "assessment", "tempo_verdict", "actor_quote", "inference_gap",
    "decision_consequence", "minimal_revision",
}


def review_due(*, actor_calls: int, step: int, last_actor_calls: int,
               last_step: int, reviews: int, max_reviews: int = 20) -> bool:
    """Review before a new Actor call after two calls or two environment steps."""
    return reviews < max_reviews and actor_calls > 0 and (
        actor_calls - last_actor_calls >= 2 or step - last_step >= 2
    )


def critic_messages(actor_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Give the critic the exact rendered Actor context as quoted evidence."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": json.dumps({
            "actor_request_as_evidence": actor_messages,
            "scope": "Review the pending Actor reasoning in its full context; no fault is stipulated.",
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
        raise ValueError("six string critic fields required")
    if value["assessment"] not in ("concern", "no_concern", "insufficient_context"):
        raise ValueError("invalid assessment")
    if value["tempo_verdict"] not in ("ACT_NOW", "THINK_FIRST", "CONTINUE"):
        raise ValueError("invalid tempo verdict")
    limits = {"actor_quote": 25, "inference_gap": 32,
              "decision_consequence": 28, "minimal_revision": 32}
    if any(len(value[field].split()) > limit for field, limit in limits.items()):
        raise ValueError("critic field word budget exceeded")
    if value["assessment"] == "concern":
        if any(not value[field].strip() for field in limits):
            raise ValueError("concern requires four specific fields")
    elif value["assessment"] == "no_concern":
        if value["tempo_verdict"] != "CONTINUE" or any(value[f].strip() for f in limits):
            raise ValueError("no_concern must be empty CONTINUE")
    elif value["tempo_verdict"] == "ACT_NOW":
        raise ValueError("insufficient context cannot require action")
    return value  # type: ignore[return-value]


def _strings(value: Any, path: str = ""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def bind_quote(critique: dict[str, str], actor_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Require a concern quote to occur exactly once in the Actor request."""
    quote = critique["actor_quote"]
    if not quote:
        return {"quote": "", "matches": []}
    matches = []
    for index, message in enumerate(actor_messages):
        for field, text in _strings(message):
            start = 0
            while True:
                at = text.find(quote, start)
                if at < 0:
                    break
                matches.append({"message": index, "field": field,
                                "start": at, "end": at + len(quote)})
                start = at + 1
    if len(matches) != 1:
        raise ValueError(f"actor_quote has {len(matches)} source matches, expected 1")
    return {"quote": quote, "matches": matches}


def actor_injection(review_id: str, critique: dict[str, str]) -> dict[str, str]:
    return {"role": "user", "content": (
        f"HOST EPISTEMIC REVIEW {review_id} for the preceding Actor material.\n"
        "This is fallible epistemic advice, not authoritative game knowledge:\n"
        f"{json.dumps(critique, ensure_ascii=False)}\n"
        "Respond through behavior in this continuation. If ACT_NOW, submit the named "
        "reversible probe before further helper, note, or model work. If THINK_FIRST, "
        "perform only the bounded computation named and then act. If you contest the "
        "review, state the specific reason in at most 25 words, then act or name the "
        "concrete blocker. Do not spend a response acknowledging the review."
    )}
