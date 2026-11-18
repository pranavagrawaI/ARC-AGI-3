"""Event-driven epistemic questions for a pending Actor request.

The critic reads a compact commitment packet, not the transcript. Only
explicitly extracted Actor-authored events may be cited. It has no tools or
action authority. This module owns cadence, the packet, the question contract,
source binding, and injection; transport and lifecycle accounting remain with
``solver_agent``, and objection state with ``objection_ledger``.

Replaying the transcript cost the 2026-09-12 public run 18-26% of every game's
wall clock while each game ended with over 90% of its action budget unspent.
Prompt size grew from 24k to 117k tokens and correlated with call latency at
r=0.987. The packet is also the anti-drift mechanism: a critic shown changed-cell
counts and regions but never cell values cannot slide into asking which pixel
moved, which is how CD82 spent two of its seven reviews.
"""
from __future__ import annotations

import json
from typing import Any

# Actor-authored code items sent in the packet, newest last. Binding still
# resolves against the complete ledger, so a cited claim is always fully
# recorded. packet-live-probe-v4 set this: at ten items the critic never cited
# the window; at zero the objective-gap rate fell (4 of 12 against 6); at three
# it held (6 of 12) with every response valid and 40% fewer prompt tokens.
PACKET_CLAIMS = 3
PACKET_CLAIM_CHARS = 1400
PACKET_TRANSITIONS = 6
# Recent commitments carried as citable claims. packet-live-probe-v2 showed the
# ten-claim window filling with transition dumps and batch-permission repair
# while the objective-level claim lived in an earlier submit's rationale.
PACKET_COMMITMENTS = 5


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
- Affordability: at the packet's observed seconds per executed action, can the
  Actor still settle this before the wall clock ends? If not, what cheaper
  observation would settle it?
- Triviality: if the proposed objective would be achieved almost automatically,
  what observed feature makes it conditional? Treat benchmark difficulty only as a
  weak warning; a level, subgoal, or mechanism may really be simple.

Do not ask a generic checklist question, answer your own question, or use later
outcomes. Do not demand exhaustive evidence: retained uncertainty matters only when
it can change the next decision.

The user message is a commitment packet: the Actor's live commitment and recent
claims, host-measured transitions as changed-cell counts and regions, the wall and
action budget, and the objection ledger. You are never shown cell values or the
transcript, so which cell changed and why are out of scope. Only a ``claim_ref``
copied exactly from ``actor_claims`` or ``recent_commitments`` may be cited. Claims listed in ``closed`` were
already disposed of; do not restate an objection the Actor has answered. Wall clock, not the action
limit, is what usually ends a game, and your own call spends it.

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


def packet_claims(evidence: list[dict[str, str]],
                  window: int = PACKET_CLAIMS) -> list[dict[str, str]]:
    """The newest Actor claims, truncated for transport but referable in full."""
    claims = []
    for row in (evidence[-window:] if window > 0 else []):
        text = row["text"]
        if len(text) > PACKET_CLAIM_CHARS:
            text = text[:PACKET_CLAIM_CHARS] + f" …[{len(text) - PACKET_CLAIM_CHARS} chars omitted]"
        claims.append({"claim_ref": row["claim_ref"], "kind": row["kind"], "text": text})
    return claims


def commitment_claims(submitted: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The newest submits as Actor-authored, citable claims.

    A rationale is where the Actor says why it acts, so the commitment-level
    inference usually lives here rather than in the surrounding code.
    """
    claims = []
    for row in submitted[-PACKET_COMMITMENTS:]:
        actions = list(row.get("actions") or [])
        summary = (f"{len(actions)}x {actions[0]}" if actions and len(set(actions)) == 1
                   else ", ".join(actions))
        claims.append({
            "claim_ref": f"actor-commitment-turn-{row.get('turn')}",
            "kind": "actor_commitment",
            "text": f"[step {row.get('step')}; submitted {summary}] {row.get('rationale') or ''}",
        })
    return claims


def budget_view(*, wall_remaining: float | None, actions_executed: int,
                wall_elapsed: float | None) -> dict[str, Any]:
    """Host budget facts, including the price the Actor is paying per action."""
    rate = (wall_elapsed / actions_executed
            if wall_elapsed and actions_executed else None)
    return {
        "wall_seconds_remaining": (round(wall_remaining, 1)
                                   if wall_remaining is not None else None),
        "actions_executed": actions_executed,
        "seconds_per_executed_action": round(rate, 1) if rate else None,
        "affordable_actions_at_this_rate": (int(wall_remaining // rate)
                                            if rate and wall_remaining else None),
        "meaning": "Wall clock ends games here far more often than the action limit. "
                   "Model calls, including this one, spend it; actions are comparatively cheap.",
    }


def critic_packet(*, trigger: str, evidence: list[dict[str, str]],
                  commitment: dict[str, Any] | None, transitions: list[dict[str, Any]],
                  budget: dict[str, Any], ledger: dict[str, Any],
                  state: dict[str, Any],
                  commitments: list[dict[str, str]] | None = None,
                  claims_window: int = PACKET_CLAIMS) -> dict[str, Any]:
    """The complete critic input: commitment, measured change, budget, ledger."""
    return {
        "review_trigger": trigger,
        "state": {k: state.get(k) for k in
                  ("turn", "step", "level", "state", "levels_completed", "available_actions")},
        "commitment": commitment,
        "recent_commitments": commitments or [],
        "actor_claims": packet_claims(evidence, claims_window),
        "measured_transitions": transitions[-PACKET_TRANSITIONS:],
        "budget": budget,
        "objections": ledger,
        "scope": "Ask one consequential epistemic question; no fault is stipulated.",
    }


def critic_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    """Send the commitment packet; the transcript is deliberately not included."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False, default=str)},
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
    """Open an objection in the Actor's stream with its disposition contract."""
    return {"role": "user", "content": (
        f"HOST EPISTEMIC OBJECTION {review_id} about {question['claim_ref']} is now OPEN.\n"
        "This question is fallible advice about your reasoning, not game knowledge:\n"
        f"{question['question']}\n"
        f"Why it matters now: {question['reason']}\n"
        "Your next submit_actions must carry closes={\"review_id\":\"" + review_id + "\", "
        "\"disposition\":...,\"evidence\":...} with one of: defend (cite the observation "
        "that warrants the claim), narrow (give the smaller claim you still hold), "
        "withdraw (name the actions you will stop repeating in retire_actions), or "
        "probe (name actions_budgeted and settle it by acting). Defending and probing "
        "both leave your intended commitment unchanged, so dispose of this honestly "
        "rather than agreeing to clear the check. Do not spend a response acknowledging it."
    )}
