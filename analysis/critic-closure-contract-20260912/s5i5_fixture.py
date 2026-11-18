#!/usr/bin/env python3
"""Rebuild the S5I5 critic packets from the preserved trace and price them.

The primary failing fixture is S5I5 critic:3 through critic:7 — five reviews of
one objection. This script reconstructs, at each of those boundaries, what the
v9 compact packet would have contained, and compares it with the prompt the v8
critic actually read. No model is called; ``--emit`` writes the packets so a
separately carded live comparison can read them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACE = (ROOT / "analysis/deployment-probes/20260912-epistemic-questioner-public-v1"
         / "raw/qualification/solver-workspaces/s5i5-18d95033-qljywjh2/ipc")
sys.path.insert(0, str(ROOT / "deployment/kaggle/submission-runtime-20260912-v9/chassis"))

import epistemic_critic as critic  # noqa: E402
from objection_ledger import ObjectionLedger  # noqa: E402

WALL_BUDGET = 600.0
SYSTEM_CHARS = len(critic.CRITIC_SYSTEM)


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (TRACE / name).read_text().splitlines() if line.strip()]


def actor_messages_before(calls: list[dict], index: int) -> list[dict]:
    """Reconstruct the Actor-authored side of the transcript up to a review.

    Tool results were not preserved, so this is the Actor's own material only —
    which is exactly what source binding may cite, and what the packet carries.
    """
    messages = []
    for row in calls[:index]:
        if row["request_purpose"] == "critic":
            continue
        content = row.get("response_content") or None
        tool_calls = [{"function": {"name": call["name"], "arguments": call["arguments"]}}
                      for call in (row.get("tool_calls") or [])]
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
    return messages


def transition_summaries(transitions: list[dict], step: int) -> list[dict]:
    out = []
    for row in transitions:
        if row["step"] > step:
            break
        before, after = row["before"]["grid"], row["after"]["grid"]
        same = len(before) == len(after) and all(len(a) == len(b) for a, b in zip(before, after))
        changed = [(r, c) for r, (a, b) in enumerate(zip(before, after))
                   for c, (x, y) in enumerate(zip(a, b)) if x != y] if same else []
        out.append({
            "step": row["step"], "action": row["action"],
            "changed_cells": len(changed) if same else None, "shape_changed": not same,
            "changed_bbox": ([min(r for r, _ in changed), min(c for _, c in changed),
                              max(r for r, _ in changed), max(c for _, c in changed)]
                             if changed else None),
            "state": row["after"].get("state"),
            "levels_completed": row["after"].get("levels_completed"),
        })
    return out


def build() -> list[dict]:
    calls = rows("model_calls.jsonl")
    transitions = rows("transitions.jsonl")
    submitted = rows("actions.jsonl")
    states = {int(s["step"]): s for s in rows("states.jsonl")}
    ledger = ObjectionLedger()   # empty throughout; see note in the loop
    elapsed = 0.0
    out = []
    for index, row in enumerate(calls):
        if row["request_purpose"] != "critic":
            elapsed += row["wall_seconds"]
            continue
        review_id, step = row["review_id"], int(row["step"] or 0)
        messages = actor_messages_before(calls, index)
        executed = sum(1 for t in transitions if t["step"] <= step)
        commitment = None
        prior = [a for a in submitted if int(a["step"]) <= step]
        if prior:
            commitment = {k: prior[-1].get(k) for k in
                          ("turn", "step", "actions", "rationale", "predicted_path")}
        evidence = critic.actor_evidence(messages)
        packet = critic.critic_packet(
            trigger=row["review_trigger"], evidence=evidence,
            commitments=critic.commitment_claims(prior),
            commitment=commitment, transitions=transition_summaries(transitions, step),
            budget=critic.budget_view(wall_remaining=max(0.0, WALL_BUDGET - elapsed),
                                      actions_executed=executed, wall_elapsed=elapsed),
            ledger=ledger.view(), state=states.get(step, {"step": step}))
        # The ledger stays empty at every boundary: each packet is what a critic
        # would read if it were called fresh there. (packet-live-probe-v1 opened
        # an objection at critic:3 with a bogus ref, contaminating later packets.)
        rendered = json.dumps(packet, ensure_ascii=False, default=str)
        out.append({
            "review_id": review_id, "trigger": row["review_trigger"], "step": step,
            "v8_prompt_tokens": row["prompt_tokens"], "v8_wall_seconds": row["wall_seconds"],
            "packet_chars": len(rendered),
            # packet-live-probe-v1 measured reported prompt tokens at 1.18-1.27x
            # of chars/3.6 once the system prompt is included, so the estimate
            # covers both and a pessimistic 2.5 is also stated.
            "packet_tokens_estimate": round((len(rendered) + SYSTEM_CHARS) / 3.6),
            "packet_tokens_pessimistic": round((len(rendered) + SYSTEM_CHARS) / 2.5),
            "packet": packet,
            "evidence": evidence,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", type=Path, help="write the packets as JSON")
    args = parser.parse_args()
    built = build()
    print(f"{'review':10}{'trigger':24}{'v8 ptok':>9}{'v9 est':>9}{'ratio':>8}{'v8 wall':>9}")
    for row in built:
        print(f"{row['review_id']:10}{row['trigger']:24}{row['v8_prompt_tokens']:9,}"
              f"{row['packet_tokens_estimate']:9,}"
              f"{row['packet_tokens_estimate']/row['v8_prompt_tokens']:8.1%}"
              f"{row['v8_wall_seconds']:9.1f}")
    v8_total = sum(r["v8_prompt_tokens"] for r in built)
    v9_total = sum(r["packet_tokens_estimate"] for r in built)
    print(f"\n  every review, v8 prompt tokens: {v8_total:,}")
    print(f"  every review, v9 packet estimate: {v9_total:,} ({v9_total/v8_total:.1%})")
    # Under the contract, reviews 4-7 never reach a model at all.
    priced = [r for r in built if r["review_id"] not in ("critic:4", "critic:5", "critic:6", "critic:7")]
    print(f"  reviews the contract still pays a model for: "
          f"{[r['review_id'] for r in priced]}")
    print(f"  their packet estimate: {sum(r['packet_tokens_estimate'] for r in priced):,} "
          f"({sum(r['packet_tokens_estimate'] for r in priced)/v8_total:.1%} of v8)")
    worst = max(r["packet_tokens_pessimistic"] / r["v8_prompt_tokens"] for r in built)
    print(f"  worst case at 2.5 chars/token: {worst:.1%} of the v8 prompt")
    assert worst < 0.25
    assert all(r["packet"]["actor_claims"] for r in built), "every packet stays source-bound"
    for row in built:
        packet = row["packet"]
        # Actor claim text is Actor-authored and may mention anything, including
        # its own calls to grid_rows. The invariant is on what the HOST supplies:
        # no message stream, and measured change without cell values.
        assert "full_context_for_interpretation_only" not in packet
        assert all(set(t) == {"step", "action", "changed_cells", "shape_changed",
                              "changed_bbox", "state", "levels_completed"}
                   for t in packet["measured_transitions"])
        assert "grid_rows" not in json.dumps(packet["state"])
    print("\nAll assertions held: every packet is source-bound, the host supplies no "
          "transcript and no cell values, and each packet is under a quarter of the "
          "prompt v8 actually read.")
    if args.emit:
        args.emit.write_text(json.dumps(
            [{k: v for k, v in row.items()} for row in built], indent=2, default=str))
        print(f"wrote {args.emit}")


if __name__ == "__main__":
    main()
