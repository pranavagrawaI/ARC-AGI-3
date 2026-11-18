#!/usr/bin/env python3
"""Reproduce the wall-clock findings that reprioritise the critic redesign.

Read-only over the preserved 2026-09-12 public-run trace. Every number printed
here is asserted, so a changed trace fails loudly rather than silently drifting.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "deployment-probes/20260912-epistemic-questioner-public-v1"
WORKSPACES = RUN / "raw/qualification/solver-workspaces"
S5I5 = WORKSPACES / "s5i5-18d95033-qljywjh2/ipc"
WALL_BUDGET = 600.0
ACTION_LIMIT = 200
# critic:3 is the first bar-as-objective challenge; 4-7 restate it.
BAR_OBJECTION = ("critic:3", "critic:4", "critic:5", "critic:6", "critic:7")
# The Actor's own stated bar count at turn 12, the turn it resumed the challenged test.
BAR_FILLED_AT_REVERT = 18


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def workspace(game: str) -> Path:
    return next(p for p in WORKSPACES.iterdir() if p.name.startswith(game))


def budget_table() -> list[dict]:
    """Wall clock, not the action limit, ended every game."""
    out = []
    for progress in rows(RUN / "raw/qualification/game-progress.jsonl"):
        calls = rows(workspace(progress["game_id"].split("-")[0]) / "ipc/model_calls.jsonl")
        critic = sum(r["wall_seconds"] for r in calls if r["request_purpose"] == "critic")
        wall, actions = progress["game_elapsed_seconds"], progress["actions"]
        out.append({
            "game": progress["game_id"].split("-")[0], "actions": actions,
            "actions_unspent": ACTION_LIMIT - actions, "stop_reason": progress["stop_reason"],
            "wall_seconds": wall, "critic_seconds": critic,
            "critic_share": critic / wall, "seconds_per_action": wall / actions,
            "actions_forgone_to_critic": critic / (wall / actions),
        })
    return out


def repeated_objection_cost() -> dict:
    """One conceded objection, restated four more times, on the S5I5 trace."""
    calls = rows(S5I5 / "model_calls.jsonl")
    critic = [r for r in calls if r["request_purpose"] == "critic"]
    repeat = [r for r in critic if r["review_id"] in BAR_OBJECTION]
    redundant = [r for r in repeat if r["review_id"] != "critic:3"]
    return {
        "reviews_on_one_claim": len(repeat),
        "seconds_on_one_claim": sum(r["wall_seconds"] for r in repeat),
        "seconds_after_the_first_ask": sum(r["wall_seconds"] for r in redundant),
        "share_of_all_critic_seconds": sum(r["wall_seconds"] for r in repeat)
        / sum(r["wall_seconds"] for r in critic),
        "prompt_tokens_on_one_claim": sum(r["prompt_tokens"] for r in repeat),
    }


def ledger_duplication() -> list[dict]:
    """actor_evidence re-sends assistant text the full context already carries."""
    calls = rows(S5I5 / "model_calls.jsonl")
    out = []
    for index, row in enumerate(calls):
        if row["request_purpose"] != "critic":
            continue
        following = next((r for r in calls[index + 1:] if r["request_purpose"] != "critic"), None)
        if following:
            out.append({"review_id": row["review_id"], "critic_prompt_tokens": row["prompt_tokens"],
                        "actor_prompt_tokens": following["prompt_tokens"],
                        "ratio": row["prompt_tokens"] / following["prompt_tokens"]})
    return out


def s5i5_endgame() -> dict:
    """Wall remaining when the Actor resumed the challenged test, and its price.

    Critic calls are nested inside the Actor request that follows them, so Actor
    ``wall_seconds`` already contains critic time and must not be added to it.
    """
    calls = rows(S5I5 / "model_calls.jsonl")
    elapsed, remaining_at_revert = 0.0, None
    batch_turn_seconds = 0.0
    for row in calls:
        if row["request_purpose"] == "critic":
            continue
        if row["turn"] == 12 and remaining_at_revert is None:
            remaining_at_revert = WALL_BUDGET - elapsed
        if row["turn"] == 12:
            batch_turn_seconds += row["wall_seconds"]
        elapsed += row["wall_seconds"]
    # The Actor states its own count at turn 12: "Current bar=18. Batch of 10".
    # Settling "does a full bar complete the level?" therefore needs 64-18 clicks.
    needed = (64 - BAR_FILLED_AT_REVERT) / 10 * batch_turn_seconds
    return {
        "wall_remaining_at_revert": remaining_at_revert,
        "seconds_for_a_ten_click_batch_turn": batch_turn_seconds,
        "clicks_still_needed_to_settle_the_bar": 64 - BAR_FILLED_AT_REVERT,
        "seconds_needed_to_settle_the_bar": needed,
        "shortfall": needed - remaining_at_revert,
        "seconds_the_redundant_re_asks_consumed": repeated_objection_cost()["seconds_after_the_first_ask"],
    }


def main() -> None:
    table = budget_table()
    print("Wall clock ended every game; the action limit was never approached.\n")
    header = f"{'game':6}{'acts':>6}{'unspent':>9}{'wall_s':>9}{'critic_s':>10}{'critic%':>9}{'s/act':>8}{'acts_lost':>11}  stop"
    print(header)
    for row in table:
        print(f"{row['game']:6}{row['actions']:6d}{row['actions_unspent']:9d}"
              f"{row['wall_seconds']:9.0f}{row['critic_seconds']:10.1f}"
              f"{row['critic_share']:8.1%}{row['seconds_per_action']:8.1f}"
              f"{row['actions_forgone_to_critic']:11.1f}  {row['stop_reason']}")
        assert row["stop_reason"] != "action_limit"
        assert row["actions_unspent"] > 180
        assert row["critic_share"] > 0.18

    repeat = repeated_objection_cost()
    print("\nOne objection, restated:\n")
    for key, value in repeat.items():
        print(f"  {key:32} {value:,.1f}" if isinstance(value, float) else f"  {key:32} {value:,}")
    assert repeat["reviews_on_one_claim"] == 5
    assert repeat["share_of_all_critic_seconds"] > 0.85

    print("\nThe evidence ledger duplicates assistant text already in full context:\n")
    for row in ledger_duplication():
        print(f"  {row['review_id']:10} critic {row['critic_prompt_tokens']:7,} vs "
              f"actor {row['actor_prompt_tokens']:7,} = {row['ratio']:.2f}x")
    assert all(row["ratio"] > 1.1 for row in ledger_duplication())

    end = s5i5_endgame()
    print("\nS5I5 endgame: the re-asks consumed the budget that would have settled them:\n")
    for key, value in end.items():
        print(f"  {key:38} {value:,.1f}" if isinstance(value, float) else f"  {key:38} {value:,}")
    assert end["shortfall"] > 0
    assert end["seconds_the_redundant_re_asks_consumed"] > end["shortfall"]
    print("\nAll assertions held.")


if __name__ == "__main__":
    main()
