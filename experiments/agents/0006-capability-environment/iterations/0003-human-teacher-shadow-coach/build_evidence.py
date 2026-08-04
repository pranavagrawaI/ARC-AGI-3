#!/usr/bin/env python3
"""Build a compact mechanical index for human review of shadow-coach Q0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def load_result(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name / "result.json").read_text())


def response_finish_reason(run: str, stage: str) -> str | None:
    raw = json.loads((ARTIFACTS / run / "responses" / f"{stage}.raw").read_text())
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    return choices[0].get("finish_reason")


def stage_index(stage: Mapping[str, Any], *, run: str) -> dict[str, Any]:
    raw_message = stage.get("message")
    message: Mapping[str, Any] = raw_message if isinstance(raw_message, Mapping) else {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content")
    raw_tool_calls = message.get("tool_calls")
    tool_calls: list[Any] = raw_tool_calls if isinstance(raw_tool_calls, list) else []
    raw_usage = stage.get("usage")
    usage: Mapping[str, Any] = raw_usage if isinstance(raw_usage, Mapping) else {}
    raw_metadata = stage.get("response_metadata")
    metadata: Mapping[str, Any] = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    lowered = str(reasoning).lower()
    return {
        "name": stage["name"],
        "status": stage["status"],
        "provider": metadata.get("provider"),
        "finish_reason": response_finish_reason(run, str(stage["name"])),
        "response_id": stage.get("response_id"),
        "content": content,
        "tool_calls": tool_calls,
        "markers": {
            "invented_user_intent_mentions": lowered.count("the user wants"),
            "explicit_unknown_mentions": lowered.count("unknown"),
            "action_6_mentions": lowered.count("action 6"),
            "python_requested": any(
                isinstance(call, Mapping)
                and isinstance(call.get("function"), Mapping)
                and call["function"].get("name") == "python"
                for call in tool_calls
            ),
        },
        "reasoning_head": str(reasoning)[:2000],
        "reasoning_tail": str(reasoning)[-1200:],
        "usage": usage,
        "request_sha256": stage.get("request_sha256"),
        "response_sha256": stage.get("response_sha256"),
    }


def main() -> None:
    failed_provider = load_result("q0-live-v1")
    treatment = load_result("q0-live-v2-modelrun")
    human_review = json.loads((HERE / "human-review.json").read_text())
    completed = [stage for stage in treatment["stages"] if stage["status"] == "completed"]
    indexed = [stage_index(stage, run="q0-live-v2-modelrun") for stage in completed]
    costs = [
        stage.get("usage", {}).get("cost")
        for stage in completed
        if isinstance(stage.get("usage", {}).get("cost"), (int, float))
    ]
    evidence = {
        "schema": "0006-shadow-coach-q0-evidence-v1",
        "human_note": json.loads((HERE / "human-note.json").read_text()),
        "source": {
            "after_step": 26,
            "request_sha256": treatment["source_request_sha256"],
            "prior_action_6_count": 26,
        },
        "attempts": {
            "deepinfra": {
                "status": failed_provider["status"],
                "calls_attempted": failed_provider["calls_attempted"],
                "terminal_stage": failed_provider["stages"][-1],
            },
            "modelrun": {
                "status": treatment["status"],
                "calls_attempted": treatment["calls_attempted"],
                "terminal_error": treatment.get("error"),
                "stages": indexed,
                "total_reported_cost": sum(costs),
                "shadow_actor_executed": any(
                    stage["name"] == "shadow_note_actor" for stage in treatment["stages"]
                ),
            },
        },
        "mechanical_contrast": {
            "no_note_next_move": "action 6" if indexed[0].get("content") else None,
            "human_note_next_move": "python tool call" if indexed[1]["markers"]["python_requested"] else None,
            "human_note_retained_invented_user_intent": (
                indexed[1]["markers"]["invented_user_intent_mentions"] > 0
            ),
            "human_note_explicit_unknown_mentions": indexed[1]["markers"]["explicit_unknown_mentions"],
            "shadow_coach_finish_reason": indexed[2]["finish_reason"],
            "shadow_coach_emitted_content": indexed[2]["content"] is not None,
        },
        "human_review": human_review,
        "claims_not_licensed": [
            "No shadow-note actor response exists.",
            "The one-step branch does not establish gameplay benefit.",
            "Temporal response differences do not prove the note caused a useful change.",
            "Mechanical string markers do not replace human semantic review.",
        ],
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": treatment["status"],
        "calls_attempted": treatment["calls_attempted"],
        "reported_cost": evidence["attempts"]["modelrun"]["total_reported_cost"],
        "human_review": human_review["status"],
    }))


if __name__ == "__main__":
    main()
