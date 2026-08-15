"""Run discriminating old-evidence probes over the fixed v4 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from compaction import HistoryArchive, render_recovery_packet, validate_checkpoint
from compare_compaction import (
    OpenRouter,
    _base_recovery,
    _json_schema,
    _tool_arguments,
)

CONDITIONS = ("suffix_control", "checkpoint", "checkpoint_search")
MAX_RETRIEVAL_ROUNDS = 4
FIELDS: dict[str, Any] = {
    "turn_3_predicted_direction": ["UP", "DOWN", "LEFT", "RIGHT", "unknown"],
    "turn_3_observed_direction": ["UP", "DOWN", "LEFT", "RIGHT", "unknown"],
    "notes_position_turn": "integer or null",
    "notes_next_action": ["UP", "DOWN", "LEFT", "RIGHT", "unknown"],
    "initial_request_messages": "integer or null",
    "recovery_request_messages": "integer or null",
    "initial_request_images": "integer or null",
    "recovery_request_images": "integer or null",
    "initial_request_text_chars": "integer or null",
    "recovery_request_text_chars": "integer or null",
    "first_commitment_tool": ["execute_python", "return_result", "unknown"],
    "corrected_commitment_tool": ["execute_python", "return_result", "unknown"],
    "turn_16_failure": [
        "account_prompt_allowance",
        "model_context_window",
        "ordinary_generation",
        "unknown",
    ],
    "two_failures_are_distinct": [True, False, None],
    "evidence_event_ids": "list of archive event ids",
    "uncertainties": "list of short texts",
}

PROPERTIES: dict[str, Any] = {
    "turn_3_predicted_direction": {
        "type": "string",
        "enum": FIELDS["turn_3_predicted_direction"],
    },
    "turn_3_observed_direction": {
        "type": "string",
        "enum": FIELDS["turn_3_observed_direction"],
    },
    "notes_position_turn": {"type": ["integer", "null"]},
    "notes_next_action": {"type": "string", "enum": FIELDS["notes_next_action"]},
    **{
        key: {"type": ["integer", "null"]}
        for key in (
            "initial_request_messages",
            "recovery_request_messages",
            "initial_request_images",
            "recovery_request_images",
            "initial_request_text_chars",
            "recovery_request_text_chars",
        )
    },
    "first_commitment_tool": {
        "type": "string",
        "enum": FIELDS["first_commitment_tool"],
    },
    "corrected_commitment_tool": {
        "type": "string",
        "enum": FIELDS["corrected_commitment_tool"],
    },
    "turn_16_failure": {"type": "string", "enum": FIELDS["turn_16_failure"]},
    "two_failures_are_distinct": {"type": ["boolean", "null"]},
    "evidence_event_ids": {"type": "array", "items": {"type": "string"}},
    "uncertainties": {"type": "array", "items": {"type": "string"}},
}
SUBMIT = _json_schema("submit_retrieval_probe", PROPERTIES, list(PROPERTIES))

STRUCTURED_HISTORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "history_turn",
            "description": (
                "Return one turn's transcript and compact mechanical transition. "
                "Provider noise and raw grids are omitted. Each changed value reports "
                "before/after bounding boxes and row/column delta."
            ),
            "parameters": {
                "type": "object",
                "properties": {"turn": {"type": "integer"}},
                "required": ["turn"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "history_around",
            "description": (
                "Return a bounded same-stream neighborhood around a stable event ID. "
                "Use when causal adjacency before/after a known event matters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "before": {"type": "integer", "minimum": 0, "maximum": 12},
                    "after": {"type": "integer", "minimum": 0, "maximum": 12},
                },
                "required": ["event_id", "before", "after"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "history_provider_episode",
            "description": (
                "Return a causally grouped provider phase. context_recovery labels the "
                "initial over-limit request and first recovered request; commitment "
                "groups its attempts; failures returns bounded transport errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {"type": "integer"},
                    "phase": {
                        "type": "string",
                        "enum": ["context_recovery", "commitment", "failures"],
                    },
                },
                "required": ["sequence", "phase"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "history_note",
            "description": "Return one exact durable note by its manifest filename.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
]

EXPECTED = {
    "turn_3_predicted_direction": "LEFT",
    "turn_3_observed_direction": "DOWN",
    "notes_position_turn": 3,
    "notes_next_action": "LEFT",
    "initial_request_messages": 72,
    "recovery_request_messages": 27,
    "initial_request_images": 2,
    "recovery_request_images": 1,
    "initial_request_text_chars": 227_029,
    "recovery_request_text_chars": 110_986,
    "first_commitment_tool": "execute_python",
    "corrected_commitment_tool": "return_result",
    "turn_16_failure": "account_prompt_allowance",
    "two_failures_are_distinct": True,
}


def prompt(context: str, *, search_available: bool) -> str:
    search = (
        """Structured retrieval tools are available. You MUST retrieve when the
supplied context does not contain an exact requested value. For this probe, call
history_turn(turn=3), history_note(name="hypotheses.md"), and
history_provider_episode for sequence 15 phases context_recovery and commitment,
plus sequence 16 phase failures. Multiple tool calls may be made in one round."""
        if search_available
        else "Retrieval tools are unavailable. Return unknown or null rather than infer an absent exact value."
    )
    return f"""{context}

## Discriminating recovery probe
{search}

Recover the exact older facts below. This is diagnosis only: do not choose or
execute a game action. Submit one `submit_retrieval_probe` result. Distinguish
the turn-15 model context-window failure from the separate turn-16 account/key
prompt-allowance failure. Cite only archive events actually supporting answers.

Required fields:
{json.dumps(FIELDS, indent=2, sort_keys=True)}
"""


def run_probe(
    client: OpenRouter,
    archive: HistoryArchive,
    context: str,
    *,
    search_available: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "Recover exact historical evidence. Never infer absent counts or invent event IDs.",
        },
        {"role": "user", "content": prompt(context, search_available=search_available)},
    ]
    tools = [*STRUCTURED_HISTORY_TOOLS, SUBMIT] if search_available else [SUBMIT]
    retrievals: list[dict[str, Any]] = []
    for round_index in range(MAX_RETRIEVAL_ROUNDS + 1):
        body = client.call(messages, tools)
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        for call in calls:
            if call.get("function", {}).get("name") == "submit_retrieval_probe":
                return _tool_arguments(call), retrievals, body.get("usage", {})
        content = message.get("content")
        if not calls and isinstance(content, str) and content.strip():
            return json.loads(content), retrievals, body.get("usage", {})
        if not search_available or round_index == MAX_RETRIEVAL_ROUNDS:
            raise ValueError("probe did not submit within the retrieval cap")
        messages.append(
            {
                key: value
                for key, value in message.items()
                if key in {"role", "content", "tool_calls", "reasoning_details"}
            }
        )
        for call in calls:
            name = call.get("function", {}).get("name")
            arguments = _tool_arguments(call)
            try:
                result = execute_structured_tool(archive, name, arguments)
            except Exception as error:  # noqa: BLE001 - bounded tool feedback
                result = {"error": type(error).__name__, "message": str(error)[:500]}
            retrievals.append({"name": name, "arguments": arguments, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result, sort_keys=True),
                }
            )
    raise AssertionError("unreachable")


def score(result: dict[str, Any], archive: HistoryArchive) -> dict[str, Any]:
    fields = {key: result.get(key) == expected for key, expected in EXPECTED.items()}
    refs = result.get("evidence_event_ids")
    fields["valid_evidence_refs"] = (
        isinstance(refs, list)
        and len(refs) >= 4
        and all(isinstance(ref, str) and ref in archive.ids() for ref in refs)
    )
    return {"score": sum(fields.values()), "max_score": len(fields), "fields": fields}


def execute_structured_tool(
    archive: HistoryArchive, name: str, arguments: dict[str, Any]
) -> Any:
    if name == "history_turn":
        return archive.turn_view(arguments["turn"])
    if name == "history_around":
        return archive.around(
            arguments["event_id"],
            before=arguments["before"],
            after=arguments["after"],
        )
    if name == "history_provider_episode":
        return archive.provider_episode(arguments["sequence"], arguments["phase"])
    if name == "history_note":
        return archive.note(arguments["name"])
    raise ValueError(f"unknown structured history tool {name}")


def write_card(path: Path, repeats: int) -> None:
    text = f"""# 0159-compaction-retrieval-comparison-v3
Question: Do decision-shaped retrieval views fix the v6 motion and recovery-request errors efficiently?
Posture: offline frozen-boundary discriminating probe.
Contrast: suffix vs fixed v4 checkpoint vs checkpoint+structured retrieval; same model/probe.
Prediction: search reaches 15/15 and uses fewer prompt tokens than v6; non-search arms abstain.
Measure: probe-results.jsonl exact 15-field score, tool calls, raw responses.
Cap: {repeats} repeats/condition; <= {repeats * (1 + 1 + 5)} calls; 8192 output tokens/call; 0 ARC actions; 30 min; spend unknown.
Stop: conditions finish, cap, or provider/tool failure.
Invalid if: wrong route, checkpoint/source drift, fallback, host repair, or invented IDs accepted.
Result map: 15/15 cheaper -> promote views; residual error -> revise the failing view; no gain -> reject this representation.
Outputs: this run directory.
"""
    if len(text.encode()) > 1_024:
        raise ValueError("experiment card exceeds 1 KiB")
    path.write_text(text, encoding="utf-8")


def run(
    run_dir: Path,
    source_run: Path,
    checkpoint_path: Path,
    *,
    repeats: int,
) -> dict[str, Any]:
    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("ARC_LLM_API_KEY or OPENROUTER_API_KEY is required")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_card(run_dir / "EXPERIMENT-CARD.md", repeats)
    archive = HistoryArchive(source_run)
    checkpoint = validate_checkpoint(
        json.loads(checkpoint_path.read_text(encoding="utf-8")), archive
    )
    (run_dir / "fixed-checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    base = _base_recovery(archive, resume_turn=16, recent_turns=6)
    packet = render_recovery_packet(checkpoint, archive, resume_turn=16, recent_turns=6)
    contexts = {
        "suffix_control": base,
        "checkpoint": base + "\n\n## Semantic checkpoint\n" + packet,
        "checkpoint_search": base + "\n\n## Semantic checkpoint\n" + packet,
    }
    client = OpenRouter(api_key, run_dir / "raw-provider")
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    error: dict[str, str] | None = None
    try:
        for repeat in range(repeats):
            for condition in CONDITIONS:
                result, retrievals, usage = run_probe(
                    client,
                    archive,
                    contexts[condition],
                    search_available=condition == "checkpoint_search",
                )
                record = {
                    "condition": condition,
                    "repeat": repeat,
                    "result": result,
                    "score": score(result, archive),
                    "retrievals": retrievals,
                    "usage": usage,
                }
                records.append(record)
                with (run_dir / "probe-results.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
        status = "complete"
    except Exception as caught:  # noqa: BLE001 - preserve partial run
        status = "invalid"
        error = {"type": type(caught).__name__, "message": str(caught)}
        (run_dir / "failure.json").write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    conditions = {}
    for condition in CONDITIONS:
        subset = [record for record in records if record["condition"] == condition]
        scores = [record["score"]["score"] for record in subset]
        successes = Counter(
            key
            for record in subset
            for key, correct in record["score"]["fields"].items()
            if correct
        )
        conditions[condition] = {
            "completed": len(subset),
            "scores": scores,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "field_successes": dict(successes),
            "retrieval_calls": sum(len(record["retrievals"]) for record in subset),
        }
    summary = {
        "schema": "0159-compaction-retrieval-summary-v1",
        "status": status,
        "error": error,
        "model": "meta/muse-glimmer-30b",
        "provider": "deepinfra/bf16",
        "source_run": str(source_run.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "repeats": repeats,
        "provider_calls": client.call_count,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "conditions": conditions,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.run_dir,
                args.source_run,
                args.checkpoint,
                repeats=args.repeats,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
