"""Compare recovery from a suffix, a checkpoint, and checkpoint plus search."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from compaction import (
    SCHEMA,
    HistoryArchive,
    render_recovery_packet,
    validate_checkpoint,
)

MODEL = "meta/muse-glimmer-30b"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_OUTPUT_TOKENS = 8_192
MAX_RETRIEVAL_ROUNDS = 3
CONDITIONS = ("suffix_control", "checkpoint", "checkpoint_search")
PROVIDERS = (
    ("deepinfra-bf16", "deepinfra/bf16", ["bf16"]),
    ("together", "together", None),
    ("fireworks", "fireworks", None),
    ("phala", "phala", None),
)
PROVIDER_ATTEMPT_TIMEOUT_SECONDS = 90.0

PROBE_FIELDS = {
    "continuity": ["same_game", "new_game", "unknown"],
    "turn": "integer or null",
    "action_1": ["UP", "DOWN", "LEFT", "RIGHT", "unknown"],
    "action_2": ["UP", "DOWN", "LEFT", "RIGHT", "unknown"],
    "action_3": ["UP", "DOWN", "LEFT", "RIGHT", "unknown"],
    "action_4": ["UP", "DOWN", "LEFT", "RIGHT", "unknown"],
    "turn_10_description": ["consistent", "contradicted", "unknown"],
    "turn_15_object_motion": ["moved_down", "unchanged", "unknown"],
    "turn_15_interpretation": [
        "blocked_or_inapplicable",
        "mapping_changed",
        "unknown",
    ],
    "notes_position_status": ["current", "stale", "unknown"],
    "context_transition": [
        "recovery_after_context_limit",
        "ordinary_turn",
        "unknown",
    ],
    "commitment_interface": [
        "return_result_only",
        "execute_python_allowed",
        "unknown",
    ],
    "next_check": "short text",
    "evidence_event_ids": "list of archive event ids",
    "uncertainties": "list of short texts",
}


def _json_schema(
    name: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Submit the final {name.removeprefix('submit_')}.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


PROBE_PROPERTIES: dict[str, Any] = {
    "continuity": {"type": "string", "enum": PROBE_FIELDS["continuity"]},
    "turn": {"type": ["integer", "null"]},
    **{
        f"action_{index}": {
            "type": "string",
            "enum": PROBE_FIELDS[f"action_{index}"],
        }
        for index in range(1, 5)
    },
    "turn_10_description": {
        "type": "string",
        "enum": PROBE_FIELDS["turn_10_description"],
    },
    "turn_15_object_motion": {
        "type": "string",
        "enum": PROBE_FIELDS["turn_15_object_motion"],
    },
    "turn_15_interpretation": {
        "type": "string",
        "enum": PROBE_FIELDS["turn_15_interpretation"],
    },
    "notes_position_status": {
        "type": "string",
        "enum": PROBE_FIELDS["notes_position_status"],
    },
    "context_transition": {
        "type": "string",
        "enum": PROBE_FIELDS["context_transition"],
    },
    "commitment_interface": {
        "type": "string",
        "enum": PROBE_FIELDS["commitment_interface"],
    },
    "next_check": {"type": "string"},
    "evidence_event_ids": {"type": "array", "items": {"type": "string"}},
    "uncertainties": {"type": "array", "items": {"type": "string"}},
}
SUBMIT_PROBE = _json_schema("submit_probe", PROBE_PROPERTIES, list(PROBE_PROPERTIES))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _bbox(grid: list[list[int]], value: int) -> list[int] | None:
    points = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell == value
    ]
    if not points:
        return None
    rows = [point[0] for point in points]
    columns = [point[1] for point in points]
    return [min(rows), min(columns), max(rows), max(columns)]


def _transition_digest(event_id: str, record: dict[str, Any]) -> dict[str, Any]:
    before = record["before"]
    after = record["after"]
    changed = [
        [row, column, before_value, after_value]
        for row, (before_row, after_row) in enumerate(zip(before, after, strict=True))
        for column, (before_value, after_value) in enumerate(
            zip(before_row, after_row, strict=True)
        )
        if before_value != after_value
    ]
    changed_values = sorted(
        {
            value
            for _, _, before_value, after_value in changed
            for value in (before_value, after_value)
        }
    )
    return {
        "event_id": event_id,
        "turn": record["turn"],
        "action": record["action"],
        "levels_completed": record["levels_completed"],
        "state": record["state"],
        "changed_count": len(changed),
        "changed_cells": changed
        if len(changed) <= 40
        else changed[:20] + changed[-20:],
        "changed_cells_truncated": len(changed) > 40,
        "value_bboxes": {
            str(value): {"before": _bbox(before, value), "after": _bbox(after, value)}
            for value in changed_values
        },
    }


def consolidation_source(
    archive: HistoryArchive, *, resume_turn: int
) -> dict[str, Any]:
    transcript = [
        event
        for event in archive.turns(0, resume_turn - 1, streams={"transcript"})
        if event["kind"] in {"observation", "decision", "action_result"}
    ]
    trajectories = [
        _transition_digest(event["event_id"], event["record"])
        for event in archive.turns(0, resume_turn - 1, streams={"trajectory"})
    ]
    provider = archive.turns(
        max(0, resume_turn - 2),
        resume_turn - 1,
        streams={"provider"},
    )[-80:]
    notes = archive.get(
        sorted(
            event_id for event_id in archive.ids() if event_id.startswith("knowledge:")
        )
    )
    current = archive.turns(resume_turn, resume_turn, streams={"transcript"})
    artifacts = []
    for event_id in sorted(
        event_id for event_id in archive.ids() if event_id.startswith("knowledge:")
    ):
        event = archive.events[event_id]
        artifacts.append(
            {
                "artifact": f"knowledge/{event.record['name']}",
                "role": "model-authored durable beliefs",
                "status": "assess from evidence",
                "read_when": "Recovering stable mechanics; verify volatile state.",
                "sha256": event.record["sha256"],
                "event_ids": [event_id],
            }
        )
    trajectory_file = archive.source_files.get("trajectory")
    trajectory_ids = sorted(
        event_id for event_id in archive.ids() if event_id.startswith("trajectory:")
    )
    if trajectory_file is not None and trajectory_ids:
        artifacts.append(
            {
                "artifact": "trajectory.jsonl",
                "role": "host-authored immutable executed transitions",
                "status": "authoritative mechanical evidence",
                "read_when": "Checking action effects, movement, collision, or route history.",
                "sha256": next(
                    item["sha256"]
                    for item in archive.manifest()["source_files"]
                    if item["stream"] == "trajectory"
                ),
                "event_ids": [trajectory_ids[0], trajectory_ids[-1]],
            }
        )
    return {
        "source_contract": (
            "This is a bounded projection of immutable evidence. Every record carries "
            "a stable event_id; raw originals remain available through history retrieval."
        ),
        "resume_turn": resume_turn,
        "current_observation": current,
        "transcript": transcript,
        "transition_digests": trajectories,
        "recent_provider_events": provider,
        "durable_notes": notes,
        "artifact_manifest_exact": artifacts,
    }


def _checkpoint_prompt(source: dict[str, Any]) -> str:
    template = {
        "schema": SCHEMA,
        "resume_for": "specific purpose for the next turn",
        "episode_summary": "coverage of every consequential thread",
        "episode_index": [
            {"topic": "...", "gist": "...", "event_ids": ["stream:000000"]}
        ],
        "beliefs": [
            {
                "claim": "...",
                "status": "supported|provisional|contradicted",
                "evidence_event_ids": [],
            }
        ],
        "contradictions": [
            {"claim": "...", "resistance": "...", "evidence_event_ids": []}
        ],
        "open_questions": [{"question": "...", "evidence_event_ids": []}],
        "current_plan": ["..."],
        "artifact_manifest": "copy artifact_manifest_exact exactly, changing only status if evidence warrants",
        "review": [
            {
                "observation": "...",
                "possible_diagnosis": "...",
                "alternative_explanation": "...",
                "recommended_check": "...",
                "evidence_event_ids": [],
            }
        ],
        "next_moves": ["..."],
        "suggested_capabilities": [{"capability": "...", "use_when": "..."}],
    }
    return f"""Create a continuation checkpoint for the same agent after context compaction.

Return ONLY one exact JSON object following this shape:
{json.dumps(template, indent=2)}

Rules:
- Preserve what happened, what it means now, contradictions, uncertainties, and next orientation.
- Cite only supplied event_ids. Supported claims require evidence_event_ids.
- The review is advisory, evidence-linked, and must include an alternative explanation.
- Do not duplicate raw grids or durable files.
- Make older details searchable through the episode index.
- Keep under 16000 JSON characters.

SOURCE:
{json.dumps(source, sort_keys=True)}"""


def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("function", {}).get("arguments")
    if isinstance(arguments, str):
        return json.loads(arguments)
    if isinstance(arguments, dict):
        return arguments
    raise TypeError("tool arguments are not a JSON object")


class OpenRouter:
    def __init__(self, api_key: str, raw_dir: Path):
        self.api_key = api_key
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0
        self.attempt_count = 0
        self.preferred_index = 0
        self.unhealthy: set[int] = set()
        self.client = httpx.Client(timeout=PROVIDER_ATTEMPT_TIMEOUT_SECONDS)

    def call(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.call_count += 1
        base_payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
        }
        if tools:
            base_payload["tools"] = tools
            base_payload["tool_choice"] = "auto"
        order = list(range(self.preferred_index, len(PROVIDERS))) + list(
            range(self.preferred_index)
        )
        failures = []
        for index in order:
            if index in self.unhealthy:
                continue
            name, route, quantizations = PROVIDERS[index]
            provider = {
                "order": [route],
                "only": [route],
                "allow_fallbacks": False,
                "require_parameters": False,
            }
            if quantizations:
                provider["quantizations"] = quantizations
            payload = {**base_payload, "provider": provider}
            started = time.monotonic()
            self.attempt_count += 1
            response = None
            body: dict[str, Any] = {}
            try:
                response = self.client.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                try:
                    body = response.json()
                except ValueError:
                    body = {"non_json_body": response.text[:10_000]}
            except httpx.HTTPError as error:
                body = {"transport_error": type(error).__name__, "message": str(error)}
            record = {
                "call": self.call_count,
                "attempt": self.attempt_count,
                "provider": name,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "status_code": None if response is None else response.status_code,
                "request": {**payload, "messages": messages},
                "response": body,
            }
            (self.raw_dir / f"call-{self.call_count:03d}-{name}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            status = None if response is None else response.status_code
            error_text = str(body.get("error") or body)[:500]
            request_invalid = status in {400, 401, 403, 404, 413, 422} or any(
                marker in error_text.lower()
                for marker in ("maximum context length", "context window")
            )
            if request_invalid:
                raise RuntimeError(f"provider rejected request: {error_text}")
            if (
                response is None
                or status == 429
                or (status is not None and status >= 500)
            ):
                self.unhealthy.add(index)
                failures.append(f"{name}: {status or 'transport'} {error_text}")
                continue
            if status != 200:
                raise RuntimeError(f"provider HTTP {status}: {error_text}")
            if body.get("error") is not None:
                self.unhealthy.add(index)
                failures.append(f"{name}: error envelope {error_text}")
                continue
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("provider response contained no choices")
            self.preferred_index = index
            return body
        raise RuntimeError(f"all consolidation providers failed: {'; '.join(failures)}")


def generate_checkpoint(
    client: OpenRouter, archive: HistoryArchive, resume_turn: int
) -> dict[str, Any]:
    source = consolidation_source(archive, resume_turn=resume_turn)
    body = client.call(
        [
            {
                "role": "system",
                "content": "You consolidate evidence for continuation without taking game actions.",
            },
            {"role": "user", "content": _checkpoint_prompt(source)},
        ],
        [],
    )
    message = body["choices"][0]["message"]
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("checkpoint call did not return exact JSON text")
    candidate = json.loads(content)
    return validate_checkpoint(candidate, archive)


def _base_recovery(archive: HistoryArchive, resume_turn: int, recent_turns: int) -> str:
    observation = archive.turns(resume_turn, resume_turn, streams={"transcript"})
    recent = archive.turns(
        max(0, resume_turn - recent_turns),
        resume_turn - 1,
        streams={"transcript"},
    )
    actions = [
        _transition_digest(event["event_id"], event["record"])
        for event in archive.turns(
            max(0, resume_turn - recent_turns),
            resume_turn - 1,
            streams={"trajectory"},
        )
    ]
    return f"""# CONTEXT RECOVERY — SAME GAME, SAME AGENT

This is a context transition, not a game reset and not an action result. Continue
the existing run. During a reserved action commitment only `return_result` is
available; `execute_python` is not available in that phase.

## Host-owned current state
{json.dumps(observation, indent=2, sort_keys=True)}

## Recent contiguous suffix
{json.dumps(recent, indent=2, sort_keys=True)}

## Recent mechanical action digest
{json.dumps(actions, indent=2, sort_keys=True)}

## Durable artifact pointers
- knowledge/hypotheses.md: model-authored beliefs; may be stale.
- trajectory.jsonl: immutable executed transitions.
"""


def _probe_prompt(context: str, search_available: bool) -> str:
    search_note = (
        "Archive retrieval tools are available. Use them when the supplied orientation is insufficient or when checking a cited claim."
        if search_available
        else "Archive retrieval tools are not available in this condition."
    )
    return f"""{context}

## Recovery probe
{search_note}

Diagnose the prior run state without choosing or executing a game action. Submit
exactly one `submit_probe` result. Use `unknown` rather than guessing. Required
fields and allowed categories:
{json.dumps(PROBE_FIELDS, indent=2, sort_keys=True)}

For `next_check`, state the smallest evidence check that should precede another
action. Cite archive IDs actually supporting your answer; do not invent IDs.
"""


HISTORY_TOOLS = [
    _json_schema(
        "history_get",
        {"event_ids": {"type": "array", "items": {"type": "string"}}},
        ["event_ids"],
    ),
    _json_schema(
        "history_turns",
        {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "streams": {"type": "array", "items": {"type": "string"}},
        },
        ["start", "end", "streams"],
    ),
    _json_schema(
        "history_find",
        {
            "query": {"type": "string"},
            "streams": {"type": "array", "items": {"type": "string"}},
            "kinds": {"type": "array", "items": {"type": "string"}},
            "start_turn": {"type": ["integer", "null"]},
            "end_turn": {"type": ["integer", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ["query", "streams", "kinds", "start_turn", "end_turn", "limit"],
    ),
]


def _execute_history_tool(
    archive: HistoryArchive, name: str, arguments: dict[str, Any]
) -> Any:
    if name == "history_get":
        return archive.get(arguments["event_ids"])
    if name == "history_turns":
        streams = set(arguments["streams"]) or None
        return archive.turns(arguments["start"], arguments["end"], streams=streams)
    if name == "history_find":
        return archive.find(
            arguments["query"],
            streams=set(arguments["streams"]) or None,
            kinds=set(arguments["kinds"]) or None,
            start_turn=arguments["start_turn"],
            end_turn=arguments["end_turn"],
            limit=arguments["limit"],
        )
    raise ValueError(f"unknown history tool {name}")


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
            "content": "You are resuming an evidence-driven ARC investigation. Never invent missing evidence.",
        },
        {"role": "user", "content": _probe_prompt(context, search_available)},
    ]
    tools = [*HISTORY_TOOLS, SUBMIT_PROBE] if search_available else [SUBMIT_PROBE]
    retrievals: list[dict[str, Any]] = []
    for round_index in range(MAX_RETRIEVAL_ROUNDS + 1):
        body = client.call(messages, tools)
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        for call in calls:
            if call.get("function", {}).get("name") == "submit_probe":
                return _tool_arguments(call), retrievals, body.get("usage", {})
        content = message.get("content")
        if not calls and isinstance(content, str) and content.strip():
            return json.loads(content), retrievals, body.get("usage", {})
        if not search_available or round_index == MAX_RETRIEVAL_ROUNDS:
            raise ValueError("probe did not submit a result within the retrieval cap")
        assistant_message = {
            key: value
            for key, value in message.items()
            if key in {"role", "content", "tool_calls", "reasoning_details"}
        }
        messages.append(assistant_message)
        for call in calls:
            name = call.get("function", {}).get("name")
            arguments = _tool_arguments(call)
            try:
                result = _execute_history_tool(archive, name, arguments)
            except Exception as error:  # noqa: BLE001 - feed bounded tool failure back
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


EXPECTED = {
    "continuity": "same_game",
    "turn": 16,
    "action_1": "UP",
    "action_2": "DOWN",
    "action_3": "LEFT",
    "action_4": "RIGHT",
    "turn_10_description": "contradicted",
    "turn_15_object_motion": "unchanged",
    "turn_15_interpretation": "blocked_or_inapplicable",
    "notes_position_status": "stale",
    "context_transition": "recovery_after_context_limit",
    "commitment_interface": "return_result_only",
}


def score_probe(result: dict[str, Any], archive: HistoryArchive) -> dict[str, Any]:
    fields = {key: result.get(key) == expected for key, expected in EXPECTED.items()}
    next_check = str(result.get("next_check", "")).casefold()
    fields["next_check"] = (
        ("14" in next_check and "15" in next_check)
        or "contrast" in next_check
        or "changed cell" in next_check
        or "diff" in next_check
    )
    refs = result.get("evidence_event_ids")
    fields["valid_evidence_refs"] = (
        isinstance(refs, list)
        and len(refs) >= 2
        and all(isinstance(ref, str) and ref in archive.ids() for ref in refs)
    )
    return {
        "score": sum(fields.values()),
        "max_score": len(fields),
        "fields": fields,
    }


def write_card(path: Path, *, repeats: int) -> None:
    text = f"""# 0159-compaction-recovery-comparison-v1
Question: Does an evidence-linked checkpoint preserve frozen turn-15 orientation better than a mechanical suffix?
Posture: offline frozen-boundary probe.
Contrast: suffix vs checkpoint vs checkpoint+retrieval; same model/probe.
Prediction: checkpoint improves semantic fields; retrieval improves evidence recovery.
Measure: probe-results.jsonl categorical scores, retrievals, raw responses.
Cap: {repeats} repeats/condition; <= {1 + repeats * (1 + 1 + 4)} calls; {MAX_OUTPUT_TOKENS} output tokens/call; 0 ARC actions; 30 min; spend unknown.
Stop: conditions finish, cap, provider failure, or invalid checkpoint.
Invalid if: wrong model/provider, malformed checkpoint, source drift, fallback, or host JSON repair.
Result map: checkpoint gain -> integrate behind flag; no gain -> revise schema; search-only gain -> prioritize retrieval; ambiguous -> add probes.
Outputs: this run directory.
"""
    if len(text.encode()) > 1_024:
        raise ValueError("experiment card exceeds 1 KiB")
    path.write_text(text, encoding="utf-8")


def run(run_dir: Path, source_run: Path, *, repeats: int) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("ARC_LLM_API_KEY or OPENROUTER_API_KEY is required")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_card(run_dir / "EXPERIMENT-CARD.md", repeats=repeats)
    archive = HistoryArchive(source_run)
    client = OpenRouter(api_key, run_dir / "raw-provider")
    started = time.monotonic()
    status = "invalid"
    error: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    try:
        checkpoint = generate_checkpoint(client, archive, resume_turn=16)
        (run_dir / "model-checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        base = _base_recovery(archive, resume_turn=16, recent_turns=6)
        packet = render_recovery_packet(
            checkpoint, archive, resume_turn=16, recent_turns=6
        )
        contexts = {
            "suffix_control": base,
            "checkpoint": base + "\n\n## Semantic checkpoint\n" + packet,
            "checkpoint_search": base + "\n\n## Semantic checkpoint\n" + packet,
        }
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
                    "score": score_probe(result, archive),
                    "retrievals": retrievals,
                    "usage": usage,
                }
                results.append(record)
                with (run_dir / "probe-results.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
        status = "complete"
    except Exception as caught:  # noqa: BLE001 - preserve the failed run
        error = {"type": type(caught).__name__, "message": str(caught)}
        (run_dir / "failure.json").write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_results = [
            record for record in results if record["condition"] == condition
        ]
        scores = [record["score"]["score"] for record in condition_results]
        field_counts = Counter(
            field
            for record in condition_results
            for field, correct in record["score"]["fields"].items()
            if correct
        )
        by_condition[condition] = {
            "completed": len(condition_results),
            "scores": scores,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "field_successes": dict(field_counts),
            "retrieval_calls": sum(
                len(record["retrievals"]) for record in condition_results
            ),
        }
    summary = {
        "schema": "0159-compaction-comparison-summary-v1",
        "status": status,
        "error": error,
        "model": MODEL,
        "provider": "deepinfra/bf16",
        "source_run": str(source_run.resolve()),
        "repeats": repeats,
        "provider_calls": client.call_count,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "conditions": by_condition,
        "comparison_limit": "suffix control is reconstructed; exact NOOA retry prompt was not preserved",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(run(args.run_dir, args.source_run, repeats=args.repeats), indent=2)
    )


if __name__ == "__main__":
    main()
