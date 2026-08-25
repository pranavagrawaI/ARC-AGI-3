#!/usr/bin/env python3
"""Measure production-context growth over the preserved 60-action trajectory."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.production_shadow_replay import execute_production_shadow_replay

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
TRANSCRIPT = (
    ROOT / "runs" / "tycho-nooa-level-start-ar25-20260818T175557Z" / "transcript.jsonl"
)


def response(decision: dict, call_id: str) -> LLMResponse:
    arguments = {key: value for key, value in decision.items() if value is not None}
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="take_action",
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


async def main() -> None:
    report_path = OUTPUT / "long-shadow-after-report.json"
    assessment_path = OUTPUT / "long-shadow-after-assessment.json"
    if report_path.exists() or assessment_path.exists():
        raise FileExistsError("long shadow artifacts already exist")
    events = [json.loads(line) for line in TRANSCRIPT.read_text().splitlines()]
    actions = [event for event in events if event.get("type") == "action_result"]
    llm = FakeLLMClient(
        scripted_responses=[
            response(event["decision"], f"shadow-{index}")
            for index, event in enumerate(actions)
        ]
    )
    os.environ["TYCHO_SANDBOX_RUNTIME"] = "host"
    with tempfile.TemporaryDirectory(prefix="tycho-production-shadow-") as temporary:
        report = await execute_production_shadow_replay(
            transcript=TRANSCRIPT,
            destination=Path(temporary) / "run",
            llm=llm,
            game_id="ar25",
            mode="single",
            treatment="control",
        )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    request = [row["provider_request_bytes"] for row in report["trajectory"]]
    evidence = [row["available_exact_evidence_bytes"] for row in report["trajectory"]]
    assessment = {
        "schema": "tycho-nooa-production-shadow-growth-assessment-v1",
        "purpose": (
            "zero-provider structural context-growth measurement; scripted historical "
            "actions are not cognitive-performance evidence"
        ),
        "turns": report["turns"],
        "provider_request_bytes": _series_summary(request),
        "provider_request_bytes_after_warmup": _series_summary(request[5:]),
        "available_exact_evidence_bytes": _series_summary(evidence),
        "curves_decoupled": _growth_ratio(request) < _growth_ratio(evidence),
        "provider_context_bounded_after_warmup": (
            max(request[5:]) - min(request[5:]) < 2_048
            and abs(_slope(request[5:])) < 10
        ),
        "environment_actions_executed": report["environment_actions_executed"],
        "first_action_divergence_sequence": report["first_action_divergence_sequence"],
    }
    assessment_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(assessment, indent=2, sort_keys=True))
    if report["environment_actions_executed"] != 0 or report["turns"] != len(actions):
        raise SystemExit(1)


def _series_summary(values: list[int]) -> dict:
    return {
        "first": values[0],
        "last": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "growth_ratio": _growth_ratio(values),
        "least_squares_slope_per_turn": _slope(values),
    }


def _growth_ratio(values: list[int]) -> float:
    return values[-1] / values[0] if values[0] else float("inf")


def _slope(values: list[int]) -> float:
    count = len(values)
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    numerator = sum(
        (index - mean_x) * (value - mean_y) for index, value in enumerate(values)
    )
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    asyncio.run(main())
