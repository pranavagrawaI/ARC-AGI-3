#!/usr/bin/env python3
"""Execute the six frozen packets with deterministic tool trajectories."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.production_actor_replay import (
    execute_production_replay,
    validate_production_replay_packet,
)

OUTPUT = Path(__file__).resolve().parent
PACKETS = OUTPUT / "packets"
RESULT = OUTPUT / "scripted-probe"


def response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


async def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    os.environ["TYCHO_SANDBOX_RUNTIME"] = "host"
    records = []
    for packet in sorted(PACKETS.iterdir()):
        prepared = validate_production_replay_packet(packet)
        oracle = prepared.oracle
        actions: list[LLMResponse] = []
        if oracle.failure_class == "multi_transition_computation":
            actions.append(
                response(
                    "run_python",
                    {
                        "code": (
                            "import wmlib\n"
                            "ts = wmlib.transitions()\n"
                            "print([(t['level'], t['turn'], "
                            "int((t['prev'] != t['next']).sum())) for t in ts "
                            "if (t['level'], t['turn']) in {(0, 16), (1, 2)}])\n"
                        )
                    },
                    "compute",
                )
            )
        elif oracle.failure_class != "evidence_sufficiency":
            for index, ref in enumerate(oracle.acceptable_support_sets[0]):
                level, turn = _parse_ref(ref)
                actions.append(
                    response(
                        "read_file",
                        {"path": f"level_{level}/turn_{turn:03d}.json"},
                        f"read-{index}",
                    )
                )
        terminal = oracle.acceptable_terminal_outcomes[0]
        actions.append(
            response(
                "take_action",
                terminal.model_dump(exclude_none=True),
                "terminal",
            )
        )
        report = await execute_production_replay(
            packet,
            RESULT / packet.name,
            llm=FakeLLMClient(scripted_responses=actions),
        )
        records.append(
            {
                "packet_id": packet.name,
                "failure_class": oracle.failure_class,
                "terminal_correct": report["commit"]["terminal_correct"],
                "grounding": report["commit"]["grounding"],
                "observation_completeness": report["select"][
                    "observation_completeness"
                ],
                "provider_calls": report["resource"]["provider_calls"],
                "environment_actions_executed": report["environment_actions_executed"],
            }
        )
    summary = {
        "schema": "tycho-nooa-production-actor-phase1-scripted-report-v1",
        "purpose": "zero-provider structural proof; not cognitive evidence",
        "records": records,
        "acceptance": {
            "all_terminal_paths_reached": all(
                row["terminal_correct"] for row in records
            ),
            "no_arc_actions": all(
                row["environment_actions_executed"] == 0 for row in records
            ),
            "python_access_not_overclaimed": any(
                row["observation_completeness"] == "partial_python_access"
                and row["grounding"] == "unobservable"
                for row in records
            ),
        },
    }
    (RESULT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all(summary["acceptance"].values()):
        raise SystemExit(1)


def _parse_ref(value: str) -> tuple[int, int]:
    level, turn = value.removeprefix("L").split(":T")
    return int(level), int(turn)


if __name__ == "__main__":
    asyncio.run(main())
