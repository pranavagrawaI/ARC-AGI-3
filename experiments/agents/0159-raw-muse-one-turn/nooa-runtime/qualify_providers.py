"""Bounded live qualification of Muse providers for NOOA image/tool transport."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import RetryConfig, Tool
from nooa.unifiedllm.registry import get_llm_client
from PIL import Image
from pydantic import BaseModel

MODEL = "openai/meta/muse-glimmer-30b"
API_BASE = "https://openrouter.ai/api/v1"
PROVIDERS = (
    ("deepinfra-bf16", "deepinfra/bf16", ["bf16"]),
    ("together", "together", None),
    ("fireworks", "fireworks", None),
    ("phala", "phala", None),
)


class PythonArgs(BaseModel):
    code: str


class DecisionArgs(BaseModel):
    action: Literal["UP", "DOWN", "LEFT", "RIGHT"]
    x: int | None = None
    y: int | None = None
    explanation: str


@dataclass(frozen=True)
class CallEvidence:
    provider: str
    phase: str
    elapsed_seconds: float
    finish_reason: str | None
    tool_calls: list[dict[str, Any]]
    usage: Any
    raw_type: str | None
    error_type: str | None = None
    error: str | None = None


def qualification_image() -> tuple[str, str]:
    image = Image.new("RGB", (2, 2))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return (
        "data:image/png;base64," + base64.b64encode(data).decode(),
        hashlib.sha256(data).hexdigest(),
    )


def provider_preferences(route: str, quantizations: list[str] | None) -> dict:
    preferences: dict[str, Any] = {
        "order": [route],
        "only": [route],
        "allow_fallbacks": False,
        "require_parameters": False,
    }
    if quantizations:
        preferences["quantizations"] = quantizations
    return preferences


def tool_calls(response: Any) -> list[dict[str, Any]]:
    return [
        {"id": call.id, "name": call.name, "arguments": call.arguments}
        for call in response.tool_calls
    ]


async def bounded_call(
    client: Any,
    *,
    provider: str,
    phase: str,
    messages: list[dict[str, Any]],
    tools: list[Tool],
    timeout_seconds: float,
) -> tuple[Any | None, CallEvidence]:
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await client.acall(
                messages,
                tools=tools,
                tool_choice="required",
            )
        return response, CallEvidence(
            provider=provider,
            phase=phase,
            elapsed_seconds=round(time.monotonic() - started, 3),
            finish_reason=response.finish_reason,
            tool_calls=tool_calls(response),
            usage=response.usage,
            raw_type=type(response.raw_response).__name__,
        )
    except Exception as error:  # noqa: BLE001 - every provider failure is evidence
        return None, CallEvidence(
            provider=provider,
            phase=phase,
            elapsed_seconds=round(time.monotonic() - started, 3),
            finish_reason=None,
            tool_calls=[],
            usage=None,
            raw_type=None,
            error_type=type(error).__name__,
            error=str(error)[:1000],
        )


def execute_python_tool() -> Tool:
    return Tool(
        name="execute_python",
        description="Execute Python analysis code.",
        callable=lambda args: args.code,
        parameters_model=PythonArgs,
    )


def return_result_tool() -> Tool:
    return Tool(
        name="return_result",
        description="Commit the final game decision.",
        callable=lambda args: args,
        parameters_model=DecisionArgs,
    )


def valid_call(evidence: CallEvidence, expected_name: str) -> bool:
    return (
        evidence.error_type is None
        and evidence.finish_reason == "tool_calls"
        and len(evidence.tool_calls) == 1
        and evidence.tool_calls[0]["name"] == expected_name
    )


async def qualify_provider(
    *,
    name: str,
    route: str,
    quantizations: list[str] | None,
    api_key: str,
    image_url: str,
    timeout_seconds: float,
    max_tokens: int,
    write_event: Any,
) -> dict[str, Any]:
    client = get_llm_client(
        MODEL,
        api_base=API_BASE,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=min(timeout_seconds, 110.0),
        drop_params=True,
        extra_body={"provider": provider_preferences(route, quantizations)},
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    )
    messages = [
        {"role": "system", "content": "Follow the requested tool protocol exactly."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Inspect the supplied image transport, then call execute_python "
                        "exactly once with code print('PROVIDER_OK')."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    response, analysis = await bounded_call(
        client,
        provider=name,
        phase="execute_python",
        messages=messages,
        tools=[execute_python_tool()],
        timeout_seconds=timeout_seconds,
    )
    write_event(asdict(analysis))
    if response is None or not valid_call(analysis, "execute_python"):
        return {"provider": name, "passed": False, "failed_phase": "execute_python"}

    messages.extend(
        [
            response.assistant_message,
            {
                "role": "tool",
                "tool_call_id": response.tool_calls[0].id,
                "name": "execute_python",
                "content": "PROVIDER_OK\n",
            },
            {
                "role": "user",
                "content": (
                    "Now call return_result exactly once with action UP, null x/y, "
                    "and explanation provider qualification."
                ),
            },
        ]
    )
    _, commitment = await bounded_call(
        client,
        provider=name,
        phase="return_result",
        messages=messages,
        tools=[return_result_tool()],
        timeout_seconds=timeout_seconds,
    )
    write_event(asdict(commitment))
    passed = valid_call(commitment, "return_result")
    return {
        "provider": name,
        "passed": passed,
        "failed_phase": None if passed else "return_result",
        "latency_seconds": round(
            analysis.elapsed_seconds + commitment.elapsed_seconds, 3
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("ARC_LLM_API_KEY or OPENROUTER_API_KEY is required")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(args.card, output_dir / "EXPERIMENT-CARD.md")
    events_path = output_dir / "provider-events.jsonl"

    def write_event(event: dict[str, Any]) -> None:
        event["timestamp"] = datetime.now(UTC).isoformat()
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    image_url, image_sha256 = qualification_image()
    results = []
    started = time.monotonic()
    cap_reached = False
    for name, route, quantizations in PROVIDERS:
        remaining = args.wall_cap_seconds - (time.monotonic() - started)
        if remaining <= 0:
            cap_reached = True
            break
        try:
            async with asyncio.timeout(remaining):
                result = await qualify_provider(
                    name=name,
                    route=route,
                    quantizations=quantizations,
                    api_key=api_key,
                    image_url=image_url,
                    timeout_seconds=min(args.timeout_seconds, remaining),
                    max_tokens=args.max_tokens,
                    write_event=write_event,
                )
        except TimeoutError:
            cap_reached = True
            write_event(
                {
                    "provider": name,
                    "phase": "qualification_wall_cap",
                    "error_type": "TimeoutError",
                    "error": "global qualification wall cap reached",
                }
            )
            results.append(
                {"provider": name, "passed": False, "failed_phase": "wall_cap"}
            )
            break
        results.append(result)
    summary = {
        "schema": "0159-provider-qualification-v1",
        "model": MODEL.removeprefix("openai/"),
        "image_sha256": image_sha256,
        "timeout_seconds_per_call": args.timeout_seconds,
        "max_tokens_per_call": args.max_tokens,
        "wall_cap_seconds": args.wall_cap_seconds,
        "wall_elapsed_seconds": round(time.monotonic() - started, 3),
        "cap_reached": cap_reached,
        "results": results,
        "passing_providers": [r["provider"] for r in results if r["passed"]],
        "valid": not cap_reached,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--wall-cap-seconds", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
