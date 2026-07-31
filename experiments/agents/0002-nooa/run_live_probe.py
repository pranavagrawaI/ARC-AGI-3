"""Run one bounded 0002-NOOA provider smoke probe through the public ARC harness.

This entry point is deliberately explicit: provider settings are command-line
arguments, the credential is inherited only as ``ARC_LLM_API_KEY``, and no
``.env`` file is searched. This is not the paired G9 comparison and cannot
qualify it; it exists to test the candidate arm once offline gates are green.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
CONTROL_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = CONTROL_ROOT / "nooa-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

import arc_agi
from arc_agi.base import OperationMode

from experiments.harness import JsonlTraceWriter, run_episode
from host.qualification import PairedRunSpec, ProviderRequestProfile
from host.runner import NooaProcessAgent


def _operation_mode(value: str) -> OperationMode:
    try:
        return OperationMode(value.lower())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mode must be one of: normal, online, offline, competition"
        ) from error


def _default_run_root(game: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return CONTROL_ROOT / "artifacts" / "provider-smoke-probes" / game / stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded single-arm 0002-NOOA real-provider smoke probe."
    )
    parser.add_argument("--game", required=True)
    parser.add_argument("--mode", type=_operation_mode, default=OperationMode.OFFLINE)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wire-model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--response-timeout-seconds",
        type=float,
        help=(
            "Host wait for one action response. Defaults to the frozen request timeout "
            "times all permitted attempts, plus 30 seconds."
        ),
    )
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=12)
    parser.add_argument("--max-reported-tokens", type=int, default=32_768)
    parser.add_argument("--max-model-rounds-per-action", type=int, default=4)
    parser.add_argument("--max-decision-repairs", type=int, default=2)
    parser.add_argument("--transport-max-retries", type=int, default=2)
    parser.add_argument("--workspace-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--run-root", type=Path)
    return parser


def _paired_spec(args: argparse.Namespace) -> PairedRunSpec:
    profile = ProviderRequestProfile.responses(
        model=args.model,
        wire_model=args.wire_model,
        api_base=args.api_base,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    spec = PairedRunSpec.provider(
        profile,
        environment_profile=f"arc-agi-3:{args.mode.value}:{args.game}",
        environment_seed=args.environment_seed,
    )
    return replace(
        spec,
        max_actions=args.max_actions,
        max_model_calls=args.max_model_calls,
        max_reported_tokens=args.max_reported_tokens,
        max_model_rounds_per_action=args.max_model_rounds_per_action,
        max_decision_repairs=args.max_decision_repairs,
        transport_max_retries=args.transport_max_retries,
        workspace_timeout_seconds=args.workspace_timeout_seconds,
    )


def main() -> None:
    args = build_parser().parse_args()
    if not os.environ.get("ARC_LLM_API_KEY"):
        raise RuntimeError("ARC_LLM_API_KEY must be present in the inherited environment")
    spec = _paired_spec(args)
    run_root = (args.run_root or _default_run_root(args.game)).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "paired-run-spec.json").write_text(
        json.dumps(spec.as_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    arcade = arc_agi.Arcade(
        operation_mode=args.mode,
        recordings_dir=str(run_root / "raw-recordings"),
    )
    environment = arcade.make(
        args.game,
        seed=args.environment_seed,
        save_recording=True,
    )
    if environment is None:
        raise RuntimeError(f"could not create requested game: {args.game}")

    response_timeout = args.response_timeout_seconds
    if response_timeout is None:
        response_timeout = (
            args.max_model_rounds_per_action
            * args.request_timeout_seconds
            * (args.transport_max_retries + 1)
            + args.workspace_timeout_seconds
            + 30.0
        )

    with NooaProcessAgent(
        runtime_root=RUNTIME_ROOT,
        run_root=run_root / "worker",
        paired_run_spec=spec,
        response_timeout=response_timeout,
    ) as agent:
        with JsonlTraceWriter(run_root / "trace.jsonl") as trace:
            result = run_episode(
                environment,
                agent,
                max_actions=spec.max_actions,
                trace=trace,
            )

    summary = {
        "schema": "0002-nooa-single-arm-provider-smoke-v1",
        "game": args.game,
        "environment_seed": args.environment_seed,
        "worker_response_timeout_seconds": response_timeout,
        "actions": result.actions,
        "levels_completed": result.final.levels_completed,
        "state": result.final.state,
        "stop_reason": result.stop_reason,
        "run_root": str(run_root),
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
