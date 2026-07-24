#!/usr/bin/env python3
"""Prepare or execute the offline reconstruction-uptake replay experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from reconstruction_uptake.model_policy import ModelReplayPolicy
from reconstruction_uptake.replay import SequentialReplay
from reconstruction_uptake.runner import execute_replay


EXPERIMENT_ROOT = Path(__file__).resolve().parent
AGENT_0002_ROOT = EXPERIMENT_ROOT.parent / "0002-llm-repl-agent"
PROMPT_ROOT = EXPERIMENT_ROOT / "experiment"
MODEL_CONFIRMATION = "I_ACCEPT_OFFLINE_MODEL_COST"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _system_prompt(condition: str) -> str:
    base = (PROMPT_ROOT / "base.txt").read_text(encoding="utf-8").strip()
    treatment = (PROMPT_ROOT / f"{condition}.txt").read_text(encoding="utf-8").strip()
    return f"{base}\n\n{treatment}"


def _dry_run(args: argparse.Namespace, replay: SequentialReplay) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "replay-plan.jsonl", list(replay.plan_records()))
    manifest = {
        "schema": "reconstruction-uptake-replay-manifest-v1",
        "status": "dry_run",
        "condition": args.condition,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_trace": str(replay.source_path),
        "source_trace_sha256": replay.source_sha256,
        "eligible_steps": list(replay.eligible_steps),
        "model_calls": 0,
        "environment_actions": 0,
        "next_observation_policy": "concealed_until_prediction_commitment",
    }
    _write_json(args.output / "manifest.json", manifest)
    return manifest


def _execute_model(args: argparse.Namespace, replay: SequentialReplay) -> dict:
    if args.confirm_model_cost != MODEL_CONFIRMATION:
        raise SystemExit(
            "--execute-model requires "
            f"--confirm-model-cost {MODEL_CONFIRMATION}; no request was sent"
        )
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"required API key environment variable {args.api_key_env!r} is not set; "
            "no request was sent"
        )
    if str(AGENT_0002_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_0002_ROOT))
    from llm_repl_agent.client import OpenAICompatibleChatClient
    from llm_repl_agent.workspace import PythonWorkspace

    prompt = _system_prompt(args.condition)
    client = OpenAICompatibleChatClient(
        base_url=args.endpoint,
        model=args.model,
        api_key=api_key,
        temperature=1.0,
        max_tokens=None,
        max_completion_tokens=4096,
        top_p=0.95,
        presence_penalty=0.0,
        seed=args.model_seed,
        timeout_seconds=args.request_timeout,
        extra_body={
            "top_k": 20,
            "reasoning": {"enabled": True},
            "provider": {
                "order": ["chutes/fp8"],
                "only": ["chutes/fp8"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
        },
    )
    workspace = PythonWorkspace(timeout_seconds=args.workspace_timeout)
    policy = ModelReplayPolicy(
        client=client,
        workspace=workspace,
        condition=args.condition,
        system_prompt=prompt,
        max_rounds=args.max_rounds,
        max_context_turns=6,
    )
    result: dict | None = None
    try:
        result = execute_replay(
            replay=replay,
            policy=policy,
            condition=args.condition,
            output_dir=args.output,
        )
        return result
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        try:
            workspace.snapshot(args.output / "workspace")
        finally:
            workspace.close()
        _write_jsonl(args.output / "request-evidence.jsonl", policy.request_evidence)
        manifest_path = args.output / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_trace": str(replay.source_path),
                    "environment_actions": 0,
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "model_seed": args.model_seed,
                    "system_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                    "request_evidence_sha256": sha256(
                        (args.output / "request-evidence.jsonl").read_bytes()
                    ).hexdigest(),
                    "next_observation_policy": "concealed_until_prediction_commitment",
                }
            )
            _write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=1)
    parser.add_argument("--last-step", type=int, default=17)
    parser.add_argument(
        "--condition",
        choices=("optional", "required"),
        default="optional",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and materialize the replay schedule without model calls.",
    )
    mode.add_argument(
        "--execute-model",
        action="store_true",
        help="Make paid offline model requests; never calls the game environment.",
    )
    parser.add_argument("--confirm-model-cost")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--endpoint", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="qwen/qwen3.6-27b")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--workspace-timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must not contain existing files")

    replay = SequentialReplay.from_trace(
        args.trace,
        first_step=args.first_step,
        last_step=args.last_step,
    )
    manifest = (
        _dry_run(args, replay)
        if args.dry_run
        else _execute_model(args, replay)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
