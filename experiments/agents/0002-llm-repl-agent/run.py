"""Run the generic LLM baseline through the public observation/action harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from uuid import uuid4


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import arc_agi
from arc_agi.base import OperationMode
from dotenv import load_dotenv

from experiments.harness import JsonlTraceWriter, run_episode
from llm_repl_agent import (
    ColorPermutation,
    LlmAgentConfig,
    LlmReplAgent,
    MODEL_OUTPUT_CHAR_LIMIT,
    ModelBudgetExceeded,
    OpenAICompatibleChatClient,
    POLICY_PROMPTS,
    ProcessLog,
    PythonWorkspace,
    model_request_options,
    resolve_model_environment,
    write_annotation_template,
)


load_dotenv(REPO / ".env")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def operation_mode(value: str) -> OperationMode:
    try:
        return OperationMode(value.lower())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid operation mode: {value}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic multimodal LLM contestant.")
    parser.add_argument("--game", required=True, help="Host-only game id; never shown to the model.")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--model-profile", choices=("generic", "qwen3.6"))
    parser.add_argument(
        "--policy-prompt",
        choices=tuple(POLICY_PROMPTS),
        default="neutral",
    )
    parser.add_argument("--operation-mode", type=operation_mode, default=OperationMode.OFFLINE)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--max-context-turns", type=int, default=6)
    parser.add_argument("--max-model-rounds-per-action", type=int, default=10)
    parser.add_argument("--max-decision-repairs", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=100)
    parser.add_argument("--max-reported-tokens", type=int, default=250_000)
    parser.add_argument("--drop-stale-python-context", action="store_true")
    parser.add_argument("--workspace-timeout", type=float, default=20.0)
    parser.add_argument("--color-permutation-seed", type=int)
    parser.add_argument("--no-retrospective", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy_prompt = POLICY_PROMPTS[args.policy_prompt]
    try:
        model_environment = resolve_model_environment(
            os.environ,
            model=args.model,
            endpoint=args.endpoint,
            profile=args.model_profile,
        )
        request_options = model_request_options(
            model_environment.profile,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_completion_tokens=args.max_completion_tokens,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = args.run_dir or AGENT_ROOT / "artifacts" / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    colors = (
        ColorPermutation.from_seed(args.color_permutation_seed)
        if args.color_permutation_seed is not None
        else ColorPermutation.identity()
    )
    manifest = {
        "run_id": run_id,
        "agent": "0002-llm-repl-agent",
        "game": args.game,
        "game_identifier_exposed_to_model": False,
        "model": model_environment.model,
        "endpoint": model_environment.endpoint,
        "model_profile": model_environment.profile,
        "api_key_source": model_environment.api_key_source,
        "operation_mode": args.operation_mode.value,
        "env_seed": args.env_seed,
        "model_seed": args.model_seed,
        "model_request": request_options,
        "policy_prompt_variant": args.policy_prompt,
        "policy_prompt_sha256": sha256(policy_prompt.encode("utf-8")).hexdigest(),
        "max_actions": args.max_actions,
        "max_context_turns": args.max_context_turns,
        "max_model_rounds_per_action": args.max_model_rounds_per_action,
        "max_decision_repairs": args.max_decision_repairs,
        "max_model_calls": args.max_model_calls,
        "max_reported_tokens": args.max_reported_tokens,
        "python_context_policy": (
            "drop_stale" if args.drop_stale_python_context else "full"
        ),
        "max_python_output_chars": MODEL_OUTPUT_CHAR_LIMIT,
        "workspace_timeout": args.workspace_timeout,
        "retrospective_enabled": not args.no_retrospective,
        "color_permutation_seed": args.color_permutation_seed,
        "color_mapping": list(colors.mapping),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    _write_json(run_dir / "manifest.json", manifest)

    arcade = arc_agi.Arcade(operation_mode=args.operation_mode)
    env = arcade.make(args.game, seed=args.env_seed)
    if env is None:
        raise RuntimeError(f"could not create requested game: {args.game}")

    client = OpenAICompatibleChatClient(
        base_url=model_environment.endpoint,
        model=model_environment.model,
        api_key=model_environment.api_key,
        **request_options,
        seed=args.model_seed,
    )
    workspace = PythonWorkspace(timeout_seconds=args.workspace_timeout)
    manifest["workspace_runtime"] = workspace.runtime_info()
    _write_json(run_dir / "manifest.json", manifest)
    process_log = ProcessLog(run_dir / "process.jsonl")
    agent = LlmReplAgent(
        client=client,
        workspace=workspace,
        process_log=process_log,
        system_prompt=policy_prompt,
        color_permutation=colors,
        config=LlmAgentConfig(
            max_context_turns=args.max_context_turns,
            max_model_rounds_per_action=args.max_model_rounds_per_action,
            max_decision_repairs=args.max_decision_repairs,
            max_model_calls=args.max_model_calls,
            max_reported_tokens=args.max_reported_tokens,
            drop_stale_python_context=args.drop_stale_python_context,
        ),
    )
    result = None
    retrospective = None
    error: BaseException | None = None
    budget_exhausted = False
    workspace_snapshotted = False
    try:
        process_log.write("run_started", run_id=run_id)
        with JsonlTraceWriter(run_dir / "trace.jsonl") as trace:
            result = run_episode(env, agent, max_actions=args.max_actions, trace=trace)
        workspace.snapshot(run_dir / "workspace")
        workspace_snapshotted = True
        if not args.no_retrospective:
            retrospective = agent.finalize(result.final)
        process_log.write(
            "run_finished",
            actions=result.actions,
            levels_completed=result.final.levels_completed,
            state=result.final.state,
            stop_reason=result.stop_reason,
            usage=agent.usage,
        )
    except ModelBudgetExceeded as caught:
        error = caught
        budget_exhausted = True
        process_log.write(
            "run_stopped",
            stop_reason="model_budget_exhausted",
            detail=str(caught),
            actions=agent.actions,
        )
    except BaseException as caught:
        error = caught
        process_log.write("run_failed", error=repr(caught), actions=agent.actions)
    finally:
        try:
            if not workspace_snapshotted:
                workspace.snapshot(run_dir / "workspace")
        finally:
            workspace.close()
            process_log.close()

    write_annotation_template(
        run_dir / "process-annotation.json",
        actions=result.actions if result is not None else agent.actions,
    )
    scorecard = arcade.get_scorecard()
    _write_json(run_dir / "scorecard.json", scorecard)
    manifest.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "budget_exhausted"
            if budget_exhausted
            else "failed" if error is not None else "complete"
        ),
        "actions": result.actions if result is not None else agent.actions,
        "levels_completed": result.final.levels_completed if result is not None else None,
        "state": result.final.state if result is not None else None,
        "stop_reason": (
            result.stop_reason
            if result is not None
            else "model_budget_exhausted" if budget_exhausted else "error"
        ),
        "usage": dict(agent.usage),
        "retrospective": retrospective,
        "error": repr(error) if error is not None and not budget_exhausted else None,
        "budget_message": str(error) if budget_exhausted else None,
    })
    _write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({
        "run_id": run_id,
        "status": manifest["status"],
        "actions": manifest["actions"],
        "levels_completed": manifest["levels_completed"],
        "state": manifest["state"],
        "stop_reason": manifest["stop_reason"],
        "usage": manifest["usage"],
        "run_dir": str(run_dir),
    }, indent=2, sort_keys=True))
    if error is not None and not budget_exhausted:
        raise error


if __name__ == "__main__":
    main()
