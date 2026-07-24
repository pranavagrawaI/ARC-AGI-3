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
    MODEL_PROFILES,
    MODEL_OUTPUT_CHAR_LIMIT,
    OPENROUTER_PROFILES,
    ModelBudgetExceeded,
    OpenAICompatibleChatClient,
    POLICY_PROMPTS,
    ProcessLog,
    PythonWorkspace,
    apply_openrouter_routing,
    model_request_options,
    resolve_model_environment,
    write_annotation_template,
)
from llm_repl_agent.human_coach import (
    HUMAN_COACH_SYSTEM_ADDENDUM,
    CoachingAgent,
    InteractiveHumanCoach,
    ScriptedHumanCoach,
)
from llm_repl_agent.experiment_protocol import (
    EXPERIMENT_FILES,
    ExperimentCondition,
    compose_condition_prompt,
    endpoint_metadata_sha256,
    frozen_pair_schedule,
    frozen_request_options,
    experiment_artifact_templates,
    load_frozen_runtime,
    participating_source_hashes,
    validate_endpoint_metadata,
)
from llm_repl_agent.authority import (
    consume_member_authority,
    file_sha256,
    load_object,
    resolve_artifact_reference,
    validate_release_gate,
)
from llm_repl_agent.evidence import evidence_hashes
from llm_repl_agent.agent import DecisionFormatError


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
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--model-profile", choices=MODEL_PROFILES)
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
    parser.add_argument(
        "--human-coach",
        choices=("off", "interactive", "script"),
        default="off",
        help=(
            "Inject real human text after each env step. "
            "interactive=stdin prompts; script=JSONL via --human-coach-file. "
            "Not admissible for unsupervised contestant claims."
        ),
    )
    parser.add_argument(
        "--human-coach-file",
        type=Path,
        help="JSONL notes for --human-coach script: {\"after_step\":N,\"text\":\"...\"}",
    )
    parser.add_argument("--experiment-condition", choices=("enabled", "disabled"))
    parser.add_argument("--experiment-pair", choices=("P", "1", "2", "3"))
    parser.add_argument("--experiment-generation")
    parser.add_argument("--endpoint-metadata", type=Path)
    parser.add_argument("--release-gate-results", type=Path)
    parser.add_argument("--final-freeze-manifest", type=Path)
    parser.add_argument("--authority-consumption", type=Path)
    parser.add_argument("--run-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = None
    endpoint_metadata = None
    current_sources: dict[str, str] | None = None
    if args.experiment_condition is not None:
        forbidden = {
            "--model", "--endpoint", "--model-profile", "--policy-prompt",
            "--operation-mode", "--env-seed", "--model-seed", "--temperature",
            "--max-tokens", "--max-completion-tokens", "--max-actions",
            "--max-context-turns", "--max-model-rounds-per-action",
            "--max-decision-repairs", "--max-model-calls", "--max-reported-tokens",
            "--drop-stale-python-context", "--workspace-timeout",
            "--color-permutation-seed", "--no-retrospective",
        }
        supplied = sorted(flag for flag in forbidden if flag in sys.argv[1:])
        if supplied:
            raise SystemExit(
                "frozen experiment rejects generic runtime overrides: " + ", ".join(supplied)
            )
        if args.game != "ls20":
            raise SystemExit("the frozen experiment requires --game ls20")
        if args.experiment_pair is None or args.experiment_generation is None:
            raise SystemExit(
                "frozen experiment requires --experiment-pair and --experiment-generation"
            )
        if (
            args.endpoint_metadata is None
            or args.release_gate_results is None
            or args.authority_consumption is None
        ):
            raise SystemExit(
                "frozen experiment requires endpoint, release, and consumed authority artifacts"
            )
        endpoint_metadata = json.loads(args.endpoint_metadata.read_text(encoding="utf-8"))
        if not isinstance(endpoint_metadata, Mapping):
            raise SystemExit("endpoint metadata must be a JSON object")
        schedule = next(
            pair for pair in frozen_pair_schedule() if pair.pair_id == args.experiment_pair
        )
        experiment = load_frozen_runtime()
        try:
            validate_endpoint_metadata(endpoint_metadata, experiment)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        release_gate = json.loads(args.release_gate_results.read_text(encoding="utf-8"))
        current_sources = participating_source_hashes()
        if (
            not isinstance(release_gate, Mapping)
            or release_gate.get("passed") is not True
            or release_gate.get("pilot_authorized_by_gate") is not True
            or release_gate.get("source_hashes") != current_sources
            or not isinstance(release_gate.get("compatibility"), Mapping)
            or release_gate["compatibility"].get("endpoint_metadata_sha256")
            != endpoint_metadata_sha256(endpoint_metadata)
        ):
            raise SystemExit(
                "release-gate evidence does not authorize these exact sources and endpoint"
            )
        try:
            validate_release_gate(
                args.release_gate_results.parent,
                source_hashes=current_sources,
                endpoint_metadata_hash=endpoint_metadata_sha256(endpoint_metadata),
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if args.experiment_pair == "P" and args.final_freeze_manifest is not None:
            raise SystemExit("the excluded pilot cannot consume a final freeze")
        if args.experiment_pair != "P" and args.final_freeze_manifest is None:
            raise SystemExit("included experiment members require the final freeze path")
        args.model = experiment.model
        args.endpoint = experiment.endpoint
        args.model_profile = experiment.model_profile
        args.operation_mode = OperationMode.OFFLINE
        args.env_seed = schedule.environment_seed
        args.model_seed = schedule.requested_seed
        args.max_actions = experiment.budgets.total_actions_invariant
        args.max_context_turns = experiment.runtime.max_context_turns
        args.max_model_rounds_per_action = experiment.runtime.max_model_rounds_per_action
        args.max_decision_repairs = experiment.runtime.max_decision_repairs
        args.max_model_calls = experiment.runtime.max_model_calls
        args.max_reported_tokens = None
        args.drop_stale_python_context = True
        args.workspace_timeout = experiment.runtime.workspace_timeout_seconds
        args.color_permutation_seed = None
        args.no_retrospective = True
        args.policy_prompt = f"reconstruction-{args.experiment_condition}"
        policy_prompt = compose_condition_prompt(
            ExperimentCondition(args.experiment_condition)
        )
    else:
        if any(
            value is not None
            for value in (
                args.experiment_pair,
                args.experiment_generation,
                args.endpoint_metadata,
                args.release_gate_results,
                args.final_freeze_manifest,
                args.authority_consumption,
            )
        ):
            raise SystemExit("experiment pair fields require --experiment-condition")
        policy_prompt = POLICY_PROMPTS[args.policy_prompt]
    try:
        model_environment = resolve_model_environment(
            os.environ,
            model=args.model,
            endpoint=args.endpoint,
            profile=args.model_profile,
        )
        request_options = (
            frozen_request_options(experiment)
            if experiment is not None
            else model_request_options(
                model_environment.profile,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_completion_tokens=args.max_completion_tokens,
            )
        )
        if (
            experiment is None
            and model_environment.profile in OPENROUTER_PROFILES
        ):
            request_options = apply_openrouter_routing(request_options, os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.human_coach != "off":
        if experiment is not None:
            raise SystemExit("human coach is not allowed on frozen experiment runs")
        if args.human_coach == "script" and args.human_coach_file is None:
            raise SystemExit("--human-coach script requires --human-coach-file")
        if args.human_coach == "interactive" and args.human_coach_file is not None:
            raise SystemExit("--human-coach-file is only valid with --human-coach script")
        policy_prompt = policy_prompt.rstrip() + "\n\n" + HUMAN_COACH_SYSTEM_ADDENDUM
    elif args.human_coach_file is not None:
        raise SystemExit("--human-coach-file requires --human-coach script")

    human_coach_auto_skip = (
        args.human_coach == "interactive"
        and os.environ.get("HUMAN_COACH_AUTO_SKIP", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    try:
        if args.human_coach == "script":
            assert args.human_coach_file is not None
            coach = ScriptedHumanCoach(args.human_coach_file.resolve())
        elif args.human_coach == "interactive":
            coach = InteractiveHumanCoach(auto_skip=human_coach_auto_skip)
        else:
            coach = None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = args.run_dir or AGENT_ROOT / "artifacts" / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    member_authority_consumption = None
    if experiment is not None:
        assert current_sources is not None
        assert args.authority_consumption is not None
        assert args.experiment_generation is not None
        assert args.experiment_pair is not None
        assert args.experiment_condition is not None
        assert endpoint_metadata is not None
        try:
            member_authority_consumption = consume_member_authority(
                args.authority_consumption,
                run_dir=run_dir,
                generation=args.experiment_generation,
                pair_id=args.experiment_pair,
                condition=args.experiment_condition,
                release_gate_dir=args.release_gate_results.parent,
                source_hashes=current_sources,
                endpoint_metadata_hash=endpoint_metadata_sha256(endpoint_metadata),
            )
            pair_consumption = load_object(
                args.authority_consumption.resolve(),
                label="pair authority consumption",
            )
            consumed_authority = resolve_artifact_reference(
                pair_consumption.get("authority_path"),
                label="consumed authority",
            )
            if args.experiment_pair != "P":
                assert args.final_freeze_manifest is not None
                if consumed_authority != args.final_freeze_manifest.resolve():
                    raise ValueError("member final freeze differs from the consumed authority")
        except ValueError as error:
            raise SystemExit(str(error)) from error
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
        "human_coach": {
            "mode": args.human_coach,
            "file": (
                str(args.human_coach_file.resolve())
                if args.human_coach_file is not None
                else None
            ),
            "auto_skip": human_coach_auto_skip,
            "admissible_unsupervised_contestant": args.human_coach == "off",
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    if experiment is not None:
        assert current_sources is not None
        assert member_authority_consumption is not None
        assert args.authority_consumption is not None
        runtime_path = EXPERIMENT_FILES / "runtime-config.yaml"
        prompt_block_path = (
            EXPERIMENT_FILES / f"reconstruction-{args.experiment_condition}.txt"
        )
        schedule = next(
            pair for pair in frozen_pair_schedule() if pair.pair_id == args.experiment_pair
        )
        manifest["experiment"] = {
            "schema": experiment.schema,
            "generation": args.experiment_generation,
            "pair_id": args.experiment_pair,
            "included": schedule.included,
            "condition": args.experiment_condition,
            "pair_order": list(schedule.order),
            "runtime_config_sha256": sha256(runtime_path.read_bytes()).hexdigest(),
            "condition_block_sha256": sha256(prompt_block_path.read_bytes()).hexdigest(),
            "prompt_sha256": sha256(policy_prompt.encode("utf-8")).hexdigest(),
            "endpoint_metadata_sha256": endpoint_metadata_sha256(endpoint_metadata),
            "endpoint_metadata": endpoint_metadata,
            "level_action_budgets": [
                experiment.budgets.level_1_actions,
                experiment.budgets.level_2_actions_after_transition,
            ],
            "stop_after_levels": 2,
            "level_3_observation_policy": "suppress_before_public_conversion",
            "source_hashes": current_sources,
            "release_gate_results_sha256": _file_sha256(args.release_gate_results),
            "pair_authority_consumption": str(args.authority_consumption.resolve()),
            "pair_authority_consumption_sha256": file_sha256(
                args.authority_consumption.resolve()
            ),
            "member_authority_consumption": str(member_authority_consumption),
            "member_authority_consumption_sha256": file_sha256(
                member_authority_consumption
            ),
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
        rate_limit_retries=int(os.environ.get("OPENROUTER_RATE_LIMIT_RETRIES", "0")),
        rate_limit_backoff_seconds=float(
            os.environ.get("OPENROUTER_RATE_LIMIT_BACKOFF_SECONDS", "20")
        ),
    )
    workspace = PythonWorkspace(timeout_seconds=args.workspace_timeout)
    manifest["workspace_runtime"] = workspace.runtime_info()
    _write_json(run_dir / "manifest.json", manifest)
    process_log = ProcessLog(run_dir / "process.jsonl")

    agent: Any = LlmReplAgent(
        client=client,
        workspace=workspace,
        process_log=process_log,
        system_prompt=policy_prompt,
        color_permutation=colors,
        live_observations_dir=run_dir / "live" / "observations",
        config=LlmAgentConfig(
            max_context_turns=args.max_context_turns,
            max_model_rounds_per_action=args.max_model_rounds_per_action,
            max_decision_repairs=args.max_decision_repairs,
            max_model_calls=args.max_model_calls,
            max_reported_tokens=args.max_reported_tokens,
            drop_stale_python_context=args.drop_stale_python_context,
        ),
    )
    if coach is not None:
        agent = CoachingAgent(agent, coach)
    result = None
    retrospective = None
    error: BaseException | None = None
    budget_exhausted = False
    workspace_snapshotted = False
    try:
        process_log.write("run_started", run_id=run_id)
        with JsonlTraceWriter(run_dir / "trace.jsonl") as trace:
            result = run_episode(
                env,
                agent,
                max_actions=args.max_actions,
                trace=trace,
                level_action_budgets=(
                    (
                        experiment.budgets.level_1_actions,
                        experiment.budgets.level_2_actions_after_transition,
                    )
                    if experiment is not None
                    else None
                ),
                stop_after_levels=2 if experiment is not None else None,
                stop_on_exceptions=(
                    {
                        ModelBudgetExceeded: "model_budget_exhausted",
                        DecisionFormatError: "unrepaired_invalid_decision",
                    }
                    if experiment is not None
                    else None
                ),
            )
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

    if experiment is None:
        write_annotation_template(
            run_dir / "process-annotation.json",
            actions=result.actions if result is not None else agent.actions,
        )
    scorecard = arcade.get_scorecard()
    _write_json(run_dir / "scorecard.json", scorecard)
    committed_evidence_hashes = evidence_hashes(run_dir)
    if experiment is not None:
        templates = experiment_artifact_templates(
            ExperimentCondition(args.experiment_condition),
            evidence_hashes=committed_evidence_hashes,
        )
        _write_json(run_dir / "experiment-annotation.template.json", templates["annotation"])
        _write_json(run_dir / "candidate-workflow.template.json", templates["candidate"])
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
        "provider_usage": agent.provider_usage,
        "level_action_counts": (
            list(result.level_action_counts) if result is not None else None
        ),
        "completion_action_counts": (
            list(result.completion_action_counts) if result is not None else None
        ),
        "reset_count": result.reset_count if result is not None else None,
        "game_over_count": result.game_over_count if result is not None else None,
        "model_calls": agent.model_calls,
        "python_calls": agent.python_calls,
        "model_rounds_per_action": list(agent.model_rounds_per_action),
        "decision_repairs": agent.decision_repairs,
        "host_score_components": scorecard,
        "evidence_hashes": committed_evidence_hashes,
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
