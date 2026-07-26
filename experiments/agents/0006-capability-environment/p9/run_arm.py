"""Run one 0002-R (or treatment) arm as an experiment.

This is `run.py`'s shape minus the 0002-specific authority, freeze, and
release-gate machinery: manifest, trace, workspace snapshot, scorecard, evidence
hashes — and a friction report, because a run that is not read is not evidence.

It has two modes sharing one code path. **Live** builds a real `arc_agi` game and
a real OpenAI-compatible client. **Offline** replays a recording through the
fixture policy. Only the environment and the client differ; the arm, the loop,
and every artifact are identical. Offline exists so the runner itself is tested
before a provider is ever involved — an untested runner is how a live run gets
wasted.

Live::

    uv run python3 .../run_arm.py --game ls20 --model <id> --max-actions 40

Offline (no provider, no endpoint)::

    uv run python3 .../run_arm.py --offline --max-actions 4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "p9"

from p9.arm_diff import assert_authorized, compare_arms
from p9.arms import ArmSpec, arm_manifest, build_arm, control_spec, treatment_spec
from p9.capabilities import TREATMENT_CAPABILITIES, authorized_manifest_paths
from p9.fixtures import RecordedEnvironment, default_recording, load_recorded_frames
from p9.harness_friction import report as friction_report
from p9.task_policy import TaskPolicy

from llm_repl_agent.agent import DecisionFormatError, ModelBudgetExceeded
from llm_repl_agent.client import OpenAICompatibleChatClient
from llm_repl_agent.config import (
    MODEL_PROFILES,
    OPENROUTER_PROFILES,
    apply_openrouter_routing,
    model_request_options,
    resolve_model_environment,
)
from llm_repl_agent.evidence import evidence_hashes
from llm_repl_agent.process_log import write_annotation_template

from experiments.harness import JsonlTraceWriter, run_episode


REPO = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one 0002-R or treatment arm.")
    parser.add_argument("--arm", choices=("control", "treatment"), default="control")
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        choices=sorted(TREATMENT_CAPABILITIES),
        help="treatment capability to enable; repeatable",
    )
    parser.add_argument("--game", help="host-only game id; never shown to the model")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--model-profile", choices=MODEL_PROFILES)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--max-context-turns", type=int, default=6)
    parser.add_argument("--max-model-rounds-per-action", type=int, default=10)
    parser.add_argument("--max-model-calls", type=int, default=100)
    parser.add_argument("--max-reported-tokens", type=int, default=250_000)
    parser.add_argument("--workspace-timeout", type=float, default=20.0)
    parser.add_argument("--color-permutation-seed", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="replay a recording through the fixture policy; no provider or endpoint",
    )
    parser.add_argument("--offline-recording", type=Path)
    parser.add_argument(
        "--pair",
        action="store_true",
        help=(
            "run control and treatment from one derived spec and emit the arm diff; "
            "the two arms cannot differ except by capability, by construction"
        ),
    )
    return parser


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def spec_from_args(args: argparse.Namespace) -> ArmSpec:
    control = control_spec(
        max_actions=args.max_actions,
        max_context_turns=args.max_context_turns,
        max_model_rounds_per_action=args.max_model_rounds_per_action,
        max_model_calls=args.max_model_calls,
        max_reported_tokens=args.max_reported_tokens,
        workspace_timeout_seconds=args.workspace_timeout,
        color_permutation_seed=args.color_permutation_seed,
    )
    if args.arm == "control":
        if args.capability:
            raise SystemExit("--capability requires --arm treatment")
        return control
    capabilities = tuple(args.capability) or ("sentinel_probe_note",)
    return treatment_spec(control, treatment_capabilities=capabilities)


def build_offline(args: argparse.Namespace):
    recording = args.offline_recording or default_recording(
        REPO, minimum_frames=args.max_actions + 1
    )
    frames = load_recorded_frames(recording, count=args.max_actions + 1, crop=8)
    return RecordedEnvironment(frames=frames), TaskPolicy(), {
        "mode": "offline",
        "recording": str(recording),
        "frames": len(frames),
        "model": "fixture-task-policy",
        "admissible_contestant_claim": False,
    }


def build_live(args: argparse.Namespace):
    if not args.game:
        raise SystemExit("--game is required unless --offline")
    import arc_agi
    from arc_agi.base import OperationMode
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    try:
        environment = resolve_model_environment(
            os.environ,
            model=args.model,
            endpoint=args.endpoint,
            profile=args.model_profile,
        )
        options = model_request_options(
            environment.profile,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
        if environment.profile in OPENROUTER_PROFILES:
            options = apply_openrouter_routing(options, os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    game = arcade.make(args.game, seed=args.env_seed)
    if game is None:
        raise RuntimeError(f"could not create requested game: {args.game}")
    client = OpenAICompatibleChatClient(
        base_url=environment.endpoint,
        model=environment.model,
        api_key=environment.api_key,
        **options,
        seed=args.model_seed,
        # Retries live in RetryingChatClient so every attempt is logged.
        rate_limit_retries=0,
    )
    return game, client, {
        "mode": "live",
        "game": args.game,
        "model": environment.model,
        "endpoint": environment.endpoint,
        "model_profile": environment.profile,
        "api_key_source": environment.api_key_source,
        "model_request": options,
        "env_seed": args.env_seed,
        "model_seed": args.model_seed,
        "arcade": arcade,
    }


def run_one(spec: ArmSpec, run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run one arm to completion and return its manifest."""

    environment, client, environment_manifest = (
        build_offline(args) if args.offline else build_live(args)
    )
    arcade = environment_manifest.pop("arcade", None)

    run_id = run_dir.name
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")

    arm = build_arm(spec, run_dir=run_dir, inner_client=client)
    manifest: dict[str, Any] = {
        "schema": "p9-arm-run-v1",
        "run_id": run_id,
        "arm_id": spec.arm_id,
        "treatment_capabilities": list(spec.treatment_capabilities),
        "environment": environment_manifest,
        "arm_manifest": arm_manifest(arm),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    result = None
    error: BaseException | None = None
    stop_reason: str | None = None
    snapshotted = False
    try:
        arm.process_log.write("run_started", run_id=run_id, arm_id=spec.arm_id)
        with JsonlTraceWriter(run_dir / "trace.jsonl") as trace:
            result = run_episode(
                environment,
                arm.agent,
                max_actions=spec.max_actions,
                trace=trace,
                stop_on_exceptions={
                    ModelBudgetExceeded: "model_budget_exhausted",
                    DecisionFormatError: "unrepaired_invalid_decision",
                },
            )
        arm.workspace.snapshot(run_dir / "workspace")
        snapshotted = True
        arm.process_log.write(
            "run_finished",
            actions=result.actions,
            levels_completed=result.final.levels_completed,
            state=result.final.state,
            stop_reason=result.stop_reason,
            usage=arm.agent.usage,
        )
    except BaseException as caught:  # noqa: BLE001 - recorded, then re-raised below
        error = caught
        stop_reason = "error"
        arm.process_log.write("run_failed", error=repr(caught), actions=arm.agent.actions)
    finally:
        try:
            if not snapshotted:
                arm.workspace.snapshot(run_dir / "workspace")
        except OSError:
            pass
        arm.close()

    friction = friction_report([run_dir])
    write_json(run_dir / "friction.json", friction)
    write_annotation_template(
        run_dir / "process-annotation.json",
        actions=result.actions if result is not None else arm.agent.actions,
    )
    if arcade is not None:
        write_json(run_dir / "scorecard.json", arcade.get_scorecard())

    manifest.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if error is not None else "complete",
            "actions": result.actions if result is not None else arm.agent.actions,
            "levels_completed": result.final.levels_completed if result else None,
            "state": result.final.state if result else None,
            "stop_reason": result.stop_reason if result else stop_reason,
            "model_calls": arm.agent.model_calls,
            "python_calls": arm.agent.python_calls,
            "decision_repairs": arm.agent.decision_repairs,
            "transport_attempts": arm.client.attempts,
            "transport_retries": arm.client.retries,
            "usage": dict(arm.agent.usage),
            "friction": {
                "totals": friction["totals"],
                "by_kind": friction["friction_by_kind"],
                "by_attribution": friction["friction_by_attribution"],
            },
            "evidence_hashes": evidence_hashes(run_dir),
            "error": repr(error) if error is not None else None,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    if error is not None:
        raise error
    return manifest


def _summary(manifest: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "arm": manifest["arm_id"],
        "status": manifest["status"],
        "actions": manifest["actions"],
        "stop_reason": manifest["stop_reason"],
        "friction": manifest["friction"]["by_kind"],
        "run_dir": str(run_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid4().hex[:8]

    if not args.pair:
        spec = spec_from_args(args)
        run_dir = args.run_dir or ARTIFACTS / f"{spec.arm_id}-{stamp}-{token}"
        manifest = run_one(spec, run_dir, args)
        print(json.dumps(_summary(manifest, run_dir), indent=2, sort_keys=True))
        return 0

    # A pair derives both arms from one spec, so the two cannot differ except by
    # capability. Assembling a pair by hand is exactly how a mismatched comparison
    # gets made, and the diff below would then be measuring the operator.
    if args.arm != "control":
        raise SystemExit("--pair derives both arms; do not also pass --arm")
    control = spec_from_args(argparse.Namespace(**{**vars(args), "arm": "control", "capability": []}))
    treatment = treatment_spec(
        control, treatment_capabilities=tuple(args.capability) or ("sentinel_probe_note",)
    )
    root = args.run_dir or ARTIFACTS / f"pair-{stamp}-{token}"
    manifests = {}
    summaries = []
    for spec in (control, treatment):
        directory = root / ("control" if spec is control else "treatment")
        manifests[spec.arm_id] = run_one(spec, directory, args)
        summaries.append(_summary(manifests[spec.arm_id], directory))

    diff = compare_arms(
        manifests[control.arm_id]["arm_manifest"],
        manifests[treatment.arm_id]["arm_manifest"],
        authorized_paths=authorized_manifest_paths(treatment.treatment_capabilities),
    )
    write_json(
        root / "pair-diff.json",
        {
            "schema": "p9-arm-pair-v1",
            "control": control.arm_id,
            "treatment": treatment.arm_id,
            "treatment_capabilities": list(treatment.treatment_capabilities),
            "diff": diff.as_json(),
        },
    )
    print(
        json.dumps(
            {
                "pair_dir": str(root),
                "runs": summaries,
                "shared_manifest_paths": f"{diff.shared_paths}/{diff.compared_paths}",
                "unauthorized_differences": [item.path for item in diff.unauthorized],
            },
            indent=2,
            sort_keys=True,
        )
    )
    assert_authorized(diff, label="pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
