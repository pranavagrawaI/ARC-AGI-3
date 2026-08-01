"""Run the bounded G9 0002-R/0002-NOOA usability pair.

Offline mode validates this runner without credentials. Live mode creates two
fresh instances of the same public game from one immutable ``PairedRunSpec``.
The result licenses interface observations only, never a treatment-effect claim.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
from uuid import uuid4


REPO = Path(__file__).resolve().parents[3]
CONTROL_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = CONTROL_ROOT / "nooa-runtime"
BASELINE_ROOT = REPO / "experiments" / "agents" / "0002-llm-repl-agent"
P9_PARENT = REPO / "experiments" / "agents" / "0006-capability-environment"
for source_root in (REPO, CONTROL_ROOT, BASELINE_ROOT, P9_PARENT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

from experiments.harness import JsonlTraceWriter, Observation, run_episode
from host.qualification import PairedRunSpec, ProviderRequestProfile
from host.responses_client import OpenAICompatibleResponsesClient
from host.runner import NooaProcessAgent, NooaWorkerError
from host.workspace import (
    adapt_control_spec,
    adapt_workspace_for_direct_import,
    direct_import_control_runtime,
)
from llm_repl_agent.agent import DecisionFormatError, ModelBudgetExceeded
from llm_repl_agent.client import ModelRequestError
from p9.arms import arm_manifest, build_arm, control_spec
from p9.fixtures import (
    RecordedEnvironment,
    ScriptedChatClient,
    action_turn,
    default_recording,
    load_recorded_frames,
    tool_turn,
)


EnvironmentFactory = Callable[[str], tuple[Any, Any | None]]

_PROBE_FILE = "G9_CAPABILITY_PROBE.json"
_G9_PROBE_INSTRUCTION = """Usability probe, not game evidence: before choosing the
first action, use Python to create G9_CAPABILITY_PROBE.json with a JSON object whose
first_step is the current step. Before choosing the second action, use Python to read
that same file, add second_step with the current step, and save it. Then choose game
actions normally. This requirement is identical for both arms and supplies no game
semantics."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _operation_mode(value: str) -> OperationMode:
    try:
        return OperationMode(value.lower())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mode must be one of: normal, online, offline, competition"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded 0002-R/0002-NOOA G9 usability pair."
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--offline-recording", type=Path)
    parser.add_argument("--game")
    parser.add_argument("--mode", type=_operation_mode, default=OperationMode.OFFLINE)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--wire-model")
    parser.add_argument("--api-base")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--response-timeout-seconds", type=float)
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=12)
    parser.add_argument("--max-reported-tokens", type=int, default=32_768)
    parser.add_argument("--max-model-rounds-per-action", type=int, default=4)
    parser.add_argument("--max-decision-repairs", type=int, default=2)
    parser.add_argument("--transport-max-retries", type=int, default=2)
    parser.add_argument("--workspace-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--run-root", type=Path)
    return parser


def _spec(args: argparse.Namespace) -> PairedRunSpec:
    if args.offline:
        spec = PairedRunSpec.initial()
    else:
        if not args.game or not args.model or not args.wire_model or not args.api_base:
            raise SystemExit(
                "live mode requires --game, --model, --wire-model, and --api-base"
            )
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
        usability_probe_instruction=_G9_PROBE_INSTRUCTION,
        host_workspace_initialization=PairedRunSpec.workspace_initialization_for(
            _G9_PROBE_INSTRUCTION
        ),
    )


def _fixture_decision(
    available_actions: tuple[int, ...],
    height: int,
    width: int,
) -> dict[str, Any]:
    if not available_actions or height < 1 or width < 1:
        raise ValueError("fixture observation requires an action and positive frame bounds")
    action = available_actions[0]
    data = {"x": 0, "y": 0} if GameAction.from_id(action).is_complex() else None
    return {"action": action, "data": data, "reason": "paired runner fixture"}


def _probe_python(step: int) -> str:
    if step == 0:
        return (
            "import json\n"
            "import frames\n"
            "assert frames.grid(0).ndim == 2\n"
            f"open({_PROBE_FILE!r}, 'w').write(json.dumps({{'first_step': 0}}))"
        )
    if step == 1:
        return (
            "import json\n"
            f"probe = json.load(open({_PROBE_FILE!r}))\n"
            "assert probe['first_step'] == 0\n"
            "probe['second_step'] = 1\n"
            f"open({_PROBE_FILE!r}, 'w').write(json.dumps(probe))"
        )
    return ""


def _candidate_fixture(step: int, observation: Observation) -> list[str]:
    height, width = observation.pixels.shape[-2:]
    decision = _fixture_decision(observation.available_actions, int(height), int(width))
    probe = _probe_python(step)
    source = f"{probe}\n" if probe else ""
    return [source + f"return_result({decision!r})"]


def _control_client_for_live(
    profile: ProviderRequestProfile,
    api_key: str,
) -> OpenAICompatibleResponsesClient:
    return OpenAICompatibleResponsesClient(profile, api_key)


def _probe_file_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"valid": False, "reason": "missing", "path": path.name}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "reason": f"{type(error).__name__}: {error}",
            "path": path.name,
        }
    return {
        "valid": (
            isinstance(value, dict)
            and value.get("first_step") == 0
            and value.get("second_step") == 1
        ),
        "path": path.name,
        "content": value,
    }


def _control_python_action_points(process_path: Path) -> list[int]:
    if not process_path.is_file():
        return []
    return sorted(
        {
            int(record["action_step"])
            for line in process_path.read_text(encoding="utf-8").splitlines()
            if (record := json.loads(line)).get("type") == "python_tool"
            and type(record.get("action_step")) is int
        }
    )


def _candidate_python_action_points(process_path: Path) -> list[int]:
    if not process_path.is_file():
        return []
    points = set()
    for line in process_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "model_crossing":
            continue
        for call in record.get("audit", {}).get("model_calls", []):
            response = call.get("outcome", {}).get("response", {})
            if any(
                tool_call.get("name") == "execute_python"
                for tool_call in response.get("tool_calls", [])
                if isinstance(tool_call, dict)
            ):
                points.add(int(record["action_point_id"]))
    return sorted(points)


def _run_control(
    *,
    spec: PairedRunSpec,
    run_root: Path,
    environment: Any,
    client: Any,
) -> dict[str, Any]:
    with direct_import_control_runtime():
        with build_arm(
            adapt_control_spec(control_spec(**spec.control_overrides())),
            run_dir=run_root,
            inner_client=client,
        ) as arm:
            adapt_workspace_for_direct_import(arm.workspace.root)
            if spec.usability_probe_instruction is not None:
                (arm.workspace.root / "NOTES.md").write_text(
                    spec.usability_probe_instruction,
                    encoding="utf-8",
                )
            constructed_manifest = arm_manifest(arm)
            constructed_manifest["model_visible"]["workspace"][
                "provided_helper_sha256"
            ] = sha256((arm.workspace.root / "frames.py").read_bytes()).hexdigest()
            with JsonlTraceWriter(run_root / "trace.jsonl") as trace:
                result = run_episode(
                    environment,
                    arm.agent,
                    max_actions=spec.max_actions,
                    trace=trace,
                    stop_on_exceptions={
                        ModelBudgetExceeded: "model_budget_exhausted",
                        DecisionFormatError: "unrepaired_invalid_decision",
                        ModelRequestError: "transport_failure",
                    },
                )
            arm.workspace.snapshot(run_root / "workspace")
            summary = {
                "schema": "0002-nooa-g9-arm-v1",
                "arm": "0002-R",
                "actions": result.actions,
                "levels_completed": result.final.levels_completed,
                "state": result.final.state,
                "stop_reason": result.stop_reason,
                "model_calls": arm.agent.model_calls,
                "python_calls": arm.agent.python_calls,
                "python_action_points": _control_python_action_points(
                    run_root / "process.jsonl"
                ),
                "capability_probe": _probe_file_evidence(
                    arm.workspace.root / _PROBE_FILE
                ),
                "clean_shutdown": True,
                "trace_complete": True,
            }
    _write_json(run_root / "arm-manifest.json", constructed_manifest)
    _write_json(run_root / "summary.json", summary)
    return summary


def _run_candidate(
    *,
    spec: PairedRunSpec,
    run_root: Path,
    environment: Any,
    response_timeout: float,
    offline: bool,
) -> dict[str, Any]:
    worker_root = run_root / "worker"
    with NooaProcessAgent(
        runtime_root=RUNTIME_ROOT,
        run_root=worker_root,
        fixture_policy=_candidate_fixture if offline else None,
        paired_run_spec=spec,
        response_timeout=response_timeout,
    ) as agent:
        with JsonlTraceWriter(run_root / "trace.jsonl") as trace:
            result = run_episode(
                environment,
                agent,
                max_actions=spec.max_actions,
                trace=trace,
                stop_on_exceptions={NooaWorkerError: "nooa_worker_failure"},
            )
    summary = {
        "schema": "0002-nooa-g9-arm-v1",
        "arm": "0002-NOOA",
        "actions": result.actions,
        "levels_completed": result.final.levels_completed,
        "state": result.final.state,
        "stop_reason": result.stop_reason,
        "python_action_points": _candidate_python_action_points(
            worker_root / "process.jsonl"
        ),
        "capability_probe": _probe_file_evidence(
            worker_root / "workspace" / _PROBE_FILE
        ),
        "clean_shutdown": True,
        "trace_complete": True,
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def _offline_inputs(args: argparse.Namespace) -> tuple[EnvironmentFactory, ScriptedChatClient]:
    recording = args.offline_recording or default_recording(
        REPO,
        minimum_frames=args.max_actions + 1,
    )
    frames = load_recorded_frames(
        recording,
        count=args.max_actions + 1,
        crop=8,
    )
    script = []
    for frame in frames[:-1]:
        height = len(frame.frame[-1])
        width = len(frame.frame[-1][0])
        decision = _fixture_decision(frame.available_actions, height, width)
        action_step = len([entry for entry in script if entry["kind"] == "content"])
        probe = _probe_python(action_step)
        if probe:
            script.append(tool_turn(probe, call_id=f"probe-{action_step}"))
        script.append(
            action_turn(
                decision["action"],
                data=decision["data"],
                reason=decision["reason"],
            )
        )

    def factory(_arm: str) -> tuple[RecordedEnvironment, None]:
        return RecordedEnvironment(frames=frames), None

    return factory, ScriptedChatClient(script, model="offline-fixture")


def _live_factory(args: argparse.Namespace, run_root: Path) -> EnvironmentFactory:
    def factory(arm: str) -> tuple[Any, Any]:
        arcade = arc_agi.Arcade(
            operation_mode=args.mode,
            recordings_dir=str(run_root / arm / "raw-recordings"),
        )
        environment = arcade.make(
            args.game,
            seed=args.environment_seed,
            save_recording=True,
        )
        if environment is None:
            raise RuntimeError(f"could not create requested game: {args.game}")
        return environment, arcade

    return factory


def _failed_arm_summary(arm: str, error: BaseException) -> dict[str, Any]:
    return {
        "schema": "0002-nooa-g9-arm-v1",
        "arm": arm,
        "actions": 0,
        "levels_completed": 0,
        "state": "HOST_FAILURE",
        "stop_reason": "arm_exception",
        "clean_shutdown": False,
        "trace_complete": False,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _operational_verdict(
    summary: dict[str, Any],
    *,
    max_actions: int,
) -> dict[str, Any]:
    required_actions = 2
    allowed_stops = {"action_budget", "terminal"}
    checks = {
        "multiple_steps": summary.get("actions", 0) >= required_actions,
        "non_blocking_stop": summary.get("stop_reason") in allowed_stops,
        "clean_shutdown": summary.get("clean_shutdown") is True,
        "trace_complete": summary.get("trace_complete") is True,
        "python_on_two_action_points": len(
            set(summary.get("python_action_points", []))
        )
        >= 2,
        "persistent_file_reuse": summary.get("capability_probe", {}).get("valid")
        is True,
    }
    return {
        "success": all(checks.values()),
        "required_actions": required_actions,
        "checks": checks,
    }


def _candidate_provider_crossings(run_root: Path) -> list[dict[str, Any]]:
    process_path = run_root / "candidate" / "worker" / "process.jsonl"
    if not process_path.is_file():
        return []
    crossings: list[dict[str, Any]] = []
    for line in process_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "model_crossing":
            continue
        audit = record.get("audit", {})
        for call in audit.get("model_calls", []):
            crossings.extend(call.get("actual_provider_crossings", []))
    return crossings


def _crossing_profile(crossing: dict[str, Any]) -> dict[str, Any]:
    request = crossing.get("request", {})
    body = request.get("body", {})
    url = request.get("url")
    if isinstance(url, dict):
        path = url.get("path")
    elif isinstance(url, str):
        from urllib.parse import urlsplit

        path = urlsplit(url).path
    else:
        path = None
    reasoning = body.get("reasoning") if isinstance(body, dict) else None
    return {
        "api_surface": (
            "responses"
            if isinstance(path, str) and path.endswith("/responses")
            else path
        ),
        "model": body.get("model") if isinstance(body, dict) else None,
        "max_output_tokens": body.get("max_output_tokens") if isinstance(body, dict) else None,
        "reasoning_effort": (
            reasoning.get("effort") if isinstance(reasoning, dict) else reasoning
        ),
        "temperature": body.get("temperature", "<omitted>") if isinstance(body, dict) else None,
        "seed": body.get("seed", "<omitted>") if isinstance(body, dict) else None,
        "stream": body.get("stream", "<omitted>") if isinstance(body, dict) else None,
        "parallel_tool_calls": (
            body.get("parallel_tool_calls", "<omitted>")
            if isinstance(body, dict)
            else None
        ),
        "tool_choice": body.get("tool_choice", "<omitted>") if isinstance(body, dict) else None,
        "truncation": body.get("truncation", "<omitted>") if isinstance(body, dict) else None,
    }


def _provider_profile_report(
    profile: ProviderRequestProfile,
    *,
    control_crossings: list[dict[str, Any]],
    candidate_crossings: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_static = {
        "api_surface": profile.api_surface,
        "model": profile.wire_model,
        "max_output_tokens": profile.max_output_tokens,
        "reasoning_effort": profile.reasoning_effort,
        "temperature": profile.temperature if profile.temperature is not None else "<omitted>",
        "seed": profile.seed if profile.seed is not None else "<omitted>",
        "stream": profile.stream,
        "truncation": profile.truncation,
    }
    crossings = {
        "0002-R": control_crossings,
        "0002-NOOA": candidate_crossings,
    }
    projections = {
        arm: [_crossing_profile(item) for item in items]
        for arm, items in crossings.items()
    }
    expected_projections: dict[str, list[dict[str, Any]]] = {}
    for arm, items in crossings.items():
        expected_projections[arm] = []
        for item in items:
            body = item.get("request", {}).get("body", {})
            has_tools = isinstance(body, dict) and bool(body.get("tools"))
            expected_projections[arm].append(
                {
                    **expected_static,
                    "parallel_tool_calls": (
                        profile.parallel_tool_calls if has_tools else "<omitted>"
                    ),
                    "tool_choice": profile.tool_choice if has_tools else "<omitted>",
                }
            )
    mismatches = [
        {
            "arm": arm,
            "crossing": index,
            "expected": expected_projections[arm][index],
            "actual": actual,
        }
        for arm, actuals in projections.items()
        for index, actual in enumerate(actuals)
        if actual != expected_projections[arm][index]
    ]
    return {
        "schema": "0002-nooa-g9-provider-profile-v1",
        "success": all(projections.values()) and not mismatches,
        "expected": {
            "static": expected_static,
            "when_tools_present": {
                "parallel_tool_calls": profile.parallel_tool_calls,
                "tool_choice": profile.tool_choice,
            },
            "when_tools_absent": {
                "parallel_tool_calls": "<omitted>",
                "tool_choice": "<omitted>",
            },
        },
        "projections": projections,
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_actions < 2:
        raise SystemExit("G9 requires --max-actions of at least 2")
    api_key = os.environ.get("ARC_LLM_API_KEY")
    if not args.offline and not api_key:
        raise RuntimeError("ARC_LLM_API_KEY must be present in the inherited environment")
    spec = _spec(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pair_id = f"g9-{stamp}-{uuid4().hex[:8]}"
    run_root = (
        args.run_root
        or CONTROL_ROOT / "artifacts" / "paired-usability-probes" / pair_id
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    _write_json(run_root / "paired-run-spec.json", spec.as_record())

    if args.offline:
        environment_factory, control_client = _offline_inputs(args)
    else:
        environment_factory = _live_factory(args, run_root)
        assert spec.provider_request_profile is not None and api_key is not None
        control_client = _control_client_for_live(spec.provider_request_profile, api_key)

    control_arcade = None
    candidate_arcade = None
    try:
        control_environment, control_arcade = environment_factory("control")
        control_summary = _run_control(
            spec=spec,
            run_root=run_root / "control",
            environment=control_environment,
            client=control_client,
        )
    except Exception as error:
        control_summary = _failed_arm_summary("0002-R", error)
        (run_root / "control").mkdir(parents=True, exist_ok=True)
        _write_json(run_root / "control" / "summary.json", control_summary)
    response_timeout = args.response_timeout_seconds
    if response_timeout is None:
        response_timeout = (
            args.max_model_rounds_per_action
            * args.request_timeout_seconds
            * (args.transport_max_retries + 1)
            + args.workspace_timeout_seconds
            + 30.0
            if not args.offline
            else max(
                60.0,
                args.max_model_rounds_per_action * args.workspace_timeout_seconds
                + 30.0,
            )
        )
    try:
        candidate_environment, candidate_arcade = environment_factory("candidate")
        candidate_summary = _run_candidate(
            spec=spec,
            run_root=run_root / "candidate",
            environment=candidate_environment,
            response_timeout=response_timeout,
            offline=args.offline,
        )
    except Exception as error:
        candidate_summary = _failed_arm_summary("0002-NOOA", error)
        (run_root / "candidate").mkdir(parents=True, exist_ok=True)
        _write_json(run_root / "candidate" / "summary.json", candidate_summary)

    if control_arcade is not None:
        _write_json(run_root / "control" / "scorecard.json", control_arcade.get_scorecard())
    if candidate_arcade is not None:
        _write_json(
            run_root / "candidate" / "scorecard.json",
            candidate_arcade.get_scorecard(),
        )
    arm_verdicts = {
        run["arm"]: _operational_verdict(run, max_actions=spec.max_actions)
        for run in (control_summary, candidate_summary)
    }
    provider_report = None
    provider_success = True
    if not args.offline:
        assert isinstance(control_client, OpenAICompatibleResponsesClient)
        control_crossings = control_client.actual_crossings
        candidate_crossings = _candidate_provider_crossings(run_root)
        _write_json(run_root / "control" / "provider-crossings.json", control_crossings)
        _write_json(
            run_root / "candidate" / "provider-crossings.json",
            candidate_crossings,
        )
        assert spec.provider_request_profile is not None
        provider_report = _provider_profile_report(
            spec.provider_request_profile,
            control_crossings=control_crossings,
            candidate_crossings=candidate_crossings,
        )
        _write_json(run_root / "provider-profile-report.json", provider_report)
        provider_success = provider_report["success"]
    operational_success = all(
        verdict["success"] for verdict in arm_verdicts.values()
    ) and provider_success
    summary = {
        "schema": "0002-nooa-g9-pair-v1",
        "pair_id": pair_id,
        "mode": "offline-fixture" if args.offline else "live",
        "claim_boundary": "interface usability only",
        "operational_success": operational_success,
        "arm_verdicts": arm_verdicts,
        "provider_profile_report": provider_report,
        "paired_run_spec_sha256": sha256(
            json.dumps(spec.as_record(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "runs": [control_summary, candidate_summary],
        "run_root": str(run_root),
    }
    _write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if operational_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
