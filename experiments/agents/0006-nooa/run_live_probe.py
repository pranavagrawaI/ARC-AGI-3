"""Run one bounded 0006-NOOA provider probe through the public ARC harness.

This is a single-arm treatment smoke, not a matched treatment-effect result. The
provider profile is explicit, credentials are inherited only as
``ARC_LLM_API_KEY``, and no dotenv file is searched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPO = Path(__file__).resolve().parents[3]
TREATMENT_ROOT = Path(__file__).resolve().parent
CONTROL_ROOT = TREATMENT_ROOT.parent / "0002-nooa"
CONTROL_RUNTIME_ROOT = CONTROL_ROOT / "nooa-runtime"
PROVIDER_RUNTIME_LAUNCHER = TREATMENT_ROOT / "provider_runtime_launcher.py"
BASELINE_ROOT = TREATMENT_ROOT.parent / "0002-llm-repl-agent"
P9_ROOT = TREATMENT_ROOT.parent / "0006-capability-environment" / "p9"
for path in (REPO, CONTROL_ROOT, TREATMENT_ROOT, TREATMENT_ROOT / "host"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arc_agi
from arc_agi.base import OperationMode
from competence_runner import NooaCompetenceProcessAgent
from host.qualification import PairedRunSpec, ProviderRequestProfile
from host.runner import NooaWorkerError

from experiments.harness import JsonlTraceWriter, run_episode

FROZEN_CONTROL_REVISION = "dfb2d2b"
CONFIRMATION_PREFIX = "RUN-0006-NOOA-SINGLE-ARM:"
EXPECTED_GAME = "ls20"
EXPECTED_MODEL = "openai/google/gemma-4-31b-it"
EXPECTED_WIRE_MODEL = "google/gemma-4-31b-it"
EXPECTED_API_BASE = "https://openrouter.ai/api/v1"
SMOKE_MAXIMA = {
    "max_actions": 3,
    "max_model_calls": 12,
    "max_reported_tokens": 32_768,
    "max_output_tokens": 16_384,
    "max_model_rounds_per_action": 4,
    "max_decision_repairs": 2,
}
PROVIDER_ROUTE = {
    "order": ["modelrun/fp4"],
    "only": ["modelrun/fp4"],
    "allow_fallbacks": False,
    "require_parameters": False,
    "quantizations": ["fp4"],
}
GENERATION_AUDIT_ATTEMPTS = 6
GENERATION_AUDIT_DELAY_SECONDS = 2.0
GENERATION_AUDIT_TIMEOUT_SECONDS = 10.0
GENERATION_AUDIT_URL = "https://openrouter.ai/api/v1/generation"


def _operation_mode(value: str) -> OperationMode:
    try:
        return OperationMode(value.lower())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mode must be one of: normal, online, offline, competition"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded single-arm 0006-NOOA real-provider smoke probe."
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
            "Host wait for one action response. Defaults to request timeout times "
            "all permitted rounds and attempts, plus workspace timeout and 30 seconds."
        ),
    )
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=12)
    parser.add_argument("--max-reported-tokens", type=int, default=32_768)
    parser.add_argument("--max-model-rounds-per-action", type=int, default=4)
    parser.add_argument("--max-decision-repairs", type=int, default=2)
    parser.add_argument("--transport-max-retries", type=int, default=0)
    parser.add_argument("--workspace-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--run-root", type=Path, required=True)
    authority = parser.add_mutually_exclusive_group()
    authority.add_argument(
        "--prepare-approval-request",
        type=Path,
        help="No-spend mode: write the exact source/spec/run-bound approval request.",
    )
    authority.add_argument(
        "--approval-request",
        type=Path,
        help="Live mode: exact approval request produced by this source generation.",
    )
    parser.add_argument(
        "--confirm-live-run",
        help="Required exact digest-bound value from the approval request.",
    )
    return parser


def _validate_live_args(args: argparse.Namespace) -> None:
    exact = {
        "game": EXPECTED_GAME,
        "mode": OperationMode.OFFLINE,
        "environment_seed": 0,
        "model": EXPECTED_MODEL,
        "wire_model": EXPECTED_WIRE_MODEL,
        "api_base": EXPECTED_API_BASE,
        "reasoning_effort": "medium",
        "request_timeout_seconds": 180.0,
        "workspace_timeout_seconds": 20.0,
        "transport_max_retries": 0,
    }
    for field, expected in exact.items():
        actual = getattr(args, field)
        if actual != expected:
            raise ValueError(f"{field} must be {expected!r}, got {actual!r}")
    for field, maximum in SMOKE_MAXIMA.items():
        actual = getattr(args, field)
        if not 1 <= actual <= maximum:
            raise ValueError(f"{field} must be between 1 and {maximum}, got {actual}")
    if args.max_output_tokens > args.max_reported_tokens:
        raise ValueError("max_output_tokens cannot exceed max_reported_tokens")
    if args.max_model_rounds_per_action > args.max_model_calls:
        raise ValueError("per-action model rounds cannot exceed the total model-call gate")
    minimum_timeout = (
        args.max_model_rounds_per_action
        * (
            args.request_timeout_seconds * (args.transport_max_retries + 1)
            + args.workspace_timeout_seconds
        )
        + 30.0
    )
    if args.response_timeout_seconds is not None and not isfinite(
        args.response_timeout_seconds
    ):
        raise ValueError("response_timeout_seconds must be finite")
    if (
        args.response_timeout_seconds is not None
        and args.response_timeout_seconds < minimum_timeout
    ):
        raise ValueError(
            "response_timeout_seconds is below the configured per-round worst case "
            f"({args.response_timeout_seconds} < {minimum_timeout})"
        )


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


def _git_record() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def _source_paths() -> set[Path]:
    return {
        TREATMENT_ROOT / "competence.py",
        PROVIDER_RUNTIME_LAUNCHER,
        TREATMENT_ROOT / "run_live_probe.py",
        *(TREATMENT_ROOT / "host").glob("*.py"),
        *(CONTROL_ROOT / "host").glob("*.py"),
        *(CONTROL_RUNTIME_ROOT).glob("*.py"),
        CONTROL_RUNTIME_ROOT / "pyproject.toml",
        CONTROL_RUNTIME_ROOT / "uv.lock",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
        *(BASELINE_ROOT / "llm_repl_agent").glob("*.py"),
        *P9_ROOT.glob("*.py"),
        *(REPO / "experiments" / "harness").glob("*.py"),
    }


def _participating_sources() -> dict[str, str]:
    return {
        path.relative_to(REPO).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(_source_paths())
        if path.is_file()
    }


def _verify_source_snapshot(
    snapshot_root: Path,
    *,
    expected_sha256: dict[str, str],
) -> None:
    mismatches: list[str] = []
    for relative, expected in sorted(expected_sha256.items()):
        path = snapshot_root / relative
        actual = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"source snapshot differs from approved hashes: {mismatches}")


def _snapshot_participating_sources(
    run_root: Path,
    *,
    expected_sha256: dict[str, str],
) -> Path:
    snapshot_root = run_root / "source-snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for relative, expected in sorted(expected_sha256.items()):
        source = REPO / relative
        if not source.is_file():
            raise RuntimeError(f"approved source is missing: {relative}")
        data = source.read_bytes()
        if sha256(data).hexdigest() != expected:
            raise RuntimeError(f"approved source changed before snapshot: {relative}")
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    _verify_source_snapshot(snapshot_root, expected_sha256=expected_sha256)
    _write_json(
        snapshot_root / "record.json",
        {
            "schema": "0006-nooa-live-source-snapshot-v1",
            "source_sha256": expected_sha256,
        },
    )
    return snapshot_root


def _frozen_control_paths(revision: str) -> list[Path]:
    host_prefix = (CONTROL_ROOT / "host").relative_to(REPO)
    runtime_prefix = CONTROL_RUNTIME_ROOT.relative_to(REPO)
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            host_prefix.as_posix(),
            runtime_prefix.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    paths: list[Path] = []
    for relative_text in tracked:
        relative = Path(relative_text)
        is_host_file = relative.parent == host_prefix and relative.suffix == ".py"
        is_runtime_file = relative.parent == runtime_prefix and (
            relative.suffix == ".py" or relative.name in {"pyproject.toml", "uv.lock"}
        )
        if is_host_file or is_runtime_file:
            paths.append(REPO / relative)
    return sorted(paths)


def _frozen_control_record(
    *,
    current_source_paths: list[Path] | None = None,
) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", FROZEN_CONTROL_REVISION],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = _frozen_control_paths(revision)
    current_paths = {
        path
        for path in (
            _source_paths() if current_source_paths is None else current_source_paths
        )
        if path == CONTROL_RUNTIME_ROOT / "pyproject.toml"
        or path == CONTROL_RUNTIME_ROOT / "uv.lock"
        or path.parent in {CONTROL_ROOT / "host", CONTROL_RUNTIME_ROOT}
    }
    mismatches = [
        path.relative_to(REPO).as_posix()
        for path in sorted(current_paths - set(paths))
    ]
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        frozen = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{revision}:{relative}"],
            capture_output=True,
            check=False,
        )
        current = path.read_bytes() if path.is_file() else None
        if frozen.returncode != 0 or current is None or frozen.stdout != current:
            mismatches.append(relative)
    return {
        "revision": revision,
        "checked_paths": [path.relative_to(REPO).as_posix() for path in paths],
        "mismatches": mismatches,
        "matches": not mismatches,
    }


def _source_state() -> dict[str, Any]:
    runtime_python = subprocess.run(
        [str(CONTROL_RUNTIME_ROOT / ".venv" / "bin" / "python"), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    frozen_control = _frozen_control_record()
    if not frozen_control["matches"]:
        raise RuntimeError(
            "frozen 0002-NOOA sources differ from "
            f"{FROZEN_CONTROL_REVISION}: {frozen_control['mismatches']}"
        )
    return {
        "schema": "0006-nooa-live-source-state-v2",
        "git": _git_record(),
        "runtime_versions": {
            "host_python": sys.version,
            "nooa_python": (runtime_python.stdout or runtime_python.stderr).strip(),
        },
        "frozen_control": frozen_control,
        "source_sha256": _participating_sources(),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _response_timeout(args: argparse.Namespace) -> float:
    if args.response_timeout_seconds is not None:
        return args.response_timeout_seconds
    return (
        args.max_model_rounds_per_action
        * (
            args.request_timeout_seconds * (args.transport_max_retries + 1)
            + args.workspace_timeout_seconds
        )
        + 30.0
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def _approval_request(
    *,
    spec: PairedRunSpec,
    run_root: Path,
    source_state: dict[str, Any],
) -> dict[str, Any]:
    authorization = {
        "schema": "0006-nooa-live-authorization-v2",
        "claim_boundary": "single-arm live interface-and-behavior smoke; no treatment effect",
        "provider_routing": {
            "mode": "openrouter-exact-endpoint-pin",
            "request": PROVIDER_ROUTE,
        },
        "provider_identity_audit": {
            "endpoint": GENERATION_AUDIT_URL,
            "attempt_limit_per_generation": GENERATION_AUDIT_ATTEMPTS,
            "delay_seconds": GENERATION_AUDIT_DELAY_SECONDS,
            "timeout_seconds": GENERATION_AUDIT_TIMEOUT_SECONDS,
        },
        "run_root": str(run_root.resolve()),
        "paired_run_spec": spec.as_record(),
        "source_state": source_state,
    }
    return {
        "schema": "0006-nooa-live-approval-request-v2",
        "authorization": authorization,
        "required_confirmation": CONFIRMATION_PREFIX + _canonical_sha256(authorization),
    }


def _require_approved_request(
    path: Path,
    *,
    expected: dict[str, Any],
    confirmation: str | None,
) -> None:
    try:
        approved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read approval request {path}: {error}") from error
    if approved != expected:
        raise RuntimeError("approval request does not match the exact current source/spec/run identity")
    if confirmation != expected["required_confirmation"]:
        raise RuntimeError("live execution is not authorized by the exact approval digest")


def _consume_approved_request(
    path: Path,
    *,
    expected: dict[str, Any],
    confirmation: str | None,
) -> Path:
    _require_approved_request(path, expected=expected, confirmation=confirmation)
    consumed = path.with_name(path.name + ".consumed")
    if consumed.exists():
        raise RuntimeError(f"approval request was already consumed: {consumed}")
    try:
        path.replace(consumed)
    except OSError as error:
        raise RuntimeError(f"could not atomically consume approval request {path}: {error}") from error
    return consumed


def _materialize_runtime_overlay(
    run_root: Path,
    *,
    expected_sha256: dict[str, str],
) -> Path:
    snapshot_root = run_root / "source-snapshot"
    _verify_source_snapshot(snapshot_root, expected_sha256=expected_sha256)
    overlay_launcher = (
        snapshot_root / "experiments/agents/0006-nooa/provider_runtime_launcher.py"
    )
    base_runtime_root = (
        snapshot_root / "experiments/agents/0002-nooa/nooa-runtime"
    )
    runtime_root = run_root / "provider-runtime"
    runtime_root.mkdir(parents=True, exist_ok=False)
    (runtime_root / "launcher.py").write_bytes(overlay_launcher.read_bytes())
    _write_json(
        runtime_root / "transport-config.json",
        {
            "schema": "0006-nooa-provider-runtime-overlay-v1",
            "base_runtime_root": str(base_runtime_root.resolve()),
            "provider": PROVIDER_ROUTE,
        },
    )
    (runtime_root / ".venv").symlink_to(
        (CONTROL_RUNTIME_ROOT / ".venv").resolve(),
        target_is_directory=True,
    )
    return runtime_root


def _candidate_provider_crossings(run_root: Path) -> list[dict[str, Any]]:
    process_path = run_root / "worker" / "process.jsonl"
    if not process_path.is_file():
        return []
    crossings: list[dict[str, Any]] = []
    for line in process_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "model_crossing":
            continue
        for call in record.get("audit", {}).get("model_calls", []):
            crossings.extend(call.get("actual_provider_crossings", []))
    return crossings


def _crossing_profile(crossing: dict[str, Any]) -> dict[str, Any]:
    request = crossing.get("request", {})
    body = request.get("body", {})
    url = request.get("url")
    if isinstance(url, dict):
        path = url.get("path")
    elif isinstance(url, str):
        path = urlsplit(url).path
    else:
        path = None
    reasoning = body.get("reasoning") if isinstance(body, dict) else None
    return {
        "method": str(request.get("method", "<missing>")).upper(),
        "url": url if isinstance(url, str) else "<missing>",
        "api_surface": "responses" if isinstance(path, str) and path.endswith("/responses") else path,
        "model": body.get("model") if isinstance(body, dict) else None,
        "max_output_tokens": body.get("max_output_tokens") if isinstance(body, dict) else None,
        "reasoning_effort": reasoning.get("effort") if isinstance(reasoning, dict) else reasoning,
        "temperature": body.get("temperature", "<omitted>") if isinstance(body, dict) else None,
        "seed": body.get("seed", "<omitted>") if isinstance(body, dict) else None,
        "stream": body.get("stream", "<omitted>") if isinstance(body, dict) else None,
        "parallel_tool_calls": (
            body.get("parallel_tool_calls", "<omitted>") if isinstance(body, dict) else None
        ),
        "tool_choice": body.get("tool_choice", "<omitted>") if isinstance(body, dict) else None,
        "truncation": body.get("truncation", "<omitted>") if isinstance(body, dict) else None,
        "provider": body.get("provider", "<omitted>") if isinstance(body, dict) else None,
    }


def _provider_profile_report(
    profile: ProviderRequestProfile,
    crossings: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_static = {
        "method": "POST",
        "url": f"{profile.api_base.rstrip('/')}/responses",
        "api_surface": profile.api_surface,
        "model": profile.wire_model,
        "max_output_tokens": profile.max_output_tokens,
        "reasoning_effort": profile.reasoning_effort,
        "temperature": profile.temperature if profile.temperature is not None else "<omitted>",
        "seed": profile.seed if profile.seed is not None else "<omitted>",
        "stream": profile.stream,
        "truncation": profile.truncation,
        "provider": PROVIDER_ROUTE,
    }
    actuals = [_crossing_profile(item) for item in crossings]
    expected: list[dict[str, Any]] = []
    for item in crossings:
        body = item.get("request", {}).get("body", {})
        has_tools = isinstance(body, dict) and bool(body.get("tools"))
        expected.append(
            {
                **expected_static,
                "parallel_tool_calls": profile.parallel_tool_calls if has_tools else "<omitted>",
                "tool_choice": profile.tool_choice if has_tools else "<omitted>",
            }
        )
    mismatches = [
        {"crossing": index, "expected": expected[index], "actual": actual}
        for index, actual in enumerate(actuals)
        if actual != expected[index]
    ]
    return {
        "schema": "0006-nooa-provider-profile-v1",
        "success": bool(actuals) and not mismatches,
        "expected_static": expected_static,
        "projections": actuals,
        "mismatches": mismatches,
    }


def _generation_ids(crossings: list[dict[str, Any]]) -> list[str]:
    generation_ids: list[str] = []
    for crossing in crossings:
        response = crossing.get("response", {})
        headers = response.get("headers", {}) if isinstance(response, dict) else {}
        body = response.get("body", {}) if isinstance(response, dict) else {}
        generation_id = None
        if isinstance(headers, dict):
            generation_id = next(
                (
                    value
                    for key, value in headers.items()
                    if str(key).lower() == "x-generation-id" and isinstance(value, str)
                ),
                None,
            )
        if generation_id is None and isinstance(body, dict):
            body_id = body.get("id")
            generation_id = body_id if isinstance(body_id, str) else None
        if generation_id is not None and generation_id not in generation_ids:
            generation_ids.append(generation_id)
    return generation_ids


def _generation_usage_by_id(
    crossings: list[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    usage_by_id: dict[str, dict[str, int | float | None]] = {}
    for crossing in crossings:
        generation_ids = _generation_ids([crossing])
        if len(generation_ids) != 1:
            continue
        response = crossing.get("response", {})
        body = response.get("body", {}) if isinstance(response, dict) else {}
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        output_details = (
            usage.get("output_tokens_details", {}) if isinstance(usage, dict) else {}
        )
        usage_by_id[generation_ids[0]] = {
            "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
            "output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
            "reasoning_tokens": (
                output_details.get("reasoning_tokens")
                if isinstance(output_details, dict)
                else None
            ),
            "cost_usd": usage.get("cost") if isinstance(usage, dict) else None,
        }
    return usage_by_id


def _model_revision_matches(model_revision: object, expected_wire_model: str) -> bool:
    return isinstance(model_revision, str) and (
        model_revision == expected_wire_model
        or model_revision.startswith(expected_wire_model + "-")
    )


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and value >= 0
    )


def _provider_identity_report(
    generation_ids: list[str],
    generation_payloads: list[dict[str, Any]],
    *,
    expected_wire_model: str,
    crossing_usage: dict[str, dict[str, int | float | None]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for payload in generation_payloads:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        provider_responses = data.get("provider_responses")
        record = {
                "id": data.get("id"),
                "model_revision": data.get("model"),
                "provider_name": data.get("provider_name"),
                "provider_responses": (
                    provider_responses if isinstance(provider_responses, list) else []
                ),
                "native_finish_reason": data.get("native_finish_reason"),
                "normalized_finish_reason": data.get("finish_reason"),
                "input_tokens": data.get("native_tokens_prompt"),
                "output_tokens": data.get("native_tokens_completion"),
                "reasoning_tokens": data.get("native_tokens_reasoning"),
                "cost_usd": data.get("total_cost", data.get("usage")),
            }
        errors: list[str] = []
        if not _model_revision_matches(record["model_revision"], expected_wire_model):
            errors.append("model_revision")
        if not isinstance(record["native_finish_reason"], str) or not record["native_finish_reason"]:
            errors.append("native_finish_reason")
        if not isinstance(record["normalized_finish_reason"], str) or not record["normalized_finish_reason"]:
            errors.append("normalized_finish_reason")
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            if not _nonnegative_integer(record[field]):
                errors.append(field)
        if not _nonnegative_number(record["cost_usd"]):
            errors.append("cost_usd")
        responses = record["provider_responses"]
        if not responses or not all(
            isinstance(response, dict)
            and response.get("provider_name") == "ModelRun"
            and response.get("status") == 200
            and _model_revision_matches(response.get("model_permaslug"), expected_wire_model)
            for response in responses
        ):
            errors.append("provider_responses")
        expected_usage = crossing_usage.get(str(record["id"]))
        actual_usage = {
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
            "reasoning_tokens": record["reasoning_tokens"],
            "cost_usd": record["cost_usd"],
        }
        if expected_usage is None or actual_usage != expected_usage:
            errors.append("crossing_usage")
        record["validation_errors"] = errors
        records.append(record)
    returned_ids = [record.get("id") for record in records]
    metadata_matches = bool(records) and all(
        record.get("provider_name") == "ModelRun"
        and not record["validation_errors"]
        for record in records
    )
    success = (
        bool(generation_ids)
        and returned_ids == generation_ids
        and metadata_matches
    )
    return {
        "schema": "0006-nooa-provider-identity-v1",
        "success": success,
        "expected_generation_ids": generation_ids,
        "actual_provider": "ModelRun" if success else None,
        "actual_quantization": "fp4" if success else None,
        "quantization_evidence": (
            "exclusive modelrun/fp4 request pin with fallbacks disabled"
            if success
            else None
        ),
        "records": records,
    }


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def _generation_artifact_path(run_root: Path, generation_id: str) -> Path:
    if re.fullmatch(r"gen-[A-Za-z0-9_-]{1,128}", generation_id) is None:
        raise ValueError(f"invalid OpenRouter generation ID: {generation_id!r}")
    directory = (run_root / "openrouter-generations").resolve()
    path = (directory / f"{generation_id}.raw.json").resolve()
    if path.parent != directory:
        raise ValueError(f"generation ID escapes artifact directory: {generation_id!r}")
    return path


def _retrieve_generation_payloads(
    generation_ids: list[str],
    *,
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_key = os.environ.get("ARC_LLM_API_KEY")
    payloads: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if not api_key:
        return payloads, {
            "schema": "0006-nooa-generation-fetch-v1",
            "success": False,
            "error": {"type": "MissingCredential", "message": "ARC_LLM_API_KEY absent"},
            "results": results,
        }
    opener = build_opener(_NoRedirectHandler())
    for generation_id in generation_ids:
        raw_path = _generation_artifact_path(run_root, generation_id)
        attempts: list[dict[str, Any]] = []
        payload: dict[str, Any] | None = None
        for attempt in range(1, GENERATION_AUDIT_ATTEMPTS + 1):
            url = f"{GENERATION_AUDIT_URL}?{urlencode({'id': generation_id})}"
            request = Request(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                method="GET",
            )
            try:
                with opener.open(
                    request,
                    timeout=GENERATION_AUDIT_TIMEOUT_SECONDS,
                ) as response:
                    raw = response.read()
                    candidate = json.loads(raw)
                    attempts.append({"attempt": attempt, "status": response.status})
                    if isinstance(candidate, dict) and isinstance(candidate.get("data"), dict):
                        payload = candidate
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_bytes(raw)
                        break
            except HTTPError as error:
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": error.code,
                        "body": error.read().decode("utf-8", errors="replace"),
                    }
                )
            except (URLError, TimeoutError, json.JSONDecodeError) as error:
                attempts.append(
                    {
                        "attempt": attempt,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
                )
            if attempt < GENERATION_AUDIT_ATTEMPTS:
                sleep(GENERATION_AUDIT_DELAY_SECONDS)
        if payload is not None:
            payloads.append(payload)
        results.append(
            {
                "generation_id": generation_id,
                "success": payload is not None,
                "attempts": attempts,
            }
        )
    return payloads, {
        "schema": "0006-nooa-generation-fetch-v1",
        "success": bool(generation_ids) and len(payloads) == len(generation_ids),
        "attempt_limit_per_generation": GENERATION_AUDIT_ATTEMPTS,
        "delay_seconds": GENERATION_AUDIT_DELAY_SECONDS,
        "timeout_seconds": GENERATION_AUDIT_TIMEOUT_SECONDS,
        "results": results,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trace_complete(path: Path) -> bool:
    try:
        records = _read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(records) and (records[-1].get("kind") or records[-1].get("type")) == "end"


def _artifact_counts(run_root: Path) -> dict[str, int]:
    try:
        trace_records = _read_jsonl(run_root / "trace.jsonl")
    except (OSError, json.JSONDecodeError):
        trace_records = []
    try:
        process_records = _read_jsonl(run_root / "worker" / "process.jsonl")
    except (OSError, json.JSONDecodeError):
        process_records = []
    trace_successors = sum(
        record.get("type") in {"transition", "suppressed_transition"}
        for record in trace_records
    )
    admitted_actions = sum(
        record.get("kind") == "competence_action_admitted" for record in process_records
    )
    recorded_transitions = sum(
        record.get("kind") == "transition_recorded" for record in process_records
    )
    return {
        "attempted_actions": max(admitted_actions, trace_successors),
        "observed_successors": max(recorded_transitions, trace_successors),
    }


def _predecessor_result_exposed(events: list[dict[str, Any]]) -> bool:
    result_indexes = [index for index, event in enumerate(events) if event.get("type") == "prediction_result"]
    context_indexes = [index for index, event in enumerate(events) if event.get("type") == "context_assembled"]
    return bool(result_indexes) and any(
        context_index > result_indexes[0] for context_index in context_indexes
    )


def _claim_continuity(events: list[dict[str, Any]]) -> bool:
    commitments = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") == "commitment"
    ]
    if len(commitments) < 2:
        return False
    first_index, first = commitments[0]
    second_index, second = commitments[1]
    first_claim_ids = set(first.get("claim_ids", []))
    second_claim_ids = set(second.get("claim_ids", []))
    if not first_claim_ids:
        return False
    if first_claim_ids & second_claim_ids:
        return True
    return any(
        event.get("type") == "ledger"
        and first_index < index < second_index
        and isinstance(event.get("operation"), dict)
        and event["operation"].get("claim_id") in first_claim_ids
        and event["operation"].get("op") in {"revise", "retire"}
        for index, event in enumerate(events)
    )


def _mismatches_reconciled_before_next_commitment(
    events: list[dict[str, Any]],
) -> bool:
    commitment_indexes = [
        index for index, event in enumerate(events) if event.get("type") == "commitment"
    ]
    for index, event in enumerate(events):
        if event.get("type") != "prediction_result" or event.get("outcome") != "mismatch":
            continue
        next_commitment = next(
            (commitment_index for commitment_index in commitment_indexes if commitment_index > index),
            None,
        )
        if next_commitment is None:
            continue
        if not any(
            candidate.get("type") == "reconciliation"
            for candidate in events[index + 1 : next_commitment]
        ):
            return False
    return True


def _operational_verdict(
    summary: dict[str, Any],
    *,
    competence_events: list[dict[str, Any]],
    provider_success: bool,
    required_actions: int,
) -> dict[str, Any]:
    event_types = [event.get("type") for event in competence_events]
    if required_actions == 1:
        checks = {
            "one_action_admitted": summary.get("actions") == 1,
            "one_action_attempted": summary.get("attempted_actions") == 1,
            "one_successor_observed": summary.get("observed_successors") == 1,
            "non_blocking_stop": summary.get("stop_reason") in {"action_budget", "terminal"},
            "clean_shutdown": summary.get("clean_shutdown") is True,
            "trace_complete": summary.get("trace_complete") is True,
            "one_commitment": event_types.count("commitment") == 1,
            "provider_profile_matches": provider_success,
        }
        return {
            "goal": "one-action-admission",
            "success": all(checks.values()),
            "checks": checks,
        }
    checks = {
        "exact_actions": summary.get("actions") == required_actions,
        "exact_attempted_actions": summary.get("attempted_actions") == required_actions,
        "exact_successors": summary.get("observed_successors") == required_actions,
        "non_blocking_stop": summary.get("stop_reason") in {"action_budget", "terminal"},
        "clean_shutdown": summary.get("clean_shutdown") is True,
        "trace_complete": summary.get("trace_complete") is True,
        "exact_commitments": event_types.count("commitment") == required_actions,
        "predecessor_prediction_result_exposed": _predecessor_result_exposed(competence_events),
        "claim_continuity": _claim_continuity(competence_events),
        "mismatches_reconciled_before_next_commitment": (
            _mismatches_reconciled_before_next_commitment(competence_events)
        ),
        "provider_profile_matches": provider_success,
    }
    return {
        "goal": f"{required_actions}-action-competence-continuity",
        "success": all(checks.values()),
        "checks": checks,
    }


def _failure_summary(
    *,
    game: str,
    environment_seed: int,
    run_root: Path,
    phase: str,
    error: BaseException,
) -> dict[str, Any]:
    counts = _artifact_counts(run_root)
    return {
        "schema": "0006-nooa-single-arm-provider-smoke-v2",
        "claim_boundary": "single-arm live interface-and-behavior smoke; no treatment effect",
        "game": game,
        "environment_seed": environment_seed,
        "actions": counts["observed_successors"],
        **counts,
        "levels_completed": 0,
        "state": "HOST_FAILURE",
        "stop_reason": "stage_exception",
        "failure_phase": phase,
        "error": {"type": type(error).__name__, "message": str(error)},
        "clean_shutdown": False,
        "trace_complete": False,
        "evidence_paths": {
            "trace": str(run_root / "trace.jsonl"),
            "worker_process": str(run_root / "worker" / "process.jsonl"),
            "competence": str(run_root / "worker" / "competence.jsonl"),
        },
        "run_root": str(run_root),
    }


def _run_stage(
    args: argparse.Namespace,
    *,
    spec: PairedRunSpec,
    run_root: Path,
    response_timeout: float,
    expected_source_sha256: dict[str, str],
) -> dict[str, Any]:
    phase = "source_snapshot"
    try:
        _snapshot_participating_sources(
            run_root,
            expected_sha256=expected_source_sha256,
        )
        phase = "arcade_init"
        arcade = arc_agi.Arcade(
            operation_mode=args.mode,
            recordings_dir=str(run_root / "raw-recordings"),
        )
        phase = "environment_create"
        environment = arcade.make(
            args.game,
            seed=args.environment_seed,
            save_recording=True,
        )
        if environment is None:
            raise RuntimeError(f"could not create requested game: {args.game}")
        phase = "provider_runtime_materialization"
        provider_runtime_root = _materialize_runtime_overlay(
            run_root,
            expected_sha256=expected_source_sha256,
        )
        phase = "worker_start_or_episode"
        with NooaCompetenceProcessAgent(
            runtime_root=provider_runtime_root,
            run_root=run_root / "worker",
            paired_run_spec=spec,
            response_timeout=response_timeout,
        ) as agent, JsonlTraceWriter(run_root / "trace.jsonl") as trace:
            result = run_episode(
                environment,
                agent,
                max_actions=spec.max_actions,
                trace=trace,
                stop_on_exceptions={NooaWorkerError: "agent_error"},
            )
            phase = "worker_shutdown"
        summary = {
            "schema": "0006-nooa-single-arm-provider-smoke-v2",
            "claim_boundary": "single-arm live interface-and-behavior smoke; no treatment effect",
            "game": args.game,
            "environment_seed": args.environment_seed,
            "actions": result.actions,
            **_artifact_counts(run_root),
            "levels_completed": result.final.levels_completed,
            "state": result.final.state,
            "stop_reason": result.stop_reason,
            "clean_shutdown": True,
            "trace_complete": _trace_complete(run_root / "trace.jsonl"),
            "run_root": str(run_root),
        }
    except Exception as error:  # noqa: BLE001 - preserve every post-manifest failure.
        summary = _failure_summary(
            game=args.game,
            environment_seed=args.environment_seed,
            run_root=run_root,
            phase=phase,
            error=error,
        )
        summary["trace_complete"] = _trace_complete(run_root / "trace.jsonl")
    _write_json(run_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_live_args(args)
    spec = _paired_spec(args)
    run_root = args.run_root.resolve()
    source_state = _source_state()
    approval_request = _approval_request(
        spec=spec,
        run_root=run_root,
        source_state=source_state,
    )
    if args.prepare_approval_request is not None:
        request_path = args.prepare_approval_request.resolve()
        if request_path.exists():
            raise RuntimeError(f"refusing to overwrite approval request: {request_path}")
        if run_root.exists():
            raise RuntimeError(f"approved run root already exists: {run_root}")
        _write_json(request_path, approval_request)
        print(json.dumps(approval_request, indent=2, sort_keys=True))
        return 0
    if args.approval_request is None:
        raise RuntimeError("live execution requires --approval-request")
    if not os.environ.get("ARC_LLM_API_KEY"):
        raise RuntimeError("ARC_LLM_API_KEY must be present in the inherited environment")
    if run_root.exists():
        raise RuntimeError(f"approved run root already exists: {run_root}")
    consumed_approval = _consume_approved_request(
        args.approval_request,
        expected=approval_request,
        confirmation=args.confirm_live_run,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    response_timeout = _response_timeout(args)
    manifest = {
        "schema": "0006-nooa-single-arm-execution-manifest-v2",
        "claim_boundary": "single-arm live interface-and-behavior smoke; no treatment effect",
        "created_at": datetime.now(UTC).isoformat(),
        "approval_request_sha256": _canonical_sha256(approval_request),
        "consumed_approval_request": str(consumed_approval),
        "source_state": source_state,
        "provider_routing": {
            "mode": "openrouter-exact-endpoint-pin",
            "request": PROVIDER_ROUTE,
        },
        "provider_identity_audit": {
            "endpoint": GENERATION_AUDIT_URL,
            "attempt_limit_per_generation": GENERATION_AUDIT_ATTEMPTS,
            "delay_seconds": GENERATION_AUDIT_DELAY_SECONDS,
            "timeout_seconds": GENERATION_AUDIT_TIMEOUT_SECONDS,
        },
        "paired_run_spec": spec.as_record(),
        "worker_response_timeout_seconds": response_timeout,
    }
    _write_json(run_root / "execution-manifest.json", manifest)
    _write_json(run_root / "paired-run-spec.json", spec.as_record())
    summary = _run_stage(
        args,
        spec=spec,
        run_root=run_root,
        response_timeout=response_timeout,
        expected_source_sha256=source_state["source_sha256"],
    )
    finalization_errors: list[dict[str, str]] = []
    crossings: list[dict[str, Any]] = []
    assert spec.provider_request_profile is not None
    try:
        crossings = _candidate_provider_crossings(run_root)
        provider_report = _provider_profile_report(spec.provider_request_profile, crossings)
    except Exception as error:  # noqa: BLE001 - evidence finalization must fail closed.
        provider_report = {
            "schema": "0006-nooa-provider-profile-v1",
            "success": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        finalization_errors.append(provider_report["error"])
    _write_json(run_root / "provider-profile-report.json", provider_report)
    try:
        generation_ids = _generation_ids(crossings)
        generation_payloads, generation_fetch = _retrieve_generation_payloads(
            generation_ids,
            run_root=run_root,
        )
        provider_identity = _provider_identity_report(
            generation_ids,
            generation_payloads,
            expected_wire_model=spec.provider_request_profile.wire_model,
            crossing_usage=_generation_usage_by_id(crossings),
        )
        provider_identity["fetch"] = generation_fetch
    except Exception as error:  # noqa: BLE001 - identity finalization must fail closed.
        provider_identity = {
            "schema": "0006-nooa-provider-identity-v1",
            "success": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        finalization_errors.append(provider_identity["error"])
    _write_json(run_root / "provider-identity-report.json", provider_identity)
    try:
        competence_events = _read_jsonl(run_root / "worker" / "competence.jsonl")
    except Exception as error:  # noqa: BLE001 - preserve malformed evidence as a verdict failure.
        competence_events = []
        finalization_errors.append({"type": type(error).__name__, "message": str(error)})
    verdict = _operational_verdict(
        summary,
        competence_events=competence_events,
        provider_success=(
            provider_report.get("success") is True
            and provider_identity.get("success") is True
        ),
        required_actions=spec.max_actions,
    )
    summary["provider_profile_report"] = provider_report
    summary["provider_identity_report"] = provider_identity
    summary["operational_verdict"] = verdict
    summary["finalization_errors"] = finalization_errors
    _write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verdict["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
