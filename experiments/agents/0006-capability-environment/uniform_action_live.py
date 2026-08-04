"""Prepare and execute one matched live uniform-action control run."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlopen

from p9.arms import participating_sources
from p9.run_arm import build_parser, run_one, spec_from_args, write_json
from uniform_action_control import corrected_control_spec


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
ITERATION_ROOT = (
    EXPERIMENT_ROOT
    / "iterations"
    / "0004-uniform-action-interface-control"
)
ARTIFACTS_ROOT = ITERATION_ROOT / "artifacts"
PLAN_PATH = ARTIFACTS_ROOT / "live-control-cd82-v1-plan.json"
RUN_DIR = ARTIFACTS_ROOT / "live-control-cd82-v1"
RECORDED_CONTROL_MANIFEST = (
    EXPERIMENT_ROOT
    / "p9"
    / "artifacts"
    / "runs"
    / "pair-20260729T080256Z-663e125e"
    / "control"
    / "manifest.json"
)
ENDPOINTS_URL = (
    "https://openrouter.ai/api/v1/models/google/gemma-4-31b-it/endpoints"
)
PROVIDER_TAG = "modelrun/fp4"
CONFIRM_EXECUTION = "EXECUTE UNIFORM ACTION CONTROL CD82 V1"
APPROVAL_QUOTE = (
    "do a live run and see if the model now does not give too much affordance "
    "to action 6, same test as before but with the new fix.."
)


def fixed_args():
    """The recorded control's live contract, represented by the production parser."""

    return build_parser().parse_args(
        [
            "--game",
            "cd82",
            "--model",
            "google/gemma-4-31b-it",
            "--model-profile",
            "openrouter-generic",
            "--env-seed",
            "0",
            "--model-seed",
            "0",
            "--temperature",
            "1.0",
            "--max-completion-tokens",
            "4096",
            "--max-actions",
            "44",
            "--max-context-turns",
            "6",
            "--max-model-rounds-per-action",
            "10",
            "--max-model-calls",
            "400",
            "--max-reported-tokens",
            "12000000",
            "--workspace-timeout",
            "20",
            "--transport-retries",
            "6",
            "--transport-backoff",
            "4",
        ]
    )


def corrected_live_spec():
    return corrected_control_spec(spec_from_args(fixed_args()))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source_commitments() -> dict[str, str]:
    committed = dict(participating_sources())
    for path in (
        EXPERIMENT_ROOT / "uniform_action_control.py",
        Path(__file__).resolve(),
    ):
        committed[path.relative_to(REPO).as_posix()] = _sha256(path)
    return dict(sorted(committed.items()))


def fetch_modelrun_endpoint() -> dict[str, Any]:
    with urlopen(ENDPOINTS_URL, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload = json.load(response)
    endpoints = payload.get("data", {}).get("endpoints", [])
    matches = [endpoint for endpoint in endpoints if endpoint.get("tag") == PROVIDER_TAG]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {PROVIDER_TAG} endpoint, found {len(matches)}"
        )
    endpoint = matches[0]
    return {
        key: endpoint.get(key)
        for key in (
            "name",
            "provider_name",
            "tag",
            "quantization",
            "model_name",
            "context_length",
            "max_completion_tokens",
            "supported_parameters",
        )
    }


def build_plan(endpoint: dict[str, Any]) -> dict[str, Any]:
    args = fixed_args()
    spec = corrected_live_spec()
    return {
        "schema": "0004-uniform-action-live-plan-v1",
        "stage": "one corrected-control live run",
        "anti_actions": [
            "no control rerun",
            "no treatment",
            "no automatic successor run",
        ],
        "approval": {
            "authority": "user",
            "quote": APPROVAL_QUOTE,
            "confirmation_required": CONFIRM_EXECUTION,
        },
        "arm_id": spec.arm_id,
        "treatment_capabilities": list(spec.treatment_capabilities),
        "game": args.game,
        "model": args.model,
        "model_profile": args.model_profile,
        "endpoint_url": ENDPOINTS_URL,
        "endpoint": endpoint,
        "provider_only": [PROVIDER_TAG],
        "allow_fallbacks": False,
        "env_seed": args.env_seed,
        "model_seed": args.model_seed,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "max_actions": args.max_actions,
        "max_context_turns": args.max_context_turns,
        "max_model_rounds_per_action": args.max_model_rounds_per_action,
        "max_model_calls": args.max_model_calls,
        "max_reported_tokens": args.max_reported_tokens,
        "workspace_timeout_seconds": args.workspace_timeout,
        "transport_max_retries": args.transport_retries,
        "transport_backoff_seconds": args.transport_backoff,
        "recorded_control_manifest": {
            "path": RECORDED_CONTROL_MANIFEST.relative_to(REPO).as_posix(),
            "sha256": _sha256(RECORDED_CONTROL_MANIFEST),
        },
        "source_sha256": _source_commitments(),
        "run_dir": RUN_DIR.relative_to(REPO).as_posix(),
    }


def require_confirmation(value: str) -> None:
    if value != CONFIRM_EXECUTION:
        raise ValueError(f"confirmation must be exactly {CONFIRM_EXECUTION!r}")


def prepare() -> dict[str, Any]:
    if RUN_DIR.exists():
        raise RuntimeError(f"run directory already exists: {RUN_DIR}")
    plan = build_plan(fetch_modelrun_endpoint())
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(PLAN_PATH, plan)
    return plan


def _load_and_revalidate_plan() -> dict[str, Any]:
    if not PLAN_PATH.exists():
        raise RuntimeError(f"prepare first: {PLAN_PATH}")
    frozen = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    current = build_plan(fetch_modelrun_endpoint())
    if current != frozen:
        raise RuntimeError("live plan drifted after preparation")
    return frozen


def _audit_run(manifest: dict[str, Any]) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in (RUN_DIR / "process.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [event for event in events if event.get("type") == "decision"]
    action_counts = Counter(str(event["action"]) for event in decisions)

    providers: set[str] = set()
    for path in sorted((RUN_DIR / "response-bodies").glob("*.raw")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        provider = body.get("provider")
        if provider:
            providers.add(str(provider))

    requested_routes: set[str] = set()
    for path in sorted((RUN_DIR / "request-payloads").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provider = payload.get("provider") or {}
        for tag in provider.get("only", []):
            requested_routes.add(str(tag))

    summary = {
        "schema": "0004-uniform-action-live-summary-v1",
        "run_dir": RUN_DIR.relative_to(REPO).as_posix(),
        "status": manifest["status"],
        "stop_reason": manifest["stop_reason"],
        "actions": manifest["actions"],
        "action_counts": dict(sorted(action_counts.items())),
        "action_sequence": [event["action"] for event in decisions],
        "first_eight_actions": [event["action"] for event in decisions[:8]],
        "levels_completed": manifest["levels_completed"],
        "model_calls": manifest["model_calls"],
        "decision_repairs": manifest["decision_repairs"],
        "transport_attempts": manifest["transport_attempts"],
        "transport_retries": manifest["transport_retries"],
        "usage": manifest["usage"],
        "response_providers": sorted(providers),
        "requested_provider_only": sorted(requested_routes),
        "plan_sha256": _sha256(PLAN_PATH),
        "process_sha256": _sha256(RUN_DIR / "process.jsonl"),
        "trace_sha256": _sha256(RUN_DIR / "trace.jsonl"),
    }
    write_json(RUN_DIR / "uniform-action-summary.json", summary)
    return summary


def execute(confirmation: str) -> dict[str, Any]:
    from dotenv import load_dotenv

    require_confirmation(confirmation)
    _load_and_revalidate_plan()
    if RUN_DIR.exists():
        raise RuntimeError(f"run directory already exists: {RUN_DIR}")
    load_dotenv(REPO / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not present in the process environment")

    os.environ["OPENROUTER_PROVIDER_ONLY"] = PROVIDER_TAG
    manifest = run_one(corrected_live_spec(), RUN_DIR, fixed_args())
    summary = _audit_run(manifest)
    if summary["requested_provider_only"] != [PROVIDER_TAG]:
        raise RuntimeError("persisted requests did not retain the provider pin")
    if summary["response_providers"] != ["ModelRun"]:
        raise RuntimeError(
            f"unexpected response providers: {summary['response_providers']}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--confirm", required=True)
    args = parser.parse_args(argv)
    result = prepare() if args.command == "prepare" else execute(args.confirm)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
