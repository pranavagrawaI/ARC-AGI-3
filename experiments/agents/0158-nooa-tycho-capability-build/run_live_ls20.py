"""Execute one frozen, non-resumable 0158 run against public online LS20."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parent
RUNTIME = EXPERIMENT / "nooa-runtime"
CARD = REPO / "experiments/cards/0158-qwen36-provisional-heuristics-44-action-v26.md"
APPROVAL = REPO / "experiments/approvals/0158-ls20-live-v25.json"
CONSUMPTION = REPO / "experiments/approvals/0158-ls20-live-v25.consumed.json"
QUALIFICATION = EXPERIMENT / "qualification-provisional-heuristics-v26.json"
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from host.canonical import append_jsonl, content_sha256, write_atomic_json
from host.evidence import EvidenceStore
from host.ipc import AUTONOMOUS_44_POLICY, ArtifactBridge
from host.kernel import GroundingKernel, QualificationPaused
from host.live import (
    ActiveTimeBudget,
    LiveRunGuard,
    LiveRunTerminated,
    UsageLedger,
)

MODEL = "qwen/qwen3.6-27b"
MODEL_ROUTE = "openai/qwen/qwen3.6-27b"
PROVIDER_ROUTE_ID = "coreweave/fp8"
PROVIDER_ROUTE = {
    "order": ["coreweave/fp8"],
    "only": ["coreweave/fp8"],
    "allow_fallbacks": False,
    "require_parameters": False,
    "quantizations": ["fp8"],
}
REASONING_EFFORT: str | None = "medium"
DROP_PARAMS = True
RUN_SLUG = "qwen36-provisional-heuristics-44"
CANDIDATE_IDENTITY_FILES: tuple[Path, ...] = ()
MAX_CODEACT_ITERATIONS = 12
MAX_CODEACT_ERRORS = 4
EVIDENCE_ACCESS_POLICY = "readable_within_bounded_artifact_v1"
COMMITMENT_SURFACE = "fixed_cell_values_and_translate_v4"
SUBMISSION_GRAMMAR = "atomic_submit_constructor_v7"
TRANSPORT_MAX_RETRIES = 5
MAX_PROVIDER_ATTEMPTS_PER_ARTIFACT = 72
MAX_OUTPUT_TOKENS = 16_384
MAX_MODEL_CALLS = 1_200
MAX_REPORTED_TOKENS = 15_000_000
MAX_ACTIVE_SECONDS = 14_400
ENVIRONMENT_READ_TIMEOUT_SECONDS = 60.0
RUN_POLICY = AUTONOMOUS_44_POLICY
AUTONOMOUS_CONTINUATION = True
MODEL_FRAME_WINDOW = 1
T = TypeVar("T")


class BudgetStop(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _consume_approval(*, run_id: str, run_root: Path, approval_digest: str, card_digest: str) -> None:
    record = {
        "schema": "0158-live-approval-consumption-v1",
        "approval_sha256": approval_digest,
        "card_sha256": card_digest,
        "run_id": run_id,
        "run_root": str(run_root),
        "resumable": False,
    }
    CONSUMPTION.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with CONSUMPTION.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _provider_profile() -> dict[str, Any]:
    """Reuse the already-qualified 0002 NOOA Responses transport profile."""
    return {
        "schema": "0002-nooa-provider-request-v2",
        "api_surface": "responses",
        "model": MODEL_ROUTE,
        "wire_model": MODEL,
        "api_base": "https://openrouter.ai/api/v1",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
        "request_timeout_seconds": 180.0,
        "temperature": None,
        "seed": None,
        "stream": False,
        "drop_params": DROP_PARAMS,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "truncation": "disabled",
        "nooa_max_retries": 0,
        "nooa_rate_limit_extra_retries": 0,
        "litellm_num_retries": 0,
    }


def _provider_config() -> dict[str, Any]:
    return {
        "provider_request_profile": _provider_profile(),
        "provider_route": PROVIDER_ROUTE,
        "transport_max_retries": TRANSPORT_MAX_RETRIES,
        "runtime_limits": {
            "max_iterations": MAX_CODEACT_ITERATIONS,
            "max_retries": MAX_CODEACT_ERRORS,
            "sandbox_timeout_seconds": 20.0,
        },
    }


def _worker_environment(llm_key: str) -> dict[str, str]:
    environment = {}
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    for key, value in os.environ.items():
        if any(marker in key.upper() for marker in secret_markers):
            continue
        environment[key] = value
    environment["ARC_LLM_API_KEY"] = llm_key
    return environment


def _candidate_manifest() -> dict[str, Any]:
    paths = [
        *sorted((EXPERIMENT / "host").glob("*.py")),
        EXPERIMENT / "model-contract.json",
        EXPERIMENT / "nooa-runtime/agent.py",
        EXPERIMENT / "nooa-runtime/launcher.py",
        EXPERIMENT / "nooa-runtime/provider.py",
        EXPERIMENT / "nooa-runtime/pyproject.toml",
        EXPERIMENT / "nooa-runtime/uv.lock",
        Path(__file__).resolve(),
        *CANDIDATE_IDENTITY_FILES,
        CARD,
        QUALIFICATION,
    ]
    files = [
        {
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in paths
    ]
    return {
        "schema": "0158-live-candidate-manifest-v1",
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "files": files,
        "manifest_sha256": content_sha256(files),
    }


def _layers(frame: Any) -> list[list[list[int]]]:
    array = np.asarray(frame.frame)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3 or not array.shape[0] or not array.shape[1] or not array.shape[2]:
        raise ValueError(f"public frame must have shape layers x rows x cols, got {array.shape}")
    return [[[int(value) for value in row] for row in layer] for layer in array]


def _state(frame: Any) -> str:
    return frame.state.value if hasattr(frame.state, "value") else str(frame.state)


def _record_environment_frame(path: Path, *, step: int, frame: Any, layers: list[Any]) -> None:
    append_jsonl(
        path,
        {
            "schema": "0158-public-environment-frame-v1",
            "step": step,
            "state": _state(frame),
            "levels_completed": int(frame.levels_completed),
            "full_reset": bool(frame.full_reset),
            "available_actions": [int(action) for action in frame.available_actions],
            "layer_count": len(layers),
            "layers_sha256": content_sha256(layers),
        },
    )


def _extend_environment_read_timeout(environment: Any, *, seconds: float) -> None:
    """Raise only the SDK read timeout; never add an unsafe POST retry."""
    session = getattr(environment, "_session", None)
    original_post = getattr(session, "post", None)
    if not callable(original_post):
        return

    def post(url: str, *args: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)) and float(timeout) < seconds:
            kwargs["timeout"] = seconds
        return original_post(url, *args, **kwargs)

    session.post = post


def _write_usage(run_root: Path, ledger: UsageLedger) -> None:
    write_atomic_json(
        run_root / "usage-ledger.json",
        {
            "schema": "0158-live-usage-v1",
            "calls": ledger.calls,
            "reported_tokens": ledger.reported_tokens,
            "invalid_reason": ledger.invalid_reason,
            "records": ledger.records,
        },
    )


def _ingest_response_usage(
    bridge: ArtifactBridge,
    ledger: UsageLedger,
    seen: set[str],
    run_root: Path,
) -> None:
    response = bridge.latest_response
    if not isinstance(response, dict):
        return
    artifact_id = response.get("artifact_id")
    if not isinstance(artifact_id, str) or artifact_id in seen:
        return
    seen.add(artifact_id)
    audit = response.get("audit", {})
    calls = audit.get("model_calls", []) if isinstance(audit, dict) else []
    usage_records = audit.get("usage", []) if isinstance(audit, dict) else []
    if not isinstance(calls, list) or not isinstance(usage_records, list):
        raise TypeError("provider audit has invalid model_calls or usage")
    usage_index = 0
    for call in calls:
        attempts = call.get("attempts", []) if isinstance(call, dict) else []
        if not isinstance(attempts, list):
            raise TypeError("provider audit attempts must be a list")
        for attempt_index, attempt in enumerate(attempts):
            outcome = attempt.get("outcome") if isinstance(attempt, dict) else None
            if outcome == "success":
                if usage_index >= len(usage_records):
                    raise ValueError("successful provider response has no required usage")
                ledger.record(usage_records[usage_index], retry=attempt_index > 0)
                usage_index += 1
            elif isinstance(outcome, str) and outcome.endswith("_failure"):
                failure = attempt.get("failure", {})
                disposition = failure.get("disposition") if isinstance(failure, dict) else None
                ledger.record_transport_failure(
                    retry=attempt_index > 0,
                    disposition=str(disposition),
                )
            else:
                raise ValueError("provider audit contains an unknown attempt outcome")
    if usage_index != len(usage_records):
        raise ValueError("provider usage has no matching successful attempt")
    _write_usage(run_root, ledger)


def _packet_for(kernel: GroundingKernel) -> Path:
    packets = sorted((kernel.run_root / "failure-packets").iterdir())
    if not packets:
        raise RuntimeError("paused kernel has no failure packet")
    return packets[-1]


def _run_root() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return EXPERIMENT / "artifacts/live-ls20" / f"{stamp}-{RUN_SLUG}"


def run(run_root: Path) -> dict[str, Any]:
    arc_key = os.environ.get("ARC_API_KEY")
    llm_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not arc_key or not llm_key:
        raise RuntimeError("ARC_API_KEY and ARC_LLM_API_KEY or OPENROUTER_API_KEY are required")
    card_bytes = CARD.read_bytes()
    if len(card_bytes) > 1_024:
        raise RuntimeError("live experiment card exceeds 1 KiB")
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    card_digest = sha256(card_bytes).hexdigest()
    manifest = _candidate_manifest()
    approval_checks = {
        "execution_authority": approval.get("execution_authority") is True,
        "approved_runs": approval.get("approved_runs") == 1,
        "approved_game": approval.get("approved_game") == "ls20",
        "approved_model": approval.get("approved_model") == MODEL,
        "approved_action_cap": approval.get("approved_action_cap") == 44,
        "approved_codeact_iterations": approval.get("approved_codeact_iterations")
        == MAX_CODEACT_ITERATIONS,
        "approved_codeact_errors": approval.get("approved_codeact_errors") == MAX_CODEACT_ERRORS,
        "approved_evidence_access_policy": approval.get("approved_evidence_access_policy")
        == EVIDENCE_ACCESS_POLICY,
        "approved_commitment_surface": approval.get("approved_commitment_surface")
        == COMMITMENT_SURFACE,
        "approved_submission_grammar": approval.get("approved_submission_grammar")
        == SUBMISSION_GRAMMAR,
        "approved_max_output_tokens": approval.get("approved_max_output_tokens") == MAX_OUTPUT_TOKENS,
        "approved_provider_route": approval.get("approved_provider_route") == PROVIDER_ROUTE_ID,
        "approved_transport_max_retries": approval.get("approved_transport_max_retries")
        == TRANSPORT_MAX_RETRIES,
        "approved_model_call_cap": approval.get("approved_model_call_cap") == MAX_MODEL_CALLS,
        "approved_reported_token_cap": approval.get("approved_reported_token_cap")
        == MAX_REPORTED_TOKENS,
        "approved_active_seconds_cap": approval.get("approved_active_seconds_cap")
        == MAX_ACTIVE_SECONDS,
        "approved_run_policy": approval.get("approved_run_policy") == RUN_POLICY["mode"],
        "approved_model_frame_window": approval.get("approved_model_frame_window")
        == MODEL_FRAME_WINDOW,
        "approved_environment_read_timeout_seconds": approval.get(
            "approved_environment_read_timeout_seconds"
        ) == ENVIRONMENT_READ_TIMEOUT_SECONDS,
        "approved_environment_post_retries": approval.get("approved_environment_post_retries")
        == 0,
        "approved_card_sha256": approval.get("approved_card_sha256") == card_digest,
        "qualification_summary_sha256": approval.get("qualification_summary_sha256")
        == _sha(QUALIFICATION),
        "approved_candidate_manifest_sha256": approval.get(
            "approved_candidate_manifest_sha256"
        ) == manifest["manifest_sha256"],
    }
    if not all(approval_checks.values()):
        raise PermissionError("approval record does not authorize exactly one live run")
    if CONSUMPTION.exists():
        raise PermissionError("this one-run approval has already been consumed")
    approval_digest = _sha(APPROVAL)
    guard = LiveRunGuard(run_root, card_digest=card_digest, approval_digest=approval_digest)
    _consume_approval(
        run_id=guard.run_id,
        run_root=run_root,
        approval_digest=approval_digest,
        card_digest=card_digest,
    )
    (run_root / "EXPERIMENT-CARD.md").write_bytes(card_bytes)
    shutil.copyfile(APPROVAL, run_root / "approval.json")
    write_atomic_json(run_root / "candidate-manifest.json", manifest)
    write_atomic_json(
        run_root / "run-config.json",
        {
            "schema": "0158-live-ls20-config-v1",
            "game": "ls20",
            "environment_seed": 0,
            "operation_mode": "online",
            "fresh_reset": True,
            "action_cap": 44,
            "model_call_cap": MAX_MODEL_CALLS,
            "codeact_iterations_per_artifact": MAX_CODEACT_ITERATIONS,
            "codeact_errors_per_artifact": MAX_CODEACT_ERRORS,
            "evidence_access_policy": EVIDENCE_ACCESS_POLICY,
            "commitment_surface": COMMITMENT_SURFACE,
            "submission_grammar": SUBMISSION_GRAMMAR,
            "transport_max_retries_per_call": TRANSPORT_MAX_RETRIES,
            "max_output_tokens_per_generation": MAX_OUTPUT_TOKENS,
            "reported_token_cap": MAX_REPORTED_TOKENS,
            "active_seconds_cap": MAX_ACTIVE_SECONDS,
            "run_policy": RUN_POLICY,
            "model_frame_window": MODEL_FRAME_WINDOW,
            "environment_read_timeout_seconds": ENVIRONMENT_READ_TIMEOUT_SECONDS,
            "environment_post_retries": 0,
            "card_sha256": card_digest,
            "approval_sha256": approval_digest,
            "candidate_manifest_sha256": manifest["manifest_sha256"],
            "provider_request_profile": _provider_profile(),
            "provider_route": PROVIDER_ROUTE,
        },
    )

    ledger = UsageLedger(
        max_calls=MAX_MODEL_CALLS,
        max_reported_tokens=MAX_REPORTED_TOKENS,
        max_output_tokens_per_generation=MAX_OUTPUT_TOKENS,
    )
    timer = ActiveTimeBudget(max_seconds=MAX_ACTIVE_SECONDS)
    bridge = ArtifactBridge(
        runtime_root=RUNTIME,
        run_root=run_root / "worker",
        provider_config=_provider_config(),
        worker_env=_worker_environment(llm_key),
        timeout_seconds=900.0,
        run_policy=RUN_POLICY,
    )
    evidence = EvidenceStore(run_root / "evidence")
    kernel = GroundingKernel(
        run_root=run_root / "kernel",
        evidence=evidence,
        bridge=bridge,
        autonomous_continuation=AUTONOMOUS_CONTINUATION,
        model_frame_window=MODEL_FRAME_WINDOW,
    )
    seen_responses: set[str] = set()
    environment_trace = run_root / "environment.jsonl"
    levels_completed = 0
    stop_reason = "host_failure"
    arcade: Any = None
    environment: Any = None

    def artifact_call(function: Callable[[], T]) -> T:
        timer.check()
        if ledger.remaining_calls < MAX_PROVIDER_ATTEMPTS_PER_ARTIFACT:
            raise BudgetStop("insufficient provider-attempt budget for one bounded artifact")
        bridge.timeout_seconds = min(900.0, max(1.0, timer.max_seconds - timer.active_seconds()))
        try:
            result = function()
        except Exception:
            try:
                _ingest_response_usage(bridge, ledger, seen_responses, run_root)
            except Exception as usage_error:  # noqa: BLE001 - preserve the original scored-run failure
                write_atomic_json(
                    run_root / "usage-audit-error.json",
                    {
                        "error_type": type(usage_error).__name__,
                        "message_sha256": sha256(str(usage_error).encode()).hexdigest(),
                    },
                )
            raise
        try:
            _ingest_response_usage(bridge, ledger, seen_responses, run_root)
        except Exception as error:  # noqa: BLE001 - any usage-audit defect closes the run
            kernel.pause_external_failure(error, phase="provider_usage")
        return result

    try:
        arcade = arc_agi.Arcade(
            arc_api_key=arc_key,
            operation_mode=OperationMode.ONLINE,
            recordings_dir=str(run_root / "raw-recordings"),
        )
        environment = arcade.make("ls20", seed=0, save_recording=True)
        if environment is None:
            raise RuntimeError("online ARC service did not create ls20")
        _extend_environment_read_timeout(
            environment,
            seconds=ENVIRONMENT_READ_TIMEOUT_SECONDS,
        )
        current = environment.reset()
        if current is None:
            raise RuntimeError("online ARC reset returned no public frame")
        initial_levels = int(current.levels_completed)
        levels_completed = initial_levels
        if initial_levels != 0:
            raise RuntimeError("fresh LS20 reset did not begin at levels_completed=0")
        current_layers = _layers(current)
        _record_environment_frame(environment_trace, step=0, frame=current, layers=current_layers)
        kernel.start(
            current_layers[-1],
            attempt_id=f"{guard.run_id}-attempt-0",
            available_actions=[int(action) for action in current.available_actions],
        )
        if len(current_layers) > 1:
            kernel.record_separate_event("animation", current_layers)
        print(json.dumps({"event": "reset", "run_root": str(run_root), "available_actions": [int(a) for a in current.available_actions]}), flush=True)

        while guard.actions < 44:
            timer.check()
            claim = artifact_call(kernel.analyze)
            coordinate_actions = [
                int(action)
                for action in current.available_actions
                if GameAction.from_id(int(action)).is_complex()
            ]
            commitment = artifact_call(
                lambda coordinate_actions=tuple(coordinate_actions): kernel.commit(
                    coordinate_actions=coordinate_actions
                )
            )
            action_envelope = kernel.issue_one_action()
            authority_token = guard.authorize_one_action(commitment.commitment_id)
            append_jsonl(
                run_root / "issued-actions.jsonl",
                {
                    "schema": "0158-issued-action-v1",
                    "step": guard.actions + 1,
                    "claim_id": claim.claim_id,
                    "commitment_id": commitment.commitment_id,
                    "action": action_envelope.action,
                    "data": action_envelope.data,
                    "authority_token": authority_token,
                },
            )
            game_action = GameAction.from_id(action_envelope.action)
            try:
                successor = environment.step(
                    game_action,
                    data=action_envelope.data,
                    reasoning={"claim_id": claim.claim_id, "commitment_id": commitment.commitment_id},
                )
                if successor is None:
                    raise RuntimeError("online ARC step returned no public frame")
            except Exception as error:  # noqa: BLE001 - uncertain external execution must close authority
                kernel.pause_external_failure(error, phase="environment_step")
            guard.record_successor(commitment.commitment_id)
            successor_layers = _layers(successor)
            _record_environment_frame(
                environment_trace,
                step=guard.actions,
                frame=successor,
                layers=successor_layers,
            )
            if len(successor_layers) > 1:
                kernel.record_separate_event("animation", successor_layers)
            state = _state(successor)
            levels_completed = int(successor.levels_completed)
            print(
                json.dumps(
                    {
                        "event": "successor",
                        "action_count": guard.actions,
                        "action": action_envelope.action,
                        "levels_completed": levels_completed,
                        "state": state,
                    }
                ),
                flush=True,
            )
            if levels_completed >= 1:
                kernel.observe_terminal_successor(successor_layers[-1])
                guard.complete(levels_completed=levels_completed)
                stop_reason = "level_completed"
                break
            if state == "GAME_OVER" or bool(successor.full_reset):
                kernel.observe_death_successor(successor_layers[-1])
            verdict = kernel.observe_successor(
                successor_layers[-1],
                available_actions=[int(action) for action in successor.available_actions],
            )
            current = successor
            append_jsonl(
                run_root / "verdicts.jsonl",
                {"step": guard.actions, **verdict.to_record(), "gate": kernel.gates.gate},
            )
            if kernel.gates.gate in {"INVALIDATED", "STOPPED"}:
                packet = _packet_for(kernel)
                guard.consequential_failure(packet.name, levels_completed=levels_completed)
            if kernel.gates.gate == "G2":
                guard.stop_at_g2(levels_completed=levels_completed)
                stop_reason = "g2_transition_eligibility"
                break
        else:
            guard.stop_without_success(reason="action_cap", levels_completed=levels_completed)
            stop_reason = "action_cap"
    except QualificationPaused as paused:
        _ingest_response_usage(bridge, ledger, seen_responses, run_root)
        try:
            if guard.pending_commitment_id is not None:
                guard.uncertain_action_failure(
                    reason="uncertain_environment_action",
                    levels_completed=levels_completed,
                    packet_id=paused.packet.name,
                )
            else:
                guard.consequential_failure(paused.packet.name, levels_completed=levels_completed)
        except LiveRunTerminated:
            pass
        stop_reason = f"{paused.failure_class}_failure"
    except BudgetStop as error:
        write_atomic_json(run_root / "budget-stop.json", {"reason": str(error), "usage": ledger.records})
        guard.stop_without_success(reason="model_call_cap", levels_completed=levels_completed)
        stop_reason = "model_call_cap"
    except LiveRunTerminated:
        stop_reason = "consequential_failure"
    except Exception as error:  # noqa: BLE001 - fail closed and preserve a redacted terminal record
        write_atomic_json(
            run_root / "host-failure.json",
            {
                "error_type": type(error).__name__,
                "message_chars": len(str(error)),
                "message_sha256": sha256(str(error).encode("utf-8", errors="replace")).hexdigest(),
            },
        )
        if guard.state == "ACTIVE":
            if guard.pending_commitment_id is not None:
                try:
                    guard.uncertain_action_failure(
                        reason="host_failure_with_pending_action",
                        levels_completed=levels_completed,
                    )
                except LiveRunTerminated:
                    pass
            else:
                guard.stop_without_success(reason="host_substrate_failure", levels_completed=levels_completed)
        stop_reason = "host_substrate_failure"
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        _write_usage(run_root, ledger)

    scorecard = json.loads((run_root / "scorecard.json").read_text(encoding="utf-8"))
    summary = {
        "schema": "0158-live-ls20-summary-v1",
        "run_root": str(run_root),
        "stop_reason": stop_reason,
        "actions": guard.actions,
        "levels_completed": levels_completed,
        "model_calls": ledger.calls,
        "reported_tokens": ledger.reported_tokens,
        "scorecard": scorecard,
    }
    write_atomic_json(run_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    summary = run((args.run_root or _run_root()).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
