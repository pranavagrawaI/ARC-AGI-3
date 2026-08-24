"""Durable paid launcher for the frozen Matched Actor pilot."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel

from tycho_nooa.compatibility_authorization import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
)
from tycho_nooa.compatibility_launch import AuthorizedCompatibilityLLM
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample
from tycho_nooa.matched_actor_pilot_authorization import (
    MatchedActorPilotAuthorization,
)
from tycho_nooa.matched_actor_pilot_manifest import (
    MatchedActorPilotManifest,
    load_matched_actor_pilot_manifest,
)
from tycho_nooa.matched_actor_pilot_run import (
    PilotAssessment,
    PilotPairReport,
    assess_pilot,
    execute_pilot_pair,
)


class AuthorizedMatchedActorLLM(AuthorizedCompatibilityLLM):
    """Expose ledger-priced cost to the treatment's resource observation."""

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        response = await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )
        latest = max(
            self._ledger.state.calls.values(), key=lambda call: call.call_index
        )
        usage = dict(response.usage) if isinstance(response.usage, dict) else {}
        usage["observed_cost_usd"] = float(latest.actual_cost_usd or 0)
        response.usage = usage
        return response


async def execute_authorized_matched_actor_pilot(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
) -> PilotAssessment:
    """Execute a counterbalanced sample only after fail-closed preflight."""
    manifest, authorization, authorization_hash = preflight_matched_actor_pilot(
        manifest_path,
        authorization_path,
        repository_root=repository_root,
        native_llm=native_llm,
        codeact_llm=codeact_llm,
    )
    _require_new_destination(
        destination,
        authorization=authorization,
        authorization_hash=authorization_hash,
    )
    sample, packets = load_matched_actor_pilot_sample(
        repository_root / manifest.sample.path,
        repository_root=repository_root,
    )
    destination.mkdir(parents=True)
    (destination / "experiment.json").write_bytes(manifest_path.read_bytes())
    (destination / "authorization.json").write_bytes(authorization_path.read_bytes())
    ledger = DurableAuthorizationLedger.create(
        destination / "authorization-events.jsonl",
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    reports: list[PilotPairReport] = []
    try:
        for scheduled, packet in zip(sample.packets, packets, strict=True):
            ledger.start_gate(scheduled.packet_id)
            report = await execute_pilot_pair(
                packet,
                stratum=scheduled.stratum,
                arm_order=scheduled.arm_order,
                repository_root=repository_root,
                destination=destination / "pairs" / scheduled.packet_id,
                native_llm=AuthorizedMatchedActorLLM(
                    native_llm,
                    gate_id=scheduled.packet_id,
                    arm="native",
                    ledger=ledger,
                ),
                codeact_llm=AuthorizedMatchedActorLLM(
                    codeact_llm,
                    gate_id=scheduled.packet_id,
                    arm="codeact",
                    ledger=ledger,
                ),
                max_provider_calls=manifest.caps.calls_per_session,
                max_output_tokens=manifest.caps.output_tokens_per_call,
                session_timeout_seconds=manifest.caps.wall_seconds_per_session,
                cell_timeout_seconds=min(5.0, manifest.caps.wall_seconds_per_session),
                continuation_allowed=lambda: ledger.state.run_state == "armed",
            )
            reports.append(report)
            if ledger.state.run_state != "armed":
                raise RuntimeError(ledger.state.stop_reason or "authorization stopped")
            ledger.complete_gate(
                scheduled.packet_id,
                passed=report.complete,
                report=report.model_dump(mode="json"),
            )
        assessment = assess_pilot(tuple(reports))
        _write_json(destination / "assessment.json", assessment.model_dump(mode="json"))
        _write_json(
            destination / "pilot-report.json",
            {
                "pairs": [report.model_dump(mode="json") for report in reports],
                "assessment": assessment.model_dump(mode="json"),
            },
        )
        ledger.finish_ladder(
            assessment.disposition, report=assessment.model_dump(mode="json")
        )
        return assessment
    # The trusted launcher records every ordinary control-plane failure before
    # propagation; KeyboardInterrupt/SystemExit remain outside this boundary.
    except Exception as error:
        if ledger.state.run_state == "armed":
            ledger.stop(f"pilot_launcher_failure:{type(error).__name__}")
        _write_json(
            destination / "launch-failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def preflight_matched_actor_pilot(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
) -> tuple[MatchedActorPilotManifest, MatchedActorPilotAuthorization, str]:
    manifest, _ = load_matched_actor_pilot_manifest(
        manifest_path, repository_root=repository_root
    )
    manifest.require_execution_authorized()
    authorization = MatchedActorPilotAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    authorization.require_current()
    manifest_hash = sha256_file(manifest_path)
    authorization_hash = sha256_file(authorization_path)
    _validate_authorization(authorization, manifest, manifest_hash)
    _validate_configured_llm(native_llm, authorization)
    _validate_configured_llm(codeact_llm, authorization)
    return manifest, authorization, authorization_hash


def _validate_authorization(
    authorization: CompatibilityAuthorization,
    manifest: MatchedActorPilotManifest,
    manifest_hash: str,
) -> None:
    if authorization.experiment_manifest_sha256 != manifest_hash:
        raise ValueError("authorization does not bind this pilot manifest")
    _validate_authorized_route(authorization, manifest)
    _validate_authorized_caps(authorization, manifest)


def _validate_authorized_route(
    authorization: CompatibilityAuthorization,
    manifest: MatchedActorPilotManifest,
) -> None:
    route = manifest.model_route
    authorized_route = (
        authorization.requested_model == route.model
        and authorization.expected_reported_model == route.reported_model
        and authorization.allowed_provider == route.provider
        and authorization.allowed_reasoning_effort == route.reasoning_effort
        and authorization.allowed_temperature == route.temperature
    )
    if not authorized_route:
        raise ValueError("authorization route does not match pilot route")


def _validate_authorized_caps(
    authorization: CompatibilityAuthorization,
    manifest: MatchedActorPilotManifest,
) -> None:
    caps = manifest.caps
    if authorization.maximum_calls != caps.calls_total:
        raise ValueError("authorization call cap does not match pilot cap")
    if authorization.max_input_tokens_per_call != caps.input_tokens_per_call:
        raise ValueError("authorization input-token cap does not match pilot cap")
    if authorization.max_output_tokens_per_call != caps.output_tokens_per_call:
        raise ValueError("authorization output-token cap does not match pilot cap")
    if caps.max_cost_usd is None or authorization.maximum_spend_usd != Decimal(
        str(caps.max_cost_usd)
    ):
        raise ValueError("authorization spend cap does not match pilot cap")


def _validate_configured_llm(
    llm: UnifiedLLM, authorization: CompatibilityAuthorization
) -> None:
    if llm.model != authorization.requested_model:
        raise ValueError("configured model does not match pilot authorization")
    effort = llm.config.get("reasoning_effort")
    reasoning = llm.config.get("reasoning")
    if effort is None and isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
        if effort is None and reasoning.get("enabled") is True:
            effort = "enabled"
    if effort != authorization.allowed_reasoning_effort:
        raise ValueError("configured reasoning does not match pilot authorization")
    if llm.config.get("temperature") != authorization.allowed_temperature:
        raise ValueError("configured temperature does not match pilot authorization")


def _require_new_destination(
    destination: Path,
    *,
    authorization: CompatibilityAuthorization,
    authorization_hash: str,
) -> None:
    if not destination.exists():
        return
    journal = destination / "authorization-events.jsonl"
    if not journal.is_file():
        raise FileExistsError(destination)
    ledger = DurableAuthorizationLedger(
        journal,
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    ledger.recover()
    raise RuntimeError(
        f"pilot authorization run already exists with state {ledger.state.run_state}"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
