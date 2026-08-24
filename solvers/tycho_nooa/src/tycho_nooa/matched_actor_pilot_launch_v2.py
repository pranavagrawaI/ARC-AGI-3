"""Durable paid launcher for the frozen Matched Actor pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel

from tycho_nooa.compatibility_authorization_v2 import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
)
from tycho_nooa.compatibility_launch_v2 import (
    AuthorizedCompatibilityLLM,
    require_resource_safe_delegate,
)
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample
from tycho_nooa.matched_actor_pilot_authorization_v2 import (
    MatchedActorPilotAuthorizationV2,
)
from tycho_nooa.matched_actor_pilot_launch import (
    _validate_authorization,
    _validate_configured_llm,
    _write_json,
)
from tycho_nooa.matched_actor_pilot_manifest_v2 import (
    MatchedActorPilotManifestV2,
    load_matched_actor_pilot_manifest_v2,
)
from tycho_nooa.matched_actor_pilot_run_v2 import (
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


async def execute_authorized_matched_actor_pilot_v2(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
) -> PilotAssessment:
    """Execute a counterbalanced sample only after fail-closed preflight."""
    manifest, authorization, authorization_hash = preflight_matched_actor_pilot_v2(
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


def preflight_matched_actor_pilot_v2(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
) -> tuple[MatchedActorPilotManifestV2, MatchedActorPilotAuthorizationV2, str]:
    manifest, _ = load_matched_actor_pilot_manifest_v2(
        manifest_path, repository_root=repository_root
    )
    manifest.require_execution_authorized()
    authorization = MatchedActorPilotAuthorizationV2.model_validate_json(
        authorization_path.read_text()
    )
    authorization.require_current()
    manifest_hash = sha256_file(manifest_path)
    authorization_hash = sha256_file(authorization_path)
    _validate_authorization(authorization, manifest, manifest_hash)
    _validate_configured_llm(native_llm, authorization)
    _validate_configured_llm(codeact_llm, authorization)
    require_resource_safe_delegate(native_llm)
    require_resource_safe_delegate(codeact_llm)
    return manifest, authorization, authorization_hash


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
