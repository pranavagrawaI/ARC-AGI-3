"""Fail-closed paid launcher for the frozen compatibility ladder."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel

from tycho_nooa.compatibility_authorization import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
)
from tycho_nooa.compatibility_experiment import (
    CompatibilityGateReport,
    execute_compatibility_gate,
)
from tycho_nooa.compatibility_ladder import (
    CompatibilityLadder,
    CompatibilityLadderReport,
    run_compatibility_ladder,
)
from tycho_nooa.compatibility_manifest import (
    CompatibilityExperimentManifest,
    load_compatibility_experiment_manifest,
    sha256_file,
)
from tycho_nooa.parity_compatibility import CompatibilityGate

CompatibilityArm = Literal["native", "codeact"]


class AuthorizationLedgerObserver:
    """Project ladder lifecycle into the durable authorization journal."""

    def __init__(self, ledger: DurableAuthorizationLedger):
        self.ledger = ledger

    def gate_started(self, gate: CompatibilityGate) -> None:
        self.ledger.start_gate(gate.task.gate_id)

    def gate_completed(
        self, gate: CompatibilityGate, report: CompatibilityGateReport
    ) -> None:
        self.ledger.complete_gate(
            gate.task.gate_id,
            passed=report.passed,
            report=report.model_dump(mode="json"),
        )

    def ladder_completed(self, report: CompatibilityLadderReport) -> None:
        self.ledger.finish_ladder(
            report.disposition, report=report.model_dump(mode="json")
        )


class AuthorizedCompatibilityLLM(UnifiedLLM):
    """One gate/arm provider adapter governed entirely by a durable ledger."""

    def __init__(
        self,
        delegate: UnifiedLLM,
        *,
        gate_id: str,
        arm: CompatibilityArm,
        ledger: DurableAuthorizationLedger,
    ):
        super().__init__(delegate.model, **delegate.config)
        self._delegate = delegate
        self._gate_id = gate_id
        self._arm = arm
        self._ledger = ledger
        self.cache_control_injection_points = delegate.cache_control_injection_points

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del messages, tools, output_model, kwargs
        raise RuntimeError("compatibility launcher uses async provider calls only")

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        _validate_output_ceiling(kwargs, self._ledger.authorization)
        request = _request_value(messages, tools, output_model, kwargs)
        call = self._ledger.plan_call(
            gate_id=self._gate_id, arm=self._arm, request=request
        )
        self._ledger.dispatch(call.call_id)
        try:
            response = await self._delegate.acall(
                messages, tools=tools, output_model=output_model, **kwargs
            )
        except BaseException as error:
            self._ledger.fail(
                call.call_id,
                reason=f"{type(error).__name__}: {error}",
                response=_exception_response(error),
            )
            raise
        usage = response.usage if isinstance(response.usage, dict) else {}
        route = _provider_route(response.raw_response)
        self._ledger.receive(
            call.call_id,
            response=_response_value(response),
            usage=usage,
            observed_provider=route.get("provider"),
            observed_model=route.get("model"),
            observed_reasoning_effort=route.get("reasoning_effort"),
        )
        self._ledger.reconcile(call.call_id)
        if self._ledger.state.run_state == "stopped":
            raise RuntimeError(
                f"authorized provider call stopped: {self._ledger.state.stop_reason}"
            )
        return response


async def execute_authorized_compatibility_ladder(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
) -> CompatibilityLadderReport:
    """Validate every authority input before creating a run or calling a provider."""
    manifest, ladder, authorization, authorization_hash = (
        preflight_authorized_compatibility_ladder(
            manifest_path,
            authorization_path,
            repository_root=repository_root,
            native_llm=native_llm,
            codeact_llm=codeact_llm,
        )
    )
    if destination.exists():
        _reject_existing_run(
            destination,
            authorization=authorization,
            authorization_sha256=authorization_hash,
        )

    destination.mkdir(parents=True)
    (destination / "experiment.json").write_bytes(manifest_path.read_bytes())
    (destination / "authorization.json").write_bytes(authorization_path.read_bytes())
    ledger = DurableAuthorizationLedger.create(
        destination / "authorization-events.jsonl",
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    observer = AuthorizationLedgerObserver(ledger)

    async def execute(
        gate: CompatibilityGate, gate_destination: Path
    ) -> CompatibilityGateReport:
        native = AuthorizedCompatibilityLLM(
            native_llm,
            gate_id=gate.task.gate_id,
            arm="native",
            ledger=ledger,
        )
        codeact = AuthorizedCompatibilityLLM(
            codeact_llm,
            gate_id=gate.task.gate_id,
            arm="codeact",
            ledger=ledger,
        )
        return await execute_compatibility_gate(
            gate,
            repository_root=repository_root,
            destination=gate_destination,
            native_llm=native,
            codeact_llm=codeact,
            max_llm_calls=manifest.caps.calls_per_arm_gate,
            max_output_tokens=manifest.caps.output_tokens_per_call,
            session_timeout_seconds=manifest.caps.wall_seconds_per_arm_gate,
            cell_timeout_seconds=min(
                5.0, manifest.caps.wall_seconds_per_arm_gate
            ),
        )

    try:
        return await run_compatibility_ladder(
            ladder,
            destination=destination / "ladder",
            execute_gate=execute,
            observer=observer,
        )
    except BaseException as error:
        if ledger.state.run_state == "armed":
            ledger.stop(f"launcher_failure:{type(error).__name__}")
        _write_json(
            destination / "launch-failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def preflight_authorized_compatibility_ladder(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
) -> tuple[
    CompatibilityExperimentManifest,
    CompatibilityLadder,
    CompatibilityAuthorization,
    str,
]:
    """Validate package, authority, route, expiry, and caps without side effects."""
    manifest, ladder = load_compatibility_experiment_manifest(
        manifest_path, repository_root=repository_root
    )
    manifest.require_execution_authorized()
    authorization = CompatibilityAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    authorization.require_current()
    manifest_hash = sha256_file(manifest_path)
    authorization_hash = sha256_file(authorization_path)
    _validate_authorization(authorization, manifest, manifest_hash)
    _validate_configured_llm(native_llm, authorization)
    _validate_configured_llm(codeact_llm, authorization)
    return manifest, ladder, authorization, authorization_hash


def _validate_authorization(
    authorization: CompatibilityAuthorization,
    manifest: CompatibilityExperimentManifest,
    manifest_sha256: str,
) -> None:
    if authorization.experiment_manifest_sha256 != manifest_sha256:
        raise ValueError("authorization does not bind this experiment manifest")
    route = manifest.model_route
    exact_route = (
        authorization.requested_model == route.model
        and authorization.expected_reported_model == route.reported_model
        and authorization.allowed_provider == route.provider
        and authorization.allowed_reasoning_effort == route.reasoning_effort
        and authorization.allowed_temperature == route.temperature
    )
    if not exact_route:
        raise ValueError("authorization route does not match experiment route")
    caps = manifest.caps
    if authorization.maximum_calls != caps.calls_total:
        raise ValueError("authorization call cap does not match experiment cap")
    if authorization.max_input_tokens_per_call != caps.input_tokens_per_call:
        raise ValueError("authorization input-token cap does not match experiment cap")
    if authorization.max_output_tokens_per_call != caps.output_tokens_per_call:
        raise ValueError("authorization output-token cap does not match experiment cap")
    if caps.max_cost_usd is None or authorization.maximum_spend_usd != Decimal(
        str(caps.max_cost_usd)
    ):
        raise ValueError("authorization spend cap does not match experiment cap")


def _validate_configured_llm(
    llm: UnifiedLLM, authorization: CompatibilityAuthorization
) -> None:
    if llm.model != authorization.requested_model:
        raise ValueError("configured model does not match authorization")
    effort = llm.config.get("reasoning_effort")
    reasoning = llm.config.get("reasoning")
    if effort is None and isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
        if effort is None and reasoning.get("enabled") is True:
            effort = "enabled"
    if effort != authorization.allowed_reasoning_effort:
        raise ValueError("configured reasoning does not match authorization")
    if llm.config.get("temperature") != authorization.allowed_temperature:
        raise ValueError("configured temperature does not match authorization")


def _reject_existing_run(
    destination: Path,
    *,
    authorization: CompatibilityAuthorization,
    authorization_sha256: str,
) -> None:
    journal = destination / "authorization-events.jsonl"
    if not journal.is_file():
        raise FileExistsError(destination)
    ledger = DurableAuthorizationLedger(
        journal,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
    )
    ledger.recover()
    raise RuntimeError(
        f"authorization run already exists with state {ledger.state.run_state}"
    )


def _validate_output_ceiling(
    kwargs: dict[str, Any], authorization: CompatibilityAuthorization
) -> None:
    value = kwargs.get("max_tokens")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("provider request lacks a positive max_tokens ceiling")
    if value > authorization.max_output_tokens_per_call:
        raise ValueError("provider request exceeds authorized output-token ceiling")


def _request_value(
    messages: list[dict[str, Any]],
    tools: list[Tool] | None,
    output_model: type[BaseModel] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": _json_safe(messages),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_model.model_json_schema(),
            }
            for tool in tools or ()
        ],
        "output_model": getattr(output_model, "__name__", None),
        "parameters": _json_safe(kwargs),
    }


def _response_value(response: LLMResponse) -> dict[str, Any]:
    return {
        "content": _json_safe(response.content),
        "reasoning": _json_safe(response.reasoning),
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
        "finish_reason": response.finish_reason,
        "usage": _json_safe(response.usage),
        "raw_response": _json_safe(response.raw_response),
    }


def _exception_response(error: BaseException) -> dict[str, Any] | None:
    value = getattr(error, "response", None)
    return None if value is None else {"partial_response": _json_safe(value)}


def _provider_route(raw: Any) -> dict[str, str]:
    fields = {
        "provider": _raw_field(raw, "provider"),
        "model": _raw_field(raw, "model"),
        "reasoning_effort": _raw_field(raw, "reasoning_effort"),
    }
    return {key: value for key, value in fields.items() if isinstance(value, str)}


def _raw_field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
