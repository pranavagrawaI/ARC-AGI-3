"""Hash-bound paid launcher for a bounded production-Actor shadow replay."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.compatibility_authorization_v2 import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
)
from tycho_nooa.compatibility_launch_v2 import (
    AuthorizedCompatibilityLLM,
    require_resource_safe_delegate,
)
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.production_shadow_replay import execute_production_shadow_replay

RUNTIME_FIXED_FILES = (
    "analysis/production-actor-paid-shadow-v1/run.py",
    "solvers/tycho_nooa/pyproject.toml",
    "solvers/tycho_nooa/uv.lock",
)
RUNTIME_PACKAGE_ROOT = "solvers/tycho_nooa/src/tycho_nooa"


class ProductionShadowPrerequisite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    certificate_path: str
    short_manifest_path: str
    short_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    short_authorization_path: str
    short_authorization_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ProductionShadowCaps(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turns: int = Field(ge=1)
    calls_total: int = Field(ge=1, le=30)
    input_tokens_per_call: int = Field(ge=1)
    output_tokens_per_call: int = Field(ge=1)
    max_cost_usd: float = Field(gt=0)


class ProductionShadowRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    reported_model: str
    provider: str
    reasoning_effort: str
    temperature: float | None = None


class ProductionShadowManifest(BaseModel):
    """One feasibility run whose only consequence is a recorded shadow decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str
    execution_authorized: bool
    phase: Literal["short", "long"]
    prerequisite: ProductionShadowPrerequisite | None = None
    runtime_bundle_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    transcript_path: str
    transcript_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    controller_path: str
    controller_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    launcher_path: str
    launcher_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    experiment_card_path: str
    experiment_card_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    pricing_source_path: str
    pricing_source_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_route: ProductionShadowRoute
    caps: ProductionShadowCaps
    output_root: str
    advancement_rule: str
    invalidity_conditions: tuple[str, ...]

    def require_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError("paid production shadow execution is disabled")


async def execute_authorized_production_shadow(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    destination: Path,
    llm: UnifiedLLM,
) -> dict[str, Any]:
    """Execute one bounded shadow through the durable paid reference monitor."""

    manifest, authorization, authorization_hash = preflight_production_shadow(
        manifest_path,
        authorization_path,
        repository_root=repository_root,
        llm=llm,
    )
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    (destination / "experiment.json").write_bytes(manifest_path.read_bytes())
    (destination / "authorization.json").write_bytes(authorization_path.read_bytes())
    ledger = DurableAuthorizationLedger.create(
        destination / "authorization-events.jsonl",
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    gate_id = f"{manifest.phase}-shadow"
    ledger.start_gate(gate_id)
    try:
        report = await execute_production_shadow_replay(
            transcript=repository_root / manifest.transcript_path,
            destination=destination / "shadow",
            llm=AuthorizedCompatibilityLLM(
                llm,
                gate_id=gate_id,
                arm="native",
                ledger=ledger,
            ),
            game_id="ar25",
            mode="single",
            treatment="control",
            max_turns=manifest.caps.turns,
        )
        passed = _shadow_passed(report, manifest)
        assessment = _assessment(report, manifest, ledger, passed=passed)
        _write_json(destination / "assessment.json", assessment)
        ledger.complete_gate(gate_id, passed=passed, report=assessment)
        disposition = (
            "advance"
            if passed and manifest.phase == "short"
            else ("completed_long_shadow" if passed else "do_not_advance")
        )
        ledger.finish_ladder(disposition, report=assessment)
        if manifest.phase == "short" and passed:
            _write_advancement_certificate(
                destination,
                manifest_path=manifest_path,
                authorization_path=authorization_path,
                authorization=authorization,
                assessment_path=destination / "assessment.json",
                ledger_path=destination / "authorization-events.jsonl",
                repository_root=repository_root,
            )
        return assessment
    except Exception as error:
        if ledger.state.run_state == "armed":
            ledger.stop(f"shadow_launcher_failure:{type(error).__name__}")
        _write_json(
            destination / "launch-failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def preflight_production_shadow(
    manifest_path: Path,
    authorization_path: Path,
    *,
    repository_root: Path,
    llm: UnifiedLLM,
) -> tuple[ProductionShadowManifest, CompatibilityAuthorization, str]:
    """Validate the exact evidence, implementation, route, and authorization."""

    manifest = ProductionShadowManifest.model_validate_json(manifest_path.read_text())
    manifest.require_authorized()
    _validate_runtime_bundle(repository_root, manifest.runtime_bundle_sha256)
    _validate_import_origin(repository_root)
    _validate_committed_file(
        repository_root, manifest.transcript_path, manifest.transcript_sha256
    )
    _validate_committed_file(
        repository_root, manifest.controller_path, manifest.controller_sha256
    )
    _validate_committed_file(
        repository_root, manifest.launcher_path, manifest.launcher_sha256
    )
    _validate_committed_file(
        repository_root,
        manifest.experiment_card_path,
        manifest.experiment_card_sha256,
    )
    _validate_committed_file(
        repository_root,
        manifest.pricing_source_path,
        manifest.pricing_source_sha256,
    )
    if manifest.phase == "long":
        _validate_prerequisite(repository_root, manifest.prerequisite)
    elif manifest.prerequisite is not None:
        raise ValueError("short shadow must not declare a prerequisite")

    authorization = CompatibilityAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    authorization.require_current()
    if authorization.experiment_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("authorization does not bind the shadow manifest")
    _validate_route_and_caps(manifest, authorization, llm)
    require_resource_safe_delegate(llm)
    return manifest, authorization, sha256_file(authorization_path)


def _validate_committed_file(
    repository_root: Path, relative: str, expected: str
) -> None:
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise ValueError(f"committed file drift: {relative}")


def _validate_prerequisite(
    repository_root: Path,
    prerequisite: ProductionShadowPrerequisite | None,
) -> None:
    if prerequisite is None:
        raise ValueError("long shadow lacks its short-run prerequisite")
    manifest_path = _bound_file(
        repository_root,
        prerequisite.short_manifest_path,
        prerequisite.short_manifest_sha256,
    )
    authorization_path = _bound_file(
        repository_root,
        prerequisite.short_authorization_path,
        prerequisite.short_authorization_sha256,
    )
    short_manifest = ProductionShadowManifest.model_validate_json(
        manifest_path.read_text()
    )
    short_authorization = CompatibilityAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    if short_manifest.phase != "short":
        raise ValueError("advancement prerequisite is not a short manifest")
    if short_authorization.experiment_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("short authorization does not bind its manifest")
    certificate_path = _safe_file(repository_root, prerequisite.certificate_path)
    certificate = json.loads(certificate_path.read_text())
    expected_identity = {
        "schema": "tycho-nooa-production-shadow-advancement-v1",
        "authorization_id": short_authorization.authorization_id,
        "short_manifest_sha256": sha256_file(manifest_path),
        "short_authorization_sha256": sha256_file(authorization_path),
        "advance": True,
    }
    if any(certificate.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("short advancement certificate identity mismatch")
    assessment_path = _safe_file(repository_root, certificate["assessment_path"])
    ledger_path = _safe_file(repository_root, certificate["ledger_path"])
    if sha256_file(assessment_path) != certificate.get(
        "assessment_sha256"
    ) or sha256_file(ledger_path) != certificate.get("ledger_sha256"):
        raise ValueError("short advancement certificate artifact drift")
    assessment = json.loads(assessment_path.read_text())
    if assessment.get("passed") is not True or assessment.get("advance") is not True:
        raise RuntimeError("short shadow did not authorize advancement")
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    if not events or events[-1].get("event_type") != "RunCompleted":
        raise RuntimeError("short authorization ledger is not completed")


def runtime_bundle_sha256(repository_root: Path) -> str:
    """Hash the complete local control plane used by the paid shadow."""

    files = [_safe_file(repository_root, relative) for relative in RUNTIME_FIXED_FILES]
    package_root = (repository_root / RUNTIME_PACKAGE_ROOT).resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    files.extend(
        path
        for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(repository_root.resolve()).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _validate_runtime_bundle(repository_root: Path, expected: str) -> None:
    if runtime_bundle_sha256(repository_root) != expected:
        raise ValueError("production Actor runtime bundle drift")


def _validate_import_origin(repository_root: Path) -> None:
    import tycho_nooa

    expected = (repository_root / RUNTIME_PACKAGE_ROOT).resolve()
    actual = Path(inspect.getfile(tycho_nooa)).resolve()
    if not actual.is_relative_to(expected):
        raise ValueError(
            "imported tycho_nooa does not resolve under the bound checkout"
        )


def _safe_file(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _bound_file(repository_root: Path, relative: str, expected: str) -> Path:
    path = _safe_file(repository_root, relative)
    if sha256_file(path) != expected:
        raise ValueError(f"committed file drift: {relative}")
    return path


def _validate_route_and_caps(
    manifest: ProductionShadowManifest,
    authorization: CompatibilityAuthorization,
    llm: UnifiedLLM,
) -> None:
    route = manifest.model_route
    caps = manifest.caps
    if (
        authorization.requested_model != route.model
        or authorization.expected_reported_model != route.reported_model
        or authorization.allowed_provider != route.provider
        or authorization.allowed_reasoning_effort != route.reasoning_effort
        or authorization.allowed_temperature != route.temperature
        or authorization.maximum_calls != caps.calls_total
        or authorization.max_input_tokens_per_call != caps.input_tokens_per_call
        or authorization.max_output_tokens_per_call != caps.output_tokens_per_call
        or float(authorization.maximum_spend_usd) != caps.max_cost_usd
    ):
        raise ValueError("authorization route or caps do not match the manifest")
    effort = llm.config.get("reasoning_effort")
    reasoning = llm.config.get("reasoning")
    if effort is None and isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
        if effort is None and reasoning.get("enabled") is True:
            effort = "enabled"
    if (
        llm.model != route.model
        or effort != route.reasoning_effort
        or llm.config.get("temperature") != route.temperature
    ):
        raise ValueError("configured LLM does not match the manifest route")


def _shadow_passed(
    report: Mapping[str, Any], manifest: ProductionShadowManifest
) -> bool:
    return (
        report.get("turns") == manifest.caps.turns
        and report.get("environment_actions_executed") == 0
        and len(report.get("trajectory", ())) == manifest.caps.turns
        and all(row.get("provider_calls", 0) >= 1 for row in report["trajectory"])
    )


def _assessment(
    report: Mapping[str, Any],
    manifest: ProductionShadowManifest,
    ledger: DurableAuthorizationLedger,
    *,
    passed: bool,
) -> dict[str, Any]:
    calls = ledger.state.committed_calls
    trajectory = report.get("trajectory", ())
    return {
        "schema": "tycho-nooa-paid-production-shadow-assessment-v1",
        "phase": manifest.phase,
        "passed": passed,
        "advance": passed and manifest.phase == "short",
        "turns": report.get("turns", 0),
        "provider_calls": calls,
        "actual_provider_spend_usd": str(ledger.state.actual_provider_spend_usd),
        "historical_action_agreement": report.get("historical_action_agreement", 0),
        "first_action_divergence_sequence": report.get(
            "first_action_divergence_sequence"
        ),
        "maximum_provider_request_bytes": max(
            (row.get("provider_request_bytes", 0) for row in trajectory),
            default=0,
        ),
        "environment_actions_executed": report.get("environment_actions_executed"),
        "claim_ceiling": (
            "feasibility, resource use, and off-policy historical agreement only; "
            "historical agreement is descriptive rather than action correctness"
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_advancement_certificate(
    destination: Path,
    *,
    manifest_path: Path,
    authorization_path: Path,
    authorization: CompatibilityAuthorization,
    assessment_path: Path,
    ledger_path: Path,
    repository_root: Path,
) -> None:
    root = repository_root.resolve()
    certificate = {
        "schema": "tycho-nooa-production-shadow-advancement-v1",
        "authorization_id": authorization.authorization_id,
        "short_manifest_sha256": sha256_file(manifest_path),
        "short_authorization_sha256": sha256_file(authorization_path),
        "assessment_path": assessment_path.resolve().relative_to(root).as_posix(),
        "assessment_sha256": sha256_file(assessment_path),
        "ledger_path": ledger_path.resolve().relative_to(root).as_posix(),
        "ledger_sha256": sha256_file(ledger_path),
        "advance": True,
    }
    target = destination / "advancement-certificate.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)
