"""Fail-closed authorization seam for two bounded on-policy ARC actions."""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.compatibility_authorization_v2 import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
)
from tycho_nooa.compatibility_launch_v2 import (
    AuthorizedCompatibilityLLM,
    require_resource_safe_delegate,
)
from tycho_nooa.matched_actor_manifest import sha256_file

MANIFEST_ENV = "TYCHO_NOOA_ON_POLICY_MANIFEST"
AUTHORIZATION_ENV = "TYCHO_NOOA_ON_POLICY_AUTHORIZATION"
LEDGER_ENV = "TYCHO_NOOA_ON_POLICY_LEDGER"
_GATE_ID = "two-action-on-policy"
_RUNTIME_FILES = (
    "arc-host/pyproject.toml",
    "arc-host/uv.lock",
    "solvers/tycho_nooa/pyproject.toml",
    "solvers/tycho_nooa/uv.lock",
    "tools/hermetic_gate.py",
    "tools/hermetic_gate_core.py",
)
_RUNTIME_ROOTS = (
    "arc-host/src/arc_host",
    "solvers/tycho_nooa/src/tycho_nooa",
)


class OnPolicyCaps(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: Literal[2]
    calls_total: Literal[24]
    input_tokens_per_call: Literal[262_144]
    output_tokens_per_call: Literal[4_096]
    max_cost_usd: float = Field(gt=0)
    worker_timeout_seconds: Literal[900]
    launcher_timeout_seconds: Literal[1860]

    @model_validator(mode="after")
    def exact_cost_cap(self) -> OnPolicyCaps:
        if self.max_cost_usd != 0.16:
            raise ValueError("on-policy spend cap must remain $0.16")
        return self


class OnPolicyRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["openai/deepseek/deepseek-v4-flash-0731"]
    reported_model: Literal["deepseek/deepseek-v4-flash-0731"]
    provider: Literal["DeepInfra"]
    reasoning_effort: Literal["enabled"]
    temperature: None = None


class OnPolicyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: Literal["production-actor-on-policy-v1-preflight"]
    execution_authorized: bool
    game_id: Literal["ar25"]
    seed: Literal[0]
    mode: Literal["no_world_model"]
    treatment: Literal["control"]
    sandbox_runtime: Literal["host"]
    route: OnPolicyRoute
    caps: OnPolicyCaps
    runtime_bundle_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    launcher_path: str
    launcher_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    experiment_card_path: str
    experiment_card_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    pricing_source_path: str
    pricing_source_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    output_root: str
    advancement_rule: str
    invalidity_conditions: tuple[str, ...]


class AuthorizedOnPolicyLLM(AuthorizedCompatibilityLLM):
    """Production client with one authorization ledger and explicit finish."""

    def __init__(
        self,
        delegate: UnifiedLLM,
        *,
        manifest: OnPolicyManifest,
        ledger: DurableAuthorizationLedger,
    ) -> None:
        super().__init__(
            delegate,
            gate_id=_GATE_ID,
            arm="native",
            ledger=ledger,
        )
        self.manifest = manifest

    def finish(self, *, stop_reason: str, actions: int, levels_completed: int) -> None:
        calls = tuple(self._ledger.state.calls.values())
        passed = (
            actions == self.manifest.caps.actions
            and stop_reason == "action_budget"
            and len(calls) >= self.manifest.caps.actions
            and all(
                call.state == "accounted" and call.route_conforming is True
                for call in calls
            )
        )
        report = {
            "schema": "tycho-nooa-on-policy-worker-finish-v1",
            "passed": passed,
            "stop_reason": stop_reason,
            "actions": actions,
            "levels_completed": levels_completed,
            "provider_calls": self._ledger.state.committed_calls,
            "actual_provider_spend_usd": str(
                self._ledger.state.actual_provider_spend_usd
            ),
        }
        self._ledger.complete_gate(_GATE_ID, passed=passed, report=report)
        self._ledger.finish_ladder(
            "completed_on_policy" if passed else "do_not_advance",
            report=report,
        )


def preflight_on_policy_package(
    manifest_path: Path,
    authorization_path: Path | None,
    *,
    repository_root: Path,
    require_enabled: bool = False,
) -> tuple[OnPolicyManifest, CompatibilityAuthorization | None, str | None]:
    """Validate package identity without constructing ARC or provider clients."""

    manifest = OnPolicyManifest.model_validate_json(manifest_path.read_text())
    _validate_bound_file(
        repository_root, manifest.launcher_path, manifest.launcher_sha256
    )
    _validate_bound_file(
        repository_root,
        manifest.experiment_card_path,
        manifest.experiment_card_sha256,
    )
    _validate_bound_file(
        repository_root, manifest.pricing_source_path, manifest.pricing_source_sha256
    )
    if runtime_bundle_sha256(repository_root) != manifest.runtime_bundle_sha256:
        raise ValueError("on-policy runtime bundle drift")
    _validate_import_origin(repository_root)
    if shutil.which("bwrap") is None:
        raise RuntimeError("host sandbox treatment requires bubblewrap")

    if not manifest.execution_authorized:
        if require_enabled:
            raise RuntimeError("on-policy paid execution is disabled")
        if authorization_path is not None:
            raise ValueError("disabled preflight must not carry an authorization")
        return manifest, None, None
    if authorization_path is None:
        raise RuntimeError("enabled on-policy run lacks an authorization")

    authorization = CompatibilityAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    authorization.require_current()
    if authorization.experiment_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("authorization does not bind the on-policy manifest")
    _validate_authorization(manifest, authorization)
    return manifest, authorization, sha256_file(authorization_path)


def authorize_on_policy_llm(
    delegate: UnifiedLLM,
    environment: Mapping[str, str] | None = None,
) -> UnifiedLLM:
    """Wrap the worker's live client only when all authority paths are present."""

    env = os.environ if environment is None else environment
    values = [env.get(name) for name in (MANIFEST_ENV, AUTHORIZATION_ENV, LEDGER_ENV)]
    if not any(values):
        return delegate
    if not all(values):
        raise RuntimeError("on-policy authority environment is incomplete")
    assert all(value is not None for value in values)
    manifest_path, authorization_path, ledger_path = (Path(value) for value in values)
    repository_root = _repository_root()
    manifest, authorization, authorization_hash = preflight_on_policy_package(
        manifest_path,
        authorization_path,
        repository_root=repository_root,
        require_enabled=True,
    )
    assert authorization is not None and authorization_hash is not None
    _validate_worker_environment(manifest, delegate, env)
    require_resource_safe_delegate(delegate)
    ledger = DurableAuthorizationLedger.create(
        ledger_path,
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    ledger.start_gate(_GATE_ID)
    return AuthorizedOnPolicyLLM(delegate, manifest=manifest, ledger=ledger)


def finalize_on_policy_llm(
    llm: UnifiedLLM | None,
    *,
    stop_reason: str,
    actions: int,
    levels_completed: int,
) -> None:
    if isinstance(llm, AuthorizedOnPolicyLLM):
        llm.finish(
            stop_reason=stop_reason,
            actions=actions,
            levels_completed=levels_completed,
        )


def runtime_bundle_sha256(repository_root: Path) -> str:
    root = repository_root.resolve()
    files = [_safe_file(root, relative) for relative in _RUNTIME_FILES]
    for relative in _RUNTIME_ROOTS:
        package_root = (root / relative).resolve()
        if not package_root.is_dir():
            raise FileNotFoundError(package_root)
        files.extend(
            path
            for path in package_root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _validate_authorization(
    manifest: OnPolicyManifest,
    authorization: CompatibilityAuthorization,
) -> None:
    route = manifest.route
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


def _validate_worker_environment(
    manifest: OnPolicyManifest,
    delegate: UnifiedLLM,
    environment: Mapping[str, str],
) -> None:
    if environment.get("TYCHO_NOOA_SCRIPTED_TOOL_CALLS"):
        raise RuntimeError("authorized on-policy execution requires a live client")
    if environment.get("TYCHO_MODE") != manifest.mode:
        raise ValueError("worker mode does not match the on-policy manifest")
    if environment.get("TYCHO_NOOA_TREATMENT", "control") != manifest.treatment:
        raise ValueError("worker treatment does not match the on-policy manifest")
    if environment.get("TYCHO_SANDBOX_RUNTIME") != manifest.sandbox_runtime:
        raise ValueError("sandbox runtime does not match the on-policy manifest")
    if environment.get("ARC_LLM_PROVIDER_ENDPOINT") != manifest.route.provider:
        raise ValueError("provider endpoint does not match the on-policy manifest")
    if int(environment.get("ARC_LLM_MAX_TOKENS", "0")) != (
        manifest.caps.output_tokens_per_call
    ):
        raise ValueError("output ceiling does not match the on-policy manifest")
    reasoning = delegate.config.get("reasoning")
    effort = delegate.config.get("reasoning_effort")
    if effort is None and isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
        if effort is None and reasoning.get("enabled") is True:
            effort = "enabled"
    if (
        delegate.model != manifest.route.model
        or effort != manifest.route.reasoning_effort
        or delegate.config.get("temperature") != manifest.route.temperature
    ):
        raise ValueError("configured LLM does not match the on-policy route")


def _validate_bound_file(
    repository_root: Path,
    relative: str,
    expected_sha256: str,
) -> None:
    path = _safe_file(repository_root.resolve(), relative)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"bound file drift: {relative}")


def _safe_file(repository_root: Path, relative: str) -> Path:
    path = (repository_root / relative).resolve()
    if not path.is_relative_to(repository_root) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_import_origin(repository_root: Path) -> None:
    import tycho_nooa

    expected = (repository_root / "solvers/tycho_nooa/src/tycho_nooa").resolve()
    actual = Path(inspect.getfile(tycho_nooa)).resolve()
    if not actual.is_relative_to(expected):
        raise ValueError("imported tycho_nooa does not resolve under the bound checkout")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]
