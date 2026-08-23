"""Committed inputs and disabled execution gate for parity compatibility v2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.compatibility_experiment import (
    CODEACT_COMPATIBILITY_PROMPT,
    NATIVE_COMPATIBILITY_PROMPT,
)
from tycho_nooa.compatibility_ladder import CompatibilityLadder
from tycho_nooa.parity_compatibility import load_compatibility_gate
from tycho_nooa.parity_manifest import FileCommitment, ModelRoute

CompatibilityGateRole = Literal[
    "termination",
    "metadata_query",
    "representation_rle",
    "representation_raw",
    "easy_positive_rle",
    "easy_positive_raw",
]
GATE_ROLES: tuple[CompatibilityGateRole, ...] = (
    "termination",
    "metadata_query",
    "representation_rle",
    "representation_raw",
    "easy_positive_rle",
    "easy_positive_raw",
)


class CompatibilityGateCommitment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: CompatibilityGateRole
    file: FileCommitment


class CompatibilityGateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    ladder_id: str = Field(min_length=1)
    gates: tuple[CompatibilityGateCommitment, ...]

    @model_validator(mode="after")
    def every_ladder_role_is_bound_once(self) -> CompatibilityGateManifest:
        roles = tuple(gate.role for gate in self.gates)
        if len(roles) != len(set(roles)):
            raise ValueError("compatibility gate roles must be unique")
        if set(roles) != set(GATE_ROLES):
            raise ValueError("compatibility gate manifest must bind every ladder role")
        return self


class CompatibilityCaps(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_gates: Literal[5] = 5
    calls_per_arm_gate: Literal[3] = 3
    calls_total: Literal[30] = 30
    output_tokens_per_call: int = Field(ge=1)
    wall_seconds_per_arm_gate: float = Field(gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    reserve_per_call_usd: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def cost_controls_are_joint(self) -> CompatibilityCaps:
        if (self.max_cost_usd is None) != (self.reserve_per_call_usd is None):
            raise ValueError("cost cap and per-call reserve must be supplied together")
        if (
            self.max_cost_usd is not None
            and self.reserve_per_call_usd is not None
            and self.reserve_per_call_usd > self.max_cost_usd
        ):
            raise ValueError("per-call cost reserve exceeds total cost cap")
        return self


class CompatibilityExperimentManifest(BaseModel):
    """A frozen eligibility experiment whose disabled state fails closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    execution_authorized: bool
    runtime_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    package_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    gate_manifest: FileCommitment
    gate_definitions: FileCommitment
    gate_executor: FileCommitment
    ladder_runner: FileCommitment
    authorization_gate: FileCommitment
    protocol: FileCommitment
    experiment_card: FileCommitment
    native_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    codeact_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_route: ModelRoute
    caps: CompatibilityCaps
    invalidity_conditions: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    result_to_action: dict[str, str]
    output_root: str

    @model_validator(mode="after")
    def authorized_manifest_is_fully_bound(self) -> CompatibilityExperimentManifest:
        if not self.execution_authorized:
            return self
        route = self.model_route
        if (
            not route.model
            or not route.reported_model
            or not route.provider
            or not route.reasoning_effort
        ):
            raise ValueError("authorized execution requires a complete model route")
        if self.caps.max_cost_usd is None:
            raise ValueError("authorized execution requires an explicit cost cap")
        return self

    def require_execution_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError(
                "paid compatibility execution is disabled by the frozen manifest"
            )


def load_gate_manifest(
    path: Path, *, repository_root: Path
) -> tuple[CompatibilityGateManifest, CompatibilityLadder]:
    manifest = CompatibilityGateManifest.model_validate_json(path.read_text())
    gates = {}
    for gate in manifest.gates:
        gate_path = _validate_file(gate.file, repository_root)
        gates[gate.role] = load_compatibility_gate(
            gate_path, repository_root=repository_root
        )
    return manifest, CompatibilityLadder.model_validate(gates)


def load_compatibility_experiment_manifest(
    path: Path, *, repository_root: Path
) -> tuple[CompatibilityExperimentManifest, CompatibilityLadder]:
    manifest = CompatibilityExperimentManifest.model_validate_json(path.read_text())
    for commitment in (
        manifest.gate_manifest,
        manifest.gate_definitions,
        manifest.gate_executor,
        manifest.ladder_runner,
        manifest.authorization_gate,
        manifest.protocol,
        manifest.experiment_card,
    ):
        _validate_file(commitment, repository_root)
    gate_manifest_path = _committed_path(
        manifest.gate_manifest.path, repository_root
    )
    _, ladder = load_gate_manifest(
        gate_manifest_path, repository_root=repository_root
    )
    _validate_prompt_hash(
        manifest.native_prompt_sha256, NATIVE_COMPATIBILITY_PROMPT, "native"
    )
    _validate_prompt_hash(
        manifest.codeact_prompt_sha256, CODEACT_COMPATIBILITY_PROMPT, "codeact"
    )
    card = _committed_path(manifest.experiment_card.path, repository_root)
    if len(card.read_bytes()) > 1024:
        raise ValueError("experiment card exceeds 1 KiB")
    return manifest, ladder


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _validate_file(commitment: FileCommitment, repository_root: Path) -> Path:
    path = _committed_path(commitment.path, repository_root)
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"file commitment mismatch: {commitment.path}")
    return path


def _committed_path(value: str, repository_root: Path) -> Path:
    root = repository_root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("committed path escapes repository root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_prompt_hash(expected: str, prompt: str, arm: str) -> None:
    if sha256_text(prompt) != expected:
        raise ValueError(f"{arm} compatibility prompt commitment mismatch")
