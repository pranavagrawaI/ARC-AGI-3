"""Precommitment and execution gate for the cognitive-parity experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.cognitive_corpus import load_corpus
from tycho_nooa.native_parity import SYSTEM_PROMPT as NATIVE_SYSTEM_PROMPT
from tycho_nooa.sterile_codeact import SYSTEM_PROMPT as CODEACT_SYSTEM_PROMPT


class FileCommitment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ModelRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str | None = None
    provider: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None


class ExperimentCaps(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions: int = Field(ge=1)
    calls_per_session: int = Field(ge=1)
    calls_total: int = Field(ge=1)
    output_tokens_per_call: int = Field(ge=1)
    wall_seconds_per_session: float = Field(gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    reserve_per_call_usd: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def calls_fit_sessions(self) -> ExperimentCaps:
        if self.calls_total > self.sessions * self.calls_per_session:
            raise ValueError("total call cap exceeds the sum of session caps")
        if (self.max_cost_usd is None) != (self.reserve_per_call_usd is None):
            raise ValueError("cost cap and per-call reserve must be supplied together")
        if (
            self.max_cost_usd is not None
            and self.reserve_per_call_usd is not None
            and self.reserve_per_call_usd > self.max_cost_usd
        ):
            raise ValueError("per-call cost reserve exceeds total cost cap")
        return self


class AdvancementRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_parity_required: bool
    integrity_violations_allowed: int = Field(ge=0)
    minimum_failure_case_improvements: int = Field(ge=0)
    successful_case_regressions_allowed: int = Field(ge=0)
    alternative_cost_reduction_percent: float = Field(ge=0, le=100)


class ParityExperimentManifest(BaseModel):
    """A frozen comparison; false authorization is an intentional hard stop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str
    execution_authorized: bool
    runtime_commit: str
    package_commit: str
    corpus_manifest: FileCommitment
    corpus_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evaluator: FileCommitment
    runner: FileCommitment
    screen: FileCommitment
    gate: FileCommitment
    protocol: FileCommitment
    experiment_card: FileCommitment
    native_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    codeact_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_route: ModelRoute
    caps: ExperimentCaps
    primary_metrics: tuple[str, ...]
    invalidity_conditions: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    advancement_rule: AdvancementRule
    output_root: str

    @model_validator(mode="after")
    def authorized_manifest_is_fully_bound(self) -> ParityExperimentManifest:
        if not self.execution_authorized:
            return self
        route = self.model_route
        if not route.model or not route.provider or not route.reasoning_effort:
            raise ValueError("authorized execution requires a complete model route")
        if self.caps.max_cost_usd is None:
            raise ValueError("authorized execution requires an explicit cost cap")
        return self

    def require_execution_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError(
                "paid parity execution is disabled by the frozen experiment manifest"
            )


def load_experiment_manifest(
    path: Path, *, repository_root: Path
) -> ParityExperimentManifest:
    """Validate every committed input before a caller may inspect authorization."""
    manifest = ParityExperimentManifest.model_validate_json(path.read_text())
    for commitment in (
        manifest.corpus_manifest,
        manifest.evaluator,
        manifest.runner,
        manifest.screen,
        manifest.gate,
        manifest.protocol,
        manifest.experiment_card,
    ):
        _validate_file(commitment, repository_root)
    corpus_path = _committed_path(manifest.corpus_manifest.path, repository_root)
    corpus, _ = load_corpus(corpus_path, repository_root=repository_root)
    if corpus.corpus_sha256 != manifest.corpus_sha256:
        raise ValueError("experiment corpus hash does not match corpus manifest")
    _validate_prompt_hash(manifest.native_prompt_sha256, NATIVE_SYSTEM_PROMPT, "native")
    _validate_prompt_hash(
        manifest.codeact_prompt_sha256, CODEACT_SYSTEM_PROMPT, "codeact"
    )
    card = _committed_path(manifest.experiment_card.path, repository_root)
    if len(card.read_bytes()) > 1024:
        raise ValueError("experiment card exceeds 1 KiB")
    return manifest


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _validate_file(commitment: FileCommitment, repository_root: Path) -> None:
    path = _committed_path(commitment.path, repository_root)
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"file commitment mismatch: {commitment.path}")


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
        raise ValueError(f"{arm} prompt commitment mismatch")
