"""Fail-closed package commitment for Matched Actor Treatment V1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.matched_actor import CODEACT_SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus
from tycho_nooa.parity_manifest import ExperimentCaps, FileCommitment, ModelRoute


class MatchedActorManifest(BaseModel):
    """One immutable comparison; authorization is deliberately a separate edit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str
    execution_authorized: bool
    base_commit: str
    corpus: FileCommitment
    corpus_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    controller: FileCommitment
    evaluator: FileCommitment
    runner: FileCommitment
    audit: FileCommitment
    audit_driver: FileCommitment
    audit_summary: FileCommitment
    shared_computation: FileCommitment
    protocol: FileCommitment
    experiment_card: FileCommitment
    native_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    codeact_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_route: ModelRoute
    caps: ExperimentCaps
    primary_metrics: tuple[str, ...]
    invalidity_conditions: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    result_map: tuple[str, ...]
    output_root: str

    @model_validator(mode="after")
    def paid_execution_is_fully_bound(self) -> MatchedActorManifest:
        if not self.execution_authorized:
            return self
        if not _route_is_exact(self.model_route):
            raise ValueError("authorized matched execution requires an exact route")
        if self.caps.max_cost_usd is None:
            raise ValueError("authorized matched execution requires a dollar cap")
        return self

    def require_execution_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError("paid Matched Actor execution is disabled")


def load_matched_actor_manifest(
    path: Path, *, repository_root: Path
) -> MatchedActorManifest:
    manifest = MatchedActorManifest.model_validate_json(path.read_text())
    for commitment in (
        manifest.corpus,
        manifest.controller,
        manifest.evaluator,
        manifest.runner,
        manifest.audit,
        manifest.audit_driver,
        manifest.audit_summary,
        manifest.shared_computation,
        manifest.protocol,
        manifest.experiment_card,
    ):
        _validate_file(commitment, repository_root)
    corpus, _ = load_matched_actor_corpus(
        repository_root / manifest.corpus.path,
        repository_root=repository_root,
    )
    if corpus.corpus_sha256 != manifest.corpus_sha256:
        raise ValueError("matched experiment corpus commitment mismatch")
    if sha256_text(NATIVE_SYSTEM_PROMPT) != manifest.native_prompt_sha256:
        raise ValueError("matched Native prompt commitment mismatch")
    if sha256_text(CODEACT_SYSTEM_PROMPT) != manifest.codeact_prompt_sha256:
        raise ValueError("matched CodeAct prompt commitment mismatch")
    if len((repository_root / manifest.experiment_card.path).read_bytes()) > 1_024:
        raise ValueError("matched experiment card exceeds 1 KiB")
    return manifest


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_file(commitment: FileCommitment, root: Path) -> None:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"matched committed file unavailable: {commitment.path}")
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"matched committed file drift: {commitment.path}")


def _route_is_exact(route: ModelRoute) -> bool:
    return all(
        (route.model, route.reported_model, route.provider, route.reasoning_effort)
    )
