"""Hash-bound, fail-closed manifest for Matched Actor V1-Pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.matched_actor import CODEACT_SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT
from tycho_nooa.matched_actor_manifest import (
    load_matched_actor_manifest,
    sha256_file,
    sha256_text,
)
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample
from tycho_nooa.parity_manifest import FileCommitment, ModelRoute


class PilotCaps(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions: Literal[12] = 12
    calls_per_session: Literal[6] = 6
    calls_total: Literal[72] = 72
    input_tokens_per_call: int = Field(ge=1)
    output_tokens_per_call: int = Field(ge=1)
    wall_seconds_per_session: float = Field(gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class PilotAdvancementRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    incumbent: Literal["native"] = "native"
    codeact_minimum_packet_wins: Literal[2] = 2
    codeact_losses_may_not_exceed_wins: Literal[True] = True
    equal_correctness_resource_reduction_percent: Literal[20] = 20
    full_treatment_requires_complete_pairs: Literal[6] = 6


class MatchedActorPilotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str
    execution_authorized: bool
    base_treatment: FileCommitment
    sample: FileCommitment
    selection_driver: FileCommitment
    sample_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runner: FileCommitment
    launcher: FileCommitment
    authorization_schema: FileCommitment
    durable_ledger: FileCommitment
    provider_wrapper: FileCommitment
    protocol: FileCommitment
    experiment_card: FileCommitment
    native_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    codeact_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_route: ModelRoute
    caps: PilotCaps
    advancement: PilotAdvancementRule
    invalidity_conditions: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    output_root: str

    @model_validator(mode="after")
    def paid_execution_is_fully_bound(self) -> MatchedActorPilotManifest:
        if not self.execution_authorized:
            return self
        if not all(
            (
                self.model_route.model,
                self.model_route.reported_model,
                self.model_route.provider,
                self.model_route.reasoning_effort,
            )
        ):
            raise ValueError("authorized pilot requires an exact model route")
        if self.caps.max_cost_usd is None:
            raise ValueError("authorized pilot requires an exact dollar cap")
        return self

    def require_execution_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError("paid Matched Actor pilot execution is disabled")


def load_matched_actor_pilot_manifest(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotManifest, tuple]:
    manifest = MatchedActorPilotManifest.model_validate_json(path.read_text())
    for commitment in (
        manifest.base_treatment,
        manifest.sample,
        manifest.selection_driver,
        manifest.runner,
        manifest.launcher,
        manifest.authorization_schema,
        manifest.durable_ledger,
        manifest.provider_wrapper,
        manifest.protocol,
        manifest.experiment_card,
    ):
        _validate_file(commitment, repository_root)
    load_matched_actor_manifest(
        repository_root / manifest.base_treatment.path,
        repository_root=repository_root,
    )
    sample, packets = load_matched_actor_pilot_sample(
        repository_root / manifest.sample.path,
        repository_root=repository_root,
    )
    if sample.sample_sha256 != manifest.sample_sha256:
        raise ValueError("pilot sample semantic commitment mismatch")
    if sha256_text(NATIVE_SYSTEM_PROMPT) != manifest.native_prompt_sha256:
        raise ValueError("pilot Native prompt commitment mismatch")
    if sha256_text(CODEACT_SYSTEM_PROMPT) != manifest.codeact_prompt_sha256:
        raise ValueError("pilot CodeAct prompt commitment mismatch")
    if len((repository_root / manifest.experiment_card.path).read_bytes()) > 1_024:
        raise ValueError("pilot experiment card exceeds 1 KiB")
    return manifest, packets


def _validate_file(commitment: FileCommitment, root: Path) -> None:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"pilot committed file unavailable: {commitment.path}")
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"pilot committed file drift: {commitment.path}")
