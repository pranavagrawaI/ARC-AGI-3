"""Hash-bound manifest for the repaired five-packet Matched Actor V2 pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.matched_actor_context_v2 import (
    CONTEXT_CODEACT_SYSTEM_PROMPT,
    CONTEXT_NATIVE_SYSTEM_PROMPT,
)
from tycho_nooa.matched_actor_manifest import sha256_file, sha256_text
from tycho_nooa.matched_actor_pilot_manifest_v3 import (
    nooa_request_preparation_sha256,
)
from tycho_nooa.matched_actor_pilot_v2 import (
    load_matched_actor_pilot_sample_v2,
)
from tycho_nooa.parity_manifest import FileCommitment, ModelRoute


class MatchedActorPilotCapsV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions: Literal[10] = 10
    calls_per_session: Literal[6] = 6
    calls_total: Literal[60] = 60
    input_tokens_per_call: int = Field(ge=1)
    output_tokens_per_call: int = Field(ge=1)
    wall_seconds_per_session: float = Field(gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class MatchedActorAdvancementV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    incumbent: Literal["native"] = "native"
    codeact_minimum_packet_wins: Literal[2] = 2
    codeact_losses_may_not_exceed_wins: Literal[True] = True
    equal_correctness_resource_reduction_percent: Literal[20] = 20
    complete_pairs_required: Literal[5] = 5


class MatchedActorPilotManifestV6(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[6] = 6
    experiment_id: Literal["matched-actor-v2-five-packet-pilot"] = "matched-actor-v2-five-packet-pilot"
    execution_authorized: bool
    base_contract_repair: FileCommitment
    dependency_lock: FileCommitment
    sample_schema: FileCommitment
    sample: FileCommitment
    sample_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runner: FileCommitment
    launcher: FileCommitment
    manifest_schema: FileCommitment
    authorization_schema: FileCommitment
    durable_ledger: FileCommitment
    provider_wrapper: FileCommitment
    actor_runner: FileCommitment
    evidence_service: FileCommitment
    context_evidence: FileCommitment
    context_projection: FileCommitment
    context_execution: FileCommitment
    context_worker: FileCommitment
    context_controller: FileCommitment
    audit_driver: FileCommitment
    replay_driver: FileCommitment
    replay_report: FileCommitment
    replay_artifact_index: FileCommitment
    checkpoint_loader: FileCommitment
    checkpoint_commitments: FileCommitment
    freeze_driver: FileCommitment
    protocol: FileCommitment
    experiment_card: FileCommitment
    native_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    codeact_prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    nooa_request_preparation_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_route: ModelRoute
    caps: MatchedActorPilotCapsV2
    advancement: MatchedActorAdvancementV2
    invalidity_conditions: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    output_root: str

    @model_validator(mode="after")
    def paid_execution_is_fully_bound(self) -> MatchedActorPilotManifestV6:
        if self.execution_authorized and self.caps.max_cost_usd is None:
            raise ValueError("authorized pilot requires an exact dollar cap")
        return self

    def require_execution_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError("paid Matched Actor V2 execution is disabled")


def load_matched_actor_pilot_manifest_v6(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotManifestV6, tuple]:
    manifest = MatchedActorPilotManifestV6.model_validate_json(path.read_text())
    for commitment in _commitments(manifest):
        _validate_file(commitment, repository_root)
    sample, packets = load_matched_actor_pilot_sample_v2(
        repository_root / manifest.sample.path,
        repository_root=repository_root,
    )
    if sample.sample_sha256 != manifest.sample_sha256:
        raise ValueError("Matched Actor V2 sample semantic commitment mismatch")
    if sha256_text(CONTEXT_NATIVE_SYSTEM_PROMPT) != manifest.native_prompt_sha256:
        raise ValueError("Matched Actor V2 Native prompt commitment mismatch")
    if sha256_text(CONTEXT_CODEACT_SYSTEM_PROMPT) != manifest.codeact_prompt_sha256:
        raise ValueError("Matched Actor V2 CodeAct prompt commitment mismatch")
    if nooa_request_preparation_sha256() != manifest.nooa_request_preparation_sha256:
        raise ValueError("Matched Actor V2 request-preparation commitment mismatch")
    if len((repository_root / manifest.experiment_card.path).read_bytes()) > 1_024:
        raise ValueError("Matched Actor V2 experiment card exceeds 1 KiB")
    replay_index = ReplayArtifactIndex.model_validate_json(
        (repository_root / manifest.replay_artifact_index.path).read_text()
    )
    for artifact in replay_index.files:
        _validate_file(artifact, repository_root)
    return manifest, packets


class ReplayArtifactIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    files: tuple[FileCommitment, ...]


def _commitments(manifest: MatchedActorPilotManifestV6) -> tuple[FileCommitment, ...]:
    return (
        manifest.base_contract_repair,
        manifest.dependency_lock,
        manifest.sample_schema,
        manifest.sample,
        manifest.runner,
        manifest.launcher,
        manifest.manifest_schema,
        manifest.authorization_schema,
        manifest.durable_ledger,
        manifest.provider_wrapper,
        manifest.actor_runner,
        manifest.evidence_service,
        manifest.context_evidence,
        manifest.context_projection,
        manifest.context_execution,
        manifest.context_worker,
        manifest.context_controller,
        manifest.audit_driver,
        manifest.replay_driver,
        manifest.replay_report,
        manifest.replay_artifact_index,
        manifest.checkpoint_loader,
        manifest.checkpoint_commitments,
        manifest.freeze_driver,
        manifest.protocol,
        manifest.experiment_card,
    )


def _validate_file(commitment: FileCommitment, root: Path) -> None:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Matched Actor V2 input unavailable: {commitment.path}")
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"Matched Actor V2 input drift: {commitment.path}")
