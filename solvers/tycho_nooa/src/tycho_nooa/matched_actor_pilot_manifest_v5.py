"""Hash-bound manifest for the clean-checkout-stable Matched Actor pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tycho_nooa.matched_actor_context import (
    CONTEXT_CODEACT_SYSTEM_PROMPT,
    CONTEXT_NATIVE_SYSTEM_PROMPT,
)
from tycho_nooa.matched_actor_manifest import sha256_file, sha256_text
from tycho_nooa.matched_actor_pilot_manifest_v3 import (
    nooa_request_preparation_sha256,
)
from tycho_nooa.matched_actor_pilot_manifest_v4 import (
    MatchedActorPilotManifestV4,
)
from tycho_nooa.parity_manifest import FileCommitment
from tycho_nooa.repaired_checkpoint import load_repaired_pilot_sample


class MatchedActorPilotManifestV5(MatchedActorPilotManifestV4):
    """V4 treatment plus an explicit semantic checkpoint identity overlay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[5] = 5
    base_v4_package: FileCommitment
    checkpoint_loader: FileCommitment
    checkpoint_commitments: FileCommitment


class ReplayArtifactIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    files: tuple[FileCommitment, ...]


def load_matched_actor_pilot_manifest_v5(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotManifestV5, tuple]:
    manifest = MatchedActorPilotManifestV5.model_validate_json(path.read_text())
    for commitment in (
        manifest.base_treatment,
        manifest.base_v3_package,
        manifest.base_v4_package,
        manifest.manifest_schema,
        manifest.dependency_lock,
        manifest.sample,
        manifest.selection_driver,
        manifest.runner,
        manifest.launcher,
        manifest.authorization_schema,
        manifest.durable_ledger,
        manifest.provider_wrapper,
        manifest.actor_runner,
        manifest.evidence_service,
        manifest.context_evidence,
        manifest.context_projection,
        manifest.context_execution,
        manifest.context_controller,
        manifest.audit_driver,
        manifest.checkpoint_loader,
        manifest.checkpoint_commitments,
        manifest.freeze_driver,
        manifest.replay_driver,
        manifest.replay_report,
        manifest.replay_artifact_index,
        manifest.protocol,
        manifest.experiment_card,
    ):
        _validate_file(commitment, repository_root)
    sample, packets = load_repaired_pilot_sample(
        repository_root / manifest.sample.path,
        repository_root / manifest.checkpoint_commitments.path,
        repository_root=repository_root,
    )
    if sample.sample_sha256 != manifest.sample_sha256:
        raise ValueError("V5 pilot sample semantic commitment mismatch")
    if sha256_text(CONTEXT_NATIVE_SYSTEM_PROMPT) != manifest.native_prompt_sha256:
        raise ValueError("V5 pilot Native prompt commitment mismatch")
    if sha256_text(CONTEXT_CODEACT_SYSTEM_PROMPT) != manifest.codeact_prompt_sha256:
        raise ValueError("V5 pilot CodeAct prompt commitment mismatch")
    if nooa_request_preparation_sha256() != manifest.nooa_request_preparation_sha256:
        raise ValueError("V5 pilot NOOA request-preparation commitment mismatch")
    replay_index = ReplayArtifactIndex.model_validate_json(
        (repository_root / manifest.replay_artifact_index.path).read_text()
    )
    for artifact in replay_index.files:
        _validate_file(artifact, repository_root)
    if len((repository_root / manifest.experiment_card.path).read_bytes()) > 1_024:
        raise ValueError("V5 pilot experiment card exceeds 1 KiB")
    return manifest, packets


def _validate_file(commitment: FileCommitment, root: Path) -> None:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"V5 pilot committed file unavailable: {commitment.path}")
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"V5 pilot committed file drift: {commitment.path}")
