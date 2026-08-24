"""Hash-bound manifest for the Context/Evidence Plane Matched Actor pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tycho_nooa.matched_actor_context import (
    CONTEXT_CODEACT_SYSTEM_PROMPT,
    CONTEXT_NATIVE_SYSTEM_PROMPT,
)
from tycho_nooa.matched_actor_manifest import (
    load_matched_actor_manifest,
    sha256_file,
    sha256_text,
)
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample
from tycho_nooa.matched_actor_pilot_manifest import MatchedActorPilotManifest
from tycho_nooa.matched_actor_pilot_manifest_v3 import (
    nooa_request_preparation_sha256,
)
from tycho_nooa.parity_manifest import FileCommitment


class MatchedActorPilotManifestV4(MatchedActorPilotManifest):
    """A disabled pilot whose only treatment difference is control language."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[4] = 4
    context_plane_base_revision: Literal["9747693"]
    base_v3_package: FileCommitment
    manifest_schema: FileCommitment
    dependency_lock: FileCommitment
    nooa_request_preparation_sha256: str
    actor_runner: FileCommitment
    evidence_service: FileCommitment
    context_evidence: FileCommitment
    context_projection: FileCommitment
    context_execution: FileCommitment
    context_controller: FileCommitment
    audit_driver: FileCommitment
    freeze_driver: FileCommitment
    replay_driver: FileCommitment
    replay_report: FileCommitment
    replay_artifact_index: FileCommitment


class ReplayArtifactIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    files: tuple[FileCommitment, ...]


def load_matched_actor_pilot_manifest_v4(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotManifestV4, tuple]:
    manifest = MatchedActorPilotManifestV4.model_validate_json(path.read_text())
    for commitment in (
        manifest.base_treatment,
        manifest.base_v3_package,
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
        manifest.freeze_driver,
        manifest.replay_driver,
        manifest.replay_report,
        manifest.replay_artifact_index,
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
        raise ValueError("V4 pilot sample semantic commitment mismatch")
    if sha256_text(CONTEXT_NATIVE_SYSTEM_PROMPT) != manifest.native_prompt_sha256:
        raise ValueError("V4 pilot Native prompt commitment mismatch")
    if sha256_text(CONTEXT_CODEACT_SYSTEM_PROMPT) != manifest.codeact_prompt_sha256:
        raise ValueError("V4 pilot CodeAct prompt commitment mismatch")
    if nooa_request_preparation_sha256() != manifest.nooa_request_preparation_sha256:
        raise ValueError("V4 pilot NOOA request-preparation commitment mismatch")
    replay_index = ReplayArtifactIndex.model_validate_json(
        (repository_root / manifest.replay_artifact_index.path).read_text()
    )
    for artifact in replay_index.files:
        _validate_file(artifact, repository_root)
    if len((repository_root / manifest.experiment_card.path).read_bytes()) > 1_024:
        raise ValueError("V4 pilot experiment card exceeds 1 KiB")
    return manifest, packets


def _validate_file(commitment: FileCommitment, root: Path) -> None:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"V4 pilot committed file unavailable: {commitment.path}")
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"V4 pilot committed file drift: {commitment.path}")
