"""Hash-bound manifest for the congruent-receipt five-packet pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from tycho_nooa import matched_actor_pilot_manifest_v6 as _v6
from tycho_nooa.matched_actor_context_v2 import (
    CONTEXT_CODEACT_SYSTEM_PROMPT,
    CONTEXT_NATIVE_SYSTEM_PROMPT,
)
from tycho_nooa.matched_actor_manifest import sha256_text
from tycho_nooa.matched_actor_pilot_manifest_v3 import (
    nooa_request_preparation_sha256,
)
from tycho_nooa.matched_actor_pilot_v2 import load_matched_actor_pilot_sample_v2
from tycho_nooa.parity_manifest import FileCommitment

MatchedActorPilotCapsV2 = _v6.MatchedActorPilotCapsV2
MatchedActorAdvancementV2 = _v6.MatchedActorAdvancementV2
ReplayArtifactIndex = _v6.ReplayArtifactIndex


class MatchedActorPilotManifestV7(_v6.MatchedActorPilotManifestV6):
    """V6 treatment with only the model-facing receipt execution path replaced."""

    schema_version: Literal[7] = 7
    experiment_id: Literal["matched-actor-v3-five-packet-pilot"] = (
        "matched-actor-v3-five-packet-pilot"
    )
    base_receipt_pilot: FileCommitment
    context_computational_evidence: FileCommitment
    context_computational_evidence_base: FileCommitment
    context_execution_base: FileCommitment
    context_worker_base: FileCommitment
    context_controller_base: FileCommitment
    acquisition_controller_base: FileCommitment
    contract_probe: FileCommitment
    contract_probe_report: FileCommitment
    sample_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    def require_execution_authorized(self) -> None:
        if not self.execution_authorized:
            raise RuntimeError("paid Matched Actor V3 receipt execution is disabled")


def load_matched_actor_pilot_manifest_v7(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotManifestV7, tuple]:
    manifest = MatchedActorPilotManifestV7.model_validate_json(path.read_text())
    for commitment in _commitments(manifest):
        _v6._validate_file(commitment, repository_root)
    sample, packets = load_matched_actor_pilot_sample_v2(
        repository_root / manifest.sample.path,
        repository_root=repository_root,
    )
    if sample.sample_sha256 != manifest.sample_sha256:
        raise ValueError("Matched Actor V3 receipt sample commitment mismatch")
    if sha256_text(CONTEXT_NATIVE_SYSTEM_PROMPT) != manifest.native_prompt_sha256:
        raise ValueError("Matched Actor V3 Native prompt commitment mismatch")
    if sha256_text(CONTEXT_CODEACT_SYSTEM_PROMPT) != manifest.codeact_prompt_sha256:
        raise ValueError("Matched Actor V3 CodeAct prompt commitment mismatch")
    if nooa_request_preparation_sha256() != manifest.nooa_request_preparation_sha256:
        raise ValueError("Matched Actor V3 request-preparation commitment mismatch")
    if len((repository_root / manifest.experiment_card.path).read_bytes()) > 1_024:
        raise ValueError("Matched Actor V3 experiment card exceeds 1 KiB")
    replay_index = ReplayArtifactIndex.model_validate_json(
        (repository_root / manifest.replay_artifact_index.path).read_text()
    )
    for artifact in replay_index.files:
        _v6._validate_file(artifact, repository_root)
    return manifest, packets


def _commitments(
    manifest: MatchedActorPilotManifestV7,
) -> tuple[FileCommitment, ...]:
    return (
        *_v6._commitments(manifest),
        manifest.base_receipt_pilot,
        manifest.context_computational_evidence,
        manifest.context_computational_evidence_base,
        manifest.context_execution_base,
        manifest.context_worker_base,
        manifest.context_controller_base,
        manifest.acquisition_controller_base,
        manifest.contract_probe,
        manifest.contract_probe_report,
    )
