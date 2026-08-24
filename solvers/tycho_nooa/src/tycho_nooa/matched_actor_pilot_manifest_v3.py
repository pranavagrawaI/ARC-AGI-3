"""Hash-bound manifest for the value-adapted Matched Actor V3 pilot."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

from nooa.unifiedllm import CompletionClient
from nooa.unifiedllm.unifiedllm import (
    _maybe_sanitize_response_format,
    _messages_have_tool_calls,
    _needs_dummy_tool,
)
from pydantic import ConfigDict

from tycho_nooa.matched_actor import CODEACT_SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT
from tycho_nooa.matched_actor_manifest import (
    load_matched_actor_manifest,
    sha256_file,
    sha256_text,
)
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample
from tycho_nooa.matched_actor_pilot_manifest import (
    MatchedActorPilotManifest,
)
from tycho_nooa.parity_manifest import FileCommitment


class MatchedActorPilotManifestV3(MatchedActorPilotManifest):
    """V2 containment plus an explicit legacy-query value boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3] = 3
    manifest_schema: FileCommitment
    dependency_lock: FileCommitment
    nooa_request_preparation_sha256: str
    actor_runner: FileCommitment
    evidence_service: FileCommitment
    request_admission: FileCommitment
    provider_wrapper_base: FileCommitment
    pilot_runner_base: FileCommitment
    pilot_launcher_base: FileCommitment
    freeze_driver: FileCommitment
    replay_driver: FileCommitment
    replay_report: FileCommitment
    failed_replay_report: FileCommitment
    after_action: FileCommitment
    base_v2_package: FileCommitment


def load_matched_actor_pilot_manifest_v3(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotManifestV3, tuple]:
    manifest = MatchedActorPilotManifestV3.model_validate_json(path.read_text())
    for commitment in (
        manifest.base_treatment,
        manifest.base_v2_package,
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
        manifest.request_admission,
        manifest.provider_wrapper_base,
        manifest.pilot_runner_base,
        manifest.pilot_launcher_base,
        manifest.freeze_driver,
        manifest.replay_driver,
        manifest.replay_report,
        manifest.failed_replay_report,
        manifest.after_action,
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
        raise ValueError("V3 pilot sample semantic commitment mismatch")
    if sha256_text(NATIVE_SYSTEM_PROMPT) != manifest.native_prompt_sha256:
        raise ValueError("V3 pilot Native prompt commitment mismatch")
    if sha256_text(CODEACT_SYSTEM_PROMPT) != manifest.codeact_prompt_sha256:
        raise ValueError("V3 pilot CodeAct prompt commitment mismatch")
    if nooa_request_preparation_sha256() != manifest.nooa_request_preparation_sha256:
        raise ValueError("V3 pilot NOOA request-preparation commitment mismatch")
    if len((repository_root / manifest.experiment_card.path).read_bytes()) > 1_024:
        raise ValueError("V3 pilot experiment card exceeds 1 KiB")
    return manifest, packets


def nooa_request_preparation_sha256() -> str:
    """Commit the installed code whose behavior the V2 mirror must match."""
    sources = (
        CompletionClient.acall,
        CompletionClient._inject_cache_control,
        CompletionClient._convert_tool_to_schema,
        _maybe_sanitize_response_format,
        _messages_have_tool_calls,
        _needs_dummy_tool,
    )
    return sha256_text("\n".join(inspect.getsource(value) for value in sources))


def _validate_file(commitment: FileCommitment, root: Path) -> None:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"V3 pilot committed file unavailable: {commitment.path}")
    if sha256_file(path) != commitment.sha256:
        raise ValueError(f"V3 pilot committed file drift: {commitment.path}")
