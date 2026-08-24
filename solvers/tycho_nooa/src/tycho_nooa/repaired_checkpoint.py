"""Versioned checkpoint identity that excludes generated Python bytecode."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.matched_actor_corpus import (
    MatchedActorCorpusManifest,
    _validate_manifest,
)
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.matched_actor_pilot import MatchedActorPilotSample


class RepairedCheckpointCommitment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_checkpoint_root: str
    frozen_checkpoint_root: str
    semantic_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RepairedCheckpointIndex(BaseModel):
    """Correction overlay; historical packets remain byte-for-byte preserved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    hash_policy: Literal["exclude_python_bytecode_v1"]
    supersedes_field: Literal["task.checkpoint_sha256"]
    checkpoints: tuple[RepairedCheckpointCommitment, ...]

    @model_validator(mode="after")
    def roots_are_unique(self) -> RepairedCheckpointIndex:
        sources = [item.source_checkpoint_root for item in self.checkpoints]
        frozen = [item.frozen_checkpoint_root for item in self.checkpoints]
        if len(sources) != len(set(sources)) or len(frozen) != len(set(frozen)):
            raise ValueError("repaired checkpoint root is duplicated")
        return self


def semantic_tree_sha256(root: Path) -> str:
    """Hash stable checkpoint meaning, never interpreter-generated cache files."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and _is_semantic(item)
    ):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def load_repaired_pilot_sample(
    sample_path: Path,
    checkpoint_index_path: Path,
    *,
    repository_root: Path,
) -> tuple[MatchedActorPilotSample, tuple[CognitiveParityPacket, ...]]:
    """Load the old selection through a new, clean-checkout-stable identity layer."""
    root = repository_root.resolve()
    sample = MatchedActorPilotSample.model_validate_json(sample_path.read_text())
    corpus_path = _committed_path(
        sample.source_corpus.path,
        sample.source_corpus.sha256,
        root,
    )
    corpus = MatchedActorCorpusManifest.model_validate_json(corpus_path.read_text())
    _validate_manifest(corpus)
    if corpus.corpus_sha256 != sample.source_corpus_sha256:
        raise ValueError("repaired pilot source corpus semantic commitment mismatch")
    index = RepairedCheckpointIndex.model_validate_json(
        checkpoint_index_path.read_text()
    )
    commitments = {item.source_checkpoint_root: item for item in index.checkpoints}
    records = {item.packet_id: item for item in corpus.records}
    packets = tuple(
        _load_repaired_packet(record.packet_id, records, commitments, root)
        for record in sample.packets
    )
    return sample, packets


def _load_repaired_packet(
    packet_id: str,
    records: dict,
    commitments: dict[str, RepairedCheckpointCommitment],
    root: Path,
) -> CognitiveParityPacket:
    record = records.get(packet_id)
    if record is None:
        raise ValueError(f"repaired pilot packet is absent: {packet_id}")
    packet_path = _committed_path(record.packet_path, record.packet_sha256, root)
    packet = CognitiveParityPacket.model_validate_json(packet_path.read_text())
    if packet.task.packet_id != packet_id:
        raise ValueError(f"repaired pilot packet identity drift: {packet_id}")
    commitment = commitments.get(packet.checkpoint_root)
    if commitment is None:
        raise ValueError(f"repaired checkpoint is uncommitted: {packet.checkpoint_root}")
    checkpoint = _confined_path(commitment.frozen_checkpoint_root, root)
    if semantic_tree_sha256(checkpoint) != commitment.semantic_sha256:
        raise ValueError(f"repaired checkpoint commitment mismatch: {packet_id}")
    return packet.model_copy(
        update={"checkpoint_root": commitment.frozen_checkpoint_root}
    )


def _committed_path(path_text: str, expected_sha256: str, root: Path) -> Path:
    path = _confined_path(path_text, root)
    if not path.is_file():
        raise ValueError(f"repaired committed file unavailable: {path_text}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"repaired committed file drift: {path_text}")
    return path


def _confined_path(path_text: str, root: Path) -> Path:
    path = (root / path_text).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"repaired path escapes repository: {path_text}")
    return path


def _is_semantic(path: Path) -> bool:
    return "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
