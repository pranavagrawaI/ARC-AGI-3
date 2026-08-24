"""Frozen A/B/C corpus for the matched Actor control-language treatment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.cognitive_parity import CognitiveParityPacket, load_packet

ActorCognitiveStratum = Literal[
    "computation_unnecessary",
    "computation_useful",
    "composition",
]


class MatchedActorCorpusRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    packet_path: str
    packet_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    stratum: ActorCognitiveStratum
    selection_rationale: str


class MatchedActorCorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    corpus_id: str
    selection_rule: str
    packet_count: int
    records: tuple[MatchedActorCorpusRecord, ...]
    corpus_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def load_matched_actor_corpus(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorCorpusManifest, tuple[CognitiveParityPacket, ...]]:
    """Validate selection balance, packet identity, and every byte commitment."""
    manifest = MatchedActorCorpusManifest.model_validate_json(path.read_text())
    _validate_manifest(manifest)
    packets = tuple(
        _load_record(record, repository_root) for record in manifest.records
    )
    return manifest, packets


def _validate_manifest(manifest: MatchedActorCorpusManifest) -> None:
    _validate_record_count(manifest)
    _validate_unique_packet_ids(manifest.records)
    _validate_stratum_balance(manifest.records)
    if _records_sha256(manifest.records) != manifest.corpus_sha256:
        raise ValueError("matched corpus commitment mismatch")


def _validate_record_count(manifest: MatchedActorCorpusManifest) -> None:
    if manifest.packet_count != len(manifest.records):
        raise ValueError("matched corpus packet_count does not match records")


def _validate_unique_packet_ids(
    records: tuple[MatchedActorCorpusRecord, ...],
) -> None:
    ids = [record.packet_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("matched corpus packet_id is duplicated")


def _validate_stratum_balance(
    records: tuple[MatchedActorCorpusRecord, ...],
) -> None:
    counts = {
        name: sum(record.stratum == name for record in records)
        for name in (
            "computation_unnecessary",
            "computation_useful",
            "composition",
        )
    }
    if len(set(counts.values())) != 1 or 0 in counts.values():
        raise ValueError("matched corpus must balance all three cognitive strata")


def _load_record(
    record: MatchedActorCorpusRecord, repository_root: Path
) -> CognitiveParityPacket:
    root = repository_root.resolve()
    path = (root / record.packet_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"matched packet path is unavailable: {record.packet_path}")
    if _file_sha256(path) != record.packet_sha256:
        raise ValueError(f"matched packet byte drift: {record.packet_id}")
    packet = load_packet(path, repository_root=repository_root)
    if packet.task.packet_id != record.packet_id:
        raise ValueError(f"matched packet identity drift: {record.packet_id}")
    return packet


def _records_sha256(records: tuple[MatchedActorCorpusRecord, ...]) -> str:
    value = [record.model_dump(mode="json") for record in records]
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
