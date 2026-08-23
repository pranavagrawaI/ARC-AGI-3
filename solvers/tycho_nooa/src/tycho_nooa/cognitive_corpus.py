"""Integrity and evidence-surface validation for a frozen parity corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.cognitive_parity import (
    CognitiveParityPacket,
    ControlKind,
    HistoricalOutcome,
    SelectionStratum,
    load_packet,
)
from tycho_nooa.evidence import EvidenceRef, WorkspaceEvidenceCapability


class CorpusRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    path: str
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    stratum: SelectionStratum
    historical_outcome: HistoricalOutcome
    control_kind: ControlKind


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    runtime_commit: str
    selection_rule: str
    historical_outcome_semantics: str
    packet_count: int
    packets: tuple[CorpusRecord, ...]
    corpus_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def load_corpus(
    manifest_path: Path, *, repository_root: Path
) -> tuple[CorpusManifest, tuple[CognitiveParityPacket, ...]]:
    """Validate corpus topology, packet bytes, checkpoints, and evidence refs."""
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text())
    _validate_manifest(manifest)
    packets = tuple(
        _load_record(record, manifest_path.parent, repository_root)
        for record in manifest.packets
    )
    _validate_selection_cross(packets)
    for packet in packets:
        _validate_evidence_surface(packet, repository_root)
    return manifest, packets


def _validate_manifest(manifest: CorpusManifest) -> None:
    if manifest.packet_count != len(manifest.packets):
        raise ValueError("packet_count does not match packet records")
    packet_ids = [record.packet_id for record in manifest.packets]
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("packet_id is duplicated")
    canonical = json.dumps(
        [record.model_dump(mode="json") for record in manifest.packets],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    actual = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    if actual != manifest.corpus_sha256:
        raise ValueError("corpus commitment mismatch")


def _load_record(
    record: CorpusRecord, manifest_root: Path, repository_root: Path
) -> CognitiveParityPacket:
    path = (manifest_root / record.path).resolve()
    if not path.is_relative_to(manifest_root.resolve()):
        raise ValueError("packet path escapes corpus directory")
    actual = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    if actual != record.sha256:
        raise ValueError(f"packet byte commitment mismatch: {record.packet_id}")
    packet = load_packet(path, repository_root=repository_root)
    identity = (
        packet.task.packet_id,
        packet.task.selection_stratum,
        packet.task.historical_outcome,
        packet.task.control_kind,
    )
    recorded = (
        record.packet_id,
        record.stratum,
        record.historical_outcome,
        record.control_kind,
    )
    if identity != recorded:
        raise ValueError(f"packet manifest metadata mismatch: {record.packet_id}")
    return packet


def _validate_selection_cross(packets: tuple[CognitiveParityPacket, ...]) -> None:
    pairs = {
        (packet.task.selection_stratum, packet.task.historical_outcome)
        for packet in packets
    }
    strata = {packet.task.selection_stratum for packet in packets}
    expected = {(stratum, outcome) for stratum in strata for outcome in ("success", "failure")}
    if pairs != expected:
        raise ValueError("corpus does not cross every stratum with success and failure")
    controls = {packet.task.control_kind for packet in packets}
    if not {"negative", "easy_positive", "composition"}.issubset(controls):
        raise ValueError("corpus is missing a required control class")


def _validate_evidence_surface(
    packet: CognitiveParityPacket, repository_root: Path
) -> None:
    root = repository_root / packet.checkpoint_root
    capability = WorkspaceEvidenceCapability.from_root(root)
    frame_refs = _all_refs(capability, "frames")
    transition_refs = _all_refs(capability, "transitions")
    available = frame_refs | transition_refs
    expected_refs = {
        (ref.level, ref.turn)
        for ref in (
            *packet.expected.required_evidence_refs,
            *packet.expected.acceptable_evidence_refs,
        )
    }
    if not expected_refs.issubset(available):
        raise ValueError(f"answer key cites absent evidence: {packet.task.packet_id}")
    frontier = capability.query({"kind": "frames", "limit": 1})["frontier"]
    if EvidenceRef.model_validate(frontier) != packet.task.evidence_frontier:
        raise ValueError(f"packet frontier is stale: {packet.task.packet_id}")


def _all_refs(
    capability: WorkspaceEvidenceCapability, kind: str
) -> set[tuple[int, int]]:
    refs: set[tuple[int, int]] = set()
    after = None
    while True:
        query = {"kind": kind, "limit": 32}
        if after is not None:
            query["after"] = after
        result = capability.query(query)
        refs.update((record["level"], record["turn"]) for record in result["records"])
        if not result["truncated"]:
            return refs
        after = result["next_after"]
