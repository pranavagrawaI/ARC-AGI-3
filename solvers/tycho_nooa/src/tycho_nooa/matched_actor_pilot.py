"""Six-packet sampling layer over the frozen Matched Actor V1 corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.matched_actor_corpus import (
    ActorCognitiveStratum,
    load_matched_actor_corpus,
)
from tycho_nooa.parity_manifest import FileCommitment

PilotArm = Literal["native", "codeact"]


class PilotPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    stratum: ActorCognitiveStratum
    selection_role: str
    arm_order: tuple[PilotArm, PilotArm]


class MatchedActorPilotSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    pilot_id: str
    source_corpus: FileCommitment
    source_corpus_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selection_rule: str
    packets: tuple[PilotPacket, ...]
    sample_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def balanced_and_counterbalanced(self) -> MatchedActorPilotSample:
        _validate_packet_count(self.packets)
        _validate_strata(self.packets)
        _validate_schedule(self.packets)
        _validate_unique_ids(self.packets)
        if _sample_sha256(self.packets) != self.sample_sha256:
            raise ValueError("matched Actor pilot sample commitment mismatch")
        return self


def _validate_packet_count(packets: tuple[PilotPacket, ...]) -> None:
    if len(packets) != 6:
        raise ValueError("matched Actor pilot requires exactly six packets")


def _validate_strata(packets: tuple[PilotPacket, ...]) -> None:
    counts = {
        stratum: sum(packet.stratum == stratum for packet in packets)
        for stratum in (
            "computation_unnecessary",
            "computation_useful",
            "composition",
        )
    }
    if set(counts.values()) != {2}:
        raise ValueError("matched Actor pilot requires two packets per stratum")


def _validate_schedule(packets: tuple[PilotPacket, ...]) -> None:
    orders = [packet.arm_order for packet in packets]
    if orders != [("native", "codeact"), ("codeact", "native")] * 3:
        raise ValueError("matched Actor pilot arm order is not counterbalanced")


def _validate_unique_ids(packets: tuple[PilotPacket, ...]) -> None:
    if len({packet.packet_id for packet in packets}) != 6:
        raise ValueError("matched Actor pilot packet ids must be unique")


def load_matched_actor_pilot_sample(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotSample, tuple[CognitiveParityPacket, ...]]:
    sample = MatchedActorPilotSample.model_validate_json(path.read_text())
    source_path = _committed_path(sample.source_corpus, repository_root)
    source, packets = load_matched_actor_corpus(
        source_path, repository_root=repository_root
    )
    if source.corpus_sha256 != sample.source_corpus_sha256:
        raise ValueError("pilot source corpus semantic commitment mismatch")
    by_id = {packet.task.packet_id: packet for packet in packets}
    selected = tuple(by_id[record.packet_id] for record in sample.packets)
    for record, packet in zip(sample.packets, selected, strict=True):
        source_record = next(
            item for item in source.records if item.packet_id == packet.task.packet_id
        )
        if source_record.stratum != record.stratum:
            raise ValueError(f"pilot stratum drift: {record.packet_id}")
    return sample, selected


def _sample_sha256(packets: tuple[PilotPacket, ...]) -> str:
    value = [packet.model_dump(mode="json") for packet in packets]
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _committed_path(commitment: FileCommitment, root: Path) -> Path:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("pilot source corpus is unavailable")
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != commitment.sha256:
        raise ValueError("pilot source corpus byte commitment mismatch")
    return path
