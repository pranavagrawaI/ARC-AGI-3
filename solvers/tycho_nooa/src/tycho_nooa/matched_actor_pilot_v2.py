"""Five-packet matched sample after separating the remote-history stress case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.matched_actor_pilot import PilotPacket
from tycho_nooa.parity_manifest import FileCommitment
from tycho_nooa.repaired_checkpoint import load_repaired_pilot_sample


class MatchedActorPilotSampleV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    pilot_id: Literal["matched-actor-v2-five-packet-pilot"] = "matched-actor-v2-five-packet-pilot"
    base_sample: FileCommitment
    checkpoint_commitments: FileCommitment
    excluded_packet: Literal["remote-failure-composition"]
    selection_rule: str
    packets: tuple[PilotPacket, ...]
    sample_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def five_matched_packets_are_frozen(self) -> MatchedActorPilotSampleV2:
        if len(self.packets) != 5:
            raise ValueError("Matched Actor V2 requires exactly five packets")
        if len({packet.packet_id for packet in self.packets}) != 5:
            raise ValueError("Matched Actor V2 packet ids must be unique")
        if [packet.arm_order for packet in self.packets] != [
            ("native", "codeact"),
            ("codeact", "native"),
            ("native", "codeact"),
            ("codeact", "native"),
            ("native", "codeact"),
        ]:
            raise ValueError("Matched Actor V2 arm order drifted")
        if _sample_sha256(self.packets) != self.sample_sha256:
            raise ValueError("Matched Actor V2 sample commitment mismatch")
        return self


def load_matched_actor_pilot_sample_v2(
    path: Path, *, repository_root: Path
) -> tuple[MatchedActorPilotSampleV2, tuple[CognitiveParityPacket, ...]]:
    sample = MatchedActorPilotSampleV2.model_validate_json(path.read_text())
    _validate_file(sample.base_sample, repository_root)
    _validate_file(sample.checkpoint_commitments, repository_root)
    _, source_packets = load_repaired_pilot_sample(
        repository_root / sample.base_sample.path,
        repository_root / sample.checkpoint_commitments.path,
        repository_root=repository_root,
    )
    by_id = {packet.task.packet_id: packet for packet in source_packets}
    selected = tuple(by_id[record.packet_id] for record in sample.packets)
    if sample.excluded_packet in {record.packet_id for record in sample.packets}:
        raise ValueError("remote stress packet leaked into matched sample")
    return sample, selected


def sample_sha256(packets: tuple[PilotPacket, ...]) -> str:
    return _sample_sha256(packets)


def _sample_sha256(packets: tuple[PilotPacket, ...]) -> str:
    encoded = json.dumps(
        [packet.model_dump(mode="json") for packet in packets],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_file(commitment: FileCommitment, root: Path) -> None:
    base = root.resolve()
    path = (base / commitment.path).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Matched Actor V2 input unavailable: {commitment.path}")
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != commitment.sha256:
        raise ValueError(f"Matched Actor V2 input drift: {commitment.path}")
