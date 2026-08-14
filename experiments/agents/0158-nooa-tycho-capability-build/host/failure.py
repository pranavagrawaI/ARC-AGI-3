"""Atomic, checksum-verifiable consequential failure packets."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from .canonical import canonical_json, content_sha256, write_atomic_json
from .evidence import diff_text

_PALETTE = (
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (57, 204, 204), (61, 153, 112),
    (255, 255, 255), (130, 130, 130), (110, 60, 170), (180, 180, 30),
)


def _grid(record: dict[str, Any]) -> list[list[int]]:
    grid = record.get("grid")
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) and row for row in grid):
        raise ValueError("failure packet frame requires a public grid")
    return [[int(value) for value in row] for row in grid]


def _write_frame(root: Path, stem: str, record: dict[str, Any]) -> None:
    grid = _grid(record)
    write_atomic_json(root / f"{stem}.json", record)
    (root / f"{stem}.txt").write_text(
        "\n".join(" ".join(str(value) for value in row) for row in grid) + "\n",
        encoding="utf-8",
    )
    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width, height))
    image.putdata([_PALETTE[value] for row in grid for value in row])
    image.resize((width * 4, height * 4), Image.Resampling.NEAREST).save(root / f"{stem}.png")


def _sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class FailurePacketWriter:
    """Build the exact review artifact in a temporary directory, then rename it."""

    def __init__(self, packets_root: Path) -> None:
        self.packets_root = packets_root
        self.packets_root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        grounding_claim: dict[str, Any] | None,
        probe_commitment: dict[str, Any] | None,
        verifier_verdict: dict[str, Any] | None,
        prompt: dict[str, Any],
        callable_surface: dict[str, Any],
        state_before: dict[str, Any],
        state_after: dict[str, Any],
        trace: Iterable[dict[str, Any]],
        promotion_history: Iterable[dict[str, Any]],
        causal_record: dict[str, Any],
    ) -> Path:
        required_causal = {
            "observation", "proximate_failure", "live_causal_hypotheses",
            "discriminating_probe", "conclusion", "system_decision", "failure_class",
        }
        missing = required_causal - set(causal_record)
        if missing:
            raise ValueError(f"causal record missing fields: {sorted(missing)}")
        trace_records = list(trace)
        promotion_records = list(promotion_history)
        packet_basis = {
            "before": before,
            "after": after,
            "claim": grounding_claim,
            "commitment": probe_commitment,
            "verdict": verifier_verdict,
            "prompt": prompt,
            "callable_surface": callable_surface,
            "state_before": state_before,
            "state_after": state_after,
            "trace": trace_records,
            "promotion_history": promotion_records,
            "causal_record": causal_record,
        }
        packet_id = "packet-" + content_sha256(packet_basis)
        destination = self.packets_root / packet_id
        if destination.exists():
            raise FileExistsError(f"failure packet already exists: {packet_id}")
        temporary = self.packets_root / f".{packet_id}.tmp-{os.getpid()}"
        temporary.mkdir()
        try:
            _write_frame(temporary, "before", before)
            _write_frame(temporary, "after", after)
            (temporary / "diff.txt").write_text(
                diff_text(_grid(before), _grid(after)) + "\n", encoding="utf-8"
            )
            for name, value in (
                ("grounding-claim.json", grounding_claim),
                ("probe-commitment.json", probe_commitment),
                ("verifier-verdict.json", verifier_verdict),
                ("prompt.json", prompt),
                ("callable-surface.json", callable_surface),
                ("state-before.json", state_before),
                ("state-after.json", state_after),
                ("causal-files.json", causal_record),
            ):
                write_atomic_json(temporary / name, value)
            self._write_jsonl(temporary / "trace.jsonl", trace_records)
            self._write_jsonl(temporary / "promotion-history.jsonl", promotion_records)
            write_atomic_json(
                temporary / "review.yaml",
                {"schema": "0158-review-v1", "status": "pending"},
            )
            immutable = sorted(
                path for path in temporary.iterdir()
                if path.is_file() and path.name not in {"review.yaml", "SHA256SUMS"}
            )
            (temporary / "SHA256SUMS").write_text(
                "".join(f"{_sha_file(path)}  {path.name}\n" for path in immutable),
                encoding="utf-8",
            )
            for path in temporary.iterdir():
                if path.is_file():
                    descriptor = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            directory_descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            os.replace(temporary, destination)
            descriptor = os.open(self.packets_root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        verify_packet(destination)
        return destination

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def verify_packet(packet: Path) -> dict[str, str]:
    checksums = packet / "SHA256SUMS"
    if not checksums.is_file():
        raise ValueError("failure packet has no SHA256SUMS")
    verified: dict[str, str] = {}
    for line_number, line in enumerate(checksums.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or Path(name).name != name:
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        path = packet / name
        if not path.is_file() or _sha_file(path) != digest:
            raise ValueError(f"failure packet checksum mismatch: {name}")
        verified[name] = digest
    if "review.yaml" in verified:
        raise ValueError("mutable review.yaml must not be checksummed")
    return verified
