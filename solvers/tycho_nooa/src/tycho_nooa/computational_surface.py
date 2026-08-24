"""Version binding for the shared Native and sterile computational surface."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tycho_nooa.computational_evidence import (
    MAX_DECODED_ROW_CELLS,
    RLE_HELPER_SURFACE,
    decode_rle_row,
)
from tycho_nooa.native_computation import NativeRLEDecodeArguments


class ComputationalSurfaceBinding(BaseModel):
    """Hashes that make computational parity replayable rather than nominal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal[1] = 1
    authority: Literal["pure_value_computation"] = "pure_value_computation"
    max_decoded_cells: int
    decoder_implementation_sha256: str
    native_adapter_schema_sha256: str
    sterile_helper_surface_sha256: str
    sterile_worker_sha256: str


def current_computational_surface_binding() -> ComputationalSurfaceBinding:
    """Bind implementation, both public contracts, and the sterile loader."""
    return ComputationalSurfaceBinding(
        max_decoded_cells=MAX_DECODED_ROW_CELLS,
        decoder_implementation_sha256=_sha256_text(inspect.getsource(decode_rle_row)),
        native_adapter_schema_sha256=_sha256_json(
            NativeRLEDecodeArguments.model_json_schema()
        ),
        sterile_helper_surface_sha256=_sha256_json(RLE_HELPER_SURFACE),
        sterile_worker_sha256=_sha256_bytes(
            Path(__file__).with_name("_computational_worker.py").read_bytes()
        ),
    )


def write_computational_surface_binding(path: Path) -> None:
    """Persist the canonical JSON commitment used by parity checks."""
    value = current_computational_surface_binding().model_dump(mode="json")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode())


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()
