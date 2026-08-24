"""Provider-native syntax over authority-free deterministic computation."""

from __future__ import annotations

from typing import Any

from nooa.unifiedllm import Tool
from pydantic import BaseModel, ConfigDict, ValidationError

from tycho_nooa.computational_evidence import RLEDecodeError, decode_rle_row


class NativeRLEDecodeArguments(BaseModel):
    """The entire model-facing input to the Native decoder."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    row: str


def rle_decode_row_adapter(*, row: str) -> list[int]:
    """Decode one supplied RLE value without retrieving or selecting evidence."""
    return list(decode_rle_row(row))


def native_rle_decode_tool() -> Tool:
    """Expose the shared decoder through provider-native tool syntax."""
    return Tool(
        name="rle_decode_row",
        description=(
            "Decode one supplied comma-separated '<hex count>:<hex cell>' RLE row. "
            "This pure tool cannot retrieve evidence or access solver state."
        ),
        callable=rle_decode_row_adapter,
        parameters_model=NativeRLEDecodeArguments,
    )


def dispatch_native_rle_decode(arguments: dict[str, Any]) -> list[int]:
    """Validate provider arguments before entering the shared pure decoder."""
    try:
        parsed = NativeRLEDecodeArguments.model_validate(arguments)
    except ValidationError as error:
        raise RLEDecodeError(
            "invalid_encoding", "Native decoder requires exactly one text row"
        ) from error
    return rle_decode_row_adapter(row=parsed.row)
