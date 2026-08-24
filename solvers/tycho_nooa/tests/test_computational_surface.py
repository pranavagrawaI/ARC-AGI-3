from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tycho_nooa.compatibility_return import RepresentationReturn
from tycho_nooa.computational_evidence import (
    MAX_DECODED_ROW_CELLS,
    RLEDecodeError,
    RLEHelpers,
)
from tycho_nooa.computational_execution import ComputationalSterileExecutor
from tycho_nooa.computational_surface import (
    current_computational_surface_binding,
)
from tycho_nooa.native_computation import (
    NativeRLEDecodeArguments,
    dispatch_native_rle_decode,
    native_rle_decode_tool,
    rle_decode_row_adapter,
)
from tycho_nooa.sterile_execution import ExecutionManifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SURFACE_BINDING = REPOSITORY_ROOT / (
    "analysis/computational-surface-v1/manifest.json"
)


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        ("3f:9,1:b", [9] * 63 + [11]),
        ("3f:9,1:5", [9] * 63 + [5]),
        ("1:0", [0]),
        ("1:0,1:f,1:0,1:f", [0, 15, 0, 15]),
        ("40:9", [9] * 64),
        ("1000:a", [10] * MAX_DECODED_ROW_CELLS),
    ],
)
def test_native_and_codeact_helpers_share_valid_semantics(
    encoded: str, expected: list[int]
) -> None:
    assert rle_decode_row_adapter(row=encoded) == expected
    assert list(RLEHelpers().decode_row(encoded)) == expected


@pytest.mark.parametrize(
    ("encoded", "code"),
    [
        ("", "empty_encoding"),
        ("not-a-run", "invalid_encoding"),
        ("1:0:2", "invalid_encoding"),
        ("-1:a", "invalid_run_length"),
        ("0:a", "invalid_run_length"),
        ("1001:a", "decoded_size_exceeded"),
    ],
)
def test_native_and_codeact_helpers_share_failure_semantics(
    encoded: str, code: str
) -> None:
    for decode in (
        lambda: rle_decode_row_adapter(row=encoded),
        lambda: RLEHelpers().decode_row(encoded),
    ):
        with pytest.raises(RLEDecodeError) as caught:
            decode()
        assert caught.value.code == code


def test_native_adapter_rejects_non_text_and_exposes_only_the_row() -> None:
    for arguments in (
        {"row": [1, 5]},
        {"row": "1:5", "evidence_ref": "L0:T1"},
    ):
        with pytest.raises(RLEDecodeError) as caught:
            dispatch_native_rle_decode(arguments)
        assert caught.value.code == "invalid_encoding"

    tool = native_rle_decode_tool()
    assert tool.name == "rle_decode_row"
    assert set(NativeRLEDecodeArguments.model_fields) == {"row"}
    assert tool.parameters_model is NativeRLEDecodeArguments


def test_both_surfaces_receive_computation_without_authority() -> None:
    closure = inspect.getclosurevars(rle_decode_row_adapter)
    assert closure.nonlocals == {}
    assert set(closure.globals) == {"decode_rle_row"}

    manifest = ExecutionManifest(
        invocation_id="computational-authority-proof",
        role="actor",
    )
    assert manifest.capabilities == ()
    assert manifest.input_values == {}


async def test_sterile_helper_namespace_has_no_solver_authority() -> None:
    manifest = ExecutionManifest(
        invocation_id="computational-namespace-proof",
        role="actor",
    )
    async with ComputationalSterileExecutor(
        manifest,
        return_type=RepresentationReturn,
    ) as executor:
        result = await executor.run_cell(
            "print([name for name in ('evidence', 'workspace', 'state', 'self') "
            "if name in globals()])"
        )

    assert result.stdout == "[]\n"


async def test_native_and_sterile_surfaces_have_canonical_json_value_parity() -> None:
    encoded = "3f:9,1:5"
    native_value = rle_decode_row_adapter(row=encoded)
    manifest = ExecutionManifest(
        invocation_id="computational-value-parity",
        role="actor",
        input_values={"encoded": encoded},
    )

    async with ComputationalSterileExecutor(
        manifest,
        return_type=RepresentationReturn,
    ) as executor:
        result = await executor.run_cell("print(list(rle.decode_row(encoded)))")

    codeact_value = json.loads(result.stdout)
    assert json.dumps(native_value, separators=(",", ":")) == json.dumps(
        codeact_value, separators=(",", ":")
    )


@pytest.mark.parametrize(
    ("encoded", "code"),
    [
        ("", "empty_encoding"),
        ([1, 5], "invalid_encoding"),
        ("0:a", "invalid_run_length"),
        ("1:0:2", "invalid_encoding"),
        ("1001:a", "decoded_size_exceeded"),
    ],
)
async def test_native_and_sterile_surfaces_reject_with_same_classification(
    encoded: object, code: str
) -> None:
    with pytest.raises(RLEDecodeError) as native_error:
        rle_decode_row_adapter(row=encoded)  # type: ignore[arg-type]

    manifest = ExecutionManifest(
        invocation_id=f"computational-failure-{code}-{type(encoded).__name__}",
        role="actor",
        input_values={"encoded": encoded},  # type: ignore[dict-item]
    )
    async with ComputationalSterileExecutor(
        manifest,
        return_type=RepresentationReturn,
    ) as executor:
        sterile_result = await executor.run_cell("rle.decode_row(encoded)")

    assert native_error.value.code == code
    assert sterile_result.status == "error"
    assert sterile_result.exception is not None
    assert sterile_result.exception.type == "RLEDecodeError"
    assert sterile_result.exception.message.startswith(f"{code}:")


def test_native_adapter_replays_the_golden_calculation_without_authority() -> None:
    before = rle_decode_row_adapter(row="3f:9,1:b")[63]
    after = rle_decode_row_adapter(row="3f:9,1:5")[63]

    assert f"{before}->{after}" == "11->5"


def test_versioned_binding_matches_all_live_surfaces() -> None:
    frozen = json.loads(SURFACE_BINDING.read_text())

    assert current_computational_surface_binding().model_dump(mode="json") == frozen
