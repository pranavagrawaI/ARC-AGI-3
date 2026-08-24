from __future__ import annotations

from pathlib import Path

import pytest

from tycho_nooa.compatibility_return import CompositionReturn
from tycho_nooa.computational_evidence import (
    EvidencePageView,
    decode_rle_row,
    describe_evidence_page,
)
from tycho_nooa.computational_execution import ComputationalSterileExecutor
from tycho_nooa.evidence import WorkspaceEvidenceCapability
from tycho_nooa.sterile_execution import ExecutionManifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = REPOSITORY_ROOT / (
    "runs/tycho-nooa-symmetry-uptake-ar25-20260820T101248Z/workspace/ar25"
)


def test_evidence_page_is_self_describing_and_mapping_compatible() -> None:
    value = describe_evidence_page(
        {
            "records": [{"ref": "L0:T1", "prev": ["1:b"], "next": ["1:5"]}],
            "grid_encoding": "rle",
        }
    )
    page = EvidencePageView(value)

    assert page.records[0].ref == "L0:T1"
    assert page["records"][0]["ref"] == "L0:T1"
    assert page.records[0].prev_rows == ["1:b"]
    assert "decode with rle.decode_row" in repr(page)


def test_rle_decoder_is_exact_and_rejects_bad_runs() -> None:
    assert decode_rle_row("3f:9,1:b")[63] == 11
    assert decode_rle_row("3f:9,1:5")[63] == 5
    with pytest.raises(ValueError, match="invalid RLE"):
        decode_rle_row("not-a-run")


def test_candidate_golden_path_has_no_undocumented_lookup() -> None:
    capability = WorkspaceEvidenceCapability.from_root(CHECKPOINT)
    raw = capability.query(
        {
            "kind": "transitions",
            "level": 0,
            "limit": 1,
            "projection": "full",
            "grid_encoding": "rle",
        }
    )
    page = EvidencePageView(describe_evidence_page(raw))
    record = page.records[0]
    before = decode_rle_row(record.prev_rows[0])[63]
    after = decode_rle_row(record.next_rows[0])[63]
    result = CompositionReturn(
        conclusion=f"{before}->{after}", evidence_ref=record.ref
    )

    assert result == CompositionReturn(conclusion="11->5", evidence_ref="L0:T1")


async def test_candidate_golden_path_crosses_the_sterile_membrane() -> None:
    capability = WorkspaceEvidenceCapability.from_root(CHECKPOINT)

    def query(arguments):
        return describe_evidence_page(capability.query(arguments))

    manifest = ExecutionManifest(
        invocation_id="computational-golden-path",
        role="actor",
        capabilities=("evidence.query",),
        capability_contracts={
            "evidence.query": {
                "kind": "transitions",
                "level": 0,
                "limit": 1,
                "projection": "full",
                "grid_encoding": "rle",
                "max_query_attempts": 1,
                "max_successful_disclosures": 1,
            }
        },
    )
    code = (
        "page = evidence.query(kind='transitions', level=0, limit=1, "
        "projection='full', grid_encoding='rle')\n"
        "record = page.records[0]\n"
        "before = rle.decode_row(record.prev_rows[0])[63]\n"
        "after = rle.decode_row(record.next_rows[0])[63]\n"
        "print(record.ref, f'{before}->{after}')"
    )
    async with ComputationalSterileExecutor(
        manifest,
        return_type=CompositionReturn,
        capability_handlers={"evidence.query": query},
    ) as executor:
        computed = await executor.run_cell(code)
        returned = await executor.run_cell(
            "return_result(conclusion=f'{before}->{after}', evidence_ref=record.ref)"
        )

    assert computed.status == "complete"
    assert computed.stdout == "L0:T1 11->5\n"
    assert returned.return_value == CompositionReturn(
        conclusion="11->5", evidence_ref="L0:T1"
    )
