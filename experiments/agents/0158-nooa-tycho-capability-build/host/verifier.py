"""Non-oracular exact verifier for primary-region effects."""

from __future__ import annotations

from collections.abc import Iterable

from .protocol import (
    GroundingClaim,
    GroundingVerdict,
    ProbeCommitment,
    Replace,
    Translate,
    Unchanged,
)


def account_visible_change(
    claim: GroundingClaim,
    before_grid: Iterable[Iterable[int]],
    after_grid: Iterable[Iterable[int]],
    *,
    transition_id: str,
) -> dict[str, object]:
    """Report neutral change coverage without choosing an object or causal role."""
    before = _grid(before_grid)
    after = _grid(after_grid)
    if not before or not after or len(before) != len(after) or len(before[0]) != len(after[0]):
        raise ValueError("change accounting requires equal nonempty frame shapes")
    changed = {
        (row, col)
        for row in range(len(before))
        for col in range(len(before[0]))
        if before[row][col] != after[row][col]
    }
    primary = set(claim.primary_region.cells())
    inside = changed & primary
    outside = changed - primary
    return {
        "schema": "0158-change-accounting-v1",
        "transition_id": transition_id,
        "changed_cells_total": len(changed),
        "changed_cells_inside_primary": len(inside),
        "changed_cells_outside_primary": len(outside),
        "primary_region_changed": bool(inside),
        "outside_primary_changed": bool(outside),
    }


def _grid(value: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def _inside(grid: tuple[tuple[int, ...], ...], row: int, col: int) -> bool:
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def verify_effect(
    claim: GroundingClaim,
    commitment: ProbeCommitment,
    before_grid: Iterable[Iterable[int]],
    after_grid: Iterable[Iterable[int]],
) -> GroundingVerdict:
    if commitment.claim_id != claim.claim_id:
        raise ValueError("commitment and claim identity mismatch")
    before = _grid(before_grid)
    after = _grid(after_grid)
    if not before or not after or len(before) != len(after) or len(before[0]) != len(after[0]):
        return _verdict(claim, commitment, "inconclusive", "FRAME_SHAPE_CHANGED", False)
    primary = set(claim.primary_region.cells())
    if any(not _inside(before, row, col) for row, col in primary):
        return _verdict(claim, commitment, "contradicted", "PRIMARY_REGION_OUT_OF_BOUNDS", False)
    effect = commitment.predicted_effect
    if isinstance(effect, Translate):
        destinations = {(row + effect.d_row, col + effect.d_col) for row, col in primary}
        if any(not _inside(after, row, col) for row, col in destinations):
            return _verdict(claim, commitment, "contradicted", "TRANSLATION_OUT_OF_BOUNDS", False)
        exact = all(
            after[row + effect.d_row][col + effect.d_col] == before[row][col]
            for row, col in primary
        )
        if not exact:
            return _verdict(claim, commitment, "contradicted", "PRIMARY_TRANSLATION_MISMATCH", False)
        observable = any(
            before[row][col] != after[row][col] for row, col in primary
        ) or any(before[row][col] != after[row][col] for row, col in destinations)
        if not observable:
            return _verdict(claim, commitment, "inconclusive", "NO_OBSERVABLE_PRIMARY_CHANGE", False)
        return _verdict(claim, commitment, "supported", "PRIMARY_TRANSLATION_EXACT", False)
    if isinstance(effect, Replace):
        if not claim.primary_region.bbox.overlaps(effect.expected_after.bbox):
            return _verdict(claim, commitment, "contradicted", "REPLACEMENT_NOT_LOCAL_TO_PRIMARY", False)
        if any(not _inside(after, assertion.row, assertion.col) for assertion in effect.assertions):
            return _verdict(claim, commitment, "contradicted", "REPLACEMENT_OUT_OF_BOUNDS", False)
        if any(after[item.row][item.col] != item.value for item in effect.assertions):
            return _verdict(claim, commitment, "contradicted", "PRIMARY_REPLACEMENT_MISMATCH", False)
        anchored = [item for item in effect.assertions if (item.row, item.col) in primary]
        if not anchored or not any(before[item.row][item.col] != item.value for item in anchored):
            return _verdict(claim, commitment, "inconclusive", "NO_ASSERTED_PRIMARY_CHANGE", False)
        return _verdict(claim, commitment, "supported", "PRIMARY_REPLACEMENT_EXACT", False)
    if isinstance(effect, Unchanged):
        if all(after[row][col] == before[row][col] for row, col in primary):
            return _verdict(claim, commitment, "supported", "PRIMARY_UNCHANGED_EXACT", False)
        return _verdict(claim, commitment, "contradicted", "PRIMARY_UNCHANGED_MISMATCH", False)
    raise TypeError(f"unsupported effect: {type(effect).__name__}")


def _verdict(
    claim: GroundingClaim,
    commitment: ProbeCommitment,
    status: str,
    check: str,
    promotable: bool,
) -> GroundingVerdict:
    return GroundingVerdict(
        claim_id=claim.claim_id,
        commitment_id=commitment.commitment_id,
        status=status,  # type: ignore[arg-type]
        checks=(check,),
        action_specificity=(
            "contradicted" if status == "contradicted" else "inconclusive"
        ),
        contrast_commitment_ids=(),
        promotion_credit=0,
        promotable=promotable,
    )
