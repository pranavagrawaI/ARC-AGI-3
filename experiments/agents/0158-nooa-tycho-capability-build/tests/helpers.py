from __future__ import annotations

from typing import Any

from host.protocol import (
    GroundingClaim,
    grounding_claim_from_draft,
    probe_commitment_from_draft,
)


def mask(row: int, col: int, height: int = 1, width: int = 1) -> dict[str, Any]:
    return {
        "bbox": {"row": row, "col": col, "height": height, "width": width},
        "rows": ["#" * width for _ in range(height)],
    }


def claim(
    observation_id: str = "frame-a",
    *,
    primary: dict[str, Any] | None = None,
    parent: GroundingClaim | None = None,
) -> GroundingClaim:
    return grounding_claim_from_draft(
        {
            "observation_id": observation_id,
            "primary_region": primary or mask(1, 1),
            "alternatives": [mask(0, 0)],
            "evidence_refs": [observation_id],
            "parent_claim_id": parent.claim_id if parent else None,
        },
        parent=parent,
    )


def commitment_for(
    claim_value: GroundingClaim,
    effect: dict[str, Any],
    *,
    action: int = 1,
    alternative_action: int = 2,
    alternative_effect: dict[str, Any] | None = None,
):
    if alternative_effect is None:
        alternative_effect = (
            {"kind": "translate", "source": "primary_region", "d_row": 1, "d_col": 0}
            if effect.get("kind") == "unchanged"
            else {"kind": "unchanged", "source": "primary_region"}
        )
    return probe_commitment_from_draft(
        {
            "claim_id": claim_value.claim_id,
            "action": action,
            "data": None,
            "predicted_effect": effect,
            "alternative_predictions": [
                {
                    "action": alternative_action,
                    "data": None,
                    "predicted_effect": alternative_effect,
                }
            ],
        },
        claim=claim_value,
    )
