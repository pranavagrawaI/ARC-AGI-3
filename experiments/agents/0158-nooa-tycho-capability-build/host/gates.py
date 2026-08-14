"""Level-local G0-G2 qualification state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .protocol import GroundingClaim, GroundingVerdict

GateName = Literal["EMPTY", "G0", "G1", "G2", "INVALIDATED", "STOPPED"]


@dataclass
class GateTracker:
    level: int
    gate: GateName = "EMPTY"
    supported_informative_effects: int = 0
    lineage_id: str | None = None
    tip_claim_id: str | None = None
    alternative_used: bool = False
    transition_capability_eligible: bool = False
    g2_cycles_completed: int = 0
    paused_for_review: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    def accept_claim(self, claim: GroundingClaim, *, observation_exists: bool, evidence_refs_exist: bool) -> None:
        if self.paused_for_review or self.gate in {"INVALIDATED", "STOPPED", "G2"}:
            raise RuntimeError("gate state does not accept another claim")
        if not observation_exists or not evidence_refs_exist:
            raise ValueError("claim references unavailable evidence")
        if self.tip_claim_id is None:
            if claim.parent_claim_id is not None:
                raise ValueError("initial G0 claim cannot have a parent")
            self.lineage_id = claim.lineage_id
            self.gate = "G0"
        else:
            if claim.parent_claim_id != self.tip_claim_id or claim.lineage_id != self.lineage_id:
                raise ValueError("successor claim is not linked to the accepted lineage tip")
        self.tip_claim_id = claim.claim_id
        self.history.append({"event": "claim_accepted", "claim_id": claim.claim_id, "gate": self.gate})

    def apply_verdict(self, verdict: GroundingVerdict) -> GateName:
        if verdict.claim_id != self.tip_claim_id:
            raise ValueError("verdict does not address the accepted lineage tip")
        if self.paused_for_review:
            raise RuntimeError("gate state is paused")
        self.history.append({"event": "verdict", **verdict.to_record()})
        if verdict.status == "contradicted":
            self.gate = "INVALIDATED"
            self.paused_for_review = True
            return self.gate
        if verdict.status == "supported" and not verdict.promotable:
            return self.gate
        if verdict.status == "inconclusive":
            if not self.alternative_used:
                self.alternative_used = True
                return self.gate
            self.gate = "STOPPED"
            self.paused_for_review = True
            return self.gate
        if verdict.promotion_credit < 1:
            raise ValueError("promotable verdict must carry positive promotion credit")
        self.supported_informative_effects += verdict.promotion_credit
        if self.supported_informative_effects < 3:
            self.gate = "G1"
        else:
            self.gate = "G2"
            self.transition_capability_eligible = True
            self.g2_cycles_completed += 1
            self.paused_for_review = True
        return self.gate

    def reset_for_level(self, level: int) -> None:
        if level == self.level:
            raise ValueError("level reset requires a new level")
        self.level = level
        self.gate = "EMPTY"
        self.supported_informative_effects = 0
        self.lineage_id = None
        self.tip_claim_id = None
        self.alternative_used = False
        self.transition_capability_eligible = False
        self.g2_cycles_completed = 0
        self.paused_for_review = False
        self.history.append({"event": "level_reset", "level": level})

    def begin_new_grounding_cycle(self, *, reason: str) -> None:
        """Reset only active grounding state after an audited autonomous cycle stop."""
        if self.gate not in {"INVALIDATED", "STOPPED", "G2"} or not self.paused_for_review:
            raise RuntimeError("new grounding cycle requires a completed or invalidated cycle")
        previous_gate = self.gate
        self.gate = "EMPTY"
        self.supported_informative_effects = 0
        self.lineage_id = None
        self.tip_claim_id = None
        self.alternative_used = False
        self.transition_capability_eligible = False
        self.paused_for_review = False
        self.history.append(
            {"event": "grounding_cycle_reset", "reason": reason, "previous_gate": previous_gate}
        )

    def stop_for_failure(self, *, failure_class: str) -> None:
        if self.paused_for_review:
            raise RuntimeError("gate state is already paused")
        self.gate = "STOPPED"
        self.paused_for_review = True
        self.history.append({"event": "failure_stop", "failure_class": failure_class})

    def to_record(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "gate": self.gate,
            "supported_informative_effects": self.supported_informative_effects,
            "lineage_id": self.lineage_id,
            "tip_claim_id": self.tip_claim_id,
            "alternative_used": self.alternative_used,
            "transition_capability_eligible": self.transition_capability_eligible,
            "g2_cycles_completed": self.g2_cycles_completed,
            "paused_for_review": self.paused_for_review,
            "history": list(self.history),
        }
