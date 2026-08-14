"""Strict typed artifacts for claim, commitment, and verdict."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .canonical import content_sha256


class ProtocolError(ValueError):
    """A model artifact or host protocol record is malformed."""


def _exact_keys(
    record: dict[str, Any], required: set[str], optional: set[str] | None = None
) -> None:
    if not isinstance(record, dict):
        raise ProtocolError("record must be an object")
    missing = required - set(record)
    extra = set(record) - required - (optional or set())
    if missing or extra:
        raise ProtocolError(f"record keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ProtocolError(f"{label} must be an integer")
    return value


@dataclass(frozen=True)
class BoundingBox:
    row: int
    col: int
    height: int
    width: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> BoundingBox:
        _exact_keys(record, {"row", "col", "height", "width"})
        box = cls(*(_strict_int(record[key], f"bbox.{key}") for key in ("row", "col", "height", "width")))
        if box.row < 0 or box.col < 0 or box.height < 1 or box.width < 1:
            raise ProtocolError("bbox origin must be non-negative and dimensions positive")
        return box

    @property
    def area(self) -> int:
        return self.height * self.width

    def to_record(self) -> dict[str, int]:
        return {"row": self.row, "col": self.col, "height": self.height, "width": self.width}

    def overlaps(self, other: BoundingBox) -> bool:
        return not (
            self.row + self.height <= other.row
            or other.row + other.height <= self.row
            or self.col + self.width <= other.col
            or other.col + other.width <= self.col
        )


@dataclass(frozen=True)
class CellMask:
    bbox: BoundingBox
    rows: tuple[str, ...]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CellMask:
        _exact_keys(record, {"bbox", "rows"})
        bbox = BoundingBox.from_record(record["bbox"])
        rows = record["rows"]
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ProtocolError("mask rows must be a list of strings")
        if len(rows) != bbox.height or any(len(row) != bbox.width for row in rows):
            raise ProtocolError("mask rows must exactly match bbox dimensions")
        if any(set(row) - {"#", "."} for row in rows):
            raise ProtocolError("mask rows may contain only # and .")
        if not any("#" in row for row in rows):
            raise ProtocolError("mask must select at least one cell")
        return cls(bbox=bbox, rows=tuple(rows))

    def to_record(self) -> dict[str, Any]:
        return {"bbox": self.bbox.to_record(), "rows": list(self.rows)}

    def cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.bbox.row + row, self.bbox.col + col)
            for row, values in enumerate(self.rows)
            for col, value in enumerate(values)
            if value == "#"
        )


@dataclass(frozen=True)
class CellAssertion:
    row: int
    col: int
    value: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CellAssertion:
        _exact_keys(record, {"row", "col", "value"})
        assertion = cls(
            row=_strict_int(record["row"], "assertion.row"),
            col=_strict_int(record["col"], "assertion.col"),
            value=_strict_int(record["value"], "assertion.value"),
        )
        if assertion.row < 0 or assertion.col < 0 or not 0 <= assertion.value <= 15:
            raise ProtocolError("assertion coordinates must be non-negative and value in 0..15")
        return assertion

    def to_record(self) -> dict[str, int]:
        return {"row": self.row, "col": self.col, "value": self.value}


@dataclass(frozen=True)
class Translate:
    source: Literal["primary_region"]
    d_row: int
    d_col: int
    kind: Literal["translate"] = "translate"

    def to_record(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source, "d_row": self.d_row, "d_col": self.d_col}


@dataclass(frozen=True)
class Unchanged:
    source: Literal["primary_region"]
    kind: Literal["unchanged"] = "unchanged"

    def to_record(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source}


@dataclass(frozen=True)
class Replace:
    source: Literal["primary_region"]
    expected_after: CellMask
    assertions: tuple[CellAssertion, ...]
    kind: Literal["replace"] = "replace"

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "expected_after": self.expected_after.to_record(),
            "assertions": [assertion.to_record() for assertion in self.assertions],
        }


type PredictedEffect = Translate | Unchanged | Replace


@dataclass(frozen=True)
class HypothesisUpdate:
    prior_probe_result_id: str
    disposition: Literal["retain", "revise", "abandon"]
    rationale: str

    def to_record(self) -> dict[str, str]:
        return {
            "prior_probe_result_id": self.prior_probe_result_id,
            "disposition": self.disposition,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ProvisionalHeuristic:
    name: str
    belief: str
    scope: str
    evidence_refs: tuple[str, ...]
    decision_role: Literal["active", "watch", "defer"]
    reconsider_when: str
    epistemic_status: Literal["provisional"] = "provisional"

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "belief": self.belief,
            "scope": self.scope,
            "evidence_refs": list(self.evidence_refs),
            "decision_role": self.decision_role,
            "reconsider_when": self.reconsider_when,
            "epistemic_status": self.epistemic_status,
        }


@dataclass(frozen=True)
class HeuristicUpdate:
    """Model-authored interpretation fields; the host owns evidence persistence."""

    name: str
    belief: str
    scope: str
    decision_role: Literal["active", "watch", "defer"]
    reconsider_when: str


def heuristic_update_from_record(record: dict[str, Any]) -> HeuristicUpdate:
    _exact_keys(record, {"name", "belief", "scope", "decision_role", "reconsider_when"})
    name = record["name"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name) is None:
        raise ProtocolError("heuristic name must be a stable lowercase identifier")
    text_limits = {"belief": 512, "scope": 256, "reconsider_when": 256}
    texts: dict[str, str] = {}
    for field, limit in text_limits.items():
        value = record[field]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ProtocolError(f"heuristic {field} must contain 1..{limit} characters")
        texts[field] = value.strip()
    role = record["decision_role"]
    if role not in {"active", "watch", "defer"}:
        raise ProtocolError("heuristic decision_role must be active, watch, or defer")
    return HeuristicUpdate(
        name=name,
        belief=texts["belief"],
        scope=texts["scope"],
        decision_role=role,
        reconsider_when=texts["reconsider_when"],
    )


def provisional_heuristic_from_record(record: dict[str, Any]) -> ProvisionalHeuristic:
    _exact_keys(
        record,
        {
            "name",
            "belief",
            "scope",
            "evidence_refs",
            "decision_role",
            "reconsider_when",
            "epistemic_status",
        },
    )
    name = record["name"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name) is None:
        raise ProtocolError("heuristic name must be a stable lowercase identifier")
    text_limits = {"belief": 512, "scope": 256, "reconsider_when": 256}
    texts: dict[str, str] = {}
    for field, limit in text_limits.items():
        value = record[field]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ProtocolError(f"heuristic {field} must contain 1..{limit} characters")
        texts[field] = value.strip()
    refs = record["evidence_refs"]
    if (
        not isinstance(refs, list)
        or not refs
        or len(refs) > 64
        or not all(isinstance(item, str) and item for item in refs)
        or len(set(refs)) != len(refs)
    ):
        raise ProtocolError("heuristic evidence_refs must contain 1..64 distinct identifiers")
    role = record["decision_role"]
    if role not in {"active", "watch", "defer"}:
        raise ProtocolError("heuristic decision_role must be active, watch, or defer")
    if record["epistemic_status"] != "provisional":
        raise ProtocolError("heuristic epistemic_status must remain provisional")
    return ProvisionalHeuristic(
        name=name,
        belief=texts["belief"],
        scope=texts["scope"],
        evidence_refs=tuple(refs),
        decision_role=role,
        reconsider_when=texts["reconsider_when"],
    )


def hypothesis_update_from_record(
    record: dict[str, Any],
    *,
    latest_probe_result_id: str,
) -> HypothesisUpdate:
    _exact_keys(record, {"prior_probe_result_id", "disposition", "rationale"})
    if record["prior_probe_result_id"] != latest_probe_result_id:
        raise ProtocolError("hypothesis update must bind the latest probe result")
    disposition = record["disposition"]
    if disposition not in {"retain", "revise", "abandon"}:
        raise ProtocolError("hypothesis disposition must be retain, revise, or abandon")
    rationale = record["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 512:
        raise ProtocolError("hypothesis update rationale must contain 1..512 characters")
    return HypothesisUpdate(
        prior_probe_result_id=latest_probe_result_id,
        disposition=disposition,
        rationale=rationale.strip(),
    )


def predicted_effect_from_record(record: dict[str, Any]) -> PredictedEffect:
    if not isinstance(record, dict):
        raise ProtocolError("predicted_effect must be an object")
    kind = record.get("kind")
    if kind == "translate":
        _exact_keys(record, {"kind", "source", "d_row", "d_col"})
        if record["source"] != "primary_region":
            raise ProtocolError("translate source must be primary_region")
        d_row = _strict_int(record["d_row"], "translate.d_row")
        d_col = _strict_int(record["d_col"], "translate.d_col")
        if d_row == 0 and d_col == 0:
            raise ProtocolError("translate must predict non-zero displacement")
        return Translate(source="primary_region", d_row=d_row, d_col=d_col)
    if kind == "unchanged":
        _exact_keys(record, {"kind", "source"})
        if record["source"] != "primary_region":
            raise ProtocolError("unchanged source must be primary_region")
        return Unchanged(source="primary_region")
    if kind == "replace":
        _exact_keys(record, {"kind", "source", "expected_after", "assertions"})
        if record["source"] != "primary_region":
            raise ProtocolError("replace source must be primary_region")
        expected = CellMask.from_record(record["expected_after"])
        if expected.bbox.area > 64:
            raise ProtocolError("replace expected_after bbox exceeds 64 cells")
        raw_assertions = record["assertions"]
        if not isinstance(raw_assertions, list) or not 1 <= len(raw_assertions) <= 16:
            raise ProtocolError("replace requires 1..16 assertions")
        assertions = tuple(CellAssertion.from_record(item) for item in raw_assertions)
        if len({(item.row, item.col) for item in assertions}) != len(assertions):
            raise ProtocolError("replace assertions must have distinct coordinates")
        expected_cells = set(expected.cells())
        if any((item.row, item.col) not in expected_cells for item in assertions):
            raise ProtocolError("replace assertions must lie inside selected expected_after cells")
        return Replace(source="primary_region", expected_after=expected, assertions=assertions)
    raise ProtocolError(f"unsupported predicted_effect kind: {kind!r}")


def predicted_effect_signature(
    effect: PredictedEffect,
    *,
    primary_region: CellMask,
) -> tuple[Any, ...]:
    """Return an absolute-position-invariant effect signature for one claimed region."""
    if isinstance(effect, Translate):
        return (effect.kind, effect.d_row, effect.d_col)
    if isinstance(effect, Unchanged):
        return (effect.kind,)
    if isinstance(effect, Replace):
        del primary_region
        assertion_values = tuple(sorted(item.value for item in effect.assertions))
        return (
            effect.kind,
            effect.expected_after.bbox.height,
            effect.expected_after.bbox.width,
            effect.expected_after.rows,
            assertion_values,
        )
    raise TypeError(f"unsupported effect: {type(effect).__name__}")


@dataclass(frozen=True)
class GroundingClaim:
    claim_id: str
    lineage_id: str
    observation_id: str
    primary_region: CellMask
    alternatives: tuple[CellMask, ...]
    evidence_refs: tuple[str, ...]
    parent_claim_id: str | None
    hypothesis_update: HypothesisUpdate | None
    heuristic_register: tuple[ProvisionalHeuristic, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "lineage_id": self.lineage_id,
            "observation_id": self.observation_id,
            "primary_region": self.primary_region.to_record(),
            "alternatives": [mask.to_record() for mask in self.alternatives],
            "evidence_refs": list(self.evidence_refs),
            "parent_claim_id": self.parent_claim_id,
            "hypothesis_update": (
                self.hypothesis_update.to_record() if self.hypothesis_update else None
            ),
            "heuristic_register": [item.to_record() for item in self.heuristic_register],
        }


def grounding_claim_from_draft(
    record: dict[str, Any],
    *,
    parent: GroundingClaim | None = None,
    latest_probe_result: dict[str, Any] | None = None,
) -> GroundingClaim:
    _exact_keys(
        record,
        {"observation_id", "primary_region", "alternatives", "evidence_refs", "parent_claim_id"},
        {"hypothesis_update", "heuristic_updates"},
    )
    observation_id = record["observation_id"]
    if not isinstance(observation_id, str) or not observation_id:
        raise ProtocolError("observation_id must be a non-empty string")
    primary = CellMask.from_record(record["primary_region"])
    raw_alternatives = record["alternatives"]
    if not isinstance(raw_alternatives, list) or len(raw_alternatives) > 3:
        raise ProtocolError("alternatives must contain 0..3 masks")
    alternatives = tuple(CellMask.from_record(item) for item in raw_alternatives)
    serialized_masks = [primary.to_record(), *(mask.to_record() for mask in alternatives)]
    if len({content_sha256(mask) for mask in serialized_masks}) != len(serialized_masks):
        raise ProtocolError("primary and alternative masks must be distinct")
    raw_refs = record["evidence_refs"]
    if not isinstance(raw_refs, list) or not raw_refs or not all(isinstance(item, str) and item for item in raw_refs):
        raise ProtocolError("evidence_refs must contain non-empty strings")
    if len(set(raw_refs)) != len(raw_refs):
        raise ProtocolError("evidence_refs must be distinct")
    parent_id = record["parent_claim_id"]
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        raise ProtocolError("parent_claim_id must be null or a non-empty string")
    if parent is None and parent_id is not None:
        raise ProtocolError("initial claim cannot name a parent")
    if parent is not None and parent_id != parent.claim_id:
        raise ProtocolError("parent_claim_id does not name the accepted lineage tip")
    raw_update = record.get("hypothesis_update")
    raw_heuristic_updates = record.get("heuristic_updates", [])
    if not isinstance(raw_heuristic_updates, list) or len(raw_heuristic_updates) > 12:
        raise ProtocolError("heuristic_updates must contain 0..12 entries")
    heuristic_updates = tuple(
        heuristic_update_from_record(item) for item in raw_heuristic_updates
    )
    update_names = [item.name for item in heuristic_updates]
    if len(set(update_names)) != len(update_names):
        raise ProtocolError("heuristic update names must be distinct")
    if latest_probe_result is None:
        if raw_update is not None:
            raise ProtocolError("initial claim cannot contain a hypothesis update")
        hypothesis_update = None
    else:
        if not isinstance(raw_update, dict):
            raise ProtocolError("a claim after a probe requires a hypothesis update")
        hypothesis_update = hypothesis_update_from_record(
            raw_update,
            latest_probe_result_id=latest_probe_result["evidence_id"],
        )
    prior_heuristics = (
        parent.heuristic_register
        if parent is not None
        else tuple(
            provisional_heuristic_from_record(item)
            for item in (latest_probe_result or {}).get("grounding_claim", {}).get(
                "heuristic_register", []
            )
        )
    )
    merged_by_name = {item.name: item for item in prior_heuristics}
    ordered_names = [item.name for item in prior_heuristics]
    binding_refs = list(raw_refs)
    if latest_probe_result is not None:
        binding_refs.append(latest_probe_result["evidence_id"])
    for update in heuristic_updates:
        if update.name not in merged_by_name:
            ordered_names.append(update.name)
            prior_refs: tuple[str, ...] = ()
        else:
            prior_refs = merged_by_name[update.name].evidence_refs
        evidence_refs = tuple(dict.fromkeys((*prior_refs, *binding_refs)))
        merged_by_name[update.name] = ProvisionalHeuristic(
            name=update.name,
            belief=update.belief,
            scope=update.scope,
            evidence_refs=evidence_refs,
            decision_role=update.decision_role,
            reconsider_when=update.reconsider_when,
        )
    heuristics = tuple(merged_by_name[name] for name in ordered_names)
    if latest_probe_result is not None and not heuristics:
        raise ProtocolError("a claim after a probe requires a provisional heuristic")
    if len(heuristics) > 12:
        raise ProtocolError("merged heuristic register exceeds 12 entries")
    draft = {
        "observation_id": observation_id,
        "primary_region": primary.to_record(),
        "alternatives": [mask.to_record() for mask in alternatives],
        "evidence_refs": list(raw_refs),
        "parent_claim_id": parent_id,
        "hypothesis_update": hypothesis_update.to_record() if hypothesis_update else None,
        "heuristic_updates": [
            {
                "name": item.name,
                "belief": item.belief,
                "scope": item.scope,
                "decision_role": item.decision_role,
                "reconsider_when": item.reconsider_when,
            }
            for item in heuristic_updates
        ],
        "heuristic_register": [item.to_record() for item in heuristics],
    }
    claim_id = "claim-" + content_sha256(draft)
    lineage_id = parent.lineage_id if parent is not None else "lineage-" + claim_id.removeprefix("claim-")
    return GroundingClaim(
        claim_id=claim_id,
        lineage_id=lineage_id,
        observation_id=observation_id,
        primary_region=primary,
        alternatives=alternatives,
        evidence_refs=tuple(raw_refs),
        parent_claim_id=parent_id,
        hypothesis_update=hypothesis_update,
        heuristic_register=heuristics,
    )


@dataclass(frozen=True)
class ActionPrediction:
    action: int
    data: dict[str, Any] | None
    predicted_effect: PredictedEffect

    def to_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "data": self.data,
            "predicted_effect": self.predicted_effect.to_record(),
        }


def _action_prediction_from_record(record: dict[str, Any]) -> ActionPrediction:
    _exact_keys(record, {"action", "data", "predicted_effect"})
    action = _strict_int(record["action"], "action prediction action")
    data = record["data"]
    if data is not None and not isinstance(data, dict):
        raise ProtocolError("action prediction data must be null or an object")
    return ActionPrediction(
        action=action,
        data=data,
        predicted_effect=predicted_effect_from_record(record["predicted_effect"]),
    )


@dataclass(frozen=True)
class ProbeCommitment:
    commitment_id: str
    claim_id: str
    action: int
    data: dict[str, Any] | None
    predicted_effect: PredictedEffect
    alternative_predictions: tuple[ActionPrediction, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "claim_id": self.claim_id,
            "action": self.action,
            "data": self.data,
            "predicted_effect": self.predicted_effect.to_record(),
            "alternative_predictions": [
                prediction.to_record() for prediction in self.alternative_predictions
            ],
        }


def probe_commitment_from_draft(record: dict[str, Any], *, claim: GroundingClaim) -> ProbeCommitment:
    _exact_keys(
        record,
        {
            "claim_id",
            "action",
            "data",
            "predicted_effect",
            "alternative_predictions",
        },
    )
    if record["claim_id"] != claim.claim_id:
        raise ProtocolError("commitment does not bind the accepted claim")
    action = _strict_int(record["action"], "action")
    data = record["data"]
    if data is not None and not isinstance(data, dict):
        raise ProtocolError("action data must be null or an object")
    effect = predicted_effect_from_record(record["predicted_effect"])
    raw_alternatives = record["alternative_predictions"]
    if not isinstance(raw_alternatives, list) or not 1 <= len(raw_alternatives) <= 3:
        raise ProtocolError("alternative_predictions must contain 1..3 action predictions")
    alternatives = tuple(_action_prediction_from_record(item) for item in raw_alternatives)
    actions = [action, *(item.action for item in alternatives)]
    if len(set(actions)) != len(actions):
        raise ProtocolError("selected and alternative action identifiers must be distinct")
    selected_signature = predicted_effect_signature(effect, primary_region=claim.primary_region)
    alternative_signatures = [
        predicted_effect_signature(item.predicted_effect, primary_region=claim.primary_region)
        for item in alternatives
    ]
    if all(signature == selected_signature for signature in alternative_signatures):
        raise ProtocolError(
            "at least one alternative action must predict a different primary-region effect"
        )
    draft = {
        "claim_id": claim.claim_id,
        "action": action,
        "data": data,
        "predicted_effect": effect.to_record(),
        "alternative_predictions": [item.to_record() for item in alternatives],
    }
    return ProbeCommitment(
        commitment_id="commitment-" + content_sha256(draft),
        claim_id=claim.claim_id,
        action=action,
        data=data,
        predicted_effect=effect,
        alternative_predictions=alternatives,
    )


type VerdictStatus = Literal["supported", "contradicted", "inconclusive"]
type ActionSpecificityStatus = Literal["supported", "contradicted", "inconclusive"]


@dataclass(frozen=True)
class GroundingVerdict:
    claim_id: str
    commitment_id: str
    status: VerdictStatus
    checks: tuple[str, ...]
    action_specificity: ActionSpecificityStatus
    contrast_commitment_ids: tuple[str, ...]
    promotion_credit: int
    promotable: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "commitment_id": self.commitment_id,
            "status": self.status,
            "checks": list(self.checks),
            "action_specificity": self.action_specificity,
            "contrast_commitment_ids": list(self.contrast_commitment_ids),
            "promotion_credit": self.promotion_credit,
            "promotable": self.promotable,
        }
