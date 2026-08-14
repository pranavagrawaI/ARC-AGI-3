"""Fixed, non-oracular model-visible contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical import canonical_json, content_sha256

MODEL_CONTRACT: dict[str, Any] = {
    "schema": "0158-model-contract-v5",
    "objective": "Complete the current public level using legal public actions.",
    "public_action_semantics": {
        "action": "Use one integer identifier from legal_actions.",
        "data": "Use null unless the public action schema requires an object.",
        "identifiers": "An identifier alone does not imply an effect.",
    },
    "observation": {
        "grid": "Rows of public integer cell values with zero-based row and column indices.",
        "history": "inspect_evidence() returns one current, immutable public evidence snapshot.",
        "events": "Auxiliary observed events are separate from ordinary transitions.",
    },
    "protocol": {
        "analysis": (
            "First return one GroundingClaim draft without an action. After a probe, bind an "
            "explicit retain, revise, or abandon update to that probe. Submit only changed "
            "provisional heuristics; the host retains omitted entries and conserves evidence."
        ),
        "commitment": (
            "Then return one selected action-effect prediction and one to three distinct "
            "alternative action-effect predictions for the accepted claim."
        ),
        "execution": "At most one legal action may execute for an accepted commitment.",
        "verdict": (
            "The host reports observed-effect and action-specificity checks only as supported, "
            "contradicted, or inconclusive; promotion requires a closed prospective contrast."
        ),
        "change_accounting": (
            "The host preserves only mechanical counts inside and outside the prior primary "
            "region. Outside change may be independent dynamics; the model remains responsible "
            "for grouping, relevance, identity, causal meaning, and operational attention."
        ),
    },
    "limits": {
        "alternatives": 3,
        "alternative_action_predictions": 3,
        "replace_area": 64,
        "replace_assertions": 16,
    },
}

MODEL_CONTRACT_SHA256 = content_sha256(MODEL_CONTRACT)

_ALLOWED_TOP_LEVEL = frozenset(MODEL_CONTRACT)
_PROHIBITED_DISCLOSURES = (
    "player",
    "target identity",
    "goal identity",
    "controllable-region location",
    "controllable-region appearance",
    "object-role",
    "wall",
    "blocked terrain",
    "passable terrain",
    "terminal condition",
    "winning condition",
)


def model_contract() -> dict[str, Any]:
    return deepcopy(MODEL_CONTRACT)


def contract_word_count(value: dict[str, Any] | None = None) -> int:
    contract = MODEL_CONTRACT if value is None else value
    words: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            words.extend(item.replace("_", " ").split())
        elif isinstance(item, dict):
            for key, child in item.items():
                collect(key)
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(contract)
    return len(words)


def validate_model_contract(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("model contract must be an object")
    extras = set(value) - _ALLOWED_TOP_LEVEL
    if extras:
        raise ValueError(f"model contract contains non-allowlisted fields: {sorted(extras)}")
    lowered = canonical_json(value).lower()
    disclosures = [term for term in _PROHIBITED_DISCLOSURES if term in lowered]
    if disclosures:
        raise ValueError(f"model contract discloses prohibited LS20 semantics: {disclosures}")
    if value != MODEL_CONTRACT:
        raise ValueError("model contract differs from the frozen allowlisted contract")
    if contract_word_count(value) > 400:
        raise ValueError("model contract exceeds 400 words")
    return deepcopy(value)
