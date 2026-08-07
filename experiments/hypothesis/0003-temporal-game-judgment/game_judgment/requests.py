from __future__ import annotations

from copy import deepcopy
from typing import Any

from .machinery import discriminator_is_decision_relevant


ARM_CONFIGS = {
    "A": {"images": "current"},
    "B": {"images": "chronological"},
    "C": {"images": "chronological", "step_labels": True},
    "D": {"images": "chronological", "step_labels": True, "numeric_transitions": True},
    "E": {"images": "chronological", "step_labels": True, "numeric_transitions": True, "reflection": True},
    "F": {"images": "chronological", "step_labels": True, "numeric_transitions": True, "reconstruction_authority": True, "candidate_scene": True, "executable_conjectures": True, "discriminators": True},
    "G": {"images": "chronological", "step_labels": True, "numeric_transitions": True, "reconstruction_authority": True, "candidate_scene": True, "executable_conjectures": True, "discriminators": True, "candidate_plans": True},
    "H": {"images": "chronological", "step_labels": True, "numeric_transitions": True, "reconstruction_authority": True, "candidate_scene": True, "executable_conjectures": True, "discriminators": True, "candidate_plans": True, "verified_route": True},
}
ARMS = tuple(ARM_CONFIGS)


def visible_case(case: dict[str, Any], arm: str) -> dict[str, Any]:
    try:
        config = ARM_CONFIGS[arm]
    except KeyError as exc:
        raise ValueError(f"unknown arm {arm}") from exc
    request: dict[str, Any] = {
        "schema": "0003-model-request-v2",
        "case_id": case["case_id"],
        "burden": case["burden"],
        "legal_actions": deepcopy(case["legal_actions"]),
        "current_public": deepcopy(case["current_public"]),
        "images": deepcopy(case["images"][-1:] if config["images"] == "current" else case["images"]),
        "contract": deepcopy(case["contract"]),
    }
    for field in (
        "step_labels",
        "numeric_transitions",
        "reflection",
        "reconstruction_authority",
        "candidate_scene",
        "executable_conjectures",
        "discriminators",
        "candidate_plans",
        "verified_route",
    ):
        if config.get(field):
            request[field] = deepcopy(case[field])
    if "discriminators" in request:
        request["discriminators"] = [
            discriminator
            for discriminator in request["discriminators"]
            if discriminator_is_decision_relevant(discriminator)
        ]
    return request
