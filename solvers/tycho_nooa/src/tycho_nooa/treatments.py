"""Explicit local treatments layered over Tycho's frozen source modes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tycho_nooa.modes import ModePolicy

TREATMENT_ENV = "TYCHO_NOOA_TREATMENT"
CONTROL = "control"
TYPED_CANDIDATE_GOALS = "typed_candidate_goals"


@dataclass(frozen=True)
class TreatmentPolicy:
    """Name adaptations so they cannot masquerade as exact source controls."""

    name: str
    typed_candidate_goals: bool = False
    builder_max_tool_steps: int | None = None


TREATMENTS = {
    CONTROL: TreatmentPolicy(CONTROL),
    TYPED_CANDIDATE_GOALS: TreatmentPolicy(
        TYPED_CANDIDATE_GOALS,
        typed_candidate_goals=True,
        # A deadline-bounded actor request must finish and return evidence, not
        # inherit the source control's 40-step laboratory budget and be cancelled
        # halfway through a live decision. The exact orchestrator remains unchanged.
        builder_max_tool_steps=10,
    ),
}


def treatment_from_env(
    mode: ModePolicy,
    environment: Mapping[str, str] | None = None,
) -> TreatmentPolicy:
    env = os.environ if environment is None else environment
    name = env.get(TREATMENT_ENV) or CONTROL
    try:
        treatment = TREATMENTS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown {TREATMENT_ENV}={name!r}; choose one of {sorted(TREATMENTS)}"
        ) from error
    if treatment.typed_candidate_goals and mode.name != "orchestrator":
        raise ValueError(
            f"{TREATMENT_ENV}={name!r} requires TYCHO_MODE='orchestrator'; "
            "it is a named adaptation, not a replacement for another source mode"
        )
    return treatment
