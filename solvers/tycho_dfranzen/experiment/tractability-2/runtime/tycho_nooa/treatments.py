# Modified for the Tycho/dfranzen candidate; see canonical-candidate.patch.
"""Explicit local treatments layered over Tycho's frozen source modes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from tycho_nooa.modes import ModePolicy

TREATMENT_ENV = "TYCHO_NOOA_TREATMENT"
CONTROL = "control"
TYPED_CANDIDATE_GOALS = "typed_candidate_goals"
TRANSACTIONAL_BUILDER = "transactional_builder"
DECISION_DISCIPLINE = "decision_discipline"
SINGLE_DECISION_DISCIPLINE = "single_decision_discipline"
MODEL_SCRIBE = "model_scribe"
DETERMINISTIC_LEVEL_CHECKPOINT = "deterministic_level_checkpoint"
LevelBoundaryPolicy = Literal[
    "model_scribe",
    "deterministic_level_checkpoint",
]
INVESTIGATOR_IDENTITY = (
    "You are one persistent investigator playing an unfamiliar grid game."
)
DECISION_MAKER_IDENTITY = (
    "You are a bounded decision-maker acting in an unfamiliar grid game."
)
DECISION_DISCIPLINE_STRATEGY_PROMPT = (
    "Your objective each turn is the best timely consequential decision, not complete "
    "understanding. Time, provider calls, and environment actions are different scarce "
    "resources. Before another investigation, identify the action currently favored, "
    "the uncertainty preventing commitment, and the result that could change that action. "
    "If you cannot name such a result, act. Legal survivable actions are also experiments: "
    "predict the observable consequence, act, observe, and revise. Investigate further "
    "when an action may be fatal or irreversible and one available computation can "
    "materially reduce that risk. End the turn only by calling take_actions exactly once. "
    "Pass one action when only one step is justified; add later actions only when they need "
    "no new judgment. Each public successor is still observed and can stop the queue."
)


@dataclass(frozen=True)
class TreatmentPolicy:
    """Name adaptations so they cannot masquerade as exact source controls."""

    name: str
    typed_candidate_goals: bool = False
    transactional_builder: bool = False
    decision_discipline: bool = False
    builder_max_tool_steps: int | None = None
    level_boundary_policy: LevelBoundaryPolicy = MODEL_SCRIBE


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
    TRANSACTIONAL_BUILDER: TreatmentPolicy(
        TRANSACTIONAL_BUILDER,
        transactional_builder=True,
        # The live Gemma route can spend the entire decision allowance before
        # Tycho's 40-step laboratory ceiling. Bound the named treatment, not
        # the source control, so transactions can finish or fail closed.
        builder_max_tool_steps=12,
    ),
    DECISION_DISCIPLINE: TreatmentPolicy(
        DECISION_DISCIPLINE,
        decision_discipline=True,
    ),
    "no_scribe": TreatmentPolicy("no_scribe",
        level_boundary_policy=DETERMINISTIC_LEVEL_CHECKPOINT),
    **__import__('hybrid.lean', fromlist=['treatment_policies']).treatment_policies(
        TreatmentPolicy, DETERMINISTIC_LEVEL_CHECKPOINT),
    SINGLE_DECISION_DISCIPLINE: TreatmentPolicy(
        SINGLE_DECISION_DISCIPLINE,
        decision_discipline=True,
        level_boundary_policy=DETERMINISTIC_LEVEL_CHECKPOINT,
    ),
}


def treatment_from_env(
    mode: ModePolicy,
    environment: Mapping[str, str] | None = None,
) -> TreatmentPolicy:
    env = os.environ if environment is None else environment
    name = env.get(TREATMENT_ENV) or CONTROL
    treatment = _resolve_treatment(name)
    _validate_treatment_mode(treatment, mode, name)
    return treatment


def _resolve_treatment(name: str) -> TreatmentPolicy:
    try:
        return TREATMENTS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown {TREATMENT_ENV}={name!r}; choose one of {sorted(TREATMENTS)}"
        ) from error


def _validate_treatment_mode(
    treatment: TreatmentPolicy, mode: ModePolicy, name: str
) -> None:
    if treatment.typed_candidate_goals and mode.name != "orchestrator":
        raise ValueError(
            f"{TREATMENT_ENV}={name!r} requires TYCHO_MODE='orchestrator'; "
            "it is a named adaptation, not a replacement for another source mode"
        )
    if treatment.transactional_builder and not mode.needs_builder:
        raise ValueError(
            f"{TREATMENT_ENV}={name!r} requires a Builder-owned source mode; "
            "it does not add a Builder to single or no_world_model controls"
        )
    required_mode = (
        "single" if name == SINGLE_DECISION_DISCIPLINE else "trigger"
    )
    if treatment.decision_discipline and mode.name != required_mode:
        raise ValueError(
            f"{TREATMENT_ENV}={name!r} requires TYCHO_MODE={required_mode!r}; "
            "its first comparison is against the ordinary incumbent"
        )
