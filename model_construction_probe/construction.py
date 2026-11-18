"""Bounded construction driver. Semantic selection and repair are injected callbacks.

No LLM provider is called by this prototype. Unit tests use explicitly scripted
builders; these test control flow, not the ability to infer an environment's rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Literal

from kernel import (Assessment, ContractError, FrozenModel, Prefix, Targets,
                    assess, assimilate, initialize, predict)

Operation = Literal["ground", "contrast", "close", "compose", "test"]


@dataclass(frozen=True)
class WorkItem:
    decision: str
    obstacle: str
    operation: Operation
    claim: str
    evidence: tuple[int, ...]
    dependencies: tuple[str, ...]
    # Pairs of visible transition index and evaluator-owned required targets.
    completion_checks: tuple[tuple[int, tuple[str, ...]], ...]
    remaining_scope: tuple[str, ...] = ()

    def validate(self, prefix: Prefix, targets: Targets) -> None:
        if self.operation not in {"ground", "contrast", "close", "compose", "test"}:
            raise ContractError("Unsupported operation")
        if not all(s.strip() for s in (self.decision, self.obstacle, self.claim)):
            raise ContractError("Work needs a decision, obstacle, and specific claim")
        prefix.require_refs(self.evidence)
        domains = json.loads(targets.domains_json)
        if not self.completion_checks:
            raise ContractError("This repair driver requires executable visible-prefix checks")
        for index, required in self.completion_checks:
            if not 1 <= index <= prefix.frontier or not required:
                raise ContractError("Completion checks must reference a visible transition")
            if not set(required) <= set(domains):
                raise ContractError("Completion target is not evaluator-owned")


@dataclass(frozen=True)
class NeedEvidence:
    """Blocked work, not a successful model repair and not a simulated observation."""
    dependency: str
    action_json: str
    possible_observables_json: str


@dataclass(frozen=True)
class Attempt:
    source_id: str
    outcome: str
    feedback: str


@dataclass(frozen=True)
class ConstructionResult:
    status: str
    model: FrozenModel | None
    attempts: tuple[Attempt, ...]
    request: NeedEvidence | None = None


Builder = Callable[[WorkItem, Prefix, FrozenModel, tuple[Attempt, ...]], str | NeedEvidence]


def _regress(model: FrozenModel, prefix: Prefix, targets: Targets) -> tuple[Assessment, ...]:
    belief = initialize(model, prefix.initial)
    reports = []
    for step in prefix.steps:
        forecast = predict(model, belief, json.loads(step.action_json), targets)
        report = assess(forecast, step.observation, targets)
        reports.append(report)
        belief = assimilate(model, forecast, step.observation, report)
    return tuple(reports)


def construct(work: WorkItem, prefix: Prefix, incumbent: FrozenModel,
              targets: Targets, builder: Builder, max_attempts: int = 2) -> ConstructionResult:
    """Select candidates only using visible data. Never receive the sealed suffix.

    max_attempts is an engineering default, not an empirically optimized policy.
    The adapter must separately enforce model-token, wall-time, and tool budgets.
    Python-level object separation is not protection against hostile introspection.
    """
    if max_attempts < 1:
        raise ContractError("max_attempts must be positive")
    work.validate(prefix, targets)
    attempts: list[Attempt] = []
    for _ in range(max_attempts):
        proposal = builder(work, prefix, incumbent, tuple(attempts))
        if isinstance(proposal, NeedEvidence):
            if not proposal.dependency.strip():
                raise ContractError("Evidence request must name a dependency")
            json.loads(proposal.action_json)
            json.loads(proposal.possible_observables_json)
            return ConstructionResult("needs_evidence", None, tuple(attempts), proposal)
        candidate = None
        try:
            candidate = FrozenModel(proposal)
            reports = _regress(candidate, prefix, targets)
            for index, required in work.completion_checks:
                fields = json.loads(reports[index - 1].fields_json)
                if not all(key in fields and fields[key]["covered"] and fields[key]["hit"]
                           for key in required):
                    raise ContractError(f"Required coverage or match missing at transition {index}")
            attempts.append(Attempt(candidate.model_id, "prefix_supported", "Visible checks passed"))
            # This is deliberately not called verified, correct, or batch-authorized.
            return ConstructionResult("candidate_ready", candidate, tuple(attempts))
        except Exception as exc:
            attempts.append(Attempt(candidate.model_id if candidate else "uncompiled",
                                    "rejected", f"{type(exc).__name__}: {exc}"))
    return ConstructionResult("budget_exhausted", None, tuple(attempts))
