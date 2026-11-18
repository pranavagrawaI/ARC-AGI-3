"""Explicit allocation of the host's decision deadline."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionDeadlineConfig:
    """Reserve action time before spending a bounded slice on the builder.

    The builder is advisory and can be retried on later evidence. The actor is
    the only component that can return the environment action, so it owns the
    protected remainder. Ordinary automatic construction is capped from the
    start of the decision. An actor-requested typed objective review instead
    receives a fresh bounded slice because its proposal has already made that
    review mandatory; it still preserves a smaller post-review Actor reserve.
    """

    builder_cap_seconds: float = 240.0
    actor_reserve_seconds: float = 300.0
    objective_builder_cap_seconds: float = 240.0
    objective_actor_reserve_seconds: float = 120.0
    host_return_reserve_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, value in (
            ("builder_cap_seconds", self.builder_cap_seconds),
            ("actor_reserve_seconds", self.actor_reserve_seconds),
            ("objective_builder_cap_seconds", self.objective_builder_cap_seconds),
            ("objective_actor_reserve_seconds", self.objective_actor_reserve_seconds),
            ("host_return_reserve_seconds", self.host_return_reserve_seconds),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def start(
        self,
        total_seconds: float,
        *,
        now: float | None = None,
    ) -> DecisionDeadline:
        if total_seconds <= 0:
            raise ValueError("total_seconds must be positive")
        started_at = time.monotonic() if now is None else now
        actor_cutoff = (
            started_at
            + total_seconds
            - self.actor_reserve_seconds
            - self.host_return_reserve_seconds
        )
        return DecisionDeadline(
            total_seconds=total_seconds,
            started_at=started_at,
            builder_deadline_at=min(
                started_at + self.builder_cap_seconds,
                actor_cutoff,
            ),
            config=self,
        )


@dataclass(frozen=True)
class DecisionDeadline:
    """One decision's immutable allocation on the worker's monotonic clock."""

    total_seconds: float
    started_at: float
    builder_deadline_at: float
    config: DecisionDeadlineConfig

    def builder_seconds_remaining(
        self,
        *,
        now: float | None = None,
        objective: bool = False,
    ) -> float:
        current = time.monotonic() if now is None else now
        if objective:
            objective_cutoff = (
                self.started_at
                + self.total_seconds
                - self.config.objective_actor_reserve_seconds
                - self.config.host_return_reserve_seconds
            )
            return max(
                0.0,
                min(
                    self.config.objective_builder_cap_seconds,
                    objective_cutoff - current,
                ),
            )
        return max(0.0, self.builder_deadline_at - current)

    def trace_fields(self) -> dict[str, float]:
        return {
            "decision_timeout_seconds": self.total_seconds,
            "builder_cap_seconds": self.config.builder_cap_seconds,
            "actor_reserve_seconds": self.config.actor_reserve_seconds,
            "objective_builder_cap_seconds": (
                self.config.objective_builder_cap_seconds
            ),
            "objective_actor_reserve_seconds": (
                self.config.objective_actor_reserve_seconds
            ),
            "host_return_reserve_seconds": self.config.host_return_reserve_seconds,
        }
