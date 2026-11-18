"""v3 policy: allocate construction when the Actor requests it."""
from __future__ import annotations

from dataclasses import dataclass
import time

from tycho_nooa.deadlines import DecisionDeadline, DecisionDeadlineConfig


@dataclass(frozen=True)
class RequestedBuilderDeadline(DecisionDeadline):
    def builder_seconds_remaining(self, *, now=None, objective=False):
        current = time.monotonic() if now is None else now
        usable = max(0.0, self.started_at + self.total_seconds - current
                     - self.config.host_return_reserve_seconds)
        # Preserve half of a short remainder for the Actor, up to 120 seconds.
        # Repeated requests consume the SAME absolute decision budget.
        reserve = min(self.config.actor_reserve_seconds, usable / 2)
        return min(self.config.builder_cap_seconds, usable - reserve)

    def trace_fields(self):
        return {**super().trace_fields(),
                "allocation_policy": "actor-request-v3",
                "actor_reserve_fraction": 0.5}


@dataclass(frozen=True)
class RequestedBuilderConfig(DecisionDeadlineConfig):
    actor_reserve_seconds: float = 120.0

    def start(self, total_seconds, *, now=None):
        original = super().start(total_seconds, now=now)
        return RequestedBuilderDeadline(**original.__dict__)
