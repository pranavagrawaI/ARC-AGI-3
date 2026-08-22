from __future__ import annotations

import pytest

from tycho_nooa.deadlines import DecisionDeadlineConfig


@pytest.mark.parametrize("host_timeout", [600.0, 3_600.0])
def test_default_builder_allocation_is_capped_at_four_minutes(
    host_timeout: float,
) -> None:
    deadline = DecisionDeadlineConfig().start(host_timeout, now=100.0)

    assert deadline.builder_seconds_remaining(now=100.0) == 240.0


def test_short_host_timeout_skips_builder_to_protect_actor_and_return() -> None:
    deadline = DecisionDeadlineConfig().start(300.0, now=100.0)

    assert deadline.builder_seconds_remaining(now=100.0) == 0.0


def test_builder_allocation_shrinks_before_reserve_boundary() -> None:
    deadline = DecisionDeadlineConfig().start(600.0, now=100.0)

    assert deadline.builder_seconds_remaining(now=339.5) == 0.5
    assert deadline.builder_seconds_remaining(now=340.0) == 0.0


def test_objective_request_gets_a_fresh_protected_builder_slice() -> None:
    deadline = DecisionDeadlineConfig().start(600.0, now=100.0)

    assert deadline.builder_seconds_remaining(now=181.3) == pytest.approx(158.7)
    assert deadline.builder_seconds_remaining(now=181.3, objective=True) == 240.0


def test_objective_slice_still_protects_post_builder_actor_time() -> None:
    deadline = DecisionDeadlineConfig().start(600.0, now=100.0)

    assert deadline.builder_seconds_remaining(now=500.0, objective=True) == 75.0
    assert deadline.builder_seconds_remaining(now=575.0, objective=True) == 0.0
