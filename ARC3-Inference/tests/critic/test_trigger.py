from __future__ import annotations

from inference.critic.packet import TransitionRecord
from inference.critic.trigger import (
    CAUSE_ACTOR_REQUEST,
    CAUSE_LEVEL_START,
    CAUSE_STALL_NO_CHANGE,
    CAUSE_STALL_NO_PROGRESS,
    DuckTriggerPolicy,
)


def _t(action_num: int, *, changed: bool = True, completed: bool = False, level: int = 1) -> TransitionRecord:
    return TransitionRecord(action="UP", action_num=action_num, level=level, board_changed=changed, level_completed=completed)


def test_level_start_fires_once_per_level() -> None:
    policy = DuckTriggerPolicy()
    first = policy.decide(level=1, action_num=0)
    assert first.fired and first.cause == CAUSE_LEVEL_START
    policy.observe([_t(1)])
    assert not policy.decide(level=1, action_num=1).fired
    policy.observe([_t(2, completed=True, level=2)])
    second = policy.decide(level=2, action_num=2)
    assert second.fired and second.cause == CAUSE_LEVEL_START
    policy.observe([_t(3, level=2)])
    assert not policy.decide(level=2, action_num=3).fired


def test_no_fire_on_normal_progress() -> None:
    policy = DuckTriggerPolicy(stall_no_change=3, stall_no_progress=25)
    policy.decide(level=1, action_num=0)
    for n in range(1, 25):
        policy.observe([_t(n, changed=(n % 3 != 0))])  # never three no-change in a row
        assert not policy.decide(level=1, action_num=n).fired, n


def test_stall_no_change_arms_and_rearms() -> None:
    policy = DuckTriggerPolicy(stall_no_change=3, stall_no_progress=100)
    policy.decide(level=1, action_num=0)
    policy.observe([_t(1, changed=False), _t(2, changed=False)])
    assert not policy.decide(level=1, action_num=2).fired
    policy.observe([_t(3, changed=False)])
    decision = policy.decide(level=1, action_num=3)
    assert decision.fired and decision.cause == CAUSE_STALL_NO_CHANGE
    # re-armed: two more no-change actions are not enough, a third is
    policy.observe([_t(4, changed=False), _t(5, changed=False)])
    assert not policy.decide(level=1, action_num=5).fired
    policy.observe([_t(6, changed=False)])
    assert policy.decide(level=1, action_num=6).cause == CAUSE_STALL_NO_CHANGE
    # a board change breaks the streak
    policy.observe([_t(7, changed=False), _t(8, changed=True), _t(9, changed=False), _t(10, changed=False)])
    assert not policy.decide(level=1, action_num=10).fired


def test_stall_no_change_is_latched_across_a_batch() -> None:
    # Duck executes batches; a no-change run inside a batch whose last action changed
    # the board must still fire at the next turn start.
    policy = DuckTriggerPolicy(stall_no_change=3, stall_no_progress=100)
    policy.decide(level=1, action_num=0)
    policy.observe([_t(1, changed=False), _t(2, changed=False), _t(3, changed=False), _t(4, changed=True)])
    decision = policy.decide(level=1, action_num=4)
    assert decision.fired and decision.cause == CAUSE_STALL_NO_CHANGE
    assert "run of 3" in decision.reason
    policy.observe([_t(5, changed=True)])
    assert not policy.decide(level=1, action_num=5).fired


def test_stall_no_progress_counts_actions_and_resets_on_level() -> None:
    policy = DuckTriggerPolicy(stall_no_change=100, stall_no_progress=5)
    policy.decide(level=1, action_num=0)
    policy.observe([_t(n) for n in range(1, 5)])
    assert not policy.decide(level=1, action_num=4).fired
    policy.observe([_t(5)])
    decision = policy.decide(level=1, action_num=5)
    assert decision.fired and decision.cause == CAUSE_STALL_NO_PROGRESS
    policy.observe([_t(n) for n in range(6, 10)])
    policy.observe([_t(10, completed=True, level=2)])
    assert policy.decide(level=2, action_num=10).cause == CAUSE_LEVEL_START
    policy.observe([_t(n, level=2) for n in range(11, 16)])
    assert policy.decide(level=2, action_num=15).cause == CAUSE_STALL_NO_PROGRESS


def test_explicit_request_always_fires_and_rearms_stall() -> None:
    policy = DuckTriggerPolicy(stall_no_change=3, stall_no_progress=100)
    policy.decide(level=1, action_num=0)
    policy.observe([_t(1)])
    decision = policy.decide(level=1, action_num=1, critic_request="is the door a goal?")
    assert decision.fired and decision.cause == CAUSE_ACTOR_REQUEST
    assert "is the door a goal?" in decision.reason
    assert policy.decide(level=1, action_num=1, critic_request="again").cause == CAUSE_ACTOR_REQUEST
    assert not policy.decide(level=1, action_num=1, critic_request="   ").fired


def test_observe_is_idempotent_by_action_num() -> None:
    policy = DuckTriggerPolicy(stall_no_change=3, stall_no_progress=100)
    policy.decide(level=1, action_num=0)
    batch = [_t(1, changed=False), _t(2, changed=False)]
    assert policy.observe(batch) == 2
    assert policy.observe(batch) == 0
    assert not policy.decide(level=1, action_num=2).fired


def test_any_firing_marks_level_started() -> None:
    policy = DuckTriggerPolicy()
    decision = policy.decide(level=3, action_num=9, critic_request="help")
    assert decision.cause == CAUSE_LEVEL_START  # level start takes precedence
    assert not policy.decide(level=3, action_num=9).fired
