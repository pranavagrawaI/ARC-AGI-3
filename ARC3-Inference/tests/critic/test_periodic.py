"""periodic-v1 scheduler: one anchor, minimum spacing, nothing else (Phase 9 boundary/failure tests)."""
from __future__ import annotations

from pathlib import Path

import pytest
import requests

from inference.agent.runtime_state import HistoryEntry
from inference.critic.packet import TransitionRecord
from inference.critic.policy import CriticModelConfig, CriticPolicy, CriticPolicyConfig
from inference.critic.runner import CriticRunner
from inference.critic.trigger import CAUSE_PERIODIC, DuckTriggerPolicy, PeriodicTriggerPolicy
from inference.framework import solver as solver_module
from tests.critic.test_runner import _frame, _scripted_policy

ACTOR_CONTEXT = {"summarized_knowledge": {}, "last_step_summary": None, "last_action_result": None}


def _t(action_num: int, *, changed: bool = False, completed: bool = False, level: int = 1) -> TransitionRecord:
    return TransitionRecord(action="UP", action_num=action_num, level=level, board_changed=changed, level_completed=completed)


# --- scheduler boundaries -------------------------------------------------------------


@pytest.mark.parametrize("action_num, fires", [(0, False), (1, False), (19, False), (20, True), (21, True), (45, True)])
def test_first_attempt_needs_period_actions(action_num: int, fires: bool) -> None:
    policy = PeriodicTriggerPolicy(period=20)
    decision = policy.decide(level=1, action_num=action_num)
    assert decision.fired is fires
    assert decision.cause == (CAUSE_PERIODIC if fires else "")


def test_visited_boundaries_only_and_reanchor_on_overshoot() -> None:
    """18 -> 45 in one batch: one firing at 45 (never at 20 or 40), next due at 65+."""
    policy = PeriodicTriggerPolicy(period=20)
    assert not policy.decide(level=1, action_num=18).fired
    assert policy.decide(level=1, action_num=45).fired
    assert policy.state["anchor"] == 45
    assert not policy.decide(level=1, action_num=64).fired
    assert policy.decide(level=1, action_num=65).fired
    assert policy.state["anchor"] == 65 and policy.state["attempts"] == 2


def test_same_action_count_reentered_does_not_refire() -> None:
    """A retried actor turn re-enters the boundary at the same count: no second attempt."""
    policy = PeriodicTriggerPolicy(period=20)
    assert policy.decide(level=1, action_num=20).fired
    assert not policy.decide(level=1, action_num=20).fired
    assert not policy.decide(level=1, action_num=39).fired
    assert policy.decide(level=1, action_num=40).fired


def test_level_change_and_actor_request_are_ignored() -> None:
    policy = PeriodicTriggerPolicy(period=20)
    assert not policy.decide(level=1, action_num=0).fired  # no action-zero review
    assert not policy.decide(level=2, action_num=5).fired  # level start alone does nothing
    assert not policy.decide(level=2, action_num=7, critic_request="am I stuck?").fired  # request ignored
    policy.observe([_t(n, completed=(n == 12), level=2) for n in range(1, 20)])  # completion does not re-arm
    assert not policy.decide(level=3, action_num=19).fired
    due = policy.decide(level=3, action_num=20)  # completion coinciding with the due boundary: one periodic firing
    assert due.fired and due.cause == CAUSE_PERIODIC


def test_observe_is_inert_and_clock_is_action_num() -> None:
    """The clock is the solver's action count, not the transition stream; RESET entries count if the solver counts them."""
    policy = PeriodicTriggerPolicy(period=20)
    assert policy.observe([_t(n) for n in range(1, 100)]) == 99
    assert not policy.decide(level=1, action_num=19).fired
    assert policy.decide(level=1, action_num=20).fired


def test_fresh_policy_starts_at_anchor_zero() -> None:
    assert PeriodicTriggerPolicy(period=20).state == {"period": 20, "anchor": 0, "attempts": 0, "last_fired_action_num": None}


# --- runner wiring --------------------------------------------------------------------


def test_from_env_selects_periodic(monkeypatch) -> None:
    monkeypatch.setenv("DUCK_CRITIC_TRIGGER", "periodic")
    monkeypatch.setenv("DUCK_CRITIC_PERIOD", "20")
    runner = CriticRunner.from_env(("vllm", "http://x", "m", "k"), trace_path=None)
    assert isinstance(runner.trigger, PeriodicTriggerPolicy) and runner.trigger.period == 20
    assert runner.summary()["trigger_mode"] == "periodic"


def test_from_env_default_is_the_tree(monkeypatch) -> None:
    monkeypatch.delenv("DUCK_CRITIC_TRIGGER", raising=False)
    runner = CriticRunner.from_env(("vllm", "http://x", "m", "k"), trace_path=None)
    assert isinstance(runner.trigger, DuckTriggerPolicy)


def test_unknown_trigger_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        CriticRunner(policy=_scripted_policy([]), trace_path=None, trigger_mode="sometimes")


def test_runner_periodic_ignores_stall_and_level_start(tmp_path: Path) -> None:
    calls: list[dict] = []
    runner = CriticRunner(policy=_scripted_policy(calls), trace_path=tmp_path / "critic.jsonl", trigger_mode="periodic", period=20)
    history = [HistoryEntry(action="RESET", frame=_frame(1, 0, 0))]
    assert not runner.decide(history=history, level=1, action_num=0, critic_request="").fired
    for n in range(1, 20):  # nineteen no-change actions: the tree would have fired twice over
        history.append(HistoryEntry(action="UP", frame=_frame(1, n, 0)))
    assert not runner.decide(history=history, level=1, action_num=19, critic_request="").fired
    history.append(HistoryEntry(action="UP", frame=_frame(1, 20, 0)))
    decision = runner.decide(history=history, level=1, action_num=20, critic_request="")
    assert decision.fired and decision.cause == CAUSE_PERIODIC
    firing = runner.fire(decision, current_frame=history[-1].frame, history=history, valid_actions=["ACTION1"], actor_context=ACTOR_CONTEXT)
    assert firing.parsed and len(calls) == 1
    assert runner.summary()["by_cause"] == {"periodic": 1}


# --- failure paths through the session hook -------------------------------------------


def _session(runner: CriticRunner, monkeypatch, action_count: int):
    from inference.agent.tool_agent import ToolAgent

    class _State:
        levels_completed = 0
        available_actions = [1]

    class _Run:
        history = [None] * action_count  # the solver's clock is len(game_run.history)

    class _Game:
        current_state = _State()
        number_of_levels = 3
        game_run = _Run()

    monkeypatch.setattr(solver_module, "_level_number", lambda game: 1)
    monkeypatch.setattr(solver_module, "_engine_action_names", lambda game: ["ACTION1"])
    session = solver_module._HarnessGameSession.__new__(solver_module._HarnessGameSession)
    session.game = _Game()
    session.analyzer = ToolAgent(model="stub", base_url="http://x", provider="vllm")
    session.history_entries = [HistoryEntry(action="RESET", frame=_frame(1, 0, 0))]
    session.viewer_events = []
    session.analysis_step = 1
    session.critic = runner
    session.write_viewer_payload = lambda: None
    session._base_viewer_event = lambda frame: {"level": frame.level}
    session.current_frame = lambda: session.history_entries[-1].frame
    return session


def test_fallback_report_is_still_injected(monkeypatch) -> None:
    """Transport failure -> fallback report -> queued for the actor, exactly as in tree mode."""
    def post(payload: dict) -> dict:
        raise requests.exceptions.Timeout("slow")

    policy = CriticPolicy(CriticModelConfig(provider="vllm", base_url="http://x", model_id="m"), CriticPolicyConfig(), post=post)
    runner = CriticRunner(policy=policy, trace_path=None, trigger_mode="periodic", period=20)
    session = _session(runner, monkeypatch, action_count=20)
    session._run_critic_if_triggered()
    assert session.analyzer._pending_critic_report is not None
    assert runner.firings[-1]["fallback"] and runner.firings[-1]["error"].startswith("Timeout")
    assert session.viewer_events[-1]["type"] == "critic" and session.viewer_events[-1]["parsed"] is False


def test_escaped_exception_is_logged_and_not_retried_at_the_same_count(monkeypatch) -> None:
    class _Boom(CriticPolicy):
        def run(self, packet, notes):  # noqa: D401 - deliberately escapes the policy's own guards
            raise RuntimeError("critic exploded")

    policy = _Boom(CriticModelConfig(provider="vllm", base_url="http://x", model_id="m"), CriticPolicyConfig())
    runner = CriticRunner(policy=policy, trace_path=None, trigger_mode="periodic", period=20)
    session = _session(runner, monkeypatch, action_count=20)
    session._run_critic_if_triggered()  # must not raise
    assert session.analyzer._pending_critic_report is None
    assert session.viewer_events == [
        {"level": 1, "type": "critic", "title": "Critic (escaped exception)", "action_num": 20, "analysis_step": 1, "cause": "", "error": "RuntimeError: critic exploded"}
    ]
    # The anchor moved before the attempt: re-entering the same boundary is silent, the next attempt needs +20.
    session._run_critic_if_triggered()
    assert len(session.viewer_events) == 1
    session.game.game_run.history = [None] * 40
    session._run_critic_if_triggered()
    assert len(session.viewer_events) == 2
