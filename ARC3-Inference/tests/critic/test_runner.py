from __future__ import annotations

import json
from pathlib import Path

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.critic.policy import CriticModelConfig, CriticPolicy, CriticPolicyConfig
from inference.critic.runner import CriticRunner, critic_enabled, policy_config_from_env
from inference.framework import solver as solver_module

REPORT = (
    "confidence: low\nmodel: clicks recolor\nobjective: unknown\nplan: none\n"
    "recommended_action: MOUSE(row=2, col=3)\nsubgoal: probe\nnote: click once\nscratchpad: tiles 2x2"
)


def _frame(level: int, step: int, fill: int) -> Frame:
    return Frame(grid=tuple(tuple(fill for _ in range(4)) for _ in range(4)), step=step, level=level)


def _scripted_policy(calls: list[dict]) -> CriticPolicy:
    def post(payload: dict) -> dict:
        calls.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": REPORT}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    return CriticPolicy(CriticModelConfig(provider="vllm", base_url="http://x", model_id="m"), CriticPolicyConfig(max_tool_steps=2), post=post)


def test_runner_fires_level_start_then_stays_quiet_until_stall(tmp_path: Path) -> None:
    calls: list[dict] = []
    runner = CriticRunner(policy=_scripted_policy(calls), trace_path=tmp_path / "critic.jsonl", stall_no_change=2, stall_no_progress=50)
    history = [HistoryEntry(action="RESET", frame=_frame(1, 0, 0))]

    first = runner.decide(history=history, level=1, action_num=0, critic_request="")
    assert first.fired and first.cause == "level_start"
    firing = runner.fire(first, current_frame=history[-1].frame, history=history, valid_actions=["ACTION1", "ACTION6"], actor_context={"summarized_knowledge": {}, "last_step_summary": None, "last_action_result": None})
    assert firing.parsed and firing.report.recommended_actions == ({"action": "MOUSE", "row": 2, "col": 3},)
    assert runner.notes.render().endswith("tiles 2x2")
    trace = [json.loads(line) for line in (tmp_path / "critic.jsonl").read_text().splitlines()]
    assert trace[0]["cause"] == "level_start" and trace[0]["transcript"]

    # One board-changing action: no firing.
    history.append(HistoryEntry(action="UP", frame=_frame(1, 1, 1)))
    assert not runner.decide(history=history, level=1, action_num=1, critic_request="").fired
    # Two no-change actions: stall fires.
    history.append(HistoryEntry(action="UP", frame=_frame(1, 2, 1)))
    history.append(HistoryEntry(action="UP", frame=_frame(1, 3, 1)))
    stalled = runner.decide(history=history, level=1, action_num=3, critic_request="")
    assert stalled.fired and stalled.cause == "stall_no_change"
    assert runner.summary()["firings"] == 1  # decide() does not fire the policy; fire() does


def test_explicit_request_fires(tmp_path: Path) -> None:
    runner = CriticRunner(policy=_scripted_policy([]), trace_path=None)
    history = [HistoryEntry(action="RESET", frame=_frame(1, 0, 0)), HistoryEntry(action="UP", frame=_frame(1, 1, 1))]
    runner.decide(history=history, level=1, action_num=0, critic_request="")  # level start consumed
    decision = runner.decide(history=history, level=1, action_num=1, critic_request="is the bar a timer?")
    assert decision.fired and decision.cause == "actor_request"


def test_critic_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DUCK_CRITIC_ENABLED", raising=False)
    assert critic_enabled() is False
    solver = solver_module.HarnessSolver.__new__(solver_module.HarnessSolver)
    assert solver._make_critic(object(), "stem") is None


def test_policy_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DUCK_CRITIC_MAX_TOOL_STEPS", "2")
    monkeypatch.setenv("DUCK_CRITIC_MAX_SECONDS", "90")
    config = policy_config_from_env()
    assert (config.max_tool_steps, config.max_seconds) == (2, 90.0)


def test_session_hook_injects_report_into_next_prompt(tmp_path: Path, monkeypatch) -> None:
    """`_run_critic_if_triggered` end to end with a stub game: the next user prompt carries the block."""
    from inference.agent.tool_agent import ToolAgent

    class _State:
        levels_completed = 0
        available_actions = [1, 6]

    class _Run:
        history: list = []

    class _Game:
        current_state = _State()
        number_of_levels = 3
        game_run = _Run()

    monkeypatch.setattr(solver_module, "_level_number", lambda game: 1)
    monkeypatch.setattr(solver_module, "_engine_action_names", lambda game: ["ACTION1", "ACTION6"])
    analyzer = ToolAgent(model="stub", base_url="http://x", provider="vllm")
    session = solver_module._HarnessGameSession.__new__(solver_module._HarnessGameSession)
    session.game = _Game()
    session.analyzer = analyzer
    session.history_entries = [HistoryEntry(action="RESET", frame=_frame(1, 0, 0))]
    session.viewer_events = []
    session.analysis_step = 1
    session.critic = CriticRunner(policy=_scripted_policy([]), trace_path=None)
    session.write_viewer_payload = lambda: None
    session._base_viewer_event = lambda frame: {"level": frame.level}
    session.current_frame = lambda: session.history_entries[-1].frame
    # `action_count` is a read-only property over history; RESET alone counts as 0 actions.

    session._run_critic_if_triggered()
    assert session.viewer_events and session.viewer_events[0]["type"] == "critic"
    prompt = analyzer._build_user_prompt(0, valid_actions=["UP", "MOUSE"], current_frame=session.history_entries[-1].frame, history_entries=session.history_entries)
    assert "Critic report (advisory" in prompt and '{"action": "MOUSE", "col": 3, "row": 2}' in prompt
    # Consumed once: the following prompt is clean.
    prompt2 = analyzer._build_user_prompt(1, valid_actions=["UP", "MOUSE"], current_frame=session.history_entries[-1].frame, history_entries=session.history_entries)
    assert "Critic report" not in prompt2
