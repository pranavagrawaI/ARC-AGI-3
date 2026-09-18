from __future__ import annotations

from pathlib import Path

import pytest
import requests

from inference.agent.runtime_state import write_runtime_state
from inference.agent.tool_agent import ToolAgent, _extract_scientist_note
from inference.critic.report import REPORT_BLOCK_FOOTER, REPORT_BLOCK_HEADER, CriticReport, render_report_block
from prompt_cases import prompt_cases, render_case

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", sorted(prompt_cases()))
def test_prompt_is_byte_identical_to_upstream_without_report(name: str) -> None:
    expected = (FIXTURES / f"prompt_{name}.txt").read_text(encoding="utf-8")
    agent = ToolAgent(model="snapshot-model")
    assert agent._pending_critic_report is None
    assert render_case(agent, prompt_cases()[name]) == expected


def test_report_block_follows_end_of_world_model_and_is_consumed_once() -> None:
    case = prompt_cases()["mid_run_with_world_model"]
    baseline = (FIXTURES / "prompt_mid_run_with_world_model.txt").read_text(encoding="utf-8")
    report = CriticReport(packet_id="p1", summary="try SPACE", recommended_actions=({"action": "SPACE"},))
    agent = ToolAgent(model="snapshot-model")
    agent.set_critic_report(report)
    with_report = render_case(agent, case)
    block = "\n".join(render_report_block(report))
    assert with_report == baseline.replace("end of world model. \n", f"end of world model. \n{block}\n")
    assert with_report.index(REPORT_BLOCK_HEADER) < with_report.index(REPORT_BLOCK_FOOTER)
    assert agent._pending_critic_report is None
    assert render_case(agent, case) == baseline


def test_critic_label_is_parsed_but_kept_out_of_world_model() -> None:
    note = _extract_scientist_note("World model: a maze\nCritic: check whether the door is a goal\n  more\nPlan: go up")
    assert note["critic_request"] == "check whether the door is a goal more"
    assert note["world_model"] == "a maze" and note["current_plan"] == "go up"
    agent = ToolAgent(model="snapshot-model")
    agent._update_summarized_knowledge_from_assistant("Critic: please look\nGoal model: exit")
    assert "critic_request" not in agent._summarized_knowledge
    assert agent._summarized_knowledge["goal_model"] == "exit"
    assert agent.take_critic_request() == "please look"
    assert agent.take_critic_request() == ""


def test_injected_report_is_logged_in_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_history) -> None:
    state_path = tmp_path / "tool_runtime_state.json"
    write_runtime_state(state_path, current_frame=sample_history[-1].frame, history=sample_history)
    transcript = tmp_path / "transcript.txt"
    agent = ToolAgent(model="snapshot-model")

    def _offline(*args, **kwargs):
        raise requests.RequestException("offline test")

    monkeypatch.setattr(agent, "_chat_completion", _offline)
    report = CriticReport(packet_id="pk", summary="advice")
    agent.set_critic_report(report)
    outcome = agent.analyze(state_path, 4, valid_actions=["ACTION1"], transcript_path=transcript)
    assert outcome is not None and outcome.retryable_failure and not outcome.step_executed
    text = transcript.read_text(encoding="utf-8")
    assert "[CRITIC REPORT]\n" in text
    assert text.index("[USER PROMPT]") < text.index("[CRITIC REPORT]")
    assert '"packet_id": "pk"' in text
    assert text.count(REPORT_BLOCK_HEADER) == 1  # once in the prompt; the log section is JSON

    transcript.write_text("", encoding="utf-8")
    agent.analyze(state_path, 4, valid_actions=["ACTION1"], transcript_path=transcript)
    assert "[CRITIC REPORT]" not in transcript.read_text(encoding="utf-8")
