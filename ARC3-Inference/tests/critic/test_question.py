from __future__ import annotations

import requests

from inference.critic.packet import CriticPacket
from inference.critic.policy import CriticModelConfig, CriticNotes, CriticPolicy, CriticPolicyConfig
from inference.critic.question import board_summary, parse_question_text, render_question_kickoff, transition_diff


def _packet(**overrides) -> CriticPacket:
    grid = [[0] * 8 for _ in range(8)]
    for r in range(2, 4):
        for c in range(2, 5):
            grid[r][c] = 3
    prev = [row[:] for row in grid]
    prev[2][2] = 0
    base = dict(
        level=1, action_num=5, trigger_cause="stall_no_change", trigger_reason="3 actions without change",
        current_frame={"grid": grid, "step": 5, "level": 1}, previous_frame={"grid": prev, "step": 4, "level": 1},
        history=(), history_total=6,
        transitions=({"action": "UP", "action_num": 5, "level": 1, "board_changed": False, "level_completed": False},), transitions_total=5,
        valid_actions=("ACTION1", "ACTION6"), last_action_result={"board_changed": False}, actor_beliefs={}, step_summary=None,
        actor_last_code="seg = current_frame.segmentation\naction(['UP'])",
    )
    base.update(overrides)
    return CriticPacket(**base)


def test_parse_question_text_tolerates_markdown_labels() -> None:
    fields = parse_question_text("**Assumption**: tiles are buttons\nquestion: are they displays?\n- probe: SPACE\nextra: ignored")
    assert fields == {"assumption": "tiles are buttons", "question": "are they displays?", "probe": "SPACE"}


def test_grounding_is_host_computed() -> None:
    packet = _packet()
    summary = board_summary([list(r) for r in packet.current_frame["grid"]])
    assert summary[0].startswith("  2 objects")
    diff = transition_diff([list(r) for r in packet.previous_frame["grid"]], [list(r) for r in packet.current_frame["grid"]])
    assert diff == ["  1 cells changed within rows 2-2, cols 2-2; recolors: W->G x1"]
    kickoff = render_question_kickoff(packet, ["already asked?"])
    assert "action(['UP'])" in kickoff and "already asked?" in kickoff and "Legal actions: UP, MOUSE" in kickoff


def test_question_mode_is_one_call_reasoning_off_and_fills_report() -> None:
    seen: list[dict] = []

    def post(payload: dict) -> dict:
        seen.append(payload)
        return {"choices": [{"message": {"content": "assumption: UP moves the avatar\nquestion: Is the grey block the avatar at all?\nprobe: MOUSE(row=2, col=3)"}}], "usage": {"prompt_tokens": 400, "completion_tokens": 30}}

    policy = CriticPolicy(CriticModelConfig(provider="openrouter", base_url="http://x", model_id="m", thinking=True), CriticPolicyConfig(), post=post)
    notes = CriticNotes()
    firing = policy.run(_packet(), notes)
    assert len(seen) == 1 and "tools" not in seen[0] and seen[0]["max_tokens"] == CriticPolicyConfig.question_max_output_tokens
    assert seen[0].get("reasoning") == {"enabled": False}
    assert firing.parsed and firing.model_calls == 1 and firing.tool_calls == 0
    assert firing.report.open_questions == ("Is the grey block the avatar at all?",)
    assert firing.report.recommended_actions == ({"action": "MOUSE", "row": 2, "col": 3},) and firing.recommended_legal is True
    assert notes.lines[-1].endswith("Is the grey block the avatar at all?")
    # the model config is restored after the call
    assert policy._model.thinking is True


def test_question_mode_falls_back_without_raising() -> None:
    """A transport failure (what a live HTTP timeout raises) yields the fallback question, not an exception."""

    def post(payload: dict) -> dict:
        raise requests.exceptions.Timeout("slow")

    policy = CriticPolicy(CriticModelConfig(provider="vllm", base_url="http://x", model_id="m"), CriticPolicyConfig(), post=post)
    notes = CriticNotes()
    firing = policy.run(_packet(), notes)
    assert firing.fallback and not firing.parsed and firing.model_calls == 0
    assert firing.error.startswith("Timeout")
    assert firing.report.recommended_actions == ()
    assert firing.report.open_questions == ("What did the last action change that you did not predict?",)
    assert notes.lines == []  # a fallback is not a question the critic asked
