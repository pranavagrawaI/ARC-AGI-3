from __future__ import annotations

from inference.critic.packet import CriticPacket
from inference.critic.sandbox import packet_to_initial_state, run_investigation_python


def test_initial_state_matches_duck_shape(sample_packet: CriticPacket) -> None:
    state = packet_to_initial_state(sample_packet)
    assert set(state) == {"current_frame", "history", "valid_actions", "last_action_result"}
    assert set(state["current_frame"]) == {"ascii", "step", "level", "shape", "grid"}
    assert state["current_frame"]["shape"] == [6, 6]
    assert len(state["history"]) == 5 and state["history"][0]["action"] == ""


def test_segmentation_and_previous_frame_work(sample_packet: CriticPacket) -> None:
    code = """
nodes = current_frame.segmentation["nodes"]
result = {
    "nodes": len(nodes),
    "prev_step": previous_frame.step,
    "cur_step": current_frame.step,
    "last_action": last_action,
    "n_transitions": len(transitions),
    "valid": valid_actions,
    "last_result": last_action_result,
}
"""
    outcome = run_investigation_python(sample_packet, code, timeout_seconds=10)
    assert outcome["error"] == "", outcome
    assert outcome["refused_action_calls"] == 0
    result = outcome["result"]
    assert result["nodes"] == 2  # uniform background plus one marked cell
    assert result["prev_step"] == 3 and result["cur_step"] == 4
    assert result["last_action"] == "LEFT" and result["n_transitions"] == 4
    assert result["valid"] == ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"]
    assert result["last_result"]["action_num"] == 4


def test_action_is_refused(sample_packet: CriticPacket) -> None:
    code = """
try:
    action(["UP"])
except RuntimeError as exc:
    print("refused:", exc)
try:
    action({"action": "MOUSE", "row": 1, "col": 2})
except RuntimeError as exc:
    print("refused again:", exc)
result = "still alive"
"""
    outcome = run_investigation_python(sample_packet, code, timeout_seconds=10)
    assert outcome["error"] == "", outcome
    assert outcome["result"] == "still alive"
    assert outcome["action_results"] == []
    assert outcome["refused_action_calls"] == 2
    assert outcome["stdout"].count("action failed in sandbox host.") == 2


def test_uncaught_action_surfaces_as_error(sample_packet: CriticPacket) -> None:
    outcome = run_investigation_python(sample_packet, "action('UP')\nresult = 1", timeout_seconds=10)
    assert "RuntimeError: action failed in sandbox host." in outcome["error"]
    assert outcome["refused_action_calls"] == 1
