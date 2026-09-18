from __future__ import annotations

import json
from pathlib import Path

from inference.agent.runtime_state import HistoryEntry, write_runtime_state
from inference.critic.packet import CriticPacket, build_packet, transitions_from_history
from inference.critic.trigger import TriggerDecision


def test_transitions_derive_board_change_and_level_completion(sample_history: list[HistoryEntry]) -> None:
    records = transitions_from_history(sample_history)
    assert [(r.action, r.action_num, r.board_changed, r.level_completed) for r in records] == [
        ("UP", 1, True, False),
        ("DOWN", 2, False, False),
        ("MOUSE(row=2, col=3)", 3, True, True),
        ("LEFT", 4, True, False),
    ]


def test_packet_round_trip_and_stable_id(sample_packet: CriticPacket) -> None:
    text = sample_packet.to_json()
    restored = CriticPacket.from_json(text)
    assert restored == sample_packet
    assert restored.packet_id == sample_packet.packet_id
    assert restored.to_json() == text
    assert json.loads(text)["packet_id"] == sample_packet.packet_id
    assert len(sample_packet.packet_id) == 16


def test_packet_id_changes_with_content(sample_packet: CriticPacket) -> None:
    payload = sample_packet.to_payload()
    payload["actor_beliefs"]["world_model"] = "something else"
    payload.pop("packet_id")
    assert CriticPacket.from_payload(payload).packet_id != sample_packet.packet_id


def test_from_json_rejects_tampered_id(sample_packet: CriticPacket) -> None:
    payload = sample_packet.to_payload()
    payload["packet_id"] = "deadbeefdeadbeef"
    try:
        CriticPacket.from_payload(payload)
    except ValueError as exc:
        assert "packet_id mismatch" in str(exc)
    else:
        raise AssertionError("tampered packet_id was accepted")


def test_packet_bounds_history_and_keeps_beliefs_verbatim(sample_history: list[HistoryEntry], tmp_path: Path) -> None:
    state_path = tmp_path / "tool_runtime_state.json"
    write_runtime_state(state_path, current_frame=sample_history[-1].frame, history=sample_history)
    beliefs = {"world_model": "  spaced   text ", "goal_model": "", "cross_level_notes": "L1 was easy"}
    packet = build_packet(
        state_path,
        summarized_knowledge=beliefs,
        last_step_summary=None,
        last_action_result=None,
        valid_actions=["UP", " ", "MOUSE"],
        trigger=TriggerDecision(True, "stall_no_change", "3 in a row"),
        history_limit=2,
        transition_limit=3,
    )
    assert packet.level == 2 and packet.action_num == 4
    assert packet.trigger_cause == "stall_no_change" and packet.trigger_reason == "3 in a row"
    assert packet.actor_beliefs == beliefs
    assert packet.history_total == 5 and len(packet.history) == 2
    assert [item["action"] for item in packet.history] == ["MOUSE(row=2, col=3)", "LEFT"]
    assert packet.transitions_total == 4 and [t["action_num"] for t in packet.transitions] == [2, 3, 4]
    assert packet.valid_actions == ("UP", "MOUSE")
    assert packet.previous_frame is not None and packet.previous_frame["step"] == 3
    assert packet.current_frame is not None and packet.current_frame["step"] == 4
    assert packet.step_summary is None and packet.last_action_result == {}
