from __future__ import annotations

from inference.critic.report import REPORT_BLOCK_FOOTER, REPORT_BLOCK_HEADER, CriticReport, render_report_block


def _report() -> CriticReport:
    return CriticReport(
        packet_id="abc123",
        summary="The red block is the agent;\nthe yellow cell is the exit.",
        recommended_actions=({"action": "UP"}, {"action": "MOUSE", "row": 3, "col": 4}),
        confidence="medium",
        open_questions=("Does SPACE toggle the door?",),
        evidence=("UP moved the block 1 cell in 3/3 trials", "no HUD change on LEFT"),
    )


def test_render_block_shape() -> None:
    lines = render_report_block(_report())
    assert lines[0] == REPORT_BLOCK_HEADER
    assert lines[0].startswith("Critic report (advisory, from a separate investigator; not part of your world model):")
    assert lines[-1] == REPORT_BLOCK_FOOTER == "end of critic report."
    assert lines[1] == "- Packet: abc123"
    assert lines[2] == "- Summary: The red block is the agent; the yellow cell is the exit."
    assert "- Confidence: medium" in lines
    assert "  - UP moved the block 1 cell in 3/3 trials" in lines
    rendered = "\n".join(lines)
    assert '[{"action": "UP"}, {"action": "MOUSE", "col": 4, "row": 3}]' in rendered
    assert "  - Does SPACE toggle the door?" in lines
    assert "Working world model" not in rendered


def test_render_minimal_report() -> None:
    lines = render_report_block(CriticReport(packet_id="", summary=""))
    assert lines == [REPORT_BLOCK_HEADER, "- Packet: unknown", "- Summary: (none)", "- Recommended actions: none", REPORT_BLOCK_FOOTER]


def test_report_round_trip() -> None:
    report = _report()
    assert CriticReport.from_json(report.to_json()) == report
