"""Critic report contract and its rendering into Duck's user prompt.

This module must stay free of `inference.agent` imports: `tool_agent.py` imports it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

REPORT_BLOCK_HEADER = "Critic report (advisory, from a separate investigator; not part of your world model):"
REPORT_BLOCK_FOOTER = "end of critic report."


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class CriticReport:
    packet_id: str
    summary: str
    recommended_actions: tuple[dict[str, Any], ...] = ()
    confidence: str = ""
    open_questions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    # "question": critic v2 questioner -- render only the observer's question, never advice.
    style: str = "report"

    def to_payload(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "summary": self.summary,
            "recommended_actions": [dict(item) for item in self.recommended_actions],
            "confidence": self.confidence,
            "open_questions": list(self.open_questions),
            "evidence": list(self.evidence),
            "style": self.style,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, indent=indent)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CriticReport:
        return cls(
            packet_id=str(payload.get("packet_id", "")),
            summary=str(payload.get("summary", "")),
            recommended_actions=tuple(
                dict(item) for item in payload.get("recommended_actions", ()) if isinstance(item, Mapping)
            ),
            confidence=str(payload.get("confidence", "")),
            open_questions=tuple(str(item) for item in payload.get("open_questions", ())),
            evidence=tuple(str(item) for item in payload.get("evidence", ())),
            style=str(payload.get("style", "report")),
        )

    @classmethod
    def from_json(cls, text: str) -> CriticReport:
        return cls.from_payload(json.loads(text))


def render_report_block(report: CriticReport) -> list[str]:
    """Render the labeled block Duck appends right after `end of world model. `."""
    if report.style == "question":
        # Imported here: questioner.py stays free of report.py and agent imports.
        from inference.critic.questioner import render_actor_block

        question = next((q for q in report.open_questions if _one_line(q)), "")
        return render_actor_block(_one_line(question)) if question else []
    lines = [REPORT_BLOCK_HEADER, f"- Packet: {_one_line(report.packet_id) or 'unknown'}"]
    lines.append(f"- Summary: {_one_line(report.summary) or '(none)'}")
    if report.confidence:
        lines.append(f"- Confidence: {_one_line(report.confidence)}")
    if report.evidence:
        lines.append("- Evidence:")
        lines.extend(f"  - {_one_line(item)}" for item in report.evidence if _one_line(item))
    if report.recommended_actions:
        rendered = json.dumps([dict(item) for item in report.recommended_actions], sort_keys=True)
        lines.append(f"- Recommended actions (in `action(...)` format; verify before use): {rendered}")
    else:
        lines.append("- Recommended actions: none")
    if report.open_questions:
        lines.append("- Open questions:")
        lines.extend(f"  - {_one_line(item)}" for item in report.open_questions if _one_line(item))
    lines.append(REPORT_BLOCK_FOOTER)
    return lines


__all__ = ["REPORT_BLOCK_FOOTER", "REPORT_BLOCK_HEADER", "CriticReport", "render_report_block"]
