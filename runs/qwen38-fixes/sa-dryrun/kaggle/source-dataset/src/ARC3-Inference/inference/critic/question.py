"""Question mode: the critic asks the one question that dislodges a stuck hypothesis.

One model call, reasoning off, no tool loop. The host computes the grounding the
model would otherwise spend tool calls on -- what objects are on the board, what
the last action changed, what the actor's own code was assuming -- and the model
answers with three short lines. Pólya's move: don't solve it, ask the question
that makes the solver see the problem differently.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from inference.agent.action_names import to_model_actions
from inference.critic.packet import CriticPacket
from inference.critic.report import CriticReport
from inference.utils.grid_utils import ARC_COLOR_CHARS, ARC_COLOR_LEGEND
from inference.utils.segmentation import segment_layer

QUESTION_FIELDS = ("assumption", "question", "probe")

QUESTION_SYSTEM_PROMPT = (
    "You are the critic for an agent playing an unfamiliar 64x64 grid game. You never act and "
    "you never solve; you ask. The agent is invoked repeatedly, keeps a working hypothesis about "
    "what its actions do and what the goal is, and sometimes gets stuck inside that hypothesis. "
    "Your job is to name the assumption it appears to be stuck on and ask the one question that "
    "would break or replace it -- a question the agent can settle with one or two actions.\n\n"
    f"Color legend: {ARC_COLOR_LEGEND}.\n\n"
    "Good questions vary the problem: What if the thing you are clicking is a display, not a "
    "control? What changed on the LAST action that you did not expect? Is the target a whole-board "
    "configuration rather than one object? Has the same action produced two different outcomes -- "
    "what differed? What would a level-complete look like, and which visible object is closest to "
    "that? Which legal action have you never tried?\n\n"
    "Reply with exactly three lines and nothing else:\n"
    "assumption: <the hypothesis the agent seems stuck on, one short line>\n"
    "question: <one question that would falsify or replace it>\n"
    "probe: <one legal action that answers the question -- UP, DOWN, LEFT, RIGHT, SPACE, "
    "MOUSE(row=r, col=c) -- or none>"
)

_LABEL_RE = re.compile(r"^\s*\**\s*([A-Za-z_]+)\s*\**\s*:\s*(.*?)\s*$")


def parse_question_text(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in str(text or "").splitlines():
        match = _LABEL_RE.match(raw.strip().lstrip("-• ").replace("**", ""))
        if match and match.group(1).lower() in QUESTION_FIELDS:
            fields[match.group(1).lower()] = match.group(2).strip()
    return fields


def _grid(frame: dict[str, Any] | None) -> list[list[int]] | None:
    if not frame or not isinstance(frame.get("grid"), list):
        return None
    return [list(row) for row in frame["grid"]]


def _bbox(cells: list[tuple[int, int]]) -> str:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return f"rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}"


def _node_bbox(node: dict[str, Any]) -> str:
    boundary = node.get("boundary") or []
    if not boundary:
        return "?"
    return _bbox([(int(r), int(c)) for r, c in boundary])


def board_summary(grid: list[list[int]] | None, *, max_nodes: int = 24) -> list[str]:
    """Objects on the board, biggest background first, then the small things that are
    usually the interactive ones. Computed with Duck's own segmentation."""
    if not grid:
        return ["  (no frame)"]
    try:
        seg = segment_layer(grid, ARC_COLOR_CHARS)
    except Exception as exc:  # noqa: BLE001 - grounding is best effort
        return [f"  (segmentation failed: {type(exc).__name__})"]
    nodes = list(seg.get("nodes") or [])
    by_color = Counter(str(n.get("color")) for n in nodes)
    lines = [f"  {len(nodes)} objects; count by color: " + ", ".join(f"{c}={n}" for c, n in by_color.most_common())]
    big = sorted(nodes, key=lambda n: -int(n.get("pixels") or 0))[:3]
    lines.append("  largest: " + "; ".join(f"{n['color']} {n['pixels']}px {_node_bbox(n)}" for n in big))
    small = [n for n in nodes if 1 <= int(n.get("pixels") or 0) <= 64]
    if small:
        shown = small[:max_nodes]
        lines.append(
            f"  small objects (<=64px, {len(small)} total, first {len(shown)}): "
            + "; ".join(f"{n['color']}{n['pixels']}px@{_node_bbox(n)}" for n in shown)
        )
    hashes = Counter(str(n.get("hash")) for n in nodes if int(n.get("pixels") or 0) <= 400)
    repeated = [(h, k) for h, k in hashes.most_common(4) if k >= 2]
    if repeated:
        lines.append("  repeated shapes: " + "; ".join(f"{k}x {next(n['color'] for n in nodes if str(n.get('hash')) == h)}{next(n['pixels'] for n in nodes if str(n.get('hash')) == h)}px" for h, k in repeated))
    return lines


def transition_diff(before: list[list[int]] | None, after: list[list[int]] | None) -> list[str]:
    if not before or not after:
        return ["  (no previous frame)"]
    changed: list[tuple[int, int]] = []
    recolors: Counter[str] = Counter()
    for r, (row_b, row_a) in enumerate(zip(before, after)):
        for c, (b, a) in enumerate(zip(row_b, row_a)):
            if b != a:
                changed.append((r, c))
                recolors[f"{ARC_COLOR_CHARS[b] if 0 <= b < len(ARC_COLOR_CHARS) else b}->{ARC_COLOR_CHARS[a] if 0 <= a < len(ARC_COLOR_CHARS) else a}"] += 1
    if not changed:
        return ["  no cells changed"]
    top = ", ".join(f"{k} x{v}" for k, v in recolors.most_common(4))
    return [f"  {len(changed)} cells changed within {_bbox(changed)}; recolors: {top}"]


def render_question_kickoff(packet: CriticPacket, previous_questions: list[str], *, max_code_chars: int = 700) -> str:
    legal = to_model_actions(list(packet.valid_actions))
    lines = [
        f"Level {packet.level}, after action {packet.action_num}. Why you were called: {packet.trigger_cause} -- {packet.trigger_reason}.",
        f"Legal actions: {', '.join(legal) or 'none'}.",
        "Board now:",
        *board_summary(_grid(packet.current_frame)),
        "What the last action changed (previous frame -> current):",
        *transition_diff(_grid(packet.previous_frame), _grid(packet.current_frame)),
    ]
    recent = packet.transitions[-10:]
    if recent:
        lines.append(
            "Last actions -> outcome: "
            + "; ".join(
                f"{t.get('action')}" + ("(changed)" if t.get("board_changed") else "(no change)") + ("(LEVEL+)" if t.get("level_completed") else "")
                for t in recent
            )
            + f" [{packet.transitions_total} actions this game]"
        )
    beliefs = {k: v for k, v in packet.actor_beliefs.items() if v}
    if beliefs:
        lines.append("The agent's stated beliefs:")
        lines.extend(f"  {k}: {v[:300]}" for k, v in beliefs.items())
    code = packet.actor_last_code.strip()
    if code:
        if len(code) > max_code_chars:
            code = code[:max_code_chars] + "\n# ... [truncated]"
        lines.append("The agent's most recent code (what it was assuming):")
        lines.append(code)
    if previous_questions:
        lines.append("Questions you already asked this game (do not repeat): " + " | ".join(previous_questions[-4:]))
    lines.append("Three lines: assumption, question, probe.")
    return "\n".join(lines)


def question_report(packet_id: str, fields: dict[str, str], recommended: list[dict[str, Any]]) -> CriticReport:
    return CriticReport(
        packet_id=packet_id,
        summary=f"assumption you may be stuck on: {fields.get('assumption', '')}",
        recommended_actions=tuple(recommended),
        confidence="",
        open_questions=(fields["question"],) if fields.get("question") else (),
        evidence=(),
    )


FALLBACK_QUESTION_TEXT = (
    "assumption: unknown -- the critic did not answer in time\n"
    "question: What did the last action change that you did not predict?\n"
    "probe: none"
)

__all__ = [
    "FALLBACK_QUESTION_TEXT",
    "QUESTION_FIELDS",
    "QUESTION_SYSTEM_PROMPT",
    "board_summary",
    "parse_question_text",
    "question_report",
    "render_question_kickoff",
    "transition_diff",
]
