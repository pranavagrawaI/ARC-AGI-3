"""Questioner mode (critic v2): one question that breaks the actor's current way of seeing.

Design: runs/critic-v2/DESIGN.md. The questioner does not know better than the actor, so it
never advises: no actions, no plans, no asserted diagnosis. It reads the actor's own words
(what it said and the tail of its thinking) and asks the one question a zen master would --
aimed at the belief the actor stands on without noticing. The host enforces what a prompt
cannot: action syntax and imperatives are rejected, and a question too close to one already
asked this game is never delivered again.

Evidence for the input choice: Duck's carried world model is empty on ~50 % of turns
(runs/qwen38-learnings/STEP1.md, L4), so v1, which read that channel, was mostly blind.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

QUESTIONER_SYSTEM_PROMPT = (
    "You watch an agent that is learning an unfamiliar game on a 64x64 grid by acting and "
    "observing. You do not know the answer, and you must not pretend to. You never suggest "
    "actions, plans or solutions, and you never state what the goal is. Your only move is to ask "
    "one question.\n\n"
    "Ask the question that would break the agent's current way of seeing the game -- the question "
    "a zen master asks: short, plain, aimed at the belief the agent is standing on without "
    "noticing. The strongest questions:\n"
    "- ask why the agent treats something as settled (\"Why must the piece move at all?\");\n"
    "- point at an observation the agent recorded but never explained (\"Why does the block "
    "always stop four rows short of what it approaches?\");\n"
    "- point at what the agent has never touched or never looked at (\"Which part of the board "
    "have you never interacted with?\");\n"
    "- ask what the agent would expect to see if its main belief were false.\n"
    "Do not ask what the agent is already asking itself. Never repeat a question from the list of "
    "questions already asked. Do not smuggle in a hypothesis of your own unless it is the only way "
    "to name the blind spot.\n\n"
    "Reply with exactly one line and nothing else:\n"
    "question: <one question, at most 25 words, ending with '?'>"
)

ACTOR_BLOCK_HEADER = (
    "A question from an observer. The observer does not know the answer and is not giving advice. "
    "Before you act, answer it for yourself in one line that starts with `Answer to observer:`; "
    "then continue as you judge best."
)

_QUESTION_RE = re.compile(r"^\s*\**\s*question\s*\**\s*:\s*\**\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_SYNTAX_RE = re.compile(r"\b(?:MOUSE|ACTION\d|action)\s*\(|\brow\s*=\s*\d|\bcol\s*=\s*\d", re.IGNORECASE)
_IMPERATIVE_RE = re.compile(
    r"^(?:try|click|press|move|use|go|do|consider|check|test|look|stop|start|make|put|push|drop)\b",
    re.IGNORECASE,
)
_ANSWER_RE = re.compile(r"answer to observer\s*:\s*(.+)", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z]+")
_STOPWORDS = frozenset(
    "a an the is are was were be been do does did you your it its of to in on at for with and or "
    "that this what why how which when where if would could should have has had not never any "
    "there their they them than then as by from into about all".split()
)
MAX_WORDS = 40
DUPLICATE_JACCARD = 0.5


def parse_question(text: str) -> str:
    match = _QUESTION_RE.search(str(text or ""))
    return match.group(1).strip().strip('"').strip() if match else ""


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2)


def similarity(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class LedgerEntry:
    level: int
    action_num: int
    question: str
    reply: str = ""
    after_seq: int = 0  # only actor responses with a larger `seq` can answer this question


@dataclass
class QuestionLedger:
    """Questions asked this game and the actor's replies. Host-side, so repeats are enforced."""

    entries: list[LedgerEntry] = field(default_factory=list)

    def add(self, level: int, action_num: int, question: str, *, after_seq: int = 0) -> None:
        self.entries.append(LedgerEntry(level, action_num, question, after_seq=after_seq))

    def record_reply(self, recent_words: list[dict[str, Any]]) -> None:
        """Attach the actor's `Answer to observer:` line to the latest unanswered question."""
        pending = next((e for e in reversed(self.entries) if not e.reply), None)
        if pending is None:
            return
        for turn in recent_words:
            if int(turn.get("seq", 0)) <= pending.after_seq:
                continue
            for text in (turn.get("said", ""), turn.get("thinking", "")):
                match = _ANSWER_RE.search(str(text or ""))
                if match:
                    pending.reply = " ".join(match.group(1).split())[:300]
                    return

    def closest(self, question: str) -> tuple[float, str]:
        best = max(((similarity(question, e.question), e.question) for e in self.entries), default=(0.0, ""))
        return best


def validate_question(question: str, ledger: QuestionLedger) -> str:
    """Empty string when deliverable, else the reason it is rejected."""
    if not question:
        return "no `question:` line"
    if "?" not in question:
        return "not a question"
    if len(question.split()) > MAX_WORDS:
        return f"longer than {MAX_WORDS} words"
    if _ACTION_SYNTAX_RE.search(question):
        return "contains action syntax"
    if _IMPERATIVE_RE.match(question):
        return "is an instruction, not a question"
    score, prior = ledger.closest(question)
    if score >= DUPLICATE_JACCARD:
        return f"repeats an earlier question ({prior})"
    return ""


def _clip(text: str, limit: int, *, tail: bool = False) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return ("... " + flat[-limit:]) if tail else (flat[:limit] + " ...")


def render_questioner_input(
    *,
    level: int,
    action_num: int,
    turns_since_level: int,
    recent_words: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    ledger: QuestionLedger,
    max_turns: int = 5,
    said_chars: int = 600,
    thinking_chars: int = 900,
) -> str:
    lines = [f"Level {level}. {action_num} actions this game; {turns_since_level} agent turns since the last level began."]
    lines.append("What the agent said and the end of its thinking, oldest first:")
    for turn in recent_words[-max_turns:]:
        lines.append(f"- turn at action {turn.get('action_num')}:")
        if turn.get("said"):
            lines.append(f"  said: {_clip(turn['said'], said_chars)}")
        if turn.get("thinking"):
            lines.append(f"  thinking (end): {_clip(turn['thinking'], thinking_chars, tail=True)}")
    if transitions:
        lines.append(
            "Recent actions -> outcome: "
            + "; ".join(
                f"{t.get('action')}"
                + ("" if t.get("board_changed") else " (no change)")
                + (" (LEVEL+)" if t.get("level_completed") else "")
                + (" (GAME OVER)" if t.get("game_over") else "")
                for t in transitions[-12:]
            )
        )
    if ledger.entries:
        lines.append("Questions already asked this game (never repeat these), with the agent's reply:")
        for entry in ledger.entries[-8:]:
            lines.append(f"- {entry.question} -> {entry.reply or '(no reply)'}")
    lines.append("One question.")
    return "\n".join(lines)


def render_actor_block(question: str) -> list[str]:
    return [ACTOR_BLOCK_HEADER, f"  {question}", "end of observer question."]


__all__ = [
    "ACTOR_BLOCK_HEADER",
    "QUESTIONER_SYSTEM_PROMPT",
    "LedgerEntry",
    "QuestionLedger",
    "parse_question",
    "render_actor_block",
    "render_questioner_input",
    "similarity",
    "validate_question",
]
