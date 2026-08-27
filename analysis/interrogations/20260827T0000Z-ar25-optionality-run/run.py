#!/usr/bin/env python3
"""Socratic interrogation: confront the run's own model with its verbatim trace.

Same route as the live run: OpenRouter -> DeepInfra, deepseek/deepseek-v4-flash-0731,
reasoning enabled, fallbacks disabled. No tools: this is judgment under questioning,
not another exploration episode.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]  # analysis/interrogations/<this>/
RUN = ROOT / "runs" / "optionality-deepseek-ar25-20260827T0912Z"
OUT = Path(__file__).resolve().parent / "transcript.jsonl"

MODEL = "deepseek/deepseek-v4-flash-0731"

SYSTEM = (
    "You are being interviewed about an ARC-AGI-3 episode you played earlier as the "
    "actor of a Tycho/Nooa solver on game ar25. The interviewer shows you verbatim "
    "excerpts of your own reasoning and actions from that episode. Answer in first "
    "person, honestly and specifically. If your past self made an error, say so "
    "plainly; do not defend it because it was yours. Distinguish what you knew at "
    "the time from what you only see now. Be concise. You have no tools."
)

EVIDENCE = """VERBATIM EXCERPTS FROM YOUR EPISODE (game ar25, seed 0, 15-action cap):

--- A. Your derived plan (turn 5, call 122 reasoning) ---
"So from current t5, solution = 6 LEFT + 10 DOWN moves. Total moves would be 5+16 = 21.
The counter (col63) fills from top... Budget ~64. Fine."

--- B. Your arithmetic at t9 (call 142) ---
"Green goes -3 per LEFT. 18,15,12,9,6,3 -> 5 LEFTs to reach col3. Gray goes +3 per LEFT...
So from t9 green col12, gray col42. Need 3 more LEFT."

--- C. Your reset decision (call 149) ---
"Perhaps the goal is that BOTH blocks must end up at bottom pads, but there are two:
the b-pad at left (rows45-53) and the green bottom row? ... Hmm. Alternatively maybe
the puzzle is that the two pieces are two halves forming a rectangle... Let me just
reset and explore mechanically. Let me RESET."

--- D. Your post-reset finding (call 166, after 11 host turns were gone) ---
"Confirmed: only bottom socket is the gray-shaped one at bottom right. No green socket
on left. Given the symmetric (mirror) linkage of the two blocks, the green block is
meant to ..." [thought trails off; next action was LEFT, continuing divergence probes]

--- E. Harness accounting (external fact, shown for calibration) ---
Host-level action budget for the whole episode was 15 actions. Actions used before
RESET: 10 (5 of them spent reaching t5). After RESET: 5 remained. Your internal move
counter reading (~64) is a game-board artifact; the harness cap was separate and the
host stops the episode when either exhausts. It is not recorded whether you ever saw
the host cap value.

--- F. Your final five actions after RESET ---
DOWN, LEFT, SPACEBAR, LEFT — then the episode ended mid "observe clamping".
"""

QUESTIONS = [
    # Round 1 — facts before judgments
    "Q1. Look at excerpt E. Before you pressed RESET at turn 10, did you know the "
    "15-action host cap existed? Not the col-63 counter you read from the board — "
    "the host cap. Answer plainly yes/no/unsure, then say how it changes whether "
    "resetting was affordable.",

    # Round 2 — the retention charge
    "Q2. Excerpt C ends 'Let me just reset and explore mechanically.' Excerpt D shows "
    "you re-derived the same mechanics afterward without executing the plan. From "
    "your seat now: was that a *decision to switch plans*, or did the plan simply "
    "stop being present in your working state? Defend whichever you claim.",

    # Round 3 — the genuine doubt
    "Q3. In excerpt D you noted gray has a socket but green does not. That is a real "
    "anomaly against hypothesis H2. What was the cheapest discriminating test "
    "available to you at that moment, and why did you not run it before or instead "
    "of RESET? Name the test concretely (moves, observation, expected split between "
    "hypotheses).",

    # Round 4 — sunk cost vs its inverse
    "Q4. Interviewer's claim: you avoided sunk-cost attachment to the plan but fell "
    "into the opposite failure — abandoning a verified incumbent for ambient doubt "
    "and never collecting the knowledge RESET paid for. Agree or rebut, using your "
    "actual post-reset behavior from excerpts D and F as evidence either way.",

    # Round 5 — policy going forward
    "Q5. State the rule you will adopt for future episodes in one or two sentences: "
    "when a live plan meets a genuine anomaly, what must happen before any RESET?",
]

TIME_BUDGET_S = 600


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ARC_LLM_API_KEY")
    if not key:
        sys.exit("ARC_LLM_API_KEY missing")
    client = OpenAI(
        base_url=os.environ.get("ARC_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=key,
        timeout=TIME_BUDGET_S,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EVIDENCE + "\n\n" + QUESTIONS[0]},
    ]
    with OUT.open("a") as log:
        for i, q in enumerate(QUESTIONS):
            if i > 0:
                messages.append({"role": "user", "content": q})
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=4096,
                extra_body={
                    "provider": {"order": ["DeepInfra"], "allow_fallbacks": False},
                    "usage": {"include": True},
                },
            )
            msg = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            cost = getattr(usage, "cost", None)
            row = {
                "schema": "tycho-nooa-interrogation-v1",
                "question_index": i,
                "question": q,
                "answer": msg.content,
                "reasoning": getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None),
                "model": MODEL,
                "reported_model": getattr(resp, "model", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "cost_usd": cost,
                "at": datetime.now(UTC).isoformat(),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{i+1}/{len(QUESTIONS)}] answered ({row['completion_tokens']} tok)")
            messages.append({"role": "assistant", "content": msg.content})
    print("complete ->", OUT)


if __name__ == "__main__":
    main()
