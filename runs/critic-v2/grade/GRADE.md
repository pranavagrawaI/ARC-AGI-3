# Questioner offline grade

## v1 (kernel duck-questioner-grade-comp v1, 2026-09-23 09:23Z) — CONTAMINATED for sp80/tn36
The system prompt's example questions were taken from the graded games themselves:
"Why must the piece move at all?" (tn36) and "Why does the block always stop four rows short of
what it approaches?" (sp80). sp80/tn36 hits therefore partly echo the prompt. Only sc25, dc22,
lf52, cn04 are clean evidence here. Results: `results-v1-contaminated/results.json`.

Mechanics (all 95 calls, 24 concurrent): 95/95 delivered, 0 host rejections, 0 retries.
| effort | median s | median completion tokens |
|---|---|---|
| none (thinking off) | 9.4 | 20 |
| low | 31 | 852 |

Clean-game examples (none = thinking off):
- sc25 t10 none: "What evidence, other than the left block's dots, actually says the white cells are the target?"
  — aims at the premise that failed 3 submissions.
- lf52 t14 low: "Why must the goal still be to reduce to one piece when the board has changed?"
  — aims at the wrong level-1 rule the actor carried (mode E).
- cn04 t28 none: "What would you see if the red squares were actually the character's own feet, not separate objects?"
  — aims at the perception confusion (mode C).
- dc22 t13 none: "Why do you assume the left and right panels are separate, when the pieces might just be one board?"
Weak forms seen: "Have you ever clicked X?" (affordance nudges; closest to advice), and lf52 t29 low
asserts a fact ("Level 1 ended by clicking the bottom box's green piece") that may be wrong.
Sequence mode (ledger carried): 15/15 distinct, no repeats.

## v2 — placeholder examples only (kernel duck-questioner-grade-comp v2, 09:57Z) — CLEAN
`results-v2/results.json`. 95/95 delivered, 0 rejections. Without the answer key in the prompt:
- **tn36** (premise "the piece must drop into the cup"): "Why must the piece end up in the cup?"
  (sequence t22); "Why must the T's be the mechanism, when 38 actions on them have changed nothing
  but themselves?" (t22 low); "Why must the piece be the thing that moves, rather than the cup or
  the grid moving toward it?" (t6/t22 none). It never named the T *bars* (the untouched affordance).
- **sp80** (anomaly: the block always stops 4 rows short): "Why does the blue stop at row 47 when
  rows 48–51 are empty?" (sequence t12); "Why do you assume the blue block is the only object you
  can move?" (t9 none).
Weaknesses: the first placeholder template ("Why must <that> be <so>?") dominates thinking-off
output (mode collapse onto one form); a few near-advice forms remain ("Why have you never tried
pressing DOWN...?"), which the host filter allows because they are questions.

## Reading
Passes the offline bar: in both worldview-lock games the questioner, blind to the answer, asks the
premise-breaking question at least once per game-length sequence, and in the clean games it aims at
the premise behind the failure (sc25 legend, lf52 carried L1 rule, cn04 feet/squares). Thinking off
is ~9 s / 20 tokens and reads as well as low effort (~31 s / 850 tokens) -> live arm uses effort none.
Offline grading cannot show uptake; that needs the live arm.
