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

## v2 — placeholder examples only (pending)
