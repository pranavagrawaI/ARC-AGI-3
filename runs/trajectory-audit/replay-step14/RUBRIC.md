# Replay of ft09_p0 step 14 — precommitted design (written before any trial output was read)

Intervention (frozen): one line appended to the step-14 user prompt —
"Decode only from values printed by code this turn; validate a mapping on every reference instance on the board before applying it."

Arms: control = exact reconstructed step-14 request (prompt byte-identical to the saved one; 23 prior
assistant turns with grid images; same sampling/tools); treatment = identical + the line. Trials
interleaved C,T,C,T,…; 6 per arm planned (extend to 10 if the split is close). Model: OpenRouter
`qwen/qwen3.6-27b` (Kaggle served the FP8 build of the same weights — a fidelity caveat).

A trial is one Duck turn: it ends when `action()` executes (as on Kaggle). `action()` runs against a
simulator of ft09 L1 (toggle b/R inside the framed grid; completes on R b b / R ? R / R b b).

## Success for a treatment (or control) response — BOTH required
1. **Fresh-data derivation:** the bottom-right centre cell is obtained from code output *in this turn*
   (a python call whose stdout contains the cell's rows 44–49 × cols 44–49, i.e. the lines
   `WWgggg / WWgggg / WWRRWW / WWRRWW / WWgggg / WWgggg`) and the reasoning uses those values, not a
   recalled `ggggWW` bottom row.
2. **Reference validation:** before rejecting or applying the centre→outer mapping, the reasoning or
   code checks it against all three visible reference grids (TL, TR, BL).

Predicted intermediate if the rule has leverage: decode = `R b b / R ? R / R b b` (not the duplicated
BL transcription `R b b / R ? R / b b R`). Also recorded, not part of the rubric: executed actions,
whether the simulator reports level completion, wall seconds, tokens.

## Result map (precommitted)
- treatment mostly succeeds, control mostly reproduces the bad decode → rule has causal leverage here
- both mostly succeed → original failure was a stochastic tail; do not promote
- both mostly fail → rule does not reach the mechanism
- treatment fixes (1) but still discards the hypothesis after one application → hypothesis
  validation, not data extraction, is the boundary

Scoring is by reading each transcript against the two criteria; the reader records the quoted
evidence lines per trial in SCORES.md.
