# Replay result — the frozen rule does not reach the failure; the failure is the model's modal behaviour here

Design and rubric: RUBRIC.md (precommitted). Scores with quotes: SCORES.md. Model: OpenRouter
`qwen/qwen3.6-27b` (Kaggle: FP8 build of the same weights). Prompt fidelity: the reconstructed step-14
prompt is byte-identical to the saved one; history = 23 reconstructed assistant turns with grid images;
tool results re-wrapped from their display form (one known approximation). One turn per trial; `action()`
against an ft09-L1 simulator. Cost ≈ 12 turns, well under $1.

| | control | treatment |
|---|---|---|
| C1 fresh-data derivation | 1/6 | 1/6 |
| C2 reference validation | 1/6 | 0/6 |
| both (rubric success) | **1/6** | **0/6** |
| correct decode `R b b / R ? R / R b b` | 1/6 | 0/6 |
| simulator level completed | 1/6 | 0/6 |

## Reading against the precommitted map
**Both arms mostly fail → the rule does not reach the mechanism.** The added line changed nothing
measurable: 5/7 treatment outputs (incl. smoke) ran no code before acting, the other 2 printed the
cell and then skipped validation. A one-line instruction appended to a long prompt in a 24-turn
context has no leverage on this model's behaviour at this state.

Second finding, unplanned but clear: **the transcription error is not a tail event; it is the modal
behaviour.** 10 of 14 outputs (both arms + smoke) asserted from memory that all four centre cells are
`W g g / W R W / g g W` — the same duplicated-BL reading as the original run — and reasoned from it.
The original ft09_p0 was not unlucky; it did the typical thing.

Third: the p1-style path *exists* in the control distribution. 04-control (808 s) printed the centres,
validated on TL and BL, saw BR did not match its current outer, concluded "encodes the TARGET",
decoded correctly and completed the level; the smoke control also decoded correctly but acted on a
recalled current pattern and missed a toggle. So ~1–2 in 7 turns take the measured route; the rest
reason from recall. Which route a turn takes is decided early in the turn (whether the first tool call
is an inspection or an `action()`).

## What this rules in and out
- Out: prompt-level meta-rules at the end of the user prompt as the intervention for this failure.
- Out: "the original failure was a stochastic tail" — it is the mode.
- Still open (not tested here): a *structural* intervention that makes the measured route the default —
  e.g. the host runs the inspection (prints the reference and target cells) and puts the values in the
  prompt, so there is nothing to recall; or a turn shape that forbids `action()` as the first tool call
  after N failed attempts. Both change the harness; neither is a one-line prompt. The cheap discriminator
  before either: replay this same state with the four centre cells *already printed in the prompt* and
  count correct decodes. That tests "is recall the bottleneck, or is validation?" without a screen.

Nothing in the harness was changed. Stopped here as instructed.
