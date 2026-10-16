# m2-lexicon screen-1 (2026-10-17)

**Question:** Does the ARCAID3 lexicon view (rules learned from the level's own steps) raise levels against the
unchanged dfranzen control in a matched 60-minute public-25 run?

**Why this candidate:** It is the only mined piece aimed at our diagnosed bottleneck, rule induction. Its evidence
so far:
- One hidden draw: 31.90, z=+1.3 against 26 public plain-dfranzen draws averaging 28.01 ± 3.03.
- A 10-game, 25-minute demo: 50 levels vs 43.0 ± 3.6 for plain runs, on different days.
- Uptake: the model's code read the view in 98 of 576 tool calls (`.rules` in 40), with no errors.

Source: notebook-mining-20261016, candidates.json rank 1.

**Arms:** `lex` = dfranzen control + the donor's lexicon cell + its counters cell. `rawctl` = dfranzen control.
`build.py` asserts that outside its config cell the donor equals our control plus those two cells. In submission
mode, `lex` is exactly the code that drew 31.90. Both arms stop 60 min after server startup, in the interactive
branch only.

**Confound:** the lexicon's prompt edits also remove the false "frame exposes only five fields" instruction
(the B1-fixed repair). A gain here belongs to the package, not to `.rules` alone.

**Rules (pre-registered; 7 levels ≈ 1 sd of a paired 60-min difference):**
- Invalid if `ARCAID3_LEXICON_READY` is missing from the log, any game crashed, or either arm has ≠ 25 games.
- `lex − rawctl ≥ +7`: SIGNAL. Queue a hidden draw of the lex kernel, and port onto F1 (B1-fixed + friction) next.
- `−7 ≤ diff < +7`: NO_HARM. Queue a hidden draw (draws cost no GPU). Port only if that draw beats 30, i.e. with
  31.90 it would make two draws above every control draw except 29.44.
- `diff < −7`: HARM. Drop.

**Cap:** one launch per arm, hard cap 6,000 s each, 2 slots. The timer waits for F1's screen to free the slots.
No retry except for a proven infrastructure failure. Hidden draws go through `queue.json` and the daily timer.
