# m2-f1 screen-2 (2026-10-17)

**Question:** Is the slow start in f1 caused by B1-fixed's frame-API rewrite? Would F1 without the rewrite play at the control's pace?

**Why:** Two arms rewrite the false "frame exposes only five fields" text: screen-1 f1 (B1-fixed + friction) and the lexicon screen's lex arm (dfranzen + lexicon). Both ran ~15% fewer actions and spent ~17% more generated tokens per action in the first 30 minutes than their same-hour controls.

| Run | Actions @ 30 min | Tokens / action @ 30 min | Levels @ 30 / 45 / 60 min |
|---|---|---|---|
| f1 | 1142 | 976 | 32 / 47 / 62 |
| lex | 1135 | 941 | 31 / 47 / 62 |
| b1ctl | 1323 | 806 | 41 / 56 / 65 |
| rawctl | 1298 | 825 | 43 / 60 / 80 |

The earlier B1-family runs fall in the control range: 1234–1281 actions and 38–42 levels at minute 30. Lex never used the raw-grid API, so the shared factor is the prompt text itself, not grid reading. This is n=2 and unreplicated. The rewrite was added the night before screen-1 without a screen of its own. Approved by Pranav on 2026-10-17.

**Arms:**
- `screen-2/f1x` = B1 (frame text untouched) + F1's preload cell + out16k config. It is F1 minus the B1-fixed overlay. `build.py` asserts that f1x minus F1's cell and config lines equals the control.
- `screen-2/b1ctl` = B1. Its code is byte-identical to the screen-1 b1ctl notebook (sha 7e0eb3…), under a fresh kernel id.

Both arms stop 60 min after server startup.

**Measures (`analyze.py screen-2 f1x`):**
- Levels at the end of the run.
- Cumulative levels and actions at minutes 30, 45 and 60, from `benchmark.json` history (`wallclock_seconds` is cumulative per game).
- Generated tokens per action at minute 30.
- The screen-1 mechanism counts.

**Pre-registered rules.** Tempo ratio R = actions@30(f1x) / actions@30(b1ctl). In the earlier runs, the two same-hour controls were within 2% of each other, and both rewrite arms sat at 0.86 and 0.87.
- **No-harm:** f1x fails if levels(f1x) < levels(b1ctl) − 7. If so, neither F1 variant is drawn; inspect first.
- **R ≥ 0.95, REWRITE_SLOWS:** the slow start belongs to the rewrite. **F1x** becomes the draw candidate, and the frame rewrite leaves the retained set.
- **R < 0.90, NOT_REWRITE:** f1x is just as slow, so the shared slow start is not the rewrite (it's noise, or friction slowed f1 and something else slowed lex). **F1**, with the truthful frame text, stays the candidate.
- **0.90 ≤ R < 0.95, AMBIGUOUS:** **F1x** by default. The rewrite has no level-gain evidence and carries the harm suspicion.
- **Also recorded, no rule attached:** b1ctl's levels at minute 60 against screen-1's 65 and the earlier 73–76. This is a control replicate for the noise yardstick.

**After:** the candidate gets a full-length kernel (no SCREEN soft_end) and enters `analysis/hidden-draws-20261010/queue.json` for the next daily draw.

**Cap:** one launch per arm, hard cap 6,000 s each, 2 slots. Admission requires quota used + 3.34 h ≤ 30 h. No retry except for a proven infrastructure failure. No submission from this experiment.
