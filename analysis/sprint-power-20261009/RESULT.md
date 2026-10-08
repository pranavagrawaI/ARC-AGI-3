# Checkpoint-sprint power, computed offline (2026-10-09)

Script: `power.py` (no GPU). It uses the byte-identical dfranzen runs (control, r2; s1 is a near-identical arm), via `stuck.levels()`.
A sprint resumes at a start state, runs T minutes, and scores 1 if the level is solved. A/B is paired on the same N starts.
Under A = B, var(diff) = 2·E[p(1-p)]/N. Level-entry states replicate across runs, so k-of-n gives an unbiased E[p(1-p)].
Cost assumes 10 streams per GPU at full throughput, with both arms on one server. The yardstick is one full-pass pair:
2 × 2.2 GPU-h, with a relative sd of the difference of about 10% (11 of ~105 levels).

## Verdict
**Sprints are Bernoulli-bound. The best design is only 1.3-2x cheaper than full passes, and the "stuck state" starts the handoff wanted to oversample are 1.5-5x *worse*.**
The A/A calibration run would mostly re-measure p(1-p), which the logs already give. It is not worth 2.5 GPU-h as designed.

| start | T | starts (replicated) | mean p | E[p(1-p)] | N for 10% rel SE | GPU-h, 2 arms | vs full-pass pair |
|---|---|---|---|---|---|---|---|
| level entry | 10 min | 131 (105) | 0.54 | 0.105 | 71 | 2.4 | 0.54x |
| level entry | 20 min | 127 (101) | 0.70 | 0.104 | 43 | 2.9 | 0.65x |
| open at min 10 | 10 min | 67 (31) | 0.31 | 0.097 | 197 | 6.6 | 1.49x |
| open at min 30 | 10 min | 36 (17) | 0.18 | 0.088 | 541 | 18.0 | 4.10x |
| open at min 30 | 20 min | 34 (17) | 0.41 | 0.118 | 139 | 9.3 | 2.10x |
(control + r2 only. Adding s1 makes every row 1.2-2x worse; full table in `power.py` output.)

## Why
- Stuck states have a low per-sprint p, and relative SE scales as sqrt((1-p)/p). The rare successes carry all the information.
- There are only about 130 distinct level-entry states in public-25, and the 10% row already uses 71 of them. Reaching 5% needs about 280, which means repeating starts, which multiplies cost by 4.
- These are optimistic. The logged p includes context carried from earlier levels, which a fresh-context resume loses. Resume prefill cost is also ignored.

## Not settled
- Whether a fresh-context resume preserves the arm ordering. That is a validity question, not a noise question, and an A/A run cannot answer it.
- Any outcome that is not binary. Censored time-to-solve gives modestly more information per start, but not 10x.
