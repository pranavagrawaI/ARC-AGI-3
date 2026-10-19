# Draw variance: how much of a hidden score is run luck that the private half shares? (2026-10-18)

## Fact from the rules
`competition-pages.json` (Kaggle pages API, data-description): "Competition evaluation uses a
**separate, private set of 110 games** ... Half of these are used for the Public Leaderboard score,
and the other half for the Private Leaderboard score." So one submission run plays all 110 games.
The private score is the other 55 games of **the same run**, not a rerun. Whatever luck is shared
across a run's games carries from public to private, and nothing else does.

## Model
s[g,r] = mu_g + u_r + e[g,r]. Across draws of the same code the hidden score varies by
sigma_u^2 + sigma_e^2/55. corr(public half, private half) = sigma_u^2 / (sigma_u^2 + sigma_e^2/55).
`decompose.py` fits the additive game x run model and writes `RESULT.json`.

## Public-25, full length (8 dfranzen-family runs, ~122 min, 25 games)
- Run effect: F = 1.08, permutation p = 0.37. sigma_e = 21 per game, so per-game noise alone
  predicts a run-mean sd of 21/5 = 4.2. The observed sd of run means is 4.3. **No run-level
  component is detectable.** Point estimate sigma_u = 1.2 gives a projected corr of 0.15 at 55 games.
- One run is an outlier: b1-r2 (55.79) beats every other run on 12-15 of 25 games (pairwise
  t 1.6-2.9). All other 21 pairs agree within |t| < 1.1. b1-r2 sits z ≈ 2.3 above the 8-run mean.
  The chance that the max of 8 runs reaches that is ≈ 0.09, so this is selection, not a found
  mechanism. Token totals for b1-r1 and b1-r2 are identical (4.58 M vs 4.56 M), so throughput didn't
  differ. Across runs, tokens don't track score (r = -0.15).
- The identical-code pair alone (b1-r1 vs b1-r2) gives "F = 8.5, p = 0.005". That number contains
  the outlier selection and is not an independent estimate.
- 60-min screens (6 runs): F = 0.89; the identical b1ctl pair has F = 0.16. No run effect.

## Hidden side (5 identical control draws: 23.87 / 29.44 / 24.97 / 29.38 / 31.31, sd 3.20)
- If hidden per-game noise equals the public sigma_e (~20-21): predicted sd with no run effect is
  2.8. That is consistent with 3.2 (n = 5, 95% CI for sd ≈ 1.9-9.2). Implied corr ≈ 0.2.
- If hidden per-game noise shrinks with the lower hidden mean (x0.6 → 12.7): predicted sd is 1.7,
  which implies sigma_u ≈ 2.7 and corr ≈ 0.7. Hidden per-game scores are not downloadable
  (DownloadSubmission 403), so the data can't choose between these scalings.

## Verdict
corr(public half, private half) of one run has a **best estimate of ~0.15-0.2**. No run-level
mechanism was found: same-code runs have identical throughput. The data cannot exclude up to ~0.7.

## Decisions it settles
1. Choose finals by highest public score among arms not shown worse. When arms are equal in
   expectation this costs nothing and gains corr x (excess). It does not hinge on the exact corr.
2. Don't read the 31.31 at face value. Expected private for a 31.31 draw is
   27.8 + corr x 3.5 ≈ 28.3 at corr 0.15, or ≈ 30.3 at corr 0.7.
3. Best-of-N can't reach a medal line by itself. About 24 draws by 11-02 put the expected public max
   near 34, and its expected private near 28.7-32. The 10-18 board: gold (rank 18) 38.2,
   silver (rank 222) 31.91, bronze (rank 445) 30.64, us #314 at 31.31. Everyone else's public best
   is also a max over draws, so the private board compresses toward true means. That helps against
   the 28-34 dfranzen-fork band, which shares our code mean, and does nothing against genuinely
   better code.
4. The D9 card's ">= 30 → confirmation draw" rule is uninformative: an identical control reaches 30
   in 2 of 5 draws. Treat each draw as one more ticket plus a weak arm signal.

## Addendum: simulated private board (`private_board.py`, `leaderboard-20261018.csv`)
Each team's public best is modelled as the max of N_eff = f x SubmissionCount draws (f = 0.25-1),
with per-draw sd 8-11.5% of the team's true mean. Each team picks its top-2 public draws and gets
the better private half (corr 0.15 or 0.7). Median private lines over 12 settings: **gold 36-39.5,
silver 30.7-32.9, bronze 28.9-30.9.** Deflating everyone barely moves the lines, because the
board is dense around 30 and the private best-of-2 adds back about +0.56 sd. Our private rank at
a true mean of 27.8 is 203-567. Gaps from 27.8: bronze +4-11%, silver +10-18%, gold +30-42%.
The 9 h notebook cap is already used (532 min of play plus startup), so runtime is not a lever.
