# Held-out-game scheduler test, 2026-10-13

The current hazard-index scheduler fails the robustness test. Its base-world gain survives holding out game identities, but relies on extrapolated completions. Keep the shipped allocator; this result does not rule out other allocation policies.

## Observation

Primary results use the three c14 cohorts. The pooled sensitivity uses all ten cohorts with the 10-to-14-stream cost correction estimated only on training games. Every comparison has 110 simulated games, 14 slots, a shared 31,840-second deadline, and 53.2 effective tokens/second per slot. Initial queue waiting counts against the deadline.

| Evaluation world | Shipped score | Held-out index score | Relative gain | Pooled-cohort gain |
|---|---:|---:|---:|---:|
| Fitted constant late hazard | 47.82 | 55.58 | +16.2% | +11.9% |
| Late hazard halved | 46.57 | 49.96 | +7.3% | +1.7% |
| No extrapolated tail completions | 46.33 | 31.29 | -32.5% | -31.3% |
| Twice the token costs | 30.50 | 31.34 | +2.7% | -1.1% |
| Recorded trajectories only | 44.36 | 29.24 | -34.1% | -31.7% |

These scores describe resampled public-game worlds, not predictions of the actual hidden score. The base model, half-hazard world, no-tail world and empirical-only world are different possible futures of censored levels. Empirical-only stops permanently at the selected recorded trajectory's first unresolved level; it is a pessimistic sensitivity, not proof those levels cannot be solved.

## Why the apparent win is fragile

The base-world net gain is 7.75 score points. Score carried by paths requiring at least one extrapolated completion rises from 2.95 to 12.81, a 9.85-point increase. The rest of the score falls by 2.10 points. This is provenance accounting within the simulated outcomes, not a causal estimate of deleting those levels.

Score from directly recorded trajectory levels falls from 43.34 to 38.81. The index leaves an average 21.8 unfinished games below 50K total tokens, versus 0.0 for the shipped allocator. With no extrapolated tail, its paired win rate is 0% over 100 mixed-set bootstraps. That supports the interpretation that it pursues assumed late rewards at the expense of recorded productive paths.

All five primary folds gain in the base world; holding out game identities does not remove the effect. Sensitivity is heterogeneous. Halving late hazard loses in two folds. Doubling costs loses in three folds; the mixed primary mean is slightly positive, while the pooled mean is slightly negative. Thus neither a simple in-sample-fit explanation nor a stable improvement across harder games is supported.

| Fold | Held-out identities | Base gain | Half-hazard gain | Double-cost gain |
|---|---|---:|---:|---:|
| 0 | bp35, sb26, tu93, sc25, ls20 | +4.1% | -4.5% | -3.6% |
| 1 | cn04, cd82, lf52, sk48, lp85 | +27.4% | +25.1% | -12.3% |
| 2 | ar25, dc22, wa30, m0r0, vc33 | +8.2% | -8.7% | +41.9% |
| 3 | s5i5, re86, su15, ka59, ft09 | +27.3% | +14.6% | +40.3% |
| 4 | tr87, sp80, r11l, tn36, g50t | +13.6% | +4.9% | -20.3% |

## Measurement and validation

Five deterministic, outcome-independent folds each contain five identities. Policy hazard, score-per-level estimates, late hazard, and handover duration distributions use only the other 20 identities. Evaluation-world trajectories come only from the five held-out identities; no evaluation outcome fits the deployed priority. The mixed-set test resamples from all 25 identities, but dispatches each instance to its own out-of-fold policy. This retains game diversity while preserving identity holdout. Isolated-fold results use only that fold's five identities resampled to 110 instances; they are heterogeneity checks, not five independent hidden populations.

Two cohorts × five world assumptions × six evaluation groups produce 3,500 paired runs, or 7,000 individual simulations. Each mixed group uses 100 bootstraps; each isolated fold uses 50. Quantiles in RESULT.json are descriptive bootstrap outcome quantiles, not confidence intervals over the unseen hidden-game distribution.

Every paired run consumed 23,714,432.0 tokens per arm. Maximum paired floating-point difference was 7.45e-09 tokens. A per-instance hold-duration tape is identical between policies regardless of dispatch order. Fixed slot rates remove the parent's startup-rate bonus, where early holds retained higher rates after other slots filled.

The common clock is benchmark ActionRecord.wallclock_seconds since Game.start_game. The older transcript curve starts at the first agent turn and omits initial waiting. In c14_E2 the benchmark counts 21/44 completions at minutes 15/30, versus 40/55 on transcript-local clocks. FT09 starts at 09:58:00.713257 but its first transcript turn is 10:10:59. Transcript turn headers also provide only approximate completion timing, especially for a final turn with no following header.

Leave-one-c14-run-out public calibration, 50 realizations per run, gives mean absolute relative errors of 2.3% for score, 2.7% for final level count, and 7.8% for the 15/30/60/90/121-minute curve. It passes the stated 10% aggregate diagnostic gate. This is aggregate calibration; B1_c14's early 15/30-minute predictions still err by about 20%/27%, so it is not exact trajectory reconstruction.

Eight behavior tests passed. The baseline priority port matched 2,500 cases against preserved runtime source. The scorer reconciled 250 saved game scores, and completion segmentation matched all 318 primary-cohort event boundaries. All protected parent input/result hashes remained unchanged.

## Remaining limits and next action

The 25 public identities are the entire observed support; resampling them cannot establish transfer to different hidden games. The three primary cohorts include a B1 variant, and the pooled cohort mixes other harness changes. Fixed effective slot speed abstracts tool/prefill gaps and cache interactions. Handover lengths are resampled independently of the chosen level. The inherited baseline action trajectory is interpolated from realized token/action costs; it is not a replay of every observed request. The latent-world model extrapolates pooled late hazard and action efficiency, and some deeper empirical costs are borrowed from other held-out game identities. Provenance tags distinguish these sources. Calibration of aggregate public outcomes does not validate any of these counterfactual tails.

Do not advance this HazardIndex to a GPU smoke or hidden draw on these results. Next, pseudo-censor held-out historical runs at a fixed token threshold and score out-of-sample late-completion forecasts by level and game difficulty. If it cannot predict observed continuations, replace the constant-tail assumption before revising the priority. If it can, the genuinely unobserved tail still requires a bounded continuation experiment. Allocation remains an open direction; this particular policy is unqualified.

## Artifacts and reproduction

- MANIFEST.json records folds, assumptions, rates, decision rule, and source hashes before the full run.
- RESULT.json contains all 60 comparisons and per-fold fit metadata; runs/*.json retain every paired observation and score provenance.
- CALIBRATION.json and VERIFICATION.json contain independent checks and clock receipts.
- DECISION.json records the scope of the failed robustness result.
- tests.log, calibration.log, verification.log and run.log preserve execution evidence.

From this directory: `python3 -B -m unittest -v test_heldout.py`. Run scripts refuse to overwrite completed outputs; use a fresh copy/destination for a reproduction. Main command was `python3 -B heldout.py --reps 50 --mixed-reps 100 --workers 8`. No model inference, training, GPU run, queue write, submission, solver change or promotion occurred.

Shared record: board thread 682, advance-research CPU test authorized by Pranav's 'do the test'.
