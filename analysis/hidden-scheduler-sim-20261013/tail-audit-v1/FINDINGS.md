# Late-solve calibration, 2026-10-13

The constant late solve rate is not rejected within the range covered by the logs. The scheduler remains unqualified because its simulated benefit depends on extrapolated solves beyond that range. This narrows the diagnosis from 'assumed late solves may be fictitious' to 'the indefinite continuation assumption is untested'.

## Observed continuations

At the fixed 80K-token landmark, the three c14 cohorts contain 59 eligible attempts across 22 game identities. There are 28 later solves and 31 naturally right-censored attempts. Each game is held out of the fit, using the same five identity folds, fitting on 20 games and evaluating on five as the prior audit. Artificially hiding a test attempt's future does not change its forecast.

The fitted rate implies 28.18 events over the actual 3,067,421 follow-up tokens; 28 occurred. This is integrated-hazard calibration over observed exposure, not a forecast that guessed each attempt's eventual exposure. Actual follow-up durations enter scoring only; they do not fit that attempt's rate. Fixed-window forecasts below are made solely from training-game rates.

| Extra token window after 80K | Mean forecast | Observed KM estimate | Known outcomes | Early-censored unknown outcomes |
|---|---:|---:|---:|---:|
| 20,000 | 16.7% | 13.9% | 56 | 3 |
| 40,000 | 30.6% | 29.4% | 50 | 9 |
| 80,000 | 51.8% | 47.5% | 37 | 22 |

The game-cluster bootstrap interval for observed/expected events is [0.68, 1.46], including 1. This rules out neither moderate underprediction nor moderate overprediction. The pooled ten-cohort sensitivity gives 83 events versus 83.51 expected. Its first 20K window forecast underpredicts the observed estimate (15.9% versus 20.7%), so aggregate event-count agreement is not exact time calibration.

KM estimates assume sufficiently noninformative censoring. Scheduler-dependent censoring can violate that assumption. RESULT.json also provides raw lower/upper fractions compatible with unknown outcomes, known-only Brier scores, coverage, and every forecast. Known-only metrics are diagnostics on a selected subset and must not be treated as unconditional performance.

The half-rate alternative underpredicts aggregate late events and has worse mean held-out survival log score. However, its game-cluster log-score difference interval includes 0, so these logs do not establish a statistically decisive advantage for the full rate. A level-conditioned, shrunk-rate alternative does not materially improve the primary mean log score. No fitted alternative was installed in the actor or scheduler.

| Landmark | Primary survivors | Later solves | Expected events over observed exposure | Observed / expected |
|---|---:|---:|---:|---:|
| 60,000 | 95 | 58 | 58.41 | 0.993 |
| 80,000 | 59 | 28 | 28.18 | 0.994 |
| 100,000 | 48 | 20 | 20.15 | 0.993 |
| 120,000 | 34 | 12 | 12.34 | 0.972 |

## Connecting calibration to the scheduler claim

The longest observed per-level attempt in the primary cohorts is 204,490 tokens; no genuinely observed continuation beyond that point exists in these inputs. The original latent-world extrapolation can create much longer solves. The following supplementary CPU test reproduces the original 100 mixed-set bootstraps, exact latent draws and identical hold tapes, changing only extrapolated costs. Every empirical completion is retained, including recorded completions above a chosen cap.

| Extrapolated completions allowed | Shipped score | Proposed score | Relative gain |
|---|---:|---:|---:|
| No upper limit | 47.82 | 55.58 | +16.2% |
| Only at <=160K total level tokens | 48.93 | 35.42 | -27.6% |
| Only at <=200K total level tokens | 48.85 | 40.14 | -17.8% |

These caps are conservative sensitivity worlds, not claims that a level becomes impossible at 160K or 200K. A sampled extrapolated solve beyond the cap is replaced with an unresolved level. The candidate's beliefs are unchanged, so this tests transfer when its indefinite-tail belief is wrong. The no-limit result reproduces the prior audit exactly. All three worlds use equal token budgets, the same benchmark/deadline clock, and an unchanged out-of-fold policy.

Across all sampled latent extrapolated level costs, 25.1% exceed 200K; their 10/50/90 percentiles are [16267, 111642, 296319]. These are all latent costs, not just completed levels. Thus agreement on the ordinary observed tail cannot by itself validate the counterfactual benefit from very long or deeper unseen completions.

## Interpretation and next action

The prior zero-tail sensitivity was deliberately pessimistic. Real data show substantial late progress, so that world should not be mistaken for an empirical finding that old levels never solve. At the same time, a well-matched rate over 80K to 200K does not imply that the rate remains constant indefinitely, nor that it applies to deeper levels never reached in the logs. The proposed scheduler's gain still changes sign when the untested extension is removed.

Keep the current allocator. The next discriminating experiment is a bounded continuation of still-unsolved levels near 200K tokens, with a fixed additional token budget and recorded rule/plan state. It should measure late completion yield rather than tune another priority curve against assumed future solves. This would require a separately bounded inference run; this CPU task did not launch one.

## Verification and artifacts

Five behavioral tests pass, including future-outcome mutation proving held-out forecasts do not change, unknown-label handling, exact horizon boundaries, a worked censored-exposure rate, and a KM oracle. Verified identity isolation and censoring labels for 991 forecast cases across both cohorts and all four landmarks. Previous audit artifacts and all hashed inputs remain unchanged.

MANIFEST.json freezes landmarks, horizons, folds, models and decision rules before the main forecast evaluation. SUPPORTED-RANGE-MANIFEST.json separately records the subsequent range sensitivity. RESULT.json contains predictions, actual outcomes, exposure, fit metadata and stratified summaries. SUPPORTED-RANGE.json retains all 300 paired range simulations. VERIFICATION.json, tests.log, run.log and supported-range.log preserve checks and execution evidence. DECISION.json records the supported inference.

Reproduction commands from this directory: `python3 -B -m unittest -v test_tail.py`, `python3 -B calibrate_tail.py`, then `python3 -B supported_range.py`. Completed-output scripts refuse overwrites; use a fresh copy/destination. CPU only, with no inference, training, GPU launch, queue write, submission, solver change or promotion.
