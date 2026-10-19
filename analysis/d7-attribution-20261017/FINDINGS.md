# D7 attribution, 2026-10-17

D7 is a high repeat of the unchanged baseline, not evidence of a D7-specific improvement. Sampling-driven differences in rule inference are the leading explanation supported by observable runs. D7-specific gameplay, hardware throughput, game order and scheduling cannot be attributed from the evidence available here.

## What was checked

Authenticated Kaggle submission details were collected for all seven recent completed draws. All episode lists returned empty. DownloadSubmission returned HTTP 403 for all seven, and still returned 403 for D7 after refreshing credentials. A credential refresh did enable pulling the raw notebook, so these two failures must not be conflated. Raw probe errors, corrected requests, successful metadata and the blocked downloads remain in this directory. The browser also timed out on two submission-page snapshots. No hidden logs or per-game results were recovered.

The current remote raw kernel is still version 1. Its 11 code cells exactly match the archived local raw notebook. The remote metadata pins both model inputs to version 1, disables internet, and identifies the same Docker image and RTX PRO 6000 machine shape. D2, D4 and D7 receipts all select this kernel/version. This excludes a new D7 source-code intervention; it does not establish identical effective runtime speed, scheduling, resolved environment or hidden inputs across executions.

## Is D7 unusually good?

- Unchanged-control scores: D0 23.87, D2 29.44, D4 24.97, D7 29.38. Mean 26.915, sample standard deviation 2.916, range 5.57 points. D2 already beat D7 by 0.06.
- B1's only hidden draw, 26.96, is 0.045 points above that control mean. D7's 2.42-point lead over B1 is smaller than the observed single-control standard deviation. This is descriptive, not a formal test or an equivalence claim.
- Against the mean of the neighboring controls, D1 m2_evidence is -0.415 points, D3 B1+c14 is -3.645, and D5 B1 is -0.215. These are distinct packages with one draw apiece, shared controls, and no randomized date assignment. Do not pool them into a causal treatment estimate.
- In the pre-existing external ledger of 26 plain-family scores, 9 are at least 29.38. D7 is only 0.45 sample standard deviations above that ledger's mean. That ledger includes selected public notebooks, mixed version bindings and a few runtime variations, so this is context rather than an iid reference distribution or calibrated p-value.

The evidence allows both ordinary execution variation and a real baseline advantage over a candidate. It does not currently distinguish them. In particular, lack of a demonstrated B1 benefit is not proof B1 is harmful.

## What generates variation in this solver?

The verified raw notebook sets temperature 0.7, top-p 0.95, top-k 20, and does not set LOCAL_ANALYZER_SEED. Its patch retains the analyzer's default seed of -1. It requests one pass. The true-submission branch disables request logs and enables minimal diagnostics. Thus these submissions are fresh sampled executions, not deterministic replay of an existing answer file. These facts do not prove sampling accounts for all of D7's difference.

Observable replication demonstrates substantial divergence without a code change. The two B1 notebooks have identical code cells, run the same 25 public game identities with the same 7254-second per-game cap, concurrency 25 and priority gate 10, and report the same installed overlay hashes:

| | B1 r1 | B1 r2 |
|---|---:|---:|
| Completed levels | 102 | 117 |
| Mean score | 46.3131 | 55.7879 |
| Generated tokens, including unfinished tails | 4,381,562 | 4,443,486 |

The higher run generated only 1.41% more tokens, yet completed 15 more levels. Six games contribute 8.683 of the 9.475 mean-score points: sc25, cn04, g50t, ls20, re86 and r11l. This is a concrete example of a few game trajectories dominating the aggregate, not uniform improvement or merely more generated text. It is public B1 replication, not a noise estimator for raw hidden scores.

The previously engine-checked cn04 level-2 comparison gives a mechanism. Both runs entered the same board. r1 spent 118.705 minutes and 35 actions without solving; r2 solved in 8.537 minutes and 39 actions. r1's search independently normalized connector/body coordinates and excluded valid simultaneous joins, then treated failure of that simulator as evidence against the all-ends goal. r2 noticed a placement prediction mismatch, computed a corrective move from the actual board, and completed. An archived offline replay of r2's 39-action segment from r1's failed prefix solves. This demonstrates that the failure was avoidable on that board; it is not evidence that a proposed model intervention would choose the successful route.

Sources: `analysis/cn04-b1-inspection-20261012/{findings.json,geometry-check.json,engine-replay.json}`, r1 transcript around lines 4784 and 5096, r2 transcript around line 5813. The trace review here checked these primary transcript passages; the engine replay was not rerun. No-action intervals confound scheduling, model reasoning and tool time, so the 119-versus-8.5-minute split cannot be attributed wholly to reasoning latency.

## Supported interpretation and next discriminating evidence

The baseline repeatedly reaches roughly 29, and D7 sits comfortably inside the observed spread. There is no supported D7-specific change to copy. A sampled trajectory can recognize the right rule, repair a model and unlock later levels, or settle on a wrong rule and spend its budget searching the wrong problem. Such differences provide a concrete route from identical software to different aggregate scores.

For D7 itself, a per-game score/action table and hidden inference log would distinguish completion changes, action efficiency, throughput or deadline losses, and game-order effects. Those measurements were not recovered, so claims that D7 was faster, drew easier games or had fewer errors remain untested. The existing lexicon screen is already aimed at the independently observed rule-induction bottleneck; no extra GPU run follows from the D7 aggregate alone. Repeated matched control/candidate draws would answer whether the baseline has a persistent advantage, but one B1 hidden draw cannot.

Reproduce all numeric comparisons with `python3 -B analysis/d7-attribution-20261017/analyze.py`. `RESULT.json` includes source hashes and per-game contributions. Investigation used API reads and CPU analysis only; no inference, submissions, launches, solver edits or queue changes.
