# Do our speed/intelligence metrics track reality? (2026-10-09)

Scripts: `validate.py` (T1-T4, full tables in `out.md`) and `stuck.py` (A, B). Both are pure offline and use `benchmark.json` only. No GPU was used.
Corpus: 14 public runs from 3 harness generations. Hidden scores are known for 5 systems (Duck 3.6 0.67, Qwen3.8 1.47, KF 2.96, P1 3.08, dfranzen 23.87).
The F submission (1.58) is excluded because it has no benchmark.json. The wave-2 control never ran (control-gate.json: timeout, 2026-10-08 08:14Z), so the lean arms are read against dfz-control truncated to 3600 s.

## Verdict
- **Rate metrics track hidden; per-token metrics don't.** Public score, solve hazard and human-baseline actions per game-hour each fit hidden with log residual sd 0.24-0.28.
  Leave-one-out, their worst miss is 1.7-2.2x. For raw levels it is 2.7x. Levels per Mtok, human actions per Mtok, orientation share and actions/baseline do not track hidden (residual 0.57-0.87).
- **Paired per-level cost is a mechanism detector, not an outcome predictor.** It sees large known mechanism changes at t = 2-8. It misses or reverses the sign on
  outcome changes that came from depth or action efficiency.
- **The level-grain noise floor replicates across stacks.** Between byte-identical reruns, sd(log) is 0.47-0.66 for tokens and 0.38-0.49 for actions.
- **Makarov replicates on our data.** In 79-82% of stuck levels the stuck run had already taken the action that finished the level in a sibling run, a median 72-84 min earlier.
- **The stuck-stream lever is small.** dfranzen's solve hazard stays at 18-24% per 10 min through minute 45 of a level. Only 10% of tokens are spent past minute 60.
  Freeing them buys at most about 3.5% levels at elasticity 0.35, minus the levels they still produce. Drop it.

## T1 — public metric vs hidden (n = 5 systems; dfranzen has large leverage)
| metric | spearman | log-log slope | resid sd | KF->P1 predicted / actual | LOO worst miss |
|---|---|---|---|---|---|
| public_score | 1.00 | 1.02 | 0.28 | 1.16 / 1.04 | 2.20x |
| hazard (levels per attempt-hour) | 1.00 | 1.72 | **0.24** | 1.29 / 1.04 | **1.71x** |
| human-baseline actions per game-hour | 1.00 | 1.14 | 0.25 | **1.15** / 1.04 | 1.70x |
| levels | 1.00 | 1.77 | 0.38 | 1.30 / 1.04 | 2.67x |
| levels per Mtok | 0.80 | 3.09 | 0.85 | 0.42 / 1.04 | 5.96x |
| tok/s per game (speed alone) | 0.90 | 2.59 | 0.51 | 3.03 / 1.04 | 2.72x |
Rank was never the problem, since raw levels already order the 5 systems correctly. Magnitude is the problem. Public score has slope 1.02, so hidden is roughly proportional to it.
Speed alone predicted 3x for P1 and P1 delivered 1.04x. Per-token yield alone predicted 0.42x. Only the product (a rate) lands near the truth.

## T2 — noise floor (byte-identical reruns, levels solved in both)
| pair | n | tokens sd(log) | actions sd(log) | seconds sd(log) |
|---|---|---|---|---|
| dfz control vs r2 | 88 | 0.66 | 0.38 | 0.90 |
| KF2 vs C1 | 32 | 0.47 | 0.49 | 0.55 |

## T3 — known differences, paired per level (B/A, t in parentheses)
| A -> B | known | levels | hazard | tokens/level | actions/level |
|---|---|---|---|---|---|
| KF2 -> dfz-control | hidden 8.1x | 44->107 | 3.03 | 0.83 (-2.3) | 0.69 (-3.4) |
| dfz-control@60m -> lean-cap4 | pair-1: tokens/level loss | 27->17 | 0.63 | **2.82 (6.4)** | 0.84 (-1.7) |
| dfz-control@60m -> lean-prune | same | 27->17 | 0.68 | **3.79 (7.7)** | 0.95 (-0.5) |
| KF2 -> G1 | 12x board costs actions | 44->41 | 0.93 | 0.89 (-1.1) | **0.75 (-2.3)** |
| Duck3.6 -> Qwen3.8 | hidden 2.2x | 17->22 | 1.29 | 1.37 (1.2) WRONG SIGN | 0.82 (-0.8) |
| KF2 -> P1 | hidden 1.04x | 44->51 | 1.16 | 1.13 (1.2) | 0.99 |
| dfz-control -> s1 | inconclusive | 107->98 | 0.87 | 1.07 (0.8) | 1.04 (0.9) |
Qwen3.8's hidden gain came from fewer actions per level (scored quadratically) and from reaching more levels. Paired cost only sees levels both runs solved, so it is blind to both.

## T4 — early read (8 runs with a 2 h horizon)
At 30 min, Spearman with final levels is 0.54 (28% of final levels are in). At 60 min it is 0.85 (58%). The earlier 15/17 within-game result does not make a 30-min cut safe. 60 min is the earliest usable screen.

## A — decisive-action contact (non-move finishing actions; clicks matched within 2 cells)
| family | stuck/solved pairs | stuck run had taken the decisive action | median min held after contact |
|---|---|---|---|
| dfranzen (3 runs) | 22 (12 levels) | 18/22 (82%) | 72 |
| flash-next (7 runs) | 123 (16 levels) | 97/123 (79%) | 84 |
Caveat: the finishing action is the last step of a sequence, so contact is necessary but not sufficient. This is an upper bound on "had the key".
Recognition latency in solved levels: dfranzen first takes the decisive action after a median 55% of the level's actions and about 99% of its time.
Deliberation comes first, then the solve lands in the final batch. The intelligence deficit is judgement and sequencing, not discovering actions.

## B — solve hazard by minute-in-level
| minute | dfz open | dfz solved next 10 min | dfz tokens past | KF-family solved next 10 min | KF-family tokens past |
|---|---|---|---|---|---|
| 0 | 358 | 51% | 100% | 11% | 100% |
| 10 | 169 | 27% | 49% | 20% | 85% |
| 20 | 115 | 18% | 34% | 18% | 68% |
| 45 | 55 | 24% | 16% | 8% | 39% |
| 60 | 34 | 12% | 10% | 9% | 26% |
| 90 | 19 | 5% | 3% | 7% | 9% |

## Not done, and why
- Surprise-rate logging. Asking the Actor to state an expected effect is an intervention, not instrumentation. It changes the prompt, and the questioner arms showed that prompts which make the
  model commit can entrench it (memory: external-questions-entrench). It belongs in an arm with its own control. A passive proxy is needed first.
- The wave-2 lean arms against a real control: the control never launched.
