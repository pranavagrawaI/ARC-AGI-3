# Action efficiency on already-solved levels — 2026-10-13

**Decision: no, (b) is not supported as a >=20% hidden-score lever.** Even perfect action efficiency on every already-solved public level can increase mean score by only **0.377048–3.019823 points (0.796–6.598%)** across the five logged runs. The original incumbent control is **52.481598 → at most 53.215584 (+1.399%)**. Optimizing only execution after insight has no larger ceiling. This is a public, fixed-completions bound; hidden-specific headroom remains unknown. Evidence: [accounting.json](accounting.json), [games.jsonl](games.jsonl), [RESULT.json](RESULT.json).

## What the scorer actually pays

For completed level index L (zero-based), `q_L = min(115, 100*(b_L/a_L)^2)`. Uncompleted levels score zero. A game with N levels and K completed levels scores

`min(sum((L+1)*q_L)/sum(1..N), 100*sum(1..K)/sum(1..N))`.

The second term caps credit by completed depth. The brief omits the factor 100 in the level formula and the second game cap. A 1.3x action cut multiplies an uncapped level score by 1.69, but it need not increase the game score at all. Baselines come directly from each exact game/version's `metadata.json`, and match every logged `base_actions_per_level`; the median across the 183 distinct public levels is 60. Event `score` is a completed-level count, not action-efficiency credit. RESET and fatal actions are retained in action counts; the initial seeded RESET is not in history. Evidence: [levels.jsonl](levels.jsonl), [source-manifest.json](source-manifest.json), [verification.json](verification.json); primary scorer `/home/pranav/projects/arc-agi-3-reference/dfranzen-m2/tufa-arc-agi-framework/src/taaf/game.py:381-413` and installed `arc_agi/scorecard.py:150-209`.

## Whole public-25 score ceiling

| Run | Solved levels | Level score already 115 | Games at depth cap | Actual mean | Fixed-depth ceiling | Gain, points | Gain, relative |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 107 | 90 | 21/25 | 52.481598 | 53.215584 | +0.733987 | +1.399% |
| control-r2 | 100 | 80 | 19/25 | 47.389475 | 47.766522 | +0.377048 | +0.796% |
| pruning-control | 103 | 77 | 18/25 | 44.099971 | 45.523521 | +1.423550 | +3.228% |
| c14 | 107 | 79 | 14/25 | 45.767190 | 48.787013 | +3.019823 | +6.598% |
| b1-c14 | 108 | 76 | 14/25 | 45.003545 | 47.576912 | +2.573367 | +5.718% |

Evidence: [accounting.json](accounting.json), [games.jsonl](games.jsonl). Primary old controls have different configured horizons (7,254 and 6,000 seconds); later c14 is a serving reference, and b1-c14 changes prompt/history too. These are descriptive per-run ceilings, not a paired intervention estimate. E1 Swift uses a different model and is excluded. Exact input paths and SHA-256 hashes: [input-manifest.json](input-manifest.json). Original-control residual credit is confined to four games: ka59 9.437005 game points, sb26 5.569406 game points, ls20 2.685518 game points, wa30 0.657735 game points; divide their sum by 25 to obtain the mean ceiling gain. In particular, raw level inefficiency in VC33 contributes **zero** original-control game headroom because other level bonuses already reach its depth cap.

## Baseline, optimal execution, and halved exploration

| Run | All solved levels at human baseline | Post-insight optimum: generous upper bound | Exploration halved: generous upper bound | Only globally certified TU93 routes substituted |
|---|---:|---:|---:|---:|
| control | 53.215584 | 53.215584 | 53.215584 | 52.481598 |
| control-r2 | 47.766522 | 47.766522 | 47.766522 | 47.391032 |
| pruning-control | 45.523521 | 45.523521 | 45.426166 | 44.210384 |
| c14 | 48.787013 | 48.787013 | 48.606783 | 45.769363 |
| b1-c14 | 47.576912 | 47.576912 | 47.336684 | 45.121957 |

Evidence: [games.jsonl](games.jsonl), [accounting.json](accounting.json). Human-baseline efficiency sets every completed level to 100, so it equals the depth ceiling. For post-insight execution, assume knowledge is available at entry and an unrealistically short one-action finish on every solved level: that still cannot exceed the displayed depth ceiling. The true knowledge boundary and global optimal route are usually unknown, so this is an upper bound, not a measured counterfactual total. For exploration E <= A, halving E removes at most `floor(A/2)` integer actions. Replacing every solved level's A by `ceil(A/2)` (including necessary execution actions) supplies an intentionally overgenerous exploration-halving bound; realistic E gives smaller gains. These bounds hold completions fixed and do not assign extra completions to time saved.

All 525 solved (run, level) observations have actual action counts, exact human baselines, level scores, baseline-efficiency scores, and known/unknown oracle fields in [levels.jsonl](levels.jsonl). An independent BFS using exact timer, all sprite properties, internal phase and direction queues certifies **TU93 L1/L2/L3 shortest distances 18/10/19**, checking all four allowed actions. The same semantic roots and routes verify across all five runs: **15 globally certified level observations**. RESET cannot improve a shortest route in this deterministic game because it returns to the same start at positive cost. Evidence: [exact_bfs.py](exact_bfs.py), [exact-bfs.json](exact-bfs.json), [verification.json](verification.json). Original TU93 actions are 18/19/24: its L1 path is already shortest, L2 has 9 excess actions, and L3 has 5. These first-three route substitutions add **zero** original-control game credit.

The archived BFS also marks LP85 L1 distance 5 and VC33 L1 distance 3 as exhausted. We retain their conditional distances and scores, but do not call them globally certified: its keys merge states by final rendered frame, and mouse candidates cover components/logged clicks rather than all screen coordinates. SC25 L1 is exhausted with no goal despite logged solves, which further prevents treating the exhaustion flag alone as proof of a global oracle. Evidence: `analysis/oracle-progress-20261009/oracle.py`, graph paths in [levels.jsonl](levels.jsonl), and [oracle-traces.jsonl](oracle-traces.jsonl). Optimal actions outside the three independently certified TU93 levels are unknown.

## Waste accounting and the knowledge boundary

| Run | Solved actions | Deaths / RESETs | Prefix through last RESET | Interior / full-frame unchanged | Repeated visual no-op | Stale-error result blocks |
|---|---:|---:|---:|---:|---:|---:|
| control | 3864 | 10 / 11 | 320 | 154 / 61 | 16 | 40 |
| control-r2 | 3681 | 4 / 5 | 94 | 124 / 45 | 2 | 37 |
| pruning-control | 4149 | 15 / 16 | 626 | 180 / 80 | 19 | 64 |
| c14 | 4253 | 15 / 16 | 478 | 260 / 118 | 64 | 60 |
| b1-c14 | 4116 | 13 / 14 | 414 | 184 / 59 | 19 | 45 |

Evidence: [actions.jsonl](actions.jsonl), [levels.jsonl](levels.jsonl), [auxiliary-observations.json](auxiliary-observations.json). Prefix counts cover entire attempts discarded by later RESETs; they may have taught essential rules and are not all safely avoidable. No-op/repeat keys crop one boundary cell on each edge; that is an observational convention, not complete engine-state identity. These columns overlap and must not be added. Stale blocks cover all reached levels and count text-result blocks reporting `StaleStateActionError`, not uniquely identified refused calls or scored actions. A refused action creates no action receipt and adds no action-efficiency penalty, though recovery can cost tokens/time.

[phase-accounting.jsonl](phase-accounting.jsonl) provides a disjoint partition for every solved level: fatal/RESET actions first, other interior-unchanged actions second, remaining moves before/after a manually audited full-plan witness where available, otherwise phase unknown. **Only five full-plan boundaries were manually validated; earliest correct-rule insight and the exploration-versus-execution split for the other 520 solved observations are unknown.** A correct-looking thought does not establish a correct transition model. Even in audited cases, the prefix is before the explicit full-plan witness, not proof that every prefix action was wasted exploration. Global excess over shortest distance is known only on the certified TU93 levels.

| Explicit full-plan witness | Actions before witness | Actual finish | Verified finish | Execution saving | Public mean gain |
|---|---:|---:|---:|---:|---:|
| control vc33 L1 | 13 | 1 | 1 | 0 | +0.000000 |
| control ka59 L1 | 69 | 7 | 7 | 0 | +0.000000 |
| control sb26 L8 | 25 | 13 | 13 | 0 | +0.000000 |
| control-r2 tr87 L5 | 82 | 18 | 9 | 9 | +0.071177 |
| c14 cd82 L5 | 26 | 22 | 22 | 0 | +0.000000 |

Evidence: [plan-audit.json](plan-audit.json), original transcript paths and exact action/analysis-step anchors therein. VC33's final click aligns its bar with the gap; KA59 corrects its goal to both rings occupied; SB26 corrects its recursive reading/permutation before six swaps and SPACE; CD82 corrects the SW stencil from strict to inclusive before its own backward painting search succeeds. These are concrete cases of late goal/rule/model correction followed by existing planner execution. The 7-, 13- and 22-action finishes were replayed, but their global optimality is not certified; zero shortening found is not proof of zero waste.

**A concrete positive execution example:** TR87 repeat L5 has 82 actions before the corrected tile-versus-glyph plan, then 18 to finish. Replace five UPs with two DOWNs, four UPs with three DOWNs, and six UPs with one DOWN; preserve cursor movements. The 9-action replacement completes from the exact logged checkpoint. It cuts that finish by 50%, but the whole level only from 100 to 91 actions and earns **+0.071177 mean public points**. This is one verified compression, not a globally optimal suffix certificate. Source: game `tr87.py:974-1005` (bidirectional cycle); routes, replay receipt, and score calculation: [plan-audit.json](plan-audit.json).

Existing execution already spends no new model generation between many finishing moves: original control **82/107** solved levels have a final suffix of at least two such actions (median suffix **6 actions**). Evidence: [auxiliary-observations.json](auxiliary-observations.json); each suffix starts at the last action charged positive generated tokens. This measures batching/execution receipts, not insight, and means generic “execute the model's own path” largely duplicates current behavior.

## Replay-verified deletion scenarios

These hindsight changes shorten successful recorded routes, preserve failed candidates, and then replay the composed routes from each game's initial state. Original routes are retained wherever a deletion trial fails. They are achievable route counterfactuals under already-known logs, not evidence that the online model can avoid the probes while learning the same rules. The three scenarios are separate and must not be summed.

| Run | Remove discarded RESET prefixes: actions / mean gain | Remove visual no-change candidates: actions / mean gain | Remove repeated visual no-op candidates: actions / mean gain |
|---|---:|---:|---:|
| control | 320 / +0.133730 | 153 / +0.032232 | 16 / +0.004134 |
| control-r2 | 94 / +0.001557 | 113 / +0.035810 | 0 / +0.000000 |
| pruning-control | 625 / +0.804120 | 189 / +0.069161 | 19 / +0.000000 |
| c14 | 416 / +0.824923 | 255 / +0.375714 | 64 / +0.049701 |
| b1-c14 | 352 / +0.995474 | 177 / +0.291048 | 16 / +0.003474 |

Evidence: [deletion-checks.jsonl](deletion-checks.jsonl), [chain-checks.jsonl](chain-checks.jsonl), [chain-summary.json](chain-summary.json). All retained composed scenarios preserve the recorded completed levels. Savings can exceed the initial candidate count when the shortened sequence completes earlier and its remaining actions are omitted.

There are **11 rejected deletion trials**. Counterexamples include LF52 repeat L2: removing 10 visually unchanged actions, or just two repeated visual no-op actions, prevents completion; and TN36 c14 L3: removing its 62-action pre-RESET prefix prevents the remaining nine actions from completing. Exact failures remain in [deletion-checks.jsonl](deletion-checks.jsonl) and [RESULT.json](RESULT.json). They invalidate blanket visual dedupe and the assumption that every RESET reproduces the original entry state. The exact underlying hidden-state cause was not established for every failure.

The first deletion checker incorrectly stopped at any death, including intentional death followed by recorded RESET. It was fixed to allow that continuation, all checks rerun, and the superseded outputs retained in [evidence-pre-death-reset-fix](evidence-pre-death-reset-fix/). Final figures above use the corrected checks.

## Concrete mechanisms within the zero-token constraint

1. **Exploration:** retain tested transitions and failed hypotheses in the model's existing Python simulator; select distinguishing probes in code when candidate models are already available. This can prevent repeated probes without another model call. There is no demonstrated generic mechanism that halves the work of inducing missing rules.
2. **Deaths/RESETs:** execute only simulator-validated paths with explicit hazard and remaining-action state; refresh observations and replan using that simulator after divergence. It helps when the model is correct; an incorrect simulator certifies incorrect plans. Track reset semantics rather than assuming reset erases every hidden state.
3. **Execution:** compile the existing planner's route, stop immediately on death/level completion, and simplify cyclic moves through already-validated inverses. TR87 supplies a concrete receipt. Model-owned planner execution already exists; adding another planner does not automatically remove rule induction.
4. **No-ops/staleness:** dedupe known transitions with phase/animation-aware state, retain an override, and refresh local state references after every action. Avoid hard suppression from identical interior pixels. Existing stale and batch guards already refuse actions; a more aggressive refusal can add reasoning/recovery tokens.

No production changes were made. These are mechanisms and qualification requirements, not claims of measured online gains. Routes/engine source used for diagnosis must not become a hidden-game oracle.

## Hidden-score decision and limits

If hidden relative headroom resembled even the largest observed public ceiling, a 27 hidden baseline would rise only to **28.782 (+6.598%)**; transferring the original control's ceiling gives about **27.378 (+1.399%)**. These are conditional illustrations, not hidden forecasts. A >=20% lift from 27 needs **5.4 hidden points** and at least **3.031 times** the largest public relative headroom. We do not have hidden per-level actions, baselines or completed-depth credit to test that possibility. There is no rigorous hidden-specific <20% bound in these public logs.

Therefore **NO for treating action polishing on solved levels as a demonstrated >=20% hidden lever**. The public bound is small even under omniscient, unrealistic execution. Avoiding a death or saving decode time could unlock new levels; those gains are outside this fixed-completion question and are unknown, not zero. The findings support prioritizing earlier correct rule induction and additional completions over generic post-plan action polishing.

## Verification, scope, and reproduction

Checked **125 game histories, 25,275 per-action frames and level counts**, **525 solved level observations**, **500 game/scenario calculations against the installed scorer**, and **15 source-state-matched certified oracle routes**. Composed deletion receipts are in [chain-checks.jsonl](chain-checks.jsonl); five exact-checkpoint model-plan finishes are in [plan-audit.json](plan-audit.json). Evidence: [verification.json](verification.json), [input-manifest.json](input-manifest.json), [source-manifest.json](source-manifest.json). No training, model calls, spending, GPU launches or competition submissions occurred. Authored files are within this directory, and the summary is posted through the brief's prescribed board operation. One initial dependency import created a temporary Matplotlib cache under `/tmp`; subsequent imports were redirected into this directory's `.runtime-cache`.

From the repository root, run these commands in order (all Python invocations use `-B`):

```bash
python3 -B analysis/action-efficiency-20261013/analyze.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/exact_bfs.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/replay_audit.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/chain_audit.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/plan_audit.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/verify.py
python3 -B analysis/action-efficiency-20261013/finalize.py
```

`auxiliary-observations.json` records the transcript-block and final-suffix counters, recomputed by `finalize.py`. `plan-candidates.jsonl` is unvalidated triage material: analysis transcripts can span a level boundary, so automatic candidate labels are not used as evidence of when a rule was known.

Setup exception: the first shared-board inventory command used `python3` without `-B`; later Python commands use `-B`.

The prescribed one-paragraph summary was posted as `codex-action-efficiency` in shared-board thread **685**; receipt: [board-receipt.json](board-receipt.json).
