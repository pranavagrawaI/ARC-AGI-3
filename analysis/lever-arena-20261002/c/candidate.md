# Candidate C — Find the event that turns a game from opaque to solvable

The most promising basic lever is **making the first correct understanding arrive reliably, then preserving it long enough to finish**. Faster serving, visual fidelity, callable code and retained evidence are complementary means. The next investigation should find which link fails; a leader's score cannot identify that link for us.

A useful surprise: on our existing public runs, polishing action efficiency on already-completed levels has very little scoring headroom. Under the local scorer, an ideal score with the same completed levels is `100·L(L+1)/(N(N+1))`, because later levels have larger weights. Computing that ceiling across all 25 games yields:

| Full run | Observed mean score | Same-completions ceiling | Recoverable action-efficiency gap |
|---|---:|---:|---:|
| control | 52.482 | 53.216 | 0.734 |
| control repeat | 47.388 | 47.767 | 0.378 |
| S1 | 39.521 | 41.621 | 2.101 |

This is a descriptive decomposition, not a controlled comparison: full runs have different durations, scores are rounded, and the hidden set may have very different penalties. It rules out one attractive public-data story: fewer moves on the same completed levels cannot produce a massive improvement here. Preventing waste can still unlock additional levels, which is the larger opportunity. Reproduce with [score_decomposition.py](score_decomposition.py); underlying formula is `audit/dfranzen-repo/tufa-arc-agi-framework/src/taaf/game.py:381` and inputs are the three `runs/out-*/summary.txt` files.

## What we can change

| Established frontier | Concrete controls we own | Measure before changing | Important interaction |
|---|---|---|---|
| F1 Speed and Action Economy | Main/draft quantization, speculative steps, admission count, cache slots, prefill chunks, startup; batching and UNDO | Useful generated tokens per job second, prefill misses, startup time; score and level completion | More active games can evict prefixes and make a nominally faster decoder slower end to end |
| F2 Tools, Math, and Representation | Image resolution and format; exact board views, animation, segmentation, code/search APIs | Which decisive fact was available, missing, or incorrectly represented at the first failed hypothesis | Bigger images consume context and prefill; extra tools can confuse or distract |
| F3 Harness and Interface | Truthful action receipts, terminal/reset semantics, exception behavior, persistent functions/data, cancellation | Invalid actions, false feedback, lost functions, replay agreement, time spent recovering | An interface can silently destroy correct reasoning; stronger prompts then merely compensate |
| F4 Metacognition and Judgment | Probe size, evidence requirements, hypothesis revision, reasoning budget, sampling | First falsifiable rule, contradictory evidence ignored, first successful level, repeated failed plans | More deliberation can improve rules or waste the clock; greedy decoding can remove useful exploration |
| F5 Learning, Evaluation, and Retention | Saved data, function scope, context trimming, retrieval; repeats and independent confirmation | Rediscovered facts after eviction, saved-data adoption, cross-level transfer; uncertainty of matched score differences | Correct memory helps; stale conclusions freeze a wrong model. Evaluation quality determines what we retain |
| F6 Portfolio and Resource Allocation | Admission order, per-game continuation budgets, scheduler priorities, restart policy where legal | Marginal levels/score per additional minute; starvation and cache residency | Promising games may deserve continuation, but early apparent failure can precede a breakthrough |

Sources for implemented controls: dfranzen's [write-up](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md), its configuration/source in the local audit, and `solvers/m2_memory/README.md`. Most F1 changes are already inherited; describing them is not discovering an unused lever.

## What Tufa actually disclosed

The July [technical write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133), freshly cached under `../sources/topic-717133.md`, attributes its main development improvements to **better base models and multimodality**. It reports that bespoke tools did not help in those experiments and acknowledges that many design decisions were not thoroughly ablated. It deliberately exposes observations and action execution as Python, allowing a coding model to inspect, search and act; the interpreter was reset between calls. Thus permanent Python state cannot be inferred to be their original secret.

Their [repository](https://github.com/Tufalabs/duck-harness) releases 25 games × 20 passes with diagnostics. This is a demonstrated evaluation asset: many opportunities to distinguish a repeatable improvement from a favorable trajectory. It does not prove their current development process or private solver is identical.

The parent retrieved the authenticated current leaderboard into `../sources/leaderboard.json`: Tufa 52.51, Yi-Chia 48.07, third place 34.77. These are leaderboard outcomes, not mechanisms. The current private leader's model, training, inference stack, memory policy and scheduler are **unknown**. Their July [blog](https://tufalabs.ai/research/duck-harness/) compares stronger-model behavior and emphasizes the coding interface. Public evidence supports investigating capability × usable observations × efficient execution; it does not establish an undisclosed trick.

## Ranked explanations and ways to kill them

1. **Correct mechanics are acquired earlier and more reliably.** Models and multimodality have the strongest disclosed support. Our same-code examples include a game reaching six levels versus none, consistent with an early understanding bottleneck, although unequal compute and scheduling can also explain differences. The arena also found a runtime-budget confound affecting scheduler timing, so a common 100-minute cutoff does not create identical treatment. Inspect successful/failed first-level traces at matched token counts, retaining admission and scheduling history. If both already implement a correct transition model and fail later, this explanation loses priority. A later bounded experiment should change only the observation or model suspected at those branch points.

2. **More useful attempts fit into the clock.** This includes startup, recurrent-state/prefix reuse, retained helper code and fewer recovery turns. Dfranzen reports long-context cache improvements and shows why KV capacity alone is insufficient. Build a ledger from request timing, cache metrics and tool errors. If incumbent already reuses almost all prompt tokens and delivers similar compute before divergence, kernel/cache work cannot explain that divergence. Time savings have value only when extra computation increases completions.

3. **Learned mechanics survive transitions and context eviction.** Our new explicit memory API is a plausible intervention, not the answer by construction. Identify cases where the model demonstrably knew a useful fact or helper, lost access, then failed or rebuilt it. Restore only that item in a checkpoint continuation. If failures occur before learning, or persist with the item supplied, memory is not the bottleneck there. Adoption and stale-data errors must count against it.

4. **Compute is allocated to the right games and stages.** Dfranzen already has a scheduler. An alternative policy might unlock late-level weights; aggressive abandonment might instead eliminate delayed breakthroughs. Reconstruct continuation curves and test policies offline only as upper-bound heuristics: a frozen trajectory does not reveal the counterfactual after different scheduling or cache pressure. Require a matched full-run test before claiming gains.

5. **The team searches and selects better interventions.** Repeated evaluation and possibly more development compute can create a durable lead with no exotic runtime mechanism. This is a process lever we can copy. Refute claims of a particular intervention through held-out repeats and all-run reporting, not the best leaderboard submission.

The histogram cannot establish an identical-code sampling distribution: version/inputs/runtime identity, excluded teams, selection and submission rules require verification. Nor does “eight points” universally imply 90% certainty. One noisy new run versus a known baseline mean differs from comparing two noisy single runs. A lead warrants comparison; it does not identify a cause.

## First move and decision rule

**Do a branch-point audit of cd82, tu93 and s5i5 in both existing controls before another solver modification.** These large discordances are useful diagnostic cases, not an unbiased evaluation subset. For each, preserve the earliest consequential divergence, exact model-visible context, board/history, helper definitions, tool results, token expenditure and serving delays. Classify the first unresolved failure as missing observation, wrong inference despite evidence, forgotten knowledge, execution defect, or insufficient compute; allow unknown.

Choose one intervention only when a mechanically reproducible defect or a single explanation supported in at least two of these three cases survives an explicit counterexample search. This two-case rule is a prioritization heuristic, not statistical evidence. If nothing wins, retain the incumbent and prioritize repeated whole-run baselines rather than stacking speculative fixes. If retention wins, the already-built M2 Memory is the candidate to qualify; if initial perception/inference wins, adding memory should not consume the first comparison budget. GPU promotion still requires operational qualification and repeated fixed-resource evaluation across all 25 games, including the failures and unresolved cases.
