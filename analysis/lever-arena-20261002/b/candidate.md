# Candidate B: the lever is conversion of computation into verified progress

**The best-supported starting hypothesis is model–observation compatibility, multiplied by sustained useful inference.** It is not yet “Tufa discovered a secret memory mechanism.” Tufa's July technical writeup names better base models and multimodality as its main historical gains; dfranzen independently reports large model-switch gains. Our first job is to identify which conversion currently fails: seeing the relevant evidence, deriving the mechanic, retaining it, executing the solution, or funding enough attempts. The parent verified Tufa52.51, Yi-Chia48.07 and third place34.77 in the authenticated leaderboard (`../sources/leaderboard.json`); the leader's present configuration is unknown.

A useful diagnostic factorization is:

`expected score ≈ Σgames P(discover useful mechanics within budget) × P(finish levels | discovery, remaining budget) × action-efficiency credit`.

These terms interact. Faster decoding can increase the first two probabilities; perception can change the first from nearly zero; memory helps only if what survives is useful and correct. This explains why adding individually sensible pieces need not yield a better solver.

## What we can change

| Frontier | Concrete controls under our ownership | What would count as a mechanism improvement |
|---|---|---|
| F1 Speed and Action Economy | Serving engine/build, quantization, speculative steps, stream admission, prefix/checkpoint retention, prefill chunks, generation limit; action batching and guarded stopping | More completed useful inference under identical wall budget without changed answers or extra wasted game actions |
| F2 Tools, Math, Representation | Image scale/crops, animation access, numeric board views, segmentation, coordinate conventions, exact diffs, model-written simulators/search | Missing evidence becomes accessible; a sampled interpretation or computation is demonstrably correct |
| F3 Harness and Interface | Action availability, result provenance, state freshness, sandbox APIs, error recovery, output truncation, terminal/reset handling | Replayed valid operations stop failing; model receives the actual executed transition |
| F4 Metacognition and Judgment | Hypothesis testing prompts, uncertainty-triggered probes, contradiction detection, reconsideration timing, generation temperature | A wrong hypothesis is abandoned after discriminating evidence, with bounded added cost |
| F5 Learning, Evaluation, Retention | Context budget/trimming, retained reasoning/functions/data, source-labelled snapshots, repeat design, evaluation telemetry | A previously acquired fact/helper survives exactly when useful; comparisons hold treatment and resources fixed |
| F6 Portfolio and Resource Allocation | Progress priority, fade schedule, stalled-game allocation, thinking restarts, multiple candidate plans at the same observable state | More score per total job budget after accounting for lost cache and duplicated thinking; no imaginary free environment resets |

Current source anchors: `solvers/m2_memory/README.md`; dfranzen `inference/agent/tool_agent.py` (`_PriorityGate`, `_maybe_handover`); `inference/framework/solver.py:614–624`; archived `live-da-fr-WRITEUP.md`. These are controls, not all recommended changes. Memory's explicit JSON API requires adoption and cannot retroactively rescue arbitrary discarded globals.

## What Tufa actually demonstrates

Their [July technical writeup](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133) identifies base models and multimodality as major development gains, reports that bespoke tools disappointed, and admits many design choices were not thoroughly ablated. It also acknowledges omitted animation evidence and imperfect prefix caching in that release. Their [public repository](https://github.com/Tufalabs/duck-harness) provides 25 games × 20 passes and a trace viewer. That is a concrete methodological advantage: repeated observation makes unreliable wins and systematic failures visible.

Their [research article](https://tufalabs.ai/research/duck-harness/) distinguishes model-dependent solvability from harness-dependent cost, based on a limited comparison. This is suggestive, not a universal law. It supplies no evidence that today's private leader uses fine-tuning, an ensemble, persistent Python, a different model, or a special scheduler. The leaderboard motivates investigating these hypotheses; it cannot identify them.

## Ranked lever hypotheses and ways to disprove them

1. **The model gets a representation on which its existing capabilities work.** Highest evidence for large historical gains. Examples include animations exposing otherwise absent clues, image scale resolving objects, and code replacing mental coordinate arithmetic. Cheap test: audit failed-versus-successful trajectories on the same public game. Was the decisive observation actually present, accurate, and accessible before failure? An unavailable observation calls for a mechanical interface repair. If it was present and correctly described, “better perception” is not the explanation for that failure. Next GPU test must hold model/runtime fixed while changing only the implicated observation channel. Do not turn every grid into a large automatic object dump.

2. **More useful inference per job, with long context remaining reusable.** Dfranzen connects admission slots, recurrent checkpoints, prefix reuse and coarse trimming. This is a coordinated mechanism, not “increase concurrency.” His reported earlier cache fraction is 93.43%; our campaign reports about94%. Another large gain from eliminating prefill alone therefore needs measured evidence; cached-token fraction is not GPU-time fraction. Cheap test: classify server intervals as decode, cold prefill, cache rebuild, queue, and idle, then calculate an upper bound on recoverable time. Compare engines on identical weights, workload and output-quality checks. Unmatched976vs750tokens/s is not an isolated engine result.

3. **Correct discoveries survive until they pay off.** Persistent helpers, evidence and executable plans can compound across levels. But saved wrong beliefs compound too. Cheap test: reconstruct a real post-trim failure and show that a previously verified necessary fact was lost, then replay restoration without extra game knowledge. Memory candidate tests currently prove storage contracts and selected repaired calls, not reduced gameplay failure. Reject this as the first intervention if failures happen before any useful discovery or if data remains visible but is ignored.

4. **Compute is allocated according to marginal opportunity.** A breakthrough can unlock several subsequent levels; spending another100Ktokens on a hopeless belief can starve another game. Dfranzen already prices progress and preserves slots until cache invalidation. Cheap test: report spend since last progress, remaining levels, queue residence and later completions. Replaying priorities over existing traces can reveal starvation or inconsistencies, but cannot predict unobserved counterfactual completions. Fresh-hypothesis thinking at the same state is plausible; actual environment restarts consume actions and may not create independent competition runs.

5. **Targeted reconsideration raises the probability of discovering the right mechanic.** The lottery interpretation suggests early hypothesis lock-in. It also suggests a stronger base model, greater sample diversity, or better observations; generic “reflect harder” is not identified. Cheap test: pair successful and stuck traces, locate the first discriminating contradiction, and check whether the model acknowledged it. Only then design a narrow response. A review that repeats the same reasoning at substantial cost fails the mechanism test even if one game improves.

## A concrete reason to fix measurement first

The current control and “identical repeat” are not identical treatments throughout the100-minute comparison. The original notebook configures7254seconds; the repeat6000. Both enable a deadline-dependent20%priority-tail fade. Runtime logs confirm fade evaluation at98.7minutes for control versus82.4minutes for repeat. Thus a retrospective100-minute cut leaves different allocation policy in the final portion. See `deadline-confound.txt`. This does not explain all variance; it prevents assigning all differences to randomness.

The screenshot is also an observational cohort, not authenticated identical-code randomized repetitions. Selection into the10–40range, modified copies, runtime differences and best-submission ranking can alter its distribution. Its3.93-pointSD does not directly estimate our25-game level-total noise or establish an8-point improvement rule.

## First investigation and decision

**Build the matched-budget failure ledger before another solver modification.** For the existing controls, record game/level, exact configured deadline, admission intervals, generated tokens, context drains, tool errors, observation availability and first verified mechanic. Inspect the three largest divergent games before generalizing. Preserve raw measurements; modify the solver levers, never massage the evidence ledger.

Decision rule: choose one mechanically reproduced missing-information/interface defect if it prevents a necessary operation. Otherwise select a measured runtime bottleneck only when its recoverable budget is material. If neither explains the divergent games, move to a targeted hypothesis-revision experiment, not more memory features. At reset, keep configured deadlines and fade schedules identical in all compared arms, with raw control retained. No claim of a winning lever survives only a score increase without its predicted mechanism moving.
