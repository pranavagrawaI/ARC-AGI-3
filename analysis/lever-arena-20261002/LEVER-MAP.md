# Search for the lever: getting the next level solved

Our strongest working hypothesis is that the decisive advantage comes from **acquiring a correct, usable understanding earlier and preserving it through execution**. Better models, observations, inference throughput, retained code and resource allocation can multiply each other's effects. This is a diagnostic hypothesis, not an identified description of Tufa's current private solver.

The authenticated Kaggle snapshot at 2026-10-02 08:49:36 UTC shows Tufa 52.51, Yi-Chia 48.07 and third place 34.77. The gap is substantial. Its mechanism is not public in the material retrieved. [Leaderboard snapshot](sources/leaderboard.json); [live leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard).

## What we can change

These are controllable mechanisms, not a proposal to turn every knob at once. Source paths below refer to our current `solvers/m2_memory/src/inference` tree unless specified.

| Frontier | Controls we own | Specific current seam | Cost or interaction |
|---|---|---|---|
| F1 Speed and Action Economy | Model precision, speculative decoding, concurrent streams, prefix/recurrent-state reuse, batching | Notebook serving launch; `ARC3_MAX_ACTIVE_STREAMS`; `agent/tool_agent.py::_PriorityGate` and `_maybe_handover` | More streams can evict the context they need; faster generation helps only if its extra work is useful. |
| F2 Tools, Math, Representation | Images, scale, animation, numeric grids, segmentation, generated simulators/search | `MULTIMODAL_UPSCALE`; `agent/vision_context.py`; sandbox `FrameView`; `transition.frames` | Richer views cost tokens and prefill. Generic measurements should not impose incorrect game semantics. |
| F3 Harness and Interface | Observation fidelity, receipts, state freshness, reset/terminal handling, timeout recovery | `ToolAgent._run_python_tool`; `run_sandboxed_python`; runtime-state serialization | Wrong feedback can invalidate an otherwise correct plan. Guards must not classify an uncertain probe as useless. |
| F4 Metacognition and Judgment | Base model, targeted training, sampling, hypothesis testing/revision, thinking limits | Model input/checkpoint and provider config; `LOCAL_ANALYZER_TEMPERATURE`, `LOCAL_ANALYZER_MAX_OUTPUT`; `agent/prompts.py` | Stronger or longer reasoning costs time; more thought can reinforce a false premise. Training is a high-cost hypothesis, not a short-window default. |
| F5 Learning, Evaluation, Retention | Reasoning history, helpers, data, eviction/retrieval; run identity and repeated evaluation | `ARC3_REASONING_HISTORY_KEY`, `ARC3_CONTEXT_DRAIN_TOKENS`, `ARC3_PERSISTENT_FUNCTIONS_SCOPE`; sandbox `save_data/load_data`; manifests | Retained facts can be stale or wrong. A storage API helps only when the model uses it. Evaluate all runs rather than selecting a lucky maximum. |
| F6 Portfolio and Resource Allocation | Admission priority, continuation budget, deadline-dependent fading, reconsideration/restart policy | `max_runtime_s_per_game`; `ARC3_PRIORITY_TAIL_FADE_FRACTION`; `agent/priority_scheduler.py` | A promising game can unlock several later levels; starving an initially slow game can destroy a future breakthrough. Changing admission also changes cache behavior. |

A working decomposition is: available time → useful computation → correct discoveries → retained and executed solutions → scored completions. It is not a fitted equation. Locate the failed conversion before optimizing it.

## What Tufa has actually shown

Their July/Milestone 1 [technical disclosure](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133) attributes the main historical development gains to better base models and multimodality. It reports disappointing bespoke-tool experiments and limited ablation of many choices. Its generic Python interface lets the model inspect observations, calculate and act. These are retrospective author observations, not isolated causal estimates. [Retrieved primary text](sources/topic-717133.md).

Their [public repository](https://github.com/Tufalabs/duck-harness) also provides 25 games × 20 passes with diagnostics. Repeated evaluation is a demonstrated asset we can emulate at an affordable scale. It does not establish today's development procedure. Their [July blog](https://tufalabs.ai/research/duck-harness/) suggests model capability affects solvability while harness efficiency affects cost, based on a limited comparison.

We have no verified evidence identifying the current leader's model/checkpoint, post-training, perception, memory, sampling, scheduler or repeat-score distribution. A better private checkpoint, several mundane improvements, or a better development process remain competing explanations. The historical public release is not the current private leader.

## Ranked hypotheses

1. **Make the right evidence usable by the model.** A missing animation frame or misread object can make an otherwise capable solver appear unintelligent. A better model can also infer rules the current one repeatedly misses. Historical evidence is strongest here. Test: identify a necessary observation at a failed branch; establish whether it was absent, incorrectly rendered, misread, or correctly understood. Change only the implicated representation first. A factual improvement on frozen prompts is a diagnostic result, not yet a gameplay improvement.
2. **Turn a correct discovery into reusable executable knowledge.** Save a verified rule/helper and use code to calculate or search, rather than regenerate it in prose. Test: find a fact the model previously demonstrated correctly, then lost or rebuilt. Restore that specific item. If the failure predates learning, or persists with the information present, memory is not the explanation there. M2 Memory remains a candidate, not a predetermined winner.
3. **Deliver more useful attempts within the time budget.** Measure decode, cold prefill, cache rebuild, queue, tool, startup and idle delays. Peak tokens/second and cached-token percentage are insufficient. Our inherited dfranzen solver already contains substantial serving optimization; first establish the remaining recoverable time. Match weights, workload and correctness checks when comparing engines. Speed still helps a reasoning-limited solver by allowing more attempts, but this must be measured.
4. **Fund continuation where it can unlock further levels.** Late levels have larger score weights. A scheduling improvement may outperform a local reasoning change. Test starvation and continuation curves, accounting for cache residency. Offline trace rescheduling cannot reveal counterfactual game completions. It is a screening tool; a full matched run is required.
5. **Escape an early wrong hypothesis without discarding valid knowledge.** A fresh interpretation or alternative plan at the same observed state may help; lowering temperature or repeating a generic critique may not. Test whether contradictory evidence was available and ignored. Compare bounded reconsideration against continued reasoning at equal cost. Environment reset is not a free independent lottery ticket.
6. **Improve the search-and-retention process across experiments.** Use repeated evidence, exact configuration identity and held-out confirmation to keep real gains. This can create an accumulating advantage without an exotic runtime trick. It also protects against optimizing a small, hand-picked diagnostic panel.

## Two verified constraints on the story

**The controls were not identical treatments.** Original runtime 7,254 seconds versus repeat 6,000 seconds changes the active 20% deadline-dependent priority fade. Actual logs first report this phase at 98.7 minutes versus 82.4 minutes. A common 100-minute retrospective cut does not equalize the prior allocation. This does not explain every disagreement or eliminate real stochasticity; it invalidates assigning the entire difference to randomness. [Exact evidence](b/deadline-confound.txt).

**Action-efficiency polishing has little remaining direct credit on these completed public levels.** The local scorer caps score by weighted completed levels. At unchanged completions, perfect efficiency raises mean score by only 0.734 points in control and 0.378 in repeat. Fewer wasted actions can still avoid deaths, save time and unlock additional levels; this bound does not apply to those benefits or establish hidden-set headroom. The calculation uses full runs of unequal duration, not a causal comparison; rounded input scores add up to 0.005 points uncertainty in each mean. [Calculation](c/score_decomposition.py); [independent check using archived scorer](verification.json).

The supplied histogram is observational. Artifact identity, modified copies, excluded entries, runtime effects and leaderboard selection remain uncertain. Its 3.93-point SD cannot estimate public-25 level-total variability or justify a universal 8-point improvement threshold. Do not reduce the substantial leaderboard gap to ordinary small noise, but do not infer its mechanism from rank.

## First move: a bounded diagnostic panel

Before another solver edit, construct at most 10 branch records: both existing controls for cd82, tu93, s5i5, plus bp35 and g50t. The first three are selected large disagreements; the latter two have agreeing incomplete outcomes (1/9 and 0/7 levels). They are diagnostic controls, not a representative evaluation sample.

For each record preserve the earliest consequential failure, exact model-visible prompt/observations and tool results, available helper/data evidence, action receipts, generated tokens, configured deadline, and any measured serving/admission delay. Mark absent telemetry unknown. Distinguish missing evidence, wrong inference, forgotten knowledge, execution failure, and insufficient compute; permit more than one cause or no conclusion. A board screenshot alone is not a restorable engine checkpoint: any continuation test must verify exact game/version/state restoration or be labelled prompt-only.

Decision order:
1. A mechanically reproduced necessary missing observation/operation gets the smallest repair and a regression replay.
2. Otherwise prioritize a mechanism supported in at least two distinct panel games and surviving a counterexample search. This is a triage heuristic, not a significance threshold. Compare completion-blocking importance, not just frequency of cheap errors.
3. A serving proposal must show material recoverable critical-path time; avoid deriving that from cached-token fractions. If evidence remains ambiguous, use one bounded probe to distinguish the top two explanations rather than adding both treatments.
4. After quota reset, operationally qualify the chosen exact artifact, then compare across all 25 games under identical configured deadlines, fade rules, weights/settings and resource budgets. Preserve the raw incumbent if evidence is inconclusive. The earlier three-repeat plan remains provisional, not a power guarantee.

This arena identifies the search space and the next discriminating investigation. It does not identify Tufa's private mechanism, run that branch-point audit, or authorize a speculative solver rewrite. No solver files or GPU jobs were changed.
