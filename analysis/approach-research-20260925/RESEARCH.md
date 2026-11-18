# ARC-AGI-3 approaches: research and decision notes

Researched 2026-09-25. Scope: public primary literature, official reporting, and existing local evidence. No new model calls, experiments, solver changes, or submissions. Published scores below were inspected, not independently reproduced.

## Judgment

For this project, retain a minimal coding-agent baseline, test state-conditional evidence handling, and establish model/context capability before adding orchestration. Treat a persistent executable simulator as an optional tool whose decision benefit must exceed its construction and maintenance cost.

Distinguish three objectives: frontier capability, efficient local competition performance, and causal research on learning. A frontier API result is useful for the first and for diagnosing failures, but does not establish a locally deployable competition solution.

## Evidence map

1. **Capability and persistent context.** ARC Prize reports Astra at 62.7% with the Standard harness and 99.9% with the Provider Adapter on Semi-Private, at different reasoning settings. At matched max effort the scores are 62.7% and 98.6%. The adapter retains opaque reasoning and compacts long conversations. This motivates controlling context continuity; it does not prove that copying an adapter feature into a local model will reproduce the gain. Source: https://arcprize.org/blog/astra

2. **Minimal coding loop.** Duck exposes game observations through a Python REPL with image/text views and bounded context. Its July release reports a public mean of 1.6002 over 20 passes per game using Qwen3.6-27B-FP8. This is an inspectable competition-oriented baseline, not a disclosure of subsequent private improvements. Sources: https://tufalabs.ai/research/duck-harness/ and https://github.com/Tufalabs/duck-harness

3. **Historical replay and conditional execution.** Retrodict logs frames, tests hypotheses against history, and submits action queues with predicted cells. Execution returns to the model at a prediction mismatch or queue end; a playbook preserves learning. It reports 99.86% on all 25 public games using GPT-5.6 Sol max. This supports feasibility, not transfer of the score to our local model. Source: https://github.com/ryanbbrown/Retrodict

4. **Executable-model ablation.** Rodionov compares textual, executable, executable+simplification, and fixed-interface verification treatments on 25 public games. Executable-only is not uniformly better; the textual variant beats it in both GPT-5.5 settings. The complete verification treatment ranks first across the main comparisons, at increased cost. That treatment also changes interfaces/templates; it is not a pure verification-only effect. Stronger-model follow-ups complete the public set even with the textual variant. Source: https://arxiv.org/html/2607.15439v2

5. **Optional model construction.** Tycho compares four orchestration policies, one public run per policy under matched budgets, and reports actor-requested model-builder delegation best with Opus 4.8. With the selected policy, stronger models reach 100 public RHAE. A single run per policy limits precision. Source: https://arxiv.org/abs/2607.28287

6. **Exploration directed at representation errors.** OPINE-World synthesizes object extraction and transition rules, checks recorded transitions, and directs exploration using object-type adequacy. The paper explicitly leaves targeted component ablations for future work. Its full-system score cannot identify the individual benefit of two agents or ontology-error prioritization. Source: https://arxiv.org/html/2607.01531v1

7. **Systematic search.** Graph-based exploration records visited states and tested actions and routes toward untested state-action pairs. Its reported third-place result and median 30/52 levels are from the six-game 2025 Preview Challenge. This supports a historical exploration baseline, not a present competition ranking. Source: https://arxiv.org/abs/2512.24156

Community scores are generally public-set and self-reported. They are not interchangeable with semi-private or Kaggle results, and costs/run-selection methods differ. Policy: https://arcprize.org/leaderboard/community

## Reconciliation with our work

The current checkout already contains Tycho/NOOA, critic, model construction, and prediction/batching experiments. Generic advice to add these mechanisms would repeat work.

The September 24 trace comparison gives a specific failure with a within-run counterexample: a marker click without a selected piece changed nothing; the same click after selection placed a piece. The failed pass discovered placement at action 39, while the successful Duck pass did so at action 8. The failed actor had considered the relevant test, then rejected it on unsupported visual-fit grounds. This distinguishes inference/experiment selection from an explanation based only on forgetting.

In the two successful lives, critic-off versus Duck-batch required 15 versus 16 actions to L1, 67 versus 46 model calls, and 743.4 versus 548.6 model-call seconds. Different assignments and diagnostic detours confound batching attribution. The retry's other pass reached zero levels. The result is partial progress, not a solved game or reliable speed improvement.

Local sources:
- ../deployment-probes/20260924-v12-trace-comparison/COMPARISON.md
- ../deployment-probes/20260924-v12-trace-comparison/SUMMARY.json
- ../deployment-probes/20260923-v12-sb26-duck-batch-retry-v2/RESULT.md
- ../solver-diagnosis-20260912/REPORT.md (older configuration; historical failures, not a current universal diagnosis)
- ../contender-investigation-20260911/REPORT.md (pinned historical competitor review)

## Proposed next discriminating work

1. Use existing saved prefixes to test whether state-scoped feedback changes the next experiment. Include changed-condition retests and unchanged-condition repeats, so indiscriminate repetition cannot count as success. Do not inject game coordinates or solutions. Compare the same actor/context with and without the single treatment. Saved-prefix response changes are mechanism evidence only; gameplay benefit requires subsequent matched episodes.
2. Establish a minimal coding baseline with correct model formatting and context handling. Compare the strongest feasible local actor against the current actor under the same harness and wall budget. An API teacher can diagnose capability or generate training candidates, but is a separate experimental condition.
3. Evaluate prediction checks as one optional aid: propose a short plan, predict decision-relevant observations, execute to a check, and reconsider after mismatch. Reuse existing machinery. Replay consistency cannot certify unseen predictions, and unchanged pixels cannot certify unchanged hidden state.
4. Only escalate to persistent world-model construction when a defined downstream choice needs it. Measure construction time, actual planner use, and completed levels, rather than the existence of a model file.
5. Longer-term candidate: distill verified experimental decisions and belief revisions into a local actor using permitted data. Evaluate on held-out mechanics families, not merely new layouts. This is a proposed research direction; this review did not establish a reproducible training recipe with demonstrated benefit for our solver.

Report separately: runtime validity, levels/games completed, RHAE, action count, wall time, model calls, and treatment effect across matched repeats. No execution is authorized by these research notes.
