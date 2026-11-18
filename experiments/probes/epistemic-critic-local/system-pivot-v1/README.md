# System-level pivot test

Status: built and offline-verified; no paid calls or new live game continuation.
Prepared cards and inputs are in analysis/local-probes/20260911-system-pivot-test-v1.

The treatment is the complete frequent-review intervention, including no_concern
feedback and the Actor response instruction. The control has neither. Reasoning
stays off. The Actor may discover the correction independently, reject the critic's
suggestion, or retain the same action: credit depends on warranted behavior and
retention, not authorship. Critic overhead counts in wall-clock performance.

Both arms replay the same seven preserved LS20 actions and receive identical prior
notes and full observed transitions. These contain the earlier vertical-step2 claim.
No later correction, evaluator label, or game source is supplied. These are matched
new workers, not a captured in-process checkpoint. Their notes are context evidence;
old helper/Python state is deliberately absent from both. Thus the test measures
correction on resumed evidence, not exact counterfactual continuation of the earlier
Actor. The selected movement claim is one known error, not a representative sample
of epistemic failure classes.

Two pairs reverse arm order to reduce a simple order effect. Each arm has the same
time, action, and dollar limits. Extra review calls spend the same arm budget as
Actor calls. This tests system benefit, not equal-inference-budget self-review.

## Human grading

Copy GRADES-TEMPLATE.json to GRADES.json after the run. Grade Actor evidence against
observations available at that time, with exact output quotations and call IDs.
For this fixture, a warranted correction withdraws the asserted2-row vertical
displacement and uses corresponding observed positions to infer displacement5 in
the tested transitions, without claiming universal mechanics from those cases.
Record whether the correction persists into three subsequent executed decisions.
An unchanged action can qualify if its rationale or active model demonstrates the
corrected belief. Merely making three moves or printing incorporate is insufficient.
Missing evidence is unresolved. A later statement cannot retrospectively certify an
earlier decision. Use false_pivot for an observed unwarranted change of commitment,
and regression for return to the predeclared error, with supporting evidence.

The evaluator criteria above never enter the model's context. Grades remain human
judgments; exact-quote binding verifies provenance, not their truth. Source-bound
responses and notes snapshots permit independent regrading. Inspect false pivots
across the entire fixed observation window, not only until the first correction.

## Commands

Run offline checks:

```bash
deployment/kaggle/.venv/bin/python -m pytest -q experiments/probes/epistemic-critic-local/system-pivot-v1/test_score.py
deployment/kaggle/.venv/bin/python experiments/probes/epistemic-critic-local/live_system_pivot.py --fixture analysis/local-probes/20260911-system-pivot-test-v1/fixture --order baseline-first --output analysis/local-probes/20260911-system-pivot-test-v1/pair-a
```

The live command is the same command with --execute. Pair-b uses --order critic-first
and its own output directory. Start pair-b only if pair-a has no unresolved transport,
accounting or retirement failure. Each pair is capped at$1; both pairs together$2.
The earlier rate-limited run is preserved; these are new cards, never retries under
its old card. Stop on another provider failure without fallback or relaunch.

After completing evidence grading:

```bash
deployment/kaggle/.venv/bin/python experiments/probes/epistemic-critic-local/system-pivot-v1/score.py analysis/local-probes/20260911-system-pivot-test-v1/pair-a analysis/local-probes/20260911-system-pivot-test-v1/pair-a/GRADES.json
```

The scorer reports first correction, actions spent, retention-confirmation time,
false pivots, regressions and operational validity. It computes paired seconds saved
only when both arms have a confirmed correction and valid execution. Otherwise the
comparison remains censored/unresolved; no invented score replaces a missing repair.
Repeated sample size is too small for a broad efficacy claim. Inspect both pairs
and their evidence rather than averaging away reversals or invalid arms.
