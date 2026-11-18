# Kaggle discovery factorial v2

Question: On the competition-attached RTX PRO 6000 runtime, does B, C, or D improve useful work over A under the same reasoning-off FP8 engine?

Arms: A=current; B=bounded measured-evidence suffix; C=one fixture-generated stable demonstration; D=both. The solver, model, sampling, action authority, prediction gates, context bounds, and time limits are otherwise identical.

Measures: real committed actions/transitions/levels; calls and tokens per action; model/tool wall; cache telemetry when reported; factor identity; evidence use; procedure creation; lifecycle and request reconciliation.

Cap: one notebook, public ls20, four arms, ten actions and 1,800 seconds per arm within the notebook's 10,800-second envelope. No MTP, submission, retry, or automatic promotion.

Stop and invalidation: stop before a successor arm on manifest, factor, reasoning, containment, request-accounting, worker, scheduler, timeout, or cleanup failure. A nonzero wrapper status is failure until separately diagnosed.

Decision: promote only a treatment with attributable useful-work improvement and clean lifecycle evidence; prefer the simpler tied treatment. Preserve negative and incomplete results.
