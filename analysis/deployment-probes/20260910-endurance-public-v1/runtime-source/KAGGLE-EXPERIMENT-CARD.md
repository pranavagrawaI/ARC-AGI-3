# Kaggle contingent-controller contrast

Question: On the competition-attached RTX PRO 6000 runtime, under the same reasoning-off FP8 engine, does exposing the bounded contingent interface change useful work per model crossing against the same solver without it?

Arms: `off` and `on`, set by `ARC_CONTINGENT_ARM`. In `off` the two tools leave the documented surface, their prompt sentences are withheld, the `contingent_controller` packet key is absent, the dispatcher fast path is skipped, and a direct call fails closed. Discovery arm A, solver arm `loop`, model, sampling, action authority, prediction gates, context bounds and time limits are identical. Nothing else differs.

Measures: real committed actions, transitions and levels; model calls and tokens per committed action; model and tool wall time; contingent proposals, selected branches and stop/rejection reasons from `contingent_events.jsonl`; factor identity; lifecycle and request reconciliation. The primary contrast is actions per model crossing; completion is reported but is not powered at this budget.

Cap: one notebook, public ls20, two arms, twenty actions and 3,000 seconds of admission per arm, with a 3,600-second launcher timeout, inside the 10,800-second envelope. No MTP, submission, retry or automatic promotion.

Budget basis: the inherited ten-action budget risked the `on` arm never proposing a plan. Twenty actions doubles the exposure. The time budget moves with it because it must: v1's off arm committed ten actions in 1,046 seconds of gameplay (104.6 s/action), so twenty actions project to about 2,091 seconds and an unchanged 1,800-second admission would have stopped the arm at `wall_budget` near action seventeen. Admission is 3,000 seconds to carry headroom over that projection, and the launcher timeout additionally covers server startup (about 400 seconds observed) and teardown. Worst case is 7,450 seconds of the 10,800-second envelope; the projection is about 5,500.

Known limit: twenty actions may still be too few for a contingent plan to arise. A run where the `on` arm proposes no plan is uninformative about the controller, not evidence against it, and must be reported as such rather than as a null result. The projection above is from a single ten-action arm and the `on` arm's pace may differ.

Stop and invalidation: stop before the successor arm on manifest, factor, reasoning, containment, request-accounting, worker, scheduler, timeout or cleanup failure. Invalid if arm identity, reasoning-off state, request provenance or terminal ownership cannot be established. A nonzero wrapper status is failure until separately diagnosed.

Decision: promote only on attributable useful-work improvement with clean lifecycle evidence; retain `off` on no benefit; prefer the simpler arm when practically tied. Preserve negative and incomplete results. Estimates inside a plan remain model conjectures, not calibrated probabilities or RHAE.
