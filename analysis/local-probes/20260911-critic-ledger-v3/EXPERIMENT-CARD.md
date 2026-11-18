# Measured-evidence critic v3

Question: Can a critic supplied with explicit mechanical measurements and the
current raw frame identify an actual contradiction without inventing corrections?

Posture: Saved-trace development regression. No Actor continuation or gameplay.

Contrast: Same six Actor claims and proposals, prefix cutoffs, model route and
inference settings as v1/v2. New evidence representation adds host-computed changed
cell counts, coverage and a lossless current frame; new instructions require a
literal comparison before at most two findings. This tests the combined candidate,
not the isolated causal effect of the prompt or representation.

Prediction: S5I5 actual catches exactly-one versus the observed 1,2,1 counts as a
literal contradiction; its second finding may address universal scope or objective
overreach. All three cautious controls get no false material objection. LS20 actual
identifies an evidence-supported position or unsupported bounce claim, without
inventing positions. CD82 overclaim identifies the unsupported generalization.

Measure: Raw JSON replies, literal_check, issues, decision_effect, evidence references,
word limits, and manual claim-grounding grades. Keep claim correctness separate
from whether an immediate exploratory action may proceed. Compare findings to the
frozen requirements, not merely status. Record calls, timings, provider and cost.

Cap: Six completion calls; 1024 output tokens each; temperature zero; thinking off;
45-second HTTP timeout per call; zero retries; $0.25 cost cap. Conservative request
reservation totals $0.06410616. No environment actions or Kaggle launch.

Stop: Complete six calls, or stop at the runner's first transport/provider/reasoning/
cost/unfinished-generation failure. Semantic or format failures are scored without
repair calls. Client timeout is not a guarantee of server computation cancellation.

Invalid if: Evidence beyond the specified cutoff is supplied; Actor text is modified;
rubric or desired answers enter messages; provider fallback; missing raw completion;
nonzero reasoning output; source or prompt modified mid-run. Exact artifact parity
to Kaggle is unverified; route is qwen/qwen3.8-27b with coreweave/fp8 only.

Result map: Missing S5I5 literal contradiction or a false material correction ->
reject candidate before integration, preserve failure. Those gates pass but other
cases fail -> partial capability, no integration. All semantic cases pass ->
qualify held-out and Actor-uptake tests only, not general reliability or gameplay.
Operational invalidity -> preserve attempted run; no automatic rerun.

Outputs: PREPARED.json, six request/response/timing files, STARTED.json,
EXECUTION.json, GRADES.json and RESULT.md in this directory. The implementation
is experiments/probes/epistemic-critic-local/ledger_screen.py; twelve offline
harness checks passed before freezing this card.

Authorization: User requested designing and testing on existing traces; prior
OpenRouter endpoint authorization applies to this bounded follow-up.
