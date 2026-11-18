# Deliberation diagnostic: thinking activated, no improvement on these cases

All 24 calls completed on Qwen3.8-27B, CoreWeave FP8, for **$0.037598**.
All frozen hashes match and every attempted request has a preserved response.
No retries, gameplay, Actor continuation, deployment change, or Kaggle run occurred.

| Arm | Valid deliveries | Expected judgments delivered | Mean latency |
|---|---:|---:|---:|
| Direct, thinking off | 8/8 | 8/8 | 2.09s |
| Thinking on | 6/8 | 6/8 | 17.50s |
| Thinking off, explicit commitment/grounds | 4/8 | 3/8 | 2.36s |

These are four authored cases (scope and shared-prediction faults, each with a sound
control), repeated twice per arm. They are new wording of familiar failure classes,
not held-out games or unaltered full Actor contexts. Direct and thinking arms share
the same prompt and 2048 completion ceiling. The explicit arm adds a wrapper request.
The direct ceiling is higher than in earlier screens; cross-run improvement cannot
be assigned to a single cause. Both repeats within a case use identical requests;
no independence guarantee or broad reliability estimate follows from two samples.

## Thinking mode

All eight thinking-on calls reported positive reasoning tokens; thinking-off calls
reported zero and no reasoning message. Thus the requested mode actually changed.
Six completed critiques had the expected judgment and identified the relevant gap
or correctly permitted the sound action. Two thinking-on calls hit finish_reason
length at 2048 completion tokens without a final critique, both on the second
sound-control samples. They are delivery failures, not inferred judgments.

Thinking averaged about 8.4 times the direct latency without improving the measured
judgments on these cases. This establishes a problem with this bounded mode for
frequent criticism, not that deliberation is useless or that a larger budget would
also fail. No budget was increased after seeing truncation.

Provider accounting caveat: summed reported completion tokens are 9115 and reported
reasoning tokens are 10402 for the thinking arm; several individual detail counts
exceed their supposed totals. Preserve these as provider-reported, inconsistent
fields, not an exact decomposition. All completion_tokens counts are <=2048 and
reported costs reconcile to the ledger. Positive reasoning reports plus distinct
reasoning output establish activation, but exact internal token allocation is
unresolved. We graded final answers only, not private reasoning text.

## Explicit commitment statements

This diagnostic exposed a real error: on the first sound scope case, the critic
described the Actor as assuming a click failed due to a lack of movement context.
The Actor had said only that one click produced no visible change, kept other clicks
available, and selected one movement probe. The critic then demanded another click
before moving. Its objection attacks an invented assumption and imposes an
unnecessary prerequisite: a harmful recommendation if followed, with no Actor
adoption measured here.

On the second sample it accepted that same sound case, but its commitment statement
again added an assumption: that click actions are valid. Keeping an action class
available is not establishing its effectiveness. The final label was right while
the representation of the Actor's commitment was wrong.

Four explicit-arm responses failed delivery: one extra closing brace and three
responses that retained the old top-level schema instead of the requested wrapper.
Their raw critiques remain inspectable; all four contain the expected judgment.
Counting raw semantic judgments gives 7/8 for this arm, but only 3/8 were both
correctly labeled and deliverable. Appending an overriding schema to a prompt that
already demands exactly five fields is an interface confound; these format failures
do not establish that explicit target identification intrinsically fails.

## Quality beyond labels

Direct-arm fault critiques consistently identify overgeneralization or
non-discriminating evidence; sound cases remain unchallenged in both samples.
Some consequences are stronger than warranted: describing CLICK as a primary
mechanism or suggesting its removal prevents discovery should remain conditional.
Thinking-on's first scope critique says the observation falsifies that specific
click, although no visible change alone need not exclude a hidden or delayed effect.
Its second suggests checking hidden state changes without establishing an observable
way to do that. These are qualifications to otherwise relevant critiques; the
8/8 and 6/8 figures are expected-judgment counts, not scores for perfect prose or
every epistemic claim.

## Decision

Keep the direct mode as the diagnostic baseline. Do not promote thinking-on for
frequent review based on this run. Explicit identification is useful for exposing
misreading, but its statements need grading and its schema needs a single owner.
The result does not support the earlier suspicion that disabled thinking alone
explained the critic failures: the direct mode can pass these cases consistently,
whereas the same model previously missed other short commitments.

The next useful boundary is robustness to wording and full captured context, with
the faulty commitment held constant, before another live Actor intervention.
Do not equate a successful short reasoning screen with retained Actor improvement.

Reproduce mechanical checks with:
`deployment/kaggle/.venv/bin/python experiments/probes/epistemic-critic-local/deliberation-v1/audit.py`.
CASES.json and the immutable card preserve criteria; each numbered request/response
and EXECUTION.json preserves final content, timing, usage, failures and settings.
AUDIT.json computes delivery/label counts, not subjective epistemic correctness.
