# Resource implications diagnostic: one useful objection, no reliable improvement

Twelve calls completed for $0.005696. All frozen hashes match, all attempts have
responses, required provider/settings checks passed, and no retries or gameplay
occurred. Three prompts reviewed identical evidence on four authored development
cases. This is neither a live Actor test nor a generalization estimate.

| Frozen case | Framework base | Plus revision self-check | Plus resource implications |
|---|---|---|---|
| Unknown counter threshold asserted as goal | no_concern: missed | no_concern: missed | no_concern: missed |
| Stop policy but preserve unresolved hypothesis | no_concern: appropriate | no_concern: appropriate | no_concern: appropriate |
| Four 8–12s calls before acting with 35s left | no_concern: missed | no_concern: missed | concern: relevant, but exceeds response word limit |
| Deadline treated as falsification; erase alternative | concern: detection, mixed revision | no_concern: missed | no_concern: missed |

The base deadline critique recommends prioritizing larger threshold tests, despite
the nearly exhausted admission budget. Correctly identifying a mistake does not make
its proposed revision appropriate. The resource critique identifies the 32–48 second
cost and permits bounded action under goal uncertainty. However, it describes the
admission cutoff as preventing any subsequent environment actions. The harness gates
new model calls; it is not equivalent to an immediate universal environment-action
ban. The critique is directionally useful but overstates the consequence, and fails
delivery validation. Count it as neither a clean correction nor a delivered benefit.

The added self-check did not fix circular or unsuitable revisions in this sample;
instead it returned no_concern on all four cases. Adding the resource explanation
elicited one relevant concern, but still missed two explicit epistemic defects.
One sample per arm cannot attribute this behavior to an instruction mechanism.
The shorter threshold wording differs from the previous screen, so cross-run
changes also cannot be attributed solely to the added rules.

## Environment-contract audit and candidate

The frozen v6 Actor already receives model_admission_seconds_remaining plus a
sentence that the budget stops further model calls, including analysis calls
(chassis/solver_agent.py:1554). Its epistemic contract also distinguishes stopping
a policy from proving its hypothesis false (:675). Thus the problem is not total
absence of these instructions. Their presence has not established reliable use.

The inspected v9 public launcher sets a 600-second per-game admission cap and
200-action cap, 16,384 maximum output tokens, thinking off and zero retries
(submission-20260910-v9/build.py). These are configured public-run constraints,
not verified universal competition limits. Official Kaggle overview and accelerator
announcement pages were queried on 2026-09-11 but returned no readable full text;
no remembered notebook-hour or VRAM number was promoted to a current rule.

The new shared-context.md is a candidate explanation for both roles: account for
obtaining evidence, interpreting it, and acting before resources expire; distinguish
wall time from environment actions; preserve hypotheses when abandoning policies;
do not manufacture certainty under pressure. It was supplied only to the resource
critic arm here. It is not wired into either deployed role, and Actor effects remain
untested. Dynamic limits must ultimately come from host telemetry, with source and
scope, rather than copying fixed numbers from this diagnostic into a general prompt.

## Decision

No promotion or automatic live relaunch. Preserve the resource explanation as a
candidate, but do not claim more instructions solved the epistemic problem. The
next discriminating test should inspect how the critic identifies the actual
commitment before judging it; this run cannot distinguish misreading the review
target from accepting its inference. Keep the present failures as development
evidence, and use fresh cases to assess any revision.

Raw requests, responses, CASES.json, EXECUTION.json and AUDIT.json retain the grading
grounds, usage and timing. Invalid output remains available for semantic analysis
but is not treated as advice delivered to an Actor.
