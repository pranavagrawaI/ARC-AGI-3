# Frequent live criticism: objections appeared, none reached the Actor

The first fresh LS20 replicate ran **265.52 seconds**, executed **nine actions**,
and completed **zero levels**. It received 29 successful Actor responses and 14
critic responses. The 30th Actor request failed with an upstream CoreWeave shared-pool
HTTP429. No retry occurred; the predeclared second replicate was not launched.

This is an interrupted diagnostic, not a valid completed endurance comparison.
The successful prefix supplies evidence about review delivery and observable
decisions, but the full-run operational criterion failed. Confirmed usage cost is
$0.3310046; the failed request retains an unresolved $0.09257982 reservation. Do not
report the confirmed amount as a fully reconciled final charge. All worker/relay
processes retired. Frozen source hashes remained unchanged; six preflight tests
passed before launch.

## What repeated review actually did

Reviews were due after two Actor calls or two actions, capped at20, using the same
Qwen model with thinking off. Fourteen reviews consumed80.56 seconds, averaging5.75s
and about30.3% of the session wall time. Ten delivered reviews were no_concern.
Four responses alleged a concern, but all exceeded the90-word non-quote limit and
were rejected. No invalid critique entered the next Actor request.

Thus delayed objections did appear, but **no critic-requested correction was
delivered**, so there is no demonstrated critic-driven pivot or retained repair.
Some early reviews had many subsequent decisions; lack of follow-up time alone
does not explain the absence of correction. The first real action took64.02s.

## Four proposed concerns, with their limits

| Review | Actions already taken | Proposed criticism | Assessment and delivery |
|---|---:|---|---|
| 3 / call9 | 0 | Arbitrary changed-cell thresholds do not identify which objects moved | Useful inferential issue, but critique adds unverified snake-growth counts and overstates the likelihood of its otherwise branch.98 words; ellipsis-joined quote occurs nowhere verbatim. Not delivered. |
| 11 / call33 | 6 | Several same-colored regions need not be one moving object | Relevant representation concern; not a demonstrated need to abandon the Actor's selected moving block, since referring to a9-block need not assert all9 pixels form one object.91 words; exact quote is in Actor31 rather than latest Actor32. Not delivered. |
| 12 / call36 | 7 | Watching bar depletion substitutes a countdown for objective discovery | Treats the Actor's 'may signal a goal' as a settled objective, then promotes the bar to a simple counter and prescribes a particular static structure. Mixed/overreaching,163 words; quote from Actor34 rather than latest35. Not delivered. |
| 13 / call39 | 8 | Stop repeating movement probes; mechanics are already known | Reasonable to question marginal value, but unjustified confidence that the mechanics were known.94 words; exact quote is in latest Actor38. Not delivered. |

CONCERN-BINDINGS.json records exact word counts and prior/latest quote locations.
Word validation rejected each first. Two would also fail the latest-output-only
source rule despite quoting actual earlier material. Review13 illustrates a
particularly narrow delivery miss (four words), but its content still needs grading;
relaxing a word limit would not establish a useful correction.

## The actual pivot was independent

The Actor repeatedly described vertical movement as2 rows and horizontal movement
as5 columns. It carried the claim into later decisions and durable notes. After the
eighth action, Actor41 recognized its own contradiction: corresponding rows had
shifted37→42,38→43,39→44. It corrected the vertical displacement to5, revisited prior
transitions, and appended an explicit correction to step_log.md. Actor43 retained
the corrected uniform step-size belief and submitted LEFT toward a possible0/1
target. The host executed that ninth action.

This is observable local repair with one later decision carrying it forward, not
three-decision retention. It was not caused by the rejected criticism. Review13's
claim that another mechanics check was redundant came immediately before this
independent correction, demonstrating why an instruction to pivot quickly cannot
assume the current mechanics are already established.

## Invocation and evidence boundary

All14 critic inputs contained the complete captured Actor request, and all10 valid
reviews entered the next Actor call. But quotation validation accepts only the
latest Actor output. Frequent review often landed after an inspection cell while
the substantive belief or policy was stated in the preceding action cell. Full
context availability and permission to cite it are therefore different things.
Earlier no_concern acknowledgements also recurred in Actor prose after failed
reviews; acknowledgement language cannot identify current review delivery.

The useful next repair to evaluate is source binding to explicitly identified
earlier Actor events, together with separate reporting of semantic quality and
brevity failures. This preserves exact quotation without requiring the critic to
pretend that the last inspection cell contains the governing commitment. It does
not license ungrounded pivots or guarantee that more objections help.

## Scope and reproducibility

Only LS20 is locally installed. The initially prepared LS20/S5I5 plan made no paid
calls and was superseded before launch by two LS20 replicas. The second replica
remains unstarted per the transport-failure stop condition. No Kaggle deployment
or competition launch occurred. The frozen deployed Actor was unchanged; the new
runner forwards action data explicitly to the local environment API (needed for
future click games, but no click gameplay was tested here).

The [experiment-card skill](/home/pranav/.codex/skills/experiment-card/SKILL.md)
says: "A retry or any change to the candidate, cap, endpoint, metric, or card
requires a new run ID and card." This run's own card additionally prohibits
starting the second replicate after unresolved operational failure.

Evidence: replicate-a/EXPERIMENT-CARD.md, RESULTS.json, AUDIT.json, raw requests and
responses, actions with full grids, snapshots, REVIEW-TIMELINE.json, ACTOR-CELLS.json,
CONCERN-BINDINGS.json and transport-errors.jsonl. The audit's empty baseline row is
inherited output formatting, not a baseline run. Recompute the prefix audit with
`deployment/kaggle/.venv/bin/python experiments/probes/epistemic-critic-local/audit_endurance.py analysis/local-probes/20260911-critic-frequent-ls20-v2/replicate-a`.
