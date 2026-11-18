# packet-live-probe-v1 result

## Validity

Partially invalid, by my own fixture. `s5i5_fixture.build()` opens a ledger
objection at critic:3 with `claim_ref = review_id` ("critic:3"), so the
condition-A packets at critic:4, :5, :6, :7 carried a *pending* objection with a
ref that exists nowhere in `actor_claims`. The card specified an empty ledger for
A. Those eight calls cannot test prediction A. They are preserved and reported
because they show something else (below).

Otherwise valid: 19 of 20 calls completed (one read timeout at A/critic:6 s1),
reported model `qwen/qwen3.8-27b` on every call, zero reasoning tokens, no tool
calls, every finish reason `stop`.

Token calibration: reported prompt tokens are 1.18-1.27x the fixture estimate
once the ~875-token system prompt the estimate omitted is added back — inside
the card's 0.7-1.4 band. The fixture's estimate must include the system prompt;
corrected in `s5i5_fixture.py`. Packet-only real size at critic:7 is about 3,700
tokens, 3.6% of the 101,648 v8 read.

## Observations, verbatim

Clean condition A (empty ledger), critic:1-3, six calls:

| boundary | class | question |
|---|---|---|
| critic:1 s1 | objective_gap | Is the goal to manipulate the single cell in the bottom bar, or to clear one of the large colored structures visible on screen? |
| critic:1 s2 | other (factual) | Does row 63 have exactly one non-background cell, and is that cell at column 63? |
| critic:2 s1 | objective_gap | Does the bar fill stop before reaching the grid's left edge, and what specific state change signals level completion? |
| critic:2 s2 | objective_gap | If repeating the exact same interior click merely fills a bottom bar leftward, does that bar actually indicate level progress toward completion? |
| critic:3 s1 | mechanics | Since background clicks sometimes add two bar cells, what rule predicts whether your next commit will advance one or two? |
| critic:3 s2 | objective_gap | With 58 cells remaining and only ~9 actions affordable, what evidence would make you stop testing clicks and commit to an alternative completion mechanism? |

Contaminated condition A (pending objection visible), critic:4-7, seven completed:

| boundary | outcome | question |
|---|---|---|
| critic:4 s1 | cited pending ref, unbindable | What concrete terminal condition—full bar, specific pattern, or other state—ends this level, and what evidence confirms it beyond continued bar growth? |
| critic:4 s2 | empty | |
| critic:5 s1 | other | If the bar fills automatically without clicks, would continuing to click be a wasted use of the wall-clock budget? |
| critic:5 s2 | cited pending ref, unbindable | What evidence confirms that fully filling row 63 is the level's completion condition rather than just a proxy for another state change? |
| critic:6 s2 | objective_gap | Does completing the bottom bar alone trigger level progression, or is that merely one intermediate state? |
| critic:7 s1 | cited pending ref, unbindable | Do the 2-cell bar deltas (steps 13, 17) correlate with clicks on specific cells, rather than occurring randomly? |
| critic:7 s2 | other (perception) | What distinguishes an object cell from a background cell so I know I am clicking the center correctly? |

Condition B (critic:3 closed as `probe` 46 actions; claims consistent with it):

| sample | outcome | question |
|---|---|---|
| s1 | malformed (question with empty ref) | Why is this plan better than the live alternatives under the wall clock limits? |
| s2 | re-raise | If bar completion is the only valid path, why are the specific 13-targets static? |
| s3 | empty | |

Condition C (critic:3 closed as `withdraw`, CLICK 32 32 retired). Confounded:
the real trace's newest claims still say "KEY TEST … let me just fill it", so
the packet contradicted its own ledger.

| sample | outcome | question |
|---|---|---|
| s1 | prose, not JSON | (began: "The Actor has already tested that specific object interaction does nothing and concluded the bar is likely just a counter…") |
| s2 | re-raise | Is the 18/64 bar the objective, or just a timer? If it is a timer, what is the deadline and what triggers the win? |
| s3 | affordability | With only 81 seconds left and clicks costing over 30 seconds each, how can you test a different mechanism before the wall clock ends? |

## Interpretation

Prediction A as written (critic:3, :5, :7) is untestable from this run. On the
clean subset the packet critic found the objective gap 4 of 6 times and asked
one mechanics and one factual question; at critic:3 it split 1-1. That is
consistent with the packet preserving the diagnosis and not sufficient to say
so.

The contaminated calls carry an unplanned observation worth keeping: shown a
pending objection, the model copied its ref and restated it near-verbatim at
critic:4 and critic:5. That is the repetition the ledger exists to prevent, and
it argues for the design's central rule — never call the critic while an
objection is pending — more directly than the planned conditions did.

B and C: two re-raises in six, one of them in the confounded condition. Not
enough to trigger the mechanical-cooldown branch on its own; enough to keep it
live for the corrected run.

Two genuinely malformed responses in 19 (a question with an empty ref; prose
instead of JSON) against v8's one in 22. Small numbers, same direction as the
smaller-context critic being slightly less disciplined about the schema.

## Permitted next action

Result map: ambiguous → report per boundary, do not promote. A corrected card
(`packet-live-probe-v2`) with an empty ledger at critic:3, :5, :7 and a coherent
probe-closure condition is the permitted follow-up. No Kaggle run.
