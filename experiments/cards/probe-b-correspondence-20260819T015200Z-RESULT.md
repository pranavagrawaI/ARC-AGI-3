# Result: probe-b-correspondence-20260819T015200Z

## Validity

Valid, after one repair. The card's `Invalid if` fired — 2 `ambiguous` per
condition, above the threshold of 1 — but it caught a defect in my **grader**,
not in the data.

`grade()` searched the last 400 characters for a colour token. Every answer
puts its verdict first and then spends its closing paragraph explaining why the
*other* body is ruled out, so the trailing text reliably mentions 5. Four
answers were scored `wrong(5)` and four `ambiguous`; hand-reading the preserved
`responses.jsonl` shows **all eight name the 4-body**. The card required raw
outputs rather than a verdict, and that is exactly what made the repair
possible.

Two process failures on my side, neither affecting interpretation:

- The first card (`...013819Z`) capped `max_tokens` at 1200. This is a
  reasoning model; it spent the whole budget on `reasoning` and returned
  `content: null`. Aborted after 1 call, superseded, preserved.
  `small-model-selection-flat-in-scale` had already recorded this exact trap. I
  did not apply it.
- This card is **1028 bytes, 4 over the 1024 limit**, and I verified size in the
  same command that launched the run instead of before it. Not edited after the
  fact.

## Observation

`meta/muse-glimmer-30b`, temperature 1.0, 4 trials per condition.

| condition | correct (hand-graded) | reasoning tokens | latency |
| --- | --- | --- | --- |
| raw opening frame | **4/4** | 3.8k–5.0k | 35–77s |
| pre-segmented bitmaps | **4/4** | 1.4k–1.8k | 17–22s |

The reasoning is correct, not just the verdict. Across the eight answers the
model independently produced every fact the live agent missed:

- the 4-body is congruent to the target by pure translation, "identical to the
  45-cell target region";
- the 5-body is its **mirror** — "a right-hand stem can never be made to
  coincide with the left-hand stem of the target by translation alone";
- the **40 vs 45** cell count — "it has only 24 top + 16 stem = 40 cells < 45";
- the exact displacement — "slid 15 right and 30 down", i.e. the 15-press
  solution Probe A confirmed;
- and, unprompted, the wall constraint separating the compartments.

The raw condition carried a legend naming the two bodies, the target, and the
two status bars. That is not a hint: the live agent had already derived all of
it and written it into `notes/world_model.md`. The raw prompt gives the model
strictly less than the agent had.

## Interpretation

**The card's third branch: `both high -> neither; the loop never asked.`**

This overturns the prior I carried in. I expected 0/4 on the strength of
`small-model-selection-flat-in-scale` and predicted raw <=1/4. The prediction
was wrong, and the two results do not conflict — they bound different tasks:

| | that probe | this probe |
| --- | --- | --- |
| candidates | 15–42 | 2 |
| evidence | digest, no win frame | one frame, fully visible |
| judgment | name an (invariant, mutable) relation | compare two footprints |
| score | 0/4 at 9B **and** 397B | 4/4 at 30B |

Two-way geometric correspondence, visible in a single frame, is inside this
model's competence and costs ~20 seconds. Abstract relation-naming over dozens
of candidates is not. The inverted-authority bet stands where it was measured;
it does not extend here.

**So the failure is architectural, not a capability gap.** The model that ran
the live episode could have answered this at any point in 60 turns for the price
of one tool call. Nothing asked it. The builder was summoned 30 times by a
trigger that fires on *transition* prediction error, and it spent its budget
where the trigger pointed — re-deriving motion rules it had already confirmed.
The one question that would have ended the episode was never on any pass's
agenda.

Segmentation halves latency and reasoning tokens but does not change the
verdict, so pre-segmentation is an efficiency, not an enabler.

## Licensed next action

1. **Layer 3 is not required for this failure.** Program-side enumeration of
   correspondence candidates was justified by a capability claim this probe
   refutes at N=2. Do not build it on this evidence.
2. **The fix is to make the loop ask.** A correspondence pass is a single call
   against the opening frame with a known-good answer rate of 8/8. Where it
   attaches — a LEVEL_START-style trigger, a builder sub-goal, or a program-side
   step whose output seeds `outcome()` — is a design question, open.
3. **`planner_key` remains a hard prerequisite** (Probe A part 2). A correct
   goal without it searches 200k nodes and finds nothing.
4. **Not licensed:** any claim that the model can do correspondence at larger
   candidate counts, on later levels, or without a legend naming the bodies.
   Contrast covers raw vs segmented only, on ar25 level 0's opening frame.

## Raw evidence

`scratchpad/probeB/responses.jsonl` (8 records: answer, reasoning, finish
reason, reasoning tokens), `probe_b.py`. Frame:
`runs/tycho-nooa-level-start-ar25-20260818T175557Z/workspace/ar25/level_0/turn_000.txt`.
