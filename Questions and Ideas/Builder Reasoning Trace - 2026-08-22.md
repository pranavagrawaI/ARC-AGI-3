# Builder reasoning trace — 2026-08-22

## Judgment

Reasoning-on made the failure legible. Both Builders formed increasingly accurate
mechanics hypotheses, but neither converted its diagnosis into a bounded edit before
the 240-second stop. The immediate defect is **investigation without commitment**:
long serial re-derivation consumed the deadline after enough evidence for a small,
reversible test was already available.

This is returned provider reasoning, not an inference about hidden thought. Raw
records are under
`../analysis/builder-continuation/20260822-matched-coreweave-reasoning/`.

## Retained 8/16 arm

| Call | Observable thought and consequence |
| ---: | --- |
| 1 | Read beliefs and old mechanics notes; correctly selected turn-4 RIGHT as the target. |
| 2 | Isolated the exact three wrong cells: the model changed underlying `10` cells to `0` while observation preserved `10`. First probe failed because it treated `wmlib.frames` as a dictionary rather than calling it. |
| 3–4 | Corrected the API mistake, confirmed the three preserved `10` cells, and read `world_model.py`. |
| 5 | Reconstructed the moving color-5 shape and recognized the important mechanism: holes in a moving shape must reveal or preserve the underlying stationary color, not always paint literal `0`. This was already a sufficient minimal-edit hypothesis, but the call spent 70 seconds and requested another probe. |
| 6–8 | Lost orientation in frame/action indexing and repeatedly recomputed object positions. It noticed its expectations contradicted the observed slice, but moved into increasingly broad reconstruction rather than testing the three-cell repair. |
| 9 | Found a second analysis error: naïve color-5 segmentation selected the HUD/budget component. After filtering column 63 it recovered the actual observed C4/C5 bounding boxes and phase changes. |
| 10 | In-flight call was cancelled at the deadline; no returned reasoning or edit exists. |

Final belief: the remaining error concerned occlusion/background restoration as the
color-5 shape crossed the stationary color-10 block, while the apparent movement
confusion partly came from frame indexing and selecting the HUD component. The
Builder never changed the candidate, so it remained 8/16.

## Canonical 0/16 arm

| Call | Observable thought and consequence |
| ---: | --- |
| 1 | Read the same notes and model; selected turn-3 RIGHT. |
| 2–3 | Compared the observed delta with the rendered shape and established that the color-5 holes were actual `0` cells, while canonical `render()` incorrectly painted them as background `9`. |
| 4–5 | Verified the initial play area, distinguished moving shapes from stationary background/color 10, and confirmed that holes move with C5. |
| 6 | Converged on a concrete model: RIGHT dynamics were already correct; rendering was wrong. Proposed clearing the play area, restoring the stationary 10 block, repainting C4/C5, and painting C5's five internal holes as `0`. It spent 91 seconds reasoning and used its final completed call to re-check the already identified 10 block instead of editing. |
| 7 | In-flight call was cancelled at the deadline; no returned reasoning or edit exists. |

Final belief: `transition()` was substantially correct and `render()` was the defect.
This was a specific executable hypothesis, but the Builder timed out with canonical
`world_model.py` byte-identical at 0/16.

## What actually blocked the Builders

1. **No commitment threshold.** A falsifiable local hypothesis did not trigger a
   small edit; another observation always looked useful.
2. **No deadline awareness.** The model saw neither remaining wall time nor a rule
   reserving one call for an edit and verifier response.
3. **Representation mistakes consumed calls.** It confused frame numbers with action
   indices and initially selected the HUD as the first color-5 component.
4. **Old notes amplified confusion.** The inherited notes mixed correct observations
   with unresolved statements such as C5 not moving horizontally.
5. **Reasoning became recursive.** Repeated “wait, recalculate” passages rediscovered
   facts instead of externalizing a stable hypothesis and testing it.
6. **Reversibility was unused.** Candidate retention and automatic verification made
   small edits safe, but the Builder behaved as if it needed a complete theory before
   touching the model.

## Implemented treatment: model-visible budget

Do not ask for more reasoning tokens. Change the Builder control loop so useful
reasoning becomes action:

```text
first exact divergence
  -> write one falsifiable local hypothesis
  -> predict the verifier change
  -> make the smallest reversible edit on the next call
  -> keep or rollback from verifier evidence
```

The first part of this repair is now implemented. Immediately before every repair
model call, the host supplies a fresh notice containing:

- whole seconds remaining from the same monotonic deadline that enforces the Builder
  timeout;
- repair model calls remaining, including the imminent call;
- the separately reserved finishing calls; and
- an `investigate` or `commit` mode. The final three repair calls, or any call with at
  most 90 seconds remaining, enter commit mode and explicitly request the smallest
  reversible edit plus its predicted verifier effect.

The notice is telemetry, not a new clock: the host remains authoritative and still
cancels the Builder at its deadline. Mandatory outcome repair and objective-review
instructions are reasserted after the notice so budget information cannot displace the
actual assignment as the latest instruction.

Tests establish that the notice is refreshed on every repair round, the countdown and
mode transition are correct, the allocated host deadline reaches the model-visible
request, and an unresolved outcome checkpoint remains the final priority. These tests
establish plumbing, **not improved Builder accuracy**.

The rest remains proposed: once a hypothesis explains the named divergence, further
probes should require stating why the next edit would otherwise be ambiguous. We may
also supply transition records by action index directly and a HUD-excluding component
query if the matched replay shows those representation errors still consume the repair
budget.

The next test is a matched replay over these same workspaces with reasoning retained:
hold model, inputs, workspace, limits, and scoring fixed; vary only the budget notice.
Primary outcome: verifier accuracy (retained arm above 8/16; canonical above 0/16).
Mechanism outcomes: whether a model edit occurs before timeout, call index of first
edit, and whether the returned reasoning changes from continued investigation to a
falsifiable edit-and-verify commitment. A single improved run is evidence for this
pair, not a general Builder-performance claim.

## Matched result: prediction contradicted

Run: `../analysis/builder-continuation/20260822-matched-budget-awareness/`

The run was valid under the card. All 17 completed provider calls used CoreWeave,
contained nonempty returned reasoning, and carried the budget notice. Source
workspaces, model, task, evidence frontier, Builder system prompt, tools, order,
reasoning request, 12 repair steps, 4096-token ceiling, and 240-second cap matched the
prior reasoning-on control. It used zero ARC calls and cost $0.0313679.

| Arm | Before | After | First model edit | Consequence |
| --- | ---: | ---: | --- | --- |
| retained candidate | 8/16 | **0/16** | provider call 10, starting 171.8 s after call 1; notice said 70 s and 3 repair calls remained | Broad transition edit produced 0/16 and 72 wrong cells; a second broad render edit remained 0/16 and increased the first divergence to 198 cells. |
| canonical | 0/16 | **0/16** | none | Entered commit mode at 38 s remaining but spent the completed commit call on another probe; the next call was cancelled at the deadline. |

### What the returned reasoning shows

The notice changed behavior, but not in the predicted direction. In the retained arm,
the Builder had correctly identified the original three-cell occlusion error. At the
commit boundary it said it had little time, abandoned that local repair, and encoded a
speculative whole-transition rule (`+9` on the first RIGHT, then `+3`). The automatic
verifier immediately reported a fall from 8/16 to 0/16. The Builder recognized the
drop, but did not restore the previous function; it instead replaced the render rule
with a larger approximation, worsening the divergence from 72 to 198 cells. Its final
report admitted `confidence: low` and `simulation_accuracy=0.00`.

In the canonical arm, one investigate call began with 124 seconds left and consumed
85.4 seconds. Budget telemetry cannot interrupt or redirect reasoning already in
flight. The next notice switched to commit mode at 38 seconds, but the Builder still
chose another diagnostic probe, leaving no completed edit before timeout.

### Judgment and licensed next move

Reject budget telemetry as a default performance treatment. It solves only
*investigation without action*; alone, it converts hesitation into unsafe action. The
implementation remains available behind
`BuilderConfig(show_budget_telemetry=True)` for controlled research and is off by
default.

The next discriminating treatment is not a louder or earlier countdown. It is a
host-enforced transactional acceptance gate: checkpoint the last verified candidate,
apply one edit, run the verifier, retain only a non-degrading result, and otherwise
restore automatically before another hypothesis. If tested, combine it with an
actually local edit contract; do not infer that time awareness itself improves
accuracy.
