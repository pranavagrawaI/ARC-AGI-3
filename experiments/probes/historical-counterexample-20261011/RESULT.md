# First detection probe: a real contradiction, a later failure, and an empty check

The local CPU probe reproduced a historical contradiction in an agent-written simulator, then distinguished it from a failure that historical replay could not detect. It also reproduced a historical validator reporting zero mismatches while making zero simulator calls. No new contestant inference, training, or game interaction was run.

Source: unhinted dfranzen `out-control`, game `s5i5-18d95033`, second level. This run completed one level in total. The case was selected deliberately after reading its simulator code; these counts do not estimate failure prevalence.

| Recorded candidate | Historical cases with successor predictions | Historical contradiction | No-op without a successor prediction | Subsequent observation |
|---|---:|---|---:|---|
| sim4 | 10 matches | Rejects the observed S- at transition 33 | 1 | Incorrect N+ prediction at transition 34 |
| sim5 | 11 matches | None found | 1 | Same incorrect N+ prediction at transition 34 |
| sim6 | 12 matches | None found | 1 | No subsequent action available |

These comparisons concern the four object states produced by the contestant's own extractor. They do not check full rendered frames, objectives, or terminal conditions. A returned `None` rejects an edge in the contestant's planner; it is not silently counted as a predicted unchanged state.

## The first historical counterexample

At transition 33, the recorded S- action shrinks the sky-colored object and moves two connected objects. sim4 returns `None`, rejecting an action already observed to change the state. The candidate checks moved objects against the shrinking object's OLD rectangle, producing a spurious collision. The agent's next revision, sim5, uses the final rectangles and reproduces this transition correctly. Both definitions were written before the next game action.

This is an executable positive case for a historical detector. It is not evidence of incremental benefit from our feedback: the original agent already diagnosed and repaired it without our intervention.

## Why historical agreement is insufficient

sim5 matches every selected historical successor it predicts, but its next N+ prediction moves the orange and sky objects down by three cells. In the recorded result those objects stay put. The agent's own execution guard catches this and halts its planned sequence after one action. It then changes the connectivity traversal in sim6: paths through objects outside the relevant direction no longer transmit movement.

We cannot claim that historical replay would have detected this rule error earlier. Transition 34 was new evidence for sim5. We also cannot claim prospective repair success for sim6: the run records no further game action after that revision.

## The empty validator

The original sim6 checking snippet, first visible at request-log line 119, selects actions with dictionary keys such as `(57, 12)`. Its parsed action suffix is actually `57, col=12)`. No key matches. With the recorded action strings, executing the original snippet reproduces:

```
checked, mismatches 0
```

It considers 16 transitions in its declared range but calls sim6 **zero times**. The probe marks this `INVALID_EMPTY`. The handmade structured-action adapter selects 13 modeled actions: 12 exact successor matches and one observed no-op on which sim6 returns `None`. Three other clicks fall outside the eight modeled controls.

Fixing coverage makes the check meaningful, but finds no additional contradiction in sim6 on this sample. It cannot currently support a score-gain claim.

## Verification and next decision

The probe executes the original extracted function definitions without editing their semantics. It cross-checks all 16 printed transition rows against saved frames and action receipts, and reproduces the separately logged sim5 predicted and observed tuples. Source SHA-256 hashes, exact generated snippets, frames, and comparison rows are preserved. Runtime is under one second locally; there are no model calls.

The first detection stage is complete. Keep the coverage safeguard and use the historical S- contradiction as a positive fixture. A matched continuation packet is prepared at request 107, before sim5 exists and before transition 34 occurs. Its feedback contains only transition 33 and requests an actual comparison count. The control retains the original request, and neither arm has been sent.

The next informative model test is whether explicit historical feedback reduces repair cost relative to the already capable ordinary continuation. It must preserve the original model and context and count inference cost. This case alone does **not** meet the agreed another-day criterion: we have not demonstrated incremental prospective prediction or decision improvement under the intervention. No numerical probability of score gain is justified.

Reproduce from repository root:

```bash
solvers/tycho_nooa/.venv/bin/python experiments/probes/historical-counterexample-20261011/probe.py
```
