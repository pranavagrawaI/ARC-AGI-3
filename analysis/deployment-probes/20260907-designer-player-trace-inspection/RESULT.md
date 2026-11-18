# Why 17 delivered calls produced one action

Read-only inspection of the repaired live recheck. No solver edits, paid calls,
or game-source inspection. Raw evidence remains in the reasoning-recheck directory.

## Finding

This was not an idle worker or an action gate rejecting the solver. It was
mostly model-request time spent inspecting, authoring helpers, and recovering
from interface misunderstandings. There are two concrete software-facing targets:
make transition evidence unambiguous, and expose the exact helper serialization
contract in a preflight. Neither requires shortening reasoning.

## Time and work

The 17 delivered requests account for 831.71 seconds of recorded model-call wall
time and 53,295 completion tokens (31,086 reported reasoning tokens). The relay
was occupied for 895.79 of the 900 admission seconds, including the unfinished
18th request. Total gaps between the 18 relay requests were 1.20 seconds; startup
before the first admission was 3.02 seconds. These are remote request/transport
timings, not GPU-kernel or pure decode attribution.

| Calls | Visible executed work | Model-call wall |
|---|---|---:|
| 1–6 | Status, grid histogram, full grid, connected components, decision packet, local windows | 186.25 s |
| 7–9 | Write world-model v1, replace with v2 and test directly, write notes | 149.58 s |
| 10 | Record a prospective experiment and submit one UP | 8.93 s |
| 11–15 | Inspect failed prediction/transition, rewrite helper, assertion failure, forbidden globals, search for initial evidence | 231.41 s |
| 16–17 | Regenerate the entire initial grid in code, diff it, then recheck current objects | 255.54 s |

All delivered responses ended with tool calls. Fifteen Python calls completed;
call12 had an AssertionError and call13 was rejected for forbidden `globals()`.
No delivered response hit the output-token ceiling (largest:10,713 tokens).
Call18 completed after shutdown and was not executed; its proposed code was
another inspection, not a second action. Do not count it as delivered work.

## 1. Transition data existed, but the model used the wrong shape

`solver_agent.py:979–1005` returns one **transition** per executed action.
Each entry includes `before_grid_rows` and successor `grid_rows`. Call11 treated
entries as individual frames and compared `traj[0]['grid_rows']` with
`traj[-1]['grid_rows']`. With one transition those are the same successor, so it
printed zero changes even though `action_results` said52.

Calls13–15 searched for `diff_summary`, earlier local variables and two trajectory
entries. The exposed handle instructions (`solver_agent.py:1460–1487`) advertise
`diff_summary` and describe diffing points in history, without giving this precise
transition schema. The actual notification has no `diff_summary`.

Call16 spent187.80 seconds generating a64-row literal for the initial grid.
Read-only evaluation of the actual trajectory method confirms its
`before_grid_rows` exactly equals the saved initial state. AST literal extraction
also confirms the hand-copied rows were accurate. Thus no evidence was lost and
the reconstruction was unnecessary: the existing exact predecessor was available.
This cost is observed; eliminating the full187.80 seconds is not a measured
counterfactual speedup.

## 2. Direct helper tests did not match the runtime contract

Call8 directly tested `predict(encode(grid),'UP')` successfully. Runtime
`prediction_feedback.py:129–134` JSON-normalizes the encoded state before calling
predict. Tuple coordinates become lists; v2's `set(z['walk'])` then raises
`TypeError: unhashable type: 'list'`.

Read-only replay reproduces direct success and normalized failure. Call11 itself
also reproduced this exact distinction. `model_predictions.jsonl:1` records the
failed preparation, so the per-action checker correctly found no committed
prediction. The helper authoring API describes syntax checking/loading but does
not expose a matching serialization preflight.

This was **not a prohibition on another exploratory action**. The qualification
gate applies to multi-action batches (`solver_agent.py:1327`); unresolved single
actions remain allowed (`prediction_feedback.py:193–195`). There was one policy
attempt and one submitted action, with no rejected second submission.

## 3. A generated repair promoted an unsupported claim to a fact

Call12 repaired JSON handling but also changed UP to row+1, claiming a confirmed
sprite-only52-pixel shift in the v3 helper's header. Its own assertion failed:
the current encoded player was still at[31,21], not the position assumed by the
assertion. Call13 then failed because it attempted `globals()` in the restricted
interpreter.

The saved transition gives an unambiguous counterexample to that claim:
all five0/1 cluster cells are unchanged. Fifty changes are in the color9/12
region (the block moved upward five rows), and two are color11→3. Call16's correct
diff and call17's fresh measurements exposed this; the helper on disk still
contains the earlier unsupported claim. The trace demonstrates both recovery
effort and a failure to keep claimed facts aligned with measurements. It does
not establish the true game objective or the identity of every object.

## Next bounded changes to consider

1. Expose a transition view with explicit before/after fields, machine-computed
   changes, and examples using the actual schema; remove promises of absent
   fields. Preserve raw evidence access. Test the first-action boundary where
   there is exactly one transition.
2. Offer a helper preflight using the same JSON normalization as runtime
   preparation. Report failures before action submission, but preserve the
   ability to take an exploratory single action without a valid world model.
3. Keep measured observations separate from hypotheses and tie “confirmed”
   claims to the relevant transition. Prefer a small new discriminating action
   once the decisive measurement exists over repeatedly reconstructing the same
   evidence. Do not enforce an arbitrary12-call or shortened-reasoning ceiling.

These findings support interface and experiment-loop work, not another serving
flag sweep. They do not measure the designer-player mindset intervention:
**only the loop arm ran**. No causal reasoning-on/off or solver-quality comparison
is available from this single trace.
