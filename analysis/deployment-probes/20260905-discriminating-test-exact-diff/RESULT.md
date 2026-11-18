# Discriminating-experiment probe, 2026-09-05

No successful grounded discriminating experiment was demonstrated. Two complete raw-frame conversations chose DOWN, executed it in the real local ls20 environment, and reviewed the result. Both invented their starting observations and later measurements. An exact-difference follow-up identified the main shift but exhausted its output budget before producing a valid committed plan. Do not add a restoration predicate on this evidence.

## Design and scope

Qwen qwen/qwen3.8-27b through OpenRouter, all six paid responses served by Reka, reasoning disabled. Each fresh conversation received the identical real before/after UP frames and the instruction to choose two plausible explanations, one action, distinct predictions, a measurement procedure and consequent decisions. DOWN and restoration were not suggested. Full grids were encoded as unnumbered row strings with hexadecimal palette symbols and zero-based coordinate instructions. These were standalone chat calls, not the previous CodeAct runtime: no Python execution or interactive measurement tools were exposed. This representation and tool-access difference limits comparison with earlier runs.

The selected action was applied only after saving the committed plan. Every environment reset and UP successor matched the recorded input exactly. The review received the full resulting grid, not a host-supplied verdict. Independent comparisons used all raw pixels and are preserved in ../20260905-discriminating-test-v2/MEASUREMENTS.json with reproducible measure.py.

## Outcomes

| Attempt | Grounded observations | Distinct committed predictions | Correct measurement | Calibrated conclusion |
|---|---|---|---|---|
| Raw A | Fail | Ambiguous: proposed shift vs toggle, but disjunctions overlap and premise is false | Fail | Fail |
| Raw B | Fail | Weak: movement/reversion vs unspecified other change, with overlap | Fail | Fail |
| Exact difference A | Partial: identifies main five-row pattern displacement, but still calls a 3x5 section 3x3 | No completed plan | Not reached | Not reached |

Raw A asserted (56,4) and (57,4) changed from 4 to 3 after UP and back to 4 after DOWN. Both cells were actually 9 in every frame. Its review nevertheless said the result strongly supported a boundary toggle.

Raw B asserted changes at (59,15) and (60,15), later claiming DOWN shifted markers to column 16. The actual values at those coordinates were respectively 4 and 5 in every frame, including column 16. Its review marked distinguished=true while describing both initial explanations as rejected. Neither run receives credit merely for writing a confident conclusion.

The actual UP changes were 50 pixels in rows 40-49, cols 34-38, plus two cells in rows 61-62, col 13. DOWN restored the main region exactly and changed two further bottom cells at col 14. Relative to the initial grid only (61,13), (62,13), (61,14), (62,14) remained different, all 11 -> 3.

The exact-difference follow-up kept the prompt and raw row strings and added a complete list of changed-pixel row/column/before/after values and the shape [64,64] to planning data (and would have added an equivalent list to review). The model correctly described a 5x5 pattern at rows 45-49 moving to 40-44. It rejected its own global-scroll explanation, repeatedly revised alternatives, and reached 4096 output tokens while still inside hypothesis_b. finish_reason=length; JSON was incomplete. No selected action executed, no review occurred, and the second start was not attempted because the precommitted card required stopping on malformed output. This is a protocol/budget failure; incomplete prose is not a committed scientific prediction.

## Failures and cost

The first directory, ../20260905-discriminating-test, preserves a DNS failure before an upstream response. A network-enabled attempt then obtained one response but encountered our action-validation bug: numeric GameAction values were compared with logged action names. It stopped before executing the chosen action. The corrected v2 compared names and was checked against the real recorded action list and response before running. The failed paid response is preserved separately and excluded from completed-run scores.

Six total paid responses cost $0.03854274: one in the failed harness attempt, four in the completed raw-frame attempts, one in the truncated exact-difference follow-up. TOTAL-COST.json includes every response and its finish reason. No automatic paid retries occurred; revised attempts have separate immutable cards and identities. All source/card pins passed final verification. The queued Kaggle package prepare.py --check passed; no launch or package edits were made. All probe processes have exited.

## Decision

The raw-frame probe did not isolate experiment selection as intended: it exposed coordinate-reading errors first. Exact pixel differences produced a more accurate account of the main change in one incomplete response, but do not establish improved experiment selection or causality. No new predicate or solver change is justified yet.

The next candidate should use the existing exact-difference evidence, give separate small output fields/budgets to observations and the final pair of predictions, and permit the model to say it cannot yet distinguish plausible explanations. It must not award credit for forcing two stories or for retrospectively declaring a winner. Test that candidate with actual code/measurement access before drawing conclusions about the full actor. This recommendation was not implemented or tested in this run.
