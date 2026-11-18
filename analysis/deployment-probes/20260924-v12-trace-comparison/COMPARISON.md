# Where the sb26 traces diverged

The strongest supported failure is an overgeneralized negative observation: the failed Duck pass treated a destination click with no piece selected as evidence that the destination was inactive. It eventually discovered select-then-place at action 39, after spending most of its wall budget. The successful Duck pass explicitly reconsidered the earlier negative result after selection and discovered placement at action 8. This comparison supports targeting that inference and experiment choice; it does not establish that any proposed prompt or feedback change will repair it.

The claimed Duck timing improvement also narrows under phase accounting. The older successful life discovered placement sooner. Its large delay came after filling the slots in the wrong order, inventing additional targets, and correcting the arrangement. Duck's successful life avoided that detour. The observed speed difference cannot be assigned to batching or the other bundled changes.

## Sources and method

- Duck failed: v2 pass 0, `sb26-c7_k1dh4`.
- Duck successful: v2 pass 1, `sb26-wov3ulk5`.
- Critic-off successful life: `sb26-ef2qddnf`, the session after the reset in critic-off pass 1. Its earlier failed life is excluded from this life-to-life comparison.
- All three initial **visible grids are exactly equal**, checked from raw transitions. This does not prove equality of unobserved internal state.
- A step is an executed action. A call recorded at step N occurs after N actions, so calls with `step < milestone` are included through the action reaching that milestone.
- Placement is detected from a colored piece appearing at a middle slot center. All-filled means all four middle centers contain one of the four piece colors. L1 is the first transition with `after.levels_completed >= 1`.
- Model time is the sum of recorded `wall_seconds`, not a replay estimate or the total episode wall. Raw file SHA-256 hashes and per-step metrics are in [SUMMARY.json](SUMMARY.json).
- [CALL-LEDGER.md](CALL-LEDGER.md) contains all action sequences and rationales, full model code through L1 for the successful lives, and the failed pass's early calls through step 16 plus its later discovery and finish. Numbered calls correspond to one-based lines in each raw `model_calls.jsonl`; numbered steps correspond to `transitions.jsonl` lines for these sessions.
- Reproduce with `python3 analysis/deployment-probes/20260924-v12-trace-comparison/compare_traces.py` from the tall-humerus root. Raw source and runtime are unchanged.

## Discovery and completion

| Milestone | Duck failed | Duck successful | Critic-off successful life |
|---|---:|---:|---:|
| First actual placement | Action 39 | Action 8 | Action 5 |
| Calls through that placement | 135 | 30 | 26 |
| Model seconds through placement | 1554.6 | 313.5 | 266.4 |
| All four slots filled | Never | Action 15, correct order | Action 11, first two swapped |
| L1 | Never | Action 16 | Action 15 |
| Calls through L1 | — | 46 | 67 |
| Model seconds through L1 | — | 548.6 | 743.4 |

The failed pass had placed only three pieces when the wall budget stopped it at action 45. Its first discovery was at 07:58:09.995 UTC, 1577.1 s after the session start. It had spent 135 model calls before encountering the useful conditional behavior.

## The decisive difference in the first 16 actions

**Failed Duck, action 4:** `CLICK 22 29`, with no piece selected, changes zero pixels. In call 18 the solver writes that the marker is non-interactive. In call 29, after selection becomes visible, its carry-forward still says clicking marker cells is blocked.

**Failed Duck, calls 36 and 38, after action 7:** the solver explicitly considers clicking a middle marker with the piece selected. It therefore has the relevant experiment available in its current context. Call 38 rejects the candidate on visual geometry: “A 4x4 block can't fit in a 2x2 slot.” It clicks a top reference square instead, then empty space, then investigates selection sequences. This is stronger evidence for an unsupported inference overriding a cheap test than for the candidate being lost solely to transcript eviction.

**Successful Duck, call 30, after action 7:** it has also observed a marker click do nothing and a top-square click do nothing. It reopens the marker hypothesis because the earlier marker test had no piece selected: “Maybe clicking on a gray block only works when a piece is selected?” It executes `CLICK 28 29` with green selected. The action changes 53 pixels, fills the second middle slot, and removes the piece from its former location. That result licenses the subsequent select/place pairs.

**Within-run counterexample, failed Duck action 39:** the same `CLICK 22 29` used at action 4 now changes 53 pixels after a blue piece is selected, filling slot 1. Thus the earlier zero-change observation did not license a general claim that the marker was inactive. This is an executed counterexample in the existing run; no hypothetical model replay is needed to establish the overgeneralization.

Both successful lives revise hypotheses when an observation discriminates them. The older life tests selection followed by a slot click at action 5. The Duck success does it at action 8. The failed pass postpones that exact class of test until action 39.

## Accounting for the 21 fewer model calls

| Phase | Critic-off calls / model seconds | Duck calls / model seconds | Duck minus critic-off |
|---|---:|---:|---:|
| Through first placement | 26 / 266.4 | 30 / 313.5 | +4 / +47.2 s |
| Placement to all four slots filled | 19 / 168.6 | 11 / 181.0 | -8 / +12.4 s |
| All filled to L1 | 22 / 308.4 | 5 / 54.0 | -17 / -254.4 s |
| Total through L1 | 67 / 743.4 | 46 / 548.6 | -21 / -194.8 s |

The filling phase has fewer Duck calls but slightly more model time. The observed total model-time advantage comes from the final phase, where the trajectories differ materially:

- The older life places colors `[14, 9, 11, 15]` against target order `[9, 14, 11, 15]`. Calls 46–52 spend seven calls and 130.9 s between filling the fourth slot and trying SPACEBAR. Calls 47–51 investigate supposed additional corner targets, then recognize they are decorative frame corners. SPACEBAR does not finish the incorrect arrangement.
- Calls 53–59 spend seven more calls and 105.3 s checking the state. Call 57 identifies that the first two colors are reversed. Actions 13 and 14 select and swap them; action 15 confirms and reaches L1.
- Duck success establishes the color-to-slot mapping before placement and fills the correct order. It still makes a coordinate error: a two-action submission selects an already emptied source at turn 9; the host stops after that first no-change action. The solver corrects the source coordinate and continues. Once all four slots are filled, five calls and 54.0 s lead to a successful SPACEBAR.

Duck executes three complete two-action batches before L1; a fourth two-action batch is truncated after its first action. This removes three opportunities for an intervening model turn, but the number and time of hypothetical calls at those boundaries are unknown. It does not causally explain a 21-call difference. Correct initial assignment, fewer diagnostic detours, batching, and the representation changes are entangled.

## What this licenses next

The next candidate should target **state-conditional negative evidence**: a no-change observation applies to the tested state, and a newly selected object or changed mode can make the same target worth testing again. A useful fixture is the preserved unselected marker click followed by selection: does the candidate retain the selected-plus-marker test as unresolved and choose it before repeating already measured selection/SPACEBAR probes?

This should use the existing observation and action path, with no sb26 coordinates or solution injected. First replay the evidence and inspect candidate feedback locally; any model-based or Kaggle efficacy test needs its own bounded card and authorization. A wording change alone is not a demonstrated repair.

The host currently calls exact grid equality `blocked` (`chassis/host_contract.py`, `classify`). That label establishes no visible change for this transition, not universal action illegality. Both successful and failed traces saw this label, so the trace does not prove that renaming it would fix the inference. The smallest candidate worth evaluating is clearer state-scoped feedback and a changed-condition retest cue, not another unisolated timing bundle.

No runtime edit, model request, or new kernel was performed for this comparison.
