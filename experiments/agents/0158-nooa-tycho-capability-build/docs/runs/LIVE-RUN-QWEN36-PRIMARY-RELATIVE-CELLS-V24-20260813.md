# 0158 Qwen3.6 primary-relative cells v24 — 2026-08-13

## Outcome

The fresh non-resumable run executed three confirmed LS20 actions, then stopped
on an HTTP 408 during the fourth claim artifact.

| Measure | Observed |
|---|---:|
| Accepted claims / commitments | 3 / 3 |
| Actions / successors / verdicts | 3 / 3 / 3 |
| Verdicts | 3 contradicted replacements |
| Reciprocal contrast / G2 | 0 / 0 |
| Levels completed | 0 |
| Provider calls / reported tokens | 32 / 343,793 |
| Provider-reported cost | $0.24733248 |

The terminal scorecard is `PAUSED_FOR_REVIEW`, `resumable=false`. Raw evidence is
under `artifacts/live-ls20/20260813T120611Z-qwen36-contrastive-action-map-44`.

## API result

The primary-relative variadic API passed its precommitted operational measure.
All three commitment artifacts used `predict_cells(cell_after(...), ...)`, were
expanded into exact absolute `Replace` records, accepted by the host, and
followed by confirmed successors. Qwen did not spend a repair call discovering a
list wrapper or translating primary-relative offsets into frame coordinates.

This establishes procedural closure only. All three semantic predictions were
wrong. Qwen repeatedly claimed cells of the static 3x3 marker around rows 31–33,
while the actual directional changes were in the moving 5x5 `9/12` block around
columns 34–38. The verifier contradicted each with
`PRIMARY_REPLACEMENT_MISMATCH`.

## Action evidence

The confirmed selected actions were `1, 1, 2`. The public transitions show:

- action 1 moved the 5x5 `9/12` block up five rows;
- action 1 moved it up five rows again;
- action 2 moved it down five rows.

The model did not bind those observations into load-bearing translation
commitments. No reciprocal action-effect contrast closed, and no G1/G2 credit
was earned.

## Terminal failure

The fourth claim artifact completed three exploratory provider calls. The next
provider crossing returned HTTP 408. The frozen transport classifier treats any
status other than 429 or 500/502/503/504 as fatal, so the configured retry budget
was not used. The host preserved a substrate failure packet and closed action
authority.

The 408 is terminal for this scored run but not evidence against the relative
cell API or Qwen action learning. The run already supplies valid negative action
learning evidence from its first three confirmed transitions. Any change to 408
classification requires an offline transport probe and a new card/approval; this
run is never resumed.
