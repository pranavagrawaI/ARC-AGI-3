# Zero-cost epistemic-funnel census

## Judgment

At a **3-action provisional-goal obligation**, all five development traces expose one supported first broken control boundary:

- **2 goal-to-probe failures**: ar25 goal-gate and LS20 Gemma feasibility;
- **2 terminal-to-consolidation failures**: both successful ar25 symmetry traces;
- **1 selected-action-to-host failure**: LS20 Gemma reasoning capture.

The only failure family that repeats **across games and model lineages** is candidate objective → named discriminator. That makes the architecture-wide goal-to-probe hypothesis stronger than before, but this five-trace development census is not a cross-game reliability estimate.

The census also rejects a tempting simplification: model fidelity is not the first traceable failure in these five episodes, yet independent saved-model tests find defects in both games. Therefore, objective acquisition should be investigated first **without assuming** mechanics/planning will become sufficient afterward.

## Census

| Run | First supported boundary | Seq | Actor before failure | Builder before failure | Evidence class |
|---|---|---:|---:|---:|---|
| ar25 goal gate | candidate objective→named discriminator | 3 | unknown | unknown | legacy_trace |
| ar25 symmetry uptake | terminal consequence→durable consolidation | 16 | $1.5470 | $1.8351 | legacy_trace |
| ar25 symmetry replication | terminal consequence→durable consolidation | 19 | $1.3109 | $1.0819 | legacy_trace |
| LS20 Gemma feasibility | candidate objective→named discriminator | 3 | $0.1201 | $0.1120 | exact_checkpoint |
| LS20 Gemma reasoning capture | selected action→host execution | 2 | $0.0230 | $0.0526 | exact_checkpoint |

**Unit:** one deduplicated first observable failure episode per preserved run-level trace. Later failures remain visible in the decision table but do not inflate the aggregate. Missing cost is reported as unknown, never zero.

### Horizon sensitivity

| Obligation (actions) | Goal → probe | Handoff | Consolidation | Censored/no supported failure |
|---:|---:|---:|---:|---:|
| 1 | 4 | 0 | 1 | 0 |
| 2 | 3 | 0 | 2 | 0 |
| 3 | 2 | 1 | 2 | 0 |
| 4 | 2 | 1 | 2 | 0 |
| 5 | 1 | 1 | 2 | 1 |
| 6 | 1 | 1 | 2 | 1 |

The exact mix is horizon-sensitive. At five or more actions the short Gemma feasibility trace becomes censored, so the cross-game recurrence becomes **unresolved**, not contradicted. At one or two actions, the obligation fires before later evidence can reveal the timeout handoff. Three actions is therefore an explicit working policy, not an empirically calibrated constant.

## What repeats

1. **Goal mechanism, across ar25 and LS20.** A candidate objective exists, but no objective-discriminating experiment appears before the bounded obligation expires.
2. **Terminal consolidation, twice on ar25.** Successful action suffixes reach host-confirmed completion, but the live trace lacks a resolved outcome checkpoint afterward. This is a repeated lifecycle defect, not cross-game capability evidence.
3. **Model-fidelity defects, independently in both games.** These are not dated as the earliest historical failure:
   - 13 of 36 legal replay branches failed (retrospective); The final saved model is tested retrospectively; this cannot date a failure to a historical decision. Evidence: `analysis/tool-funnel/20260820-latest-symmetry/derived-metrics.json`.
   - The saved model omitted required actions(), so replay/planning could not proceed. (saved_model_contract_failure); This proves a final artifact defect, not when that defect first changed actor behavior. Evidence: `runs/tycho-nooa-gemma4-31b-feasibility-20260821T073930Z/AFTER-ACTION.md`.

## Best discriminating checkpoint

**Exact actor call 36 at LS20 sequence 2** is the highest-value next paid shadow: `runs/tycho-nooa-gemma4-31b-reasoning-capture-20260821T081012Z/game/decision-checkpoints/calls/000036/request.json`.

State: candidate/target present; outcome unverified; actor about to inspect a route. The current actor inspects a route while the terminal predicate is still unverified. The one factor is: **Replace rhetorical goal language with one executable provisional-goal obligation: name the predicted terminal signature and choose the cheapest safe discriminator within N actions.**

Measure the first response for: named hypothesis, predicted observation, selected discriminator, legality, and a promote/demote rule. Two repeats per arm are enough initially; expand only if within-arm variability blocks judgment.

No single frozen shadow separates all four proposed causes. Call 36 directly separates **goal-mechanism failure** from **evidence-use failure**. **Handoff** needs the scripted only-legal-action/lifecycle control, and **model fidelity** needs the offline replay/contract control. Treating one response perturbation as a four-way diagnosis would be self-deception.

## Positive controls and validity

- PASS — complete funnel is not failed (expected `None`, observed `None`).
- PASS — missing discriminator is detected (expected `goal_mechanism_failure`, observed `goal_mechanism_failure`).
- PASS — selected action without host result is detected (expected `handoff_failure`, observed `handoff_failure`).
- PASS — terminal without outcome checkpoint is detected (expected `consolidation_failure`, observed `consolidation_failure`).

The miner processed **68 decision sequences**, **102 exact checkpoint calls**, and **694 accounting/response rows**. It made **0 model calls and 0 ARC environment calls**.

## Claim ceiling

- This is a census of first observable broken control boundaries in five development traces, not a population estimate across games.
- Model-authored hypotheses are signals; only tool/host records establish execution and consequence.
- Legacy ar25 spend is sequence-aligned by tool-call join; the two Gemma runs use exact decision checkpoints.
- Retrospective branch replay establishes saved-model counterexamples but cannot date them to historical decisions.

## Artifacts

- `funnel.json` — canonical evidence, hashes, detectors, episodes, and shadow ranking.
- `decisions.csv` — one row per decision sequence with cumulative stages and spend.
- `index.html` — filterable timeline, horizon sensitivity, evidence drill-down, and state export.
