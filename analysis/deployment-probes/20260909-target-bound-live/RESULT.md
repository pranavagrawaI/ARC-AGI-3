# Target-bound regional self-test live gate

Verdict: **VALID BEHAVIORAL FAIL — broader reliability run not started.**

The loophole demonstrated on 2026-09-08 is closed: the model could no longer pair a directional claim with `grid_changed`. Both live experiments used `target=region` and `metric=changed_bbox`, and the host preserved and scored the exact observed region.

The run nevertheless missed the precommitted gate. Its first alternatives were:

- `up_translation` -> `[0,1,1,1]`
- `no_change` -> `null`

These are distinguishable and host-scoreable, but they are not two competing coordinate regions. The actual successor moved the cell left with `changed_bbox=[1,0,1,1]`, so the host correctly returned `none_of_predictions` and contradicted both alternatives. The model inspected that result and changed its second action from the earlier run's DOWN to LEFT, then recorded another regional experiment. That demonstrates useful selection from negative evidence, but not the requested ability to generate competing regional alternatives prospectively.

Operational validity passed: four relay receipts, four forwarded requests, four provider responses, four worker model calls, zero provider errors, zero reasoning tokens, two actions within the `2 + 2` call cap, no helper/procedure detour, and no unresolved runtime error. Candidate, runner, and card identities matched the precommit.

Provider-reported cost was USD 0.0122402. Actor model time was 13.2582 seconds. Reservation was USD 0.31041544 under the USD 0.60 cap.

Per the result map, the broader five-case reliability run was not created or launched. The newly isolated boundary is alternative completeness: for a `region` target, permitting `null` lets the model fall back to one region versus no effect. A future intervention would need to distinguish a location-vs-no-effect test from a competing-regions test and require the latter to contain at least two non-null bbox predictions.

Raw evidence is under `cases/competing_regions/raw/`; structured output is `cases/competing_regions/RESULTS.json`.
