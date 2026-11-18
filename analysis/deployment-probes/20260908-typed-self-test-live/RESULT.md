# Typed self-test live discriminator

Verdict: **VALID BEHAVIORAL FAIL**.

The run completed two actions in the allowed `2 + 2` Actor calls, with four relay receipts, four forwarded requests, four provider responses, four worker model-call records, zero provider errors, zero reasoning tokens, and clean completion. The frozen candidate, runner, and card identities matched the precommit.

The model did use the new interface and the host did score its first experiment. But it avoided the precommitted regional discrimination:

- Question: `Which observable region changes after UP from the initial 3x3 state?`
- Hypotheses: `center_moves_up` versus `no_change_or_other_region`
- Chosen metric: `grid_changed`
- Predictions: `true` versus `false`
- Host observation: `true`, classified as support for `center_moves_up`

This is machine-falsifiable but epistemically too weak. Any change anywhere satisfies the first hypothesis, so the result cannot distinguish upward movement from a leftward, rightward, reshaping, or unrelated change. In fact, the successor moved the cell left and had `changed_bbox=[1,0,1,1]`; the boolean scorer nevertheless labeled the verbal `center_moves_up` hypothesis supported. The same loophole recurred in the second experiment (`cell_moves_down` versus `no_change_or_other_region`, again using `grid_changed`).

The second decision did inspect the persisted host result and exact changed region before acting, so result consumption works. The failure is earlier: the interface validates type separation, but not whether the selected metric can adjudicate the semantic claim written in the hypothesis and update.

Provider-reported cost was USD 0.0120676. Actor model time was 12.9113 seconds. Reservation was USD 0.31017856 under the USD 0.60 cap.

Per the result map, retain the raw evidence but do not claim typed-interface reliability or proceed directly to a multi-case run. The next intervention must close metric-to-claim alignment—most narrowly, make the test target itself typed (for example, a regional question requires `changed_bbox`) rather than allowing the model to pair a directional claim with the weaker `grid_changed` metric.

Raw evidence is under `cases/competing_regions/raw/`; structured run output is `cases/competing_regions/RESULTS.json`.
