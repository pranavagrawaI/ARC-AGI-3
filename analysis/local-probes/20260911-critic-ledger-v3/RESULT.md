# Measured-evidence critic v3: implemented and tested, candidate rejected

The six saved-decision tests completed through the same CoreWeave FP8 route.
Cost: $0.0194066. Summed request time: 33.790 seconds. Input/output tokens:
41,869 / 1,409. Reported reasoning tokens: zero. No retry or gameplay.

The concrete candidate adds a host-generated ledger of exact changed-cell counts,
coverage information and the full current frame, encoded losslessly as hex rows.
The critic must return a literal comparison before at most two epistemic findings,
plus whether the proposed action or commitment needs changing. The Actor claims,
proposals, prefix cutoffs and inference settings are preserved from the previous
six-case screen. Prompt and evidence representation change together, so this is a
candidate test, not an isolated causal estimate of either component.

Twelve local harness tests passed. The prefix reader stops before future records,
shape changes are not treated as zero impact, counts match literal expectations,
and frames survive a lossless round trip. AUDIT.json verifies the actual sent
frames against the saved source, frozen hashes, provider, tokens and cost.

## Model outcome

| Case | Assessment |
| --- | --- |
| S5I5 actual policy | Fail: counts 1,2,1 were explicit, but the critic checked the correct current count of four instead. It caught universal scope and missed the exact-one contradiction. |
| S5I5 cautious probe | Pass: no false objection. |
| LS20 actual batch | Mixed: obtained correct coordinates for the color-9 part, but introduced incorrect size/color-location claims and confused the part with the band. |
| LS20 cautious probe | Pass: no false objection. |
| CD82 actual exploration | Fail: challenged the Actor's correct count of 30 red cells and replaced it with 26. |
| CD82 authored overclaim | Pass: correctly rejected generalization from UP to all directions. |

Three clean passes, two failures and one mixed result. The frozen gate rejects the
candidate for missing S5I5's direct contradiction or making a false material
correction. Both failures occurred.

The CD82 correction is independently checkable: the supplied frame has 14 red
cells on row 24 and two on each of rows 25 through 32: 14 + 8*2 = 30. Bounds are
rows 24-32 and columns 25-38, exactly as the Actor stated. The critic kept LEFT as
the next action, but its factual correction was still wrong.

The LS20 frame has the local color-9 region at rows 47-49, columns 34-38, and color
12 at rows 45-46, columns 34-38. The critic called the former 5x5 despite naming
three rows, and said rows 40-44 contained color 12. Providing the correct frame
did not prevent it from making false statements about that frame.

## What this resolves

Making the counts explicit and providing the frame did not make this broad claim
audit reliable on these known failures. This rejects the sufficiency of this
particular evidence-plus-prompt intervention. It does not show the model cannot
compare 1,2,1 with one: the response selected a different claim to check. Claim
selection/coverage remains a live explanation. Likewise, raw-frame availability
does not substitute for executable measurement when counting cells.

No automatic revision or further paid call followed these outcomes. If pursued,
the next discriminating test should isolate claim selection from single-claim
verification, and keep numeric measurements outside the model's unsupported visual
counting. Do not expand to full Actor-critic orchestration on this result.

## Reproduction and evidence

- Implementation: `experiments/probes/epistemic-critic-local/ledger_screen.py`.
- Local checks: `python3 -m unittest discover -s experiments/probes/epistemic-critic-local -q`.
- Raw audit: `python3 experiments/probes/epistemic-critic-local/audit_ledger_run.py --output analysis/local-probes/20260911-critic-ledger-v3`.
- Numbered request/response/timing files preserve the actual interactions.
- PREPARED.json contains source prefix hashes, cutoffs and preserved Actor identities.
- GRADES.json preserves manual reasons; EXPERIMENT-CARD.md was frozen before calls.

This is a tested standalone critic prototype, not a runtime integration. Saved
cases are development examples, not held-out generalization or gameplay evidence.
