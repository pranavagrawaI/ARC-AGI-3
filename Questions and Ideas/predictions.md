# Risk provenance and current observations

The two theory documents name 17 useful risks. This file no longer treats them as a
six-hit predictive scoreboard. Some risks were written after relevant observations;
some describe proposed CodeAct or lifecycle mechanisms that do not yet exist; and some
were retrospectively mapped onto merely analogous current failures.

The risks remain valuable as precommitments for **future exposures**. Credit requires:

1. source text and timestamp precede the exposing evidence;
2. the mechanism under test matches the source risk;
3. the operational detector is frozen before scoring;
4. run eligibility is reviewed independently of detector output;
5. the result is reported as supported, contradicted, inconclusive, or unexposed.

## Current risk audit

| # | Risk | Judgment | Why |
| --- | --- | --- | --- |
| S1 | CodeAct safety and containment | `UNEXPOSED` | no CodeAct actor arm has run |
| S2 | mutable authority leakage | `UNEXPOSED` | no typed actor facade has run |
| S3 | loss of semantic trace grain | `UNEXPOSED` | no multi-operation CodeAct cell trace exists |
| S4 | confounded comparison | `SUPPORTED` | the later trigger replacement changed outer call/time caps while ownership treatment also differed; the deviation is documented, so no treatment-effect ranking is permitted |
| S5 | capability without uptake | `OBSERVED ANALOGUE` | typed goal tools were offered but not exercised past proposal; this pattern existed before the risk document and is not evidence about a future CodeAct facade |
| S6 | procedural overreach under Python | `UNEXPOSED` | native Builder Python volume is not the CodeAct-conditional risk |
| S7 | loss of Tycho parity | `UNEXPOSED` | no CodeAct treatment exists to compare with the native baseline |
| T1 | epoch theater | `OBSERVED ANALOGUE` | two eligible typed runs ended with revision-1 proposed goals, but no epistemic-epoch boundary was implemented or checked |
| T2 | manifest theater | `OBSERVED ANALOGUE` | low-confidence, zero-simulation Builder outputs became canonical and advisory; the strict planner gate did not trust them as verified capabilities |
| T3 | realm theater | `UNEXPOSED` | no multi-realm hypothesis treatment exists |
| T4 | reactive thrashing | `INCONCLUSIVE` | repeated rebuilds and null level outcome are observed, but three recommendations changed actions; unchanged decisions were not established counterfactually |
| T5 | false rollback | `NOT OBSERVED` | rollback did not claim to reverse sent actions, observations, or provider cost |
| T6 | lifecycle deadlock | `UNEXPOSED` | no provider/consumer lifecycle exists |
| T7 | historical erasure | `OBSERVED ANALOGUE` | three eligible runs contain superseded models without final live paths; only the trigger forensics establishes one valuable lost candidate |
| T8 | provenance overload | `CHECKABLE` | compare storage and analyst cost with findings enabled; no threshold is frozen yet |
| T9 | noncanonical publication | `UNEXPOSED` | there is only one publishing realm |
| T10 | abstraction without deletion | `UNEXPOSED` | no reusable lifecycle runtime has been added |

There is deliberately no hit count. `OBSERVED ANALOGUE` nominates a test; it does not
earn predictive credit.

## Machine-observable facts

The schema-2 report at
`../analysis/prediction-check/20260821-corpus/prediction-check.json` uses the reviewed
manifest and admits six of 12 checkpoint-bearing traces for whole-run negative-evidence
tallies:

- superseded-model unreachability: observed in 3, not observed in 3;
- declared tool activity: every eligible run offered at least one tool never attempted;
- typed-goal lifecycle: 2 frozen after proposal, 1 exposed with no goal created, 3 not
  applicable.

These are observations, not architectural verdicts. Each report row carries its claim
limit.

## Detector corrections

`tools/prediction_check.py` now:

- reads final `response.json` tool calls, preventing last-call right censoring;
- separates offered, attempted, completed, and errored activity;
- reads goal-tool events rather than raw identifier substrings;
- inspects the actual final workspace for content-identical retained models;
- does not treat an archive-like filename as retention without matching content;
- labels model value as unassessed;
- refuses inferential corpus tallies when discovered runs lack reviewed admission.

## Future prediction template

```text
ID and timestamp/hash:
Mechanism and exposure condition:
Prediction:
Competing prediction:
Measure and detector version:
Eligible population:
Result-to-action map:
Invalidity and stopping rule:
```

A theory programme earns funding by surviving this template prospectively, not by
renaming old failures after they are visible.
