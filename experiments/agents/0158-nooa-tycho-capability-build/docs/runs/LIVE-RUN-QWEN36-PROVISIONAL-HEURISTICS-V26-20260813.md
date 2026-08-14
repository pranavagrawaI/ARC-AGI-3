# 0158 Qwen3.6 provisional heuristics v26 — 2026-08-13

## Outcome

The fresh non-resumable run reached all 44 actions without provider, schema, or
ARC failure. It completed no level and earned no action-specific promotion.

| Measure | Observed |
|---|---:|
| Claims / commitments / confirmed successors | 44 / 44 / 44 |
| Selected actions | action 1: 25; action 2: 19; actions 3–4: 0 |
| Verdicts | 31 supported; 13 contradicted |
| Commitment kinds | 41 Replace; 3 Unchanged; 0 Translate |
| Action-specificity closures / promotion credit | 0 / 0 |
| Levels completed | 0 |
| Provider calls / reported tokens | 539 / 8,739,918 |
| Provider-reported cost | $5.31022584 |

The terminal scorecard is `STOPPED`, `stop_reason=action_cap`,
`resumable=false`. Evidence is under
`artifacts/live-ls20/20260813T155603Z-qwen36-provisional-heuristics-44`.

## What the treatment fixed

The register passed its epistemic non-closure mechanism. Across 127 persisted
heuristic instances, every status was `provisional`. Qwen carried rejected
`diamond_pattern` and `eight_pair_pattern` beliefs forward with their evidence
and changed their attention role to `defer`; it neither deleted them nor marked
them explained or certain.

This continuity changed the semantic behavior. After two failed proxy regions,
action 4 introduced an active `shaft_motion` heuristic. It stated that action 1
moves the 5x5 shaft up five rows and action 2 moves it down five rows. By action
12 it had also revised the heuristic to include the upper boundary where action
1 produces no change. This is stronger action understanding than v24 or v25.

## What remained broken

The learned map did not become load-bearing in the formal commitments. Qwen
mostly selected a 2x5 slice or later the full 5x5 shaft but expressed its
prediction as local cell replacement. It never used `Translate`. Local effects
were often accurate—31 supported verdicts—but all were action-specificity
inconclusive. No reciprocal effect pair closed.

The run also did not separate the bottom progression as a watched scoped
heuristic by the precommitted action-12 horizon. When action 43 changed only two
bottom-strip cells at the shaft's lower boundary, Qwen overwrote its operational
interpretation: it deferred `shaft_motion` as non-responsive and created an
active `eleven_conversion` belief claiming action 2 converts the strip one
column at a time. Action 44 locally supported that prediction, but the run had
already shown the same progression alongside both actions. The new interface
prevented certainty and deletion; it did not force causal contrast.

Finally, Qwen selected only actions 1 and 2. Actions 3 and 4 remained untested,
so even its useful shaft model was incomplete for control.

## Diagnosis

The provisional register became a durable belief ledger. It improved continuity,
revision, and recovery from wrong candidates. It did not connect three separate
model responsibilities:

1. choosing the right abstraction (`Translate` rather than a locally equivalent
   `Replace` slice);
2. prospectively contrasting a suspected step-linked phenomenon across actions;
3. exploring legal actions absent from the current partial map.

The host should not decide those semantics. The result therefore supports the
register as an epistemic substrate while refuting it as a sufficient action-map
or gameplay mechanism.

The causal review in the run root is `pending_pranav`. No further run is
authorized.
