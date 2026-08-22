# Prediction check — reviewed 2026-08-21 corpus

This is a zero-model-call, zero-ARC-call observation report. It is not a prediction
scoreboard and does not infer cognition from absent tool calls or value from a missing
artifact path.

```bash
arc-host/.venv/bin/python tools/prediction_check.py --corpus \
  --manifest analysis/prediction-check/20260821-corpus/corpus-manifest.json \
  --output analysis/prediction-check/20260821-corpus/prediction-check.json
```

## Admission

The manifest keeps all 12 checkpoint-bearing traces visible. Six are eligible for
whole-run negative-evidence tallies; six remain visible but excluded with a recorded
reason. A newly discovered, unlisted run makes corpus status `review_required` rather
than entering the tallies automatically.

## Eligible-run observations

| Observation | Tally | Maximum claim |
| --- | --- | --- |
| superseded-model final-path unreachability | observed 3; not observed 3 | a superseded body lacks a content-identical path in the final workspace; value is unassessed |
| declared tool activity | unattempted declared tools in 6 | offered, attempted, completed, and errored activity only; no comprehension or causal-uptake claim |
| typed-goal lifecycle | frozen after proposal 2; no goal created 1; not applicable 3 | actual goal events and final typed state only; no truth or gameplay-benefit claim |

## Corrections from schema 1

Schema 1 was construct-invalid for the conclusions previously attached to it:

- final assistant tool calls were right-censored;
- raw goal-identifier text was counted as lifecycle attention;
- checkpoint presence silently granted corpus eligibility;
- archive-like filenames could count as retained models without a content match;
- missing final paths were called historical erasure without independently measuring
  whether the lost version was valuable.

Schema 2 reads final responses and tool traces, requires reviewed admission for corpus
tallies, checks exact final-workspace hashes, and reports explicit claim limits. The
selected trigger-run forensics independently establish one valuable lost candidate;
this detector deliberately does not generalize that result to other superseded models.
