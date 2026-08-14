# 0158 Qwen3.6 action-semantics v22 — 2026-08-12

## Outcome

The fresh LS20 run reached its 44-action cap without a host, provider, or
environment failure. It did not complete level 1. The constructor-only grammar
fixed v21's terminal claim-9 failure: all 88 claim/commitment artifacts closed.

| Measure | Observed |
|---|---:|
| Confirmed actions / successors / verdicts | 44 / 44 / 44 |
| Levels completed | 0 |
| Supported / contradicted verdicts | 28 / 16 |
| Complete G2 cycles | 5 |
| Provider calls / HTTP 200 | 430 / 430 |
| Reported input + output tokens | 5,256,033 |

Actions were distributed as `1:36, 2:3, 3:1, 4:4`. The terminal scorecard is
`STOPPED` at `action_cap` and is non-resumable. Raw evidence is in
`artifacts/live-ls20/20260812T165325Z-qwen36-action-semantics-44`.

## Precommitted measure

The operational recurrence criterion passed:

- action 1 was contradicted at step 1;
- action 1 was prospectively supported at step 6;
- step 7 was an intervening prospective decision; and
- action 1 was prospectively supported again at step 8.

Action 2 similarly moved from contradiction at step 2 to support at steps 10
and 11. These verdicts establish that Qwen consulted history and preserved a
prospective prediction procedure across fresh artifacts.

## Why this does not establish action semantics

The selected effect was not discriminative among actions. Steps 1–4 used
actions 1, 2, 3, and 4, yet every successor included the same automatic
progression: cells at rows 61–62 changed `11→3`, advancing one column per
successor. Action 4 then received supported verdicts for predicting that
progression at steps 4–5; actions 1 and 2 received later supports for the same
progression.

Qwen itself reached the relevant alternative explanation. In commitment IPC
response sequence 18 it inferred that the specific action did not seem to
matter because each action advanced the conversion one column. In sequence 20
it chose action 2 while predicting that any legal action would produce the
same next-column effect.

Therefore same-action recurrence did not distinguish:

```text
learned effect of opaque action 1
```

from:

```text
learned action-insensitive successor progression
```

The run demonstrates prospective history use and persistence, but **does not
establish acquisition of distinct opaque action semantics**. The card's
`ambiguous → refine` branch applies. No grounding capability is promoted on
this evidence.

## Gameplay behavior

After early exploration, Qwen increasingly selected action 1: 36 of 44 total
actions. The run accumulated supported local predictions but made no level
progress. This separates local successor prediction from goal-directed action
selection; neither level-solving competence nor route planning is established.

## Next discriminator

A future action-semantics probe must require the prospective effect to
distinguish the selected action from at least one live alternative. Predicting
an effect already expected under every legal action cannot qualify. This is a
measurement correction, not a reason to teach Qwen LS20 action meanings.

## Integrity

- 44 authorization records pair with 44 consumption records.
- 44 issued actions pair with 44 successors and 44 verdicts.
- All 430 provider attempts returned HTTP 200 through the pinned route.
- All 134 content-addressed evidence objects and indexes verify.
- All 20 candidate snapshot hashes verify against the frozen manifest.
- No configured credential value occurs in the run tree.
- The approval is consumed and the trajectory cannot resume.

Machine-readable interpretation is preserved in `action-semantic-analysis.json`.
