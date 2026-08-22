# Questions and Ideas

This folder is a decision surface, not an idea archive. The source documents remain
under `sources/`; the live files say what has earned action, what remains a bet, and
what evidence would change the order.

## Current judgment

The two host-contract interventions now pass their preserved-boundary replays:

1. timed-out Builder candidates survive with contemporaneous score and evidence while
   remaining noncanonical and planner-ineligible;
2. the selected frozen goal cannot reach action without independent review and an
   explicit revision-2 resolution.

The proposed job allocator failed its discriminator: the selected timeout was already
100% mechanics-scoped. A pinned-route matched continuation preserved the retained
candidate's 8/16 advantage while canonical remained 0/16, but neither arm gained
score. Turning reasoning on exposed 6,356 reasoning tokens, yet both fixed-time arms
timed out without editing and accuracy remained 8/16 versus 0/16. The next earned
work is trace diagnosis plus explicit nonauthoritative continuation selection, not
more generic Builder calls. Persistent execution, state-by-reference, and a CodeAct
actor remain possible treatments, not explanations established by the corpus.

## Files

| File | Function |
| --- | --- |
| `Channel A - Promotion Ledger.md` | evidence-backed interventions and their acceptance tests |
| `Channel B - Attention Register.md` | unresolved architectural bets with cheap discriminators |
| `predictions.md` | provenance-aware risk audit; not a prediction hit-rate scoreboard |
| `execution-substrate.md` | which execution properties have earned a test, and in what order |
| `Three Gate Replays - 2026-08-22.md` | preserved-boundary and matched-continuation results; licensed next action |
| `Builder Reasoning Trace - 2026-08-22.md` | call-by-call returned reasoning and the diagnosed thought-to-edit failure |
| `sources/` | immutable proposal documents and vocabulary that generated the candidate ideas |

## Evidence vocabulary

Use four judgments:

- **SUPPORTED** — current evidence discriminates for the exact claim over a live
  alternative.
- **CONTRADICTED** — current evidence rules out the claim as stated.
- **INCONCLUSIVE** — the evidence is compatible with live alternatives that imply
  different actions.
- **UNEXPOSED** — the mechanism required to test the claim has not run.

An observed analogue is not a confirmed prediction. A tool declaration is not
uptake. A final-path loss is not valuable-learning loss until value is independently
measured. A retrospective final-evidence score is not a permissible live selector.

## Corpus admission

The reviewed manifest is
`../analysis/prediction-check/20260821-corpus/corpus-manifest.json`. It keeps all 12
checkpoint-bearing traces visible while admitting six for whole-run negative-evidence
tallies and recording why the other six are excluded. Discovery of an unlisted run
forces `review_required`; checkpoint presence alone never grants validity.

Reproduce the zero-call observations with:

```bash
arc-host/.venv/bin/python tools/prediction_check.py --corpus \
  --manifest analysis/prediction-check/20260821-corpus/corpus-manifest.json \
  --output analysis/prediction-check/20260821-corpus/prediction-check.json
```

The detector reports only:

- superseded-model final-path reachability;
- declared, attempted, completed, and errored tool activity;
- actual typed-goal lifecycle events and final state.

It does not establish comprehension, causal use, gameplay benefit, or transfer.

## Operating rules

1. **Active WIP is two.** The ledgers may preserve more possibilities, but only two
   interventions may consume implementation or run budget at once.
2. **Claims stale by exposure, not date alone.** New evidence marks a claim stale
   only when it exercises the mechanism or population named by that claim.
3. **Promotion requires an exact claim and a discriminator.** Analogy can nominate
   a test but cannot promote an item.
4. **Publication and retention are separate decisions.** Rejected or timed-out
   artifacts may remain inspectable without becoming canonical or actionable.
5. **Retire from the active view, preserve the decision record.** The epitaph names
   the decisive evidence so deletion does not recreate historical erasure.
6. **Reasoning and response capture are mandatory.** Live model calls request
   provider reasoning and retain returned reasoning and prose. A provider that
   returns `reasoning_tokens=0` remains observable as such; tool arguments, explicit
   hypotheses, and predictions are still required because absent provider text must
   never be replaced by invented thoughts.

## Research loop

```text
claim
  -> exact exposure condition
  -> cheapest discriminating probe
  -> supported / contradicted / inconclusive / unexposed
  -> execute, revise, or retire
```

The folder succeeds when it reduces live uncertainty and spending. Accumulating
plausible mechanisms is not progress.
