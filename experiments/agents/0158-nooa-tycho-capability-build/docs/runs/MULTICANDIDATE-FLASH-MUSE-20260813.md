# 0158 low-cost multicandidate runs — 2026-08-13

## Outcome

Three exact model/provider candidates passed a paid provider-only claim→commit
tool-loop preflight, then entered fresh seed-0 LS20 runs under the frozen v26
host. None reached the requested 44 actions or completed a level. All three runs
stopped non-resumably on model-generation failure after losing the bounded
artifact-construction loop.

| Candidate | Actions | Levels | Calls | Tokens | Cost | Primary commitments |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek V4 Flash 0731, CoreWeave fp8 | 1 | 0 | 27 | 470,213 | $0.04406495 | Replace 1 |
| Qwen3.7 Flash, Alibaba | 2 | 0 | 32 | 463,136 | $0.01724578 | Replace 2 |
| Muse Glimmer 30B, DeepInfra bf16 | 5 | 0 | 76 | 469,705 | $0.12874958 | Unchanged 1; Replace 4 |

The requested 44-action comparison is therefore incomplete. The result is
nevertheless discriminating: Qwen3.7 was cheapest; Muse remained operational
longest and learned action 1's direction; all candidates failed to make object
translation load-bearing and all were brittle under the exploration-to-
constructor transition.

## What each model did

### DeepSeek V4 Flash

DeepSeek chose the static `0/1` cross as primary and placed only part of the
movable `9/12` block among its alternatives. Action 1 moved the real block up
five rows and advanced the bottom strip, leaving its primary unchanged. Its
Replace commitment was contradicted. It then noticed relevant outside changes
but described a transforming 10x5 color span rather than a persistent object
translation. After spending exploration on invalid frame-history assumptions,
it kept using forbidden reads in constructor-only submissions and exhausted 12
iterations.

### Qwen3.7 Flash

Qwen first chose the same static cross. After contradiction it explicitly
recognized that the selected region had zero changed cells and moved attention
to the full swept region of the `9/12` block. It still modeled translation as
local value cycling and made another contradicted Replace commitment. On the
third claim it spent all exploration on the grid/components before reading the
inherited heuristic register, then could not conserve prior heuristic evidence
under constructor-only lockout. All provider calls succeeded; this was not a
provider outage.

### Muse Glimmer 30B

Muse recovered furthest. After its first probe it formed the active heuristic
`moving_5x5_up`: action 1 moves the 5x5 block upward about five rows. It tracked
the block through several positions, but commitments predicted the vacated old
footprint becoming background rather than `Translate(-5, 0)`. Because the
verifier supported one such vacated-footprint prediction, Muse confused that
supported region with the object's current identity. It eventually kept rows
30–34 as primary after the object had reached rows 25–29. The next contradiction
was followed by nine failed attempts to reconstruct the monotonic heuristic
register, ending the run. Its 131K context did not cause this stop; maximum
observed input was 11,862 tokens.

## Shared causal structure

```text
visible block translation
        ↓
attention eventually reaches changed span
        ↓
model predicts local cell replacement/vacated footprint
        ↓
verifier evaluates that scoped effect
        ↓
object identity and Translate remain non-load-bearing
        ↓
contradiction or misleading local support
        ↓
model spends exploration before preserving durable state
        ↓
constructor-only recovery fails; run stops
```

This is not solely an objective-modeling failure. No candidate survived long
enough to explore actions 2–4 or infer the game objective. The immediate shared
failures were (1) representing temporal objects as changing cell regions and
(2) failing to plan information use across the exploration/commit boundary.
Objective and constraint modeling remains necessary after these lower layers
become usable.

## Informative setup failures

- The first preflights failed locally because the frozen profile requires
  `drop_params=true`; no provider calls occurred. The candidate profile was
  repaired while still omitting nonportable `reasoning_effort`.
- The official DeepSeek route was blocked by the account's OpenRouter privacy
  policy. A new card bound the same model to CoreWeave fp8.
- Initial claim→commit preflights were invalid because the qualifier passed an
  unbound claim draft without `claim_id`. The qualifier was repaired to use the
  same `grounding_claim_from_draft(...).to_record()` binding as the live host.
  Fresh preflights then passed for all three candidates.

## Verification and evidence

Before live execution: 48 host tests and 24 runtime tests passed; host/runtime
ruff, provenance audit, compilation, and all <=1 KiB card gates passed. Each run
used an exact one-run content-bound approval and a distinct consumption record.

Raw roots:

- `artifacts/live-ls20/20260813T223036Z-deepseek-v4-flash-coreweave-matched-44-v30`
- `artifacts/live-ls20/20260813T223044Z-qwen37-flash-matched-44-v28`
- `artifacts/live-ls20/20260813T223048Z-muse-glimmer-30b-matched-44-v29`

## Decision

Do not select a final engineering substrate from this incomplete comparison.
Qwen3.7 is the cost leader; Muse supplies the strongest semantic evidence but
also the highest short-run cost and tool-error rate. The next matched test should
first remove the shared procedural confound: make durable state readable during
submission/recovery or reserve an enforced inspection budget for it, while
keeping object identity and prediction semantics model-owned. Then rerun a
small action-semantic ladder before paying for three more 44-action attempts.
