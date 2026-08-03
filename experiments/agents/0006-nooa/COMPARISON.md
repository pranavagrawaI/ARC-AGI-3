# 0002-NOOA vs 0006-NOOA, ls20, 44 actions

Status: **pre-registration**. Written and committed before either arm runs.
Nothing in this file may be edited after the first live run; results go in a
separate section appended below the pre-registration line.

## The question

Does the competence treatment — archived observations, a claim ledger, a typed
falsifiable commitment, and a reconciliation gate — change how the model acts,
against the same frozen 0002 worker it is built on?

## The prediction

**Null.** The treatment is not expected to beat the control as it stands. This
is recorded as a prediction, not a hedge: a null is the expected result and will
be reported as a result, not as a failed run.

## Why levels_completed cannot be the measure

Gemma completed **0 levels in 292 steps** on ls20 with full NOOA memory
(`experiments/probes/nooa-gemma-reproduction/results/full-game/20260729_133928_ls20_memory_gemma-pinned`).
The frozen 0002 control completed **0 levels in 44 actions**
(`provider-smoke-probes/ls20/null-normalization-full-44-20r-4m-20260801T091838Z`).
At 44 actions the target measure is a guaranteed 0-vs-0 for both arms. It is
still recorded, and it is still the thing that matters, but it cannot
discriminate here.

## Primary measure: inert action rate

An ls20 frame carries a move counter in **rows 61–62**, which ticks on nearly
every action regardless of what the action did. Play-area change is bimodal:
across 43 archived control transitions, changed cells outside the counter band
numbered **exactly 50 or exactly 0**, never anything else.

> **An action is inert when its successor observation differs only in rows ≥ 61
> — that is, when nothing in the play area changed.**

Computed on host observations (`worker/host-observations/step-*.json`), which
both arms archive identically. It does not read the rendered image, the
competence records, or anything the treatment alone produces, so it is
arm-neutral by construction.

**Control floor: 17 of 43 = 40% inert.** Four in ten actions did nothing to the
game world. That is the headroom a working predict-then-reconcile loop should
consume, and it is measurable without reaching a level.

Direction: **lower is better.** A treatment that predicts, gets falsified, and
revises should waste fewer actions on moves that do nothing.

## Secondary measure: action-choice distribution

The control picked action `2` in 25 of 44 actions (57%), with one model round of
thought each. Reported as the full distribution over actions 1–4 per arm.
Descriptive only — no threshold is pre-committed for it.

## Decision rule

Two-sided Fisher exact test on inert counts, treatment vs control, n = 44 per
arm. **p < 0.05 counts as a signal; anything else is reported as a null.** At
n = 44 the control's 40% must fall to roughly 17% before it clears that bar, so
this test can only detect a large effect. A small real improvement will be
reported as a null, and that is the correct reading of it.

## What is matched, and what cannot be

Both arms run through the same runner, the same approval flow, the same audited
launcher, and the same pinned `modelrun/fp4` route. The single authorized
difference is which agent class the runner constructs.

| Field | Both arms |
|---|---|
| game / seed | ls20 / 0, offline |
| model | `openai/google/gemma-4-31b-it`, `google/gemma-4-31b-it` on the wire |
| provider route | pinned `modelrun/fp4`, fallbacks disabled |
| max actions | 44 |
| context turns | 6 |
| rounds per action | equal ceiling in both arms |
| decision repairs | equal ceiling in both arms |
| frozen worker | `0002-NOOA` at `eeba4c0`, byte-identical |

**Necessarily unmatched, and reported per arm:** total model calls, total tokens,
and cost. The treatment authors a typed commitment and may spend repair
crossings the control has no equivalent for. Rounds-per-action is matched so
both arms get the same opportunity to think; the spend difference is therefore
an *outcome* of the treatment and is reported rather than controlled. **If the
treatment wins while spending materially more inference, the win is confounded
with compute and must be reported as such.**

## What this comparison cannot show

- **One run per arm.** Model sampling is non-deterministic (no seed, no
  temperature pin). A single run per arm cannot separate a treatment effect from
  sampling noise. A signal here justifies replication; it does not survive
  without it.
- Nothing about gameplay competence, other games, or other models.
- Nothing about whether the model uses the protocol *well* — only what its
  actions did.

---

<!-- PRE-REGISTRATION ENDS HERE. Results below this line only. -->

## Result — 2026-08-03: null/no signal with a measurement deviation

Run pair:

- control: `control-44-20260803T080119Z`
- treatment: `treatment-44-20260803T080119Z`
- source revision executed by both arms: `664668478db93f9f59f5f836a716b6c392a46e22`
- frozen control revision: `eeba4c0166eac685d81bf5c6c087e2d395564d72`

Both arms reached the 44-action budget, completed zero levels, remained
`NOT_FINISHED`, preserved complete public traces, and shut down cleanly.

### Pre-registered primary measure

The pre-registered primary cannot be evaluated exactly as written. It requires
44 transitions per arm computed only from `worker/host-observations`, but the
frozen control worker archived only steps 0–43:

| Arm | Host-only inert | Host-only non-inert | Host-only rate |
|---|---:|---:|---:|
| Control | 34 / 43 | 9 / 43 | 79.1% |
| Treatment | 34 / 44 | 10 / 44 | 77.3% |

That asymmetry is a protocol deviation, not permission to silently change the
primary after seeing the run. Two bounded post-hoc analyses establish whether
the missing successor could change the decision:

| Analysis | Control | Treatment | Two-sided Fisher |
|---|---:|---:|---:|
| Shared 43-transition host-only prefix | 34 / 43 inert | 33 / 43 inert | `p = 1.0` |
| Terminal trace supplementation | 34 / 44 inert | 34 / 44 inert | `p = 1.0` |

The second analysis uses the common runner's frozen `trace.jsonl` only for the
control successor at step 44. The executable sensitivity measure requires the
trace to equal host observations at every overlapping step, refuses every
interior host-observation gap, and supplements only a missing terminal
successor. Control step 44 has 50 changed play-area cells and is non-inert. No
transition in either arm violates the pre-registered 0-or-50 pattern.

The exact pre-registered test is therefore **not evaluable**, but both analyses
are null at `p = 1.0`. The recorded experimental outcome is no signal that the
treatment reduced inert action rate, with the measurement deviation kept
explicit rather than laundered into a clean pre-registered result.

### Secondary and resource measures

| Arm | Action 1 | Action 2 | Action 3 | Action 4 | Calls | Input tokens | Output tokens | Reasoning tokens | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 5 | 31 | 5 | 3 | 50 | 573,629 | 37,281 | 31,594 | $0.13679573 |
| Treatment | 39 | 2 | 1 | 2 | 98 | 2,747,069 | 55,752 | 32,419 | $0.50888758 |

The treatment arm had a sharply different action distribution and used 1.96×
the model calls, 4.79× the input tokens, and 3.72× the cost, without a better
result in either bounded action-performance analysis. It incurred one refused
model crossing that was repaired before an action was released. These are
descriptive arm differences, not treatment effects, in a one-pair sample.

### Verifier reconciliation

The treatment's frozen `summary.json` records three failed acceptance checks:
`linked_process_records`, `prediction_results_derived_from_commitments`, and
`reconciliation_linked_to_observed_mismatch`. Offline replay located two
acceptor defects rather than malformed run evidence:

1. a changed-cell assertion became legitimately `inconclusive` when the public
   producer changed layer count, but the acceptor allowed only boolean match or
   mismatch; and
2. the acceptor did not independently reconstruct and bind the rejection notice
   carried into the repaired crossing.

The repaired offline verifier accepts all checks on the local immutable
44-action treatment artifact. The original summary remains unchanged as part of
the frozen run; this appended record states the post-run correction rather than
rewriting historical output. The full artifacts are intentionally ignored and
therefore absent from a clean checkout; the tracked synthetic regressions prove
the repaired boundaries, while the local-artifact regression is conditional on
those exact bytes being present.

Artifact bindings used for this result:

Host-observation tree bindings are SHA-256 over the lexically ordered bytes
emitted by `sha256sum step-*.json`; this commits every relative filename and
file digest without treating directory metadata as evidence.

| Artifact | SHA-256 |
|---|---|
| control `summary.json` | `c7604a54e6540b0df03d466943eeb7d547acec00c27a44ad9a651774244a2e60` |
| control `trace.jsonl` | `28338d3c363052b2fa8ffa56098706f308792e75da01e52986ab15c15952c08c` |
| control `worker/process.jsonl` | `f9e201f8b8a6ce1f6084aa0176403111049a670f65572167de9277d5a941d247` |
| control `worker/host-observations` tree, 44 files | `eaec68c5748ccb8732e083ada61646a9fb27ee285df2375636935ba92a7062b3` |
| treatment `summary.json` | `98ca78e28e5b586e90af5ff28ab57a90acf7b9c0af2b372b0ab527ba8fbbd46f` |
| treatment `trace.jsonl` | `5830b3c852810643c71a5e1adb5388cd648aea79192e0f98c063eee925550168` |
| treatment `worker/process.jsonl` | `7dfe1d77c7620f0a4d5bcfcf6dc1f46ce78e395c7b52875a529ddba26d01c323` |
| treatment `worker/competence.jsonl` | `0da62893456918e60a45be610985c9e8fe5286b1ae6582c2c6387ef24bbbfbd3` |
| treatment `worker/host-observations` tree, 45 files | `f11d1a188738608bb37918e93addfdb6fda1a76ad8290ad6c4be6bc0af0c441f` |

### Decision

Do not replicate this treatment unchanged. The treatment arm did substantially
more work and selected different actions, but neither bounded analysis shows
better action performance. The next question is mechanistic: what work did the
protocol elicit, and why did that work fail to control action quality? A
replication could dominate this decision only if estimating sampling variation
is worth the treatment's 3.72× observed cost despite the exact tie in the
terminal-supplement analysis.
