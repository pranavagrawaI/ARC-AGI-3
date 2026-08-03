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
