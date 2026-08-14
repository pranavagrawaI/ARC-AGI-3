# 0158 atomic-interface multicandidate runs — 2026-08-14

## Result

The three candidates received the same v32 interface and a prospective
44-action cap. None reached 44 actions or completed a level. DeepSeek passed
preflight and executed 7 actions, Qwen passed preflight but failed before its
first action, and Muse failed the required no-game preflight and therefore did
not receive live-game authority.

| Candidate and exact route | Preflight | Live actions | Levels | Live calls | Live tokens | Live cost |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash 0731, `coreweave/fp8` | pass, 17 calls | 7 | 0 | 118 | 1,734,354 | $0.19612695 |
| Qwen3.7 Flash, `alibaba` | pass, 9 calls | 0 | 0 | 12 | 253,964 | $0.00917102 |
| Muse Glimmer 30B, `deepinfra/bf16` | fail, 12 calls | not authorized | 0 | — | — | — |

Including all three fresh preflights, the condition used 168 provider calls,
2,425,964 input-plus-output tokens, and $0.24651489. Every provider crossing
returned HTTP 200. The stops were artifact-construction failures, not provider
outages.

The 44-action request was a cap and intended trajectory, not permission to
ignore terminal validity gates. Each failed trajectory is non-resumable.

## Frozen comparison

The candidate-specific model and route were the only intended treatment
factors. All candidates received:

- one coherent `inspect_evidence()` snapshot;
- evidence readable throughout a 12-call artifact;
- host-owned IDs, schema bookkeeping, and heuristic-register conservation;
- `make_mask`, `submit_grounding_claim`, `predict_cell_values`,
  `predict_translate`, and `commit_contrastive` as pre-bound globals;
- atomic claim/commit submission on the first top-level submit constructor;
- an explicit 12-call budget and instruction to reserve a submission call;
- the same 44-action, 1,200-model-call, 15-million-token, and four-hour caps.

Before execution, 48 host tests and 25 runtime tests passed, Ruff passed for
host and runtime, provenance and compilation checks passed, and all experiment
cards satisfied the 1 KiB gate. Post-run recomputation matched both frozen
candidate manifests exactly:

- DeepSeek: `ef1dc052150386a8c0d3d1f51aa212da7bd2e1edf49db9baba9d7148486a9646`
- Qwen: `88725a5aabb3c41f42ec3e8f3ea0a73f7b08dcd976f5555cf871c7cab169688c`

## DeepSeek: useful semantic progress, then budget failure

### Observed trajectory

1. DeepSeek initially chose the static framed top pattern as primary. Action 1
   instead moved the 5x5 `9/12` block up five rows, so the commitment was
   contradicted.
2. In its next claim it explicitly revised to: action 1 moves the central block
   upward by five rows. For the next two commitments it nevertheless predicted
   only old-footprint cells becoming background. Those local `Replace`
   predictions were exact, but the verifier correctly marked action
   specificity inconclusive.
3. From actions 4 through 6 it used `Translate(-5, 0)` on the tracked 5x5 block.
   All three translations were exact. It tracked the object through rows
   30–34, 25–29, 20–24, and 15–19 without losing identity.
4. It still selected action 1 every time and predicted actions 2–4 as
   `Unchanged`. No alternative action was executed, so no directional action
   table became contrastively established.
5. On action 7 it anticipated interaction with the framed pattern above, but
   encoded a local replacement rather than a clean constraint/no-op
   prediction. The actual transition changed no cells, contradicting it.
6. In the next claim artifact it correctly recognized that the upward motion
   had stopped and began reasoning about an obstacle, destination, and possible
   objective. It then consumed all 12 calls on repeated inspections and never
   submitted a claim. The run stopped.

### Attribution

This is the strongest semantic result of the condition. DeepSeek learned and
prospectively tested object translation, preserved object identity, and noticed
a movement constraint. Its terminal failure is now substantially model-owned:
evidence remained readable, every inspection succeeded, the budget was stated,
and no hidden schema repair was blocking submission. It simply did not convert
analysis into a bounded decision before the disclosed limit.

One host-side weakness remains visible: exact predictions of a vacated
footprint can still receive `supported` status, although they receive no
action-specificity or promotion credit. This did not falsely establish the
action mapping, but it permits low-value cycles to continue.

DeepSeek totals: 7 actions, action distribution `{1: 7}`, primary effects
`Replace: 4, Translate: 3`, verdicts `supported: 5, contradicted: 2`, and action
specificity `inconclusive: 5, contradicted: 2`.

## Qwen: interface friction remained load-bearing

Qwen spent its first seven calls inspecting and decomposing the initial frame.
It then attempted to import the already pre-bound constructors from a nonexistent
`tools` module. After the explicit `ModuleNotFoundError`, it switched to the
global constructors but repeatedly built masks with semantic characters such
as `X` and `D`, despite `make_mask` requiring only `#` and `.`. Later attempts
fixed the alphabet but retained unequal row widths. The error messages named
each violation, yet Qwen repeated them until the 12-call limit. No claim was
accepted and no action was issued.

This cannot be classified as a clean game-reasoning failure because Qwen never
reached the game experiment. It is also not a hidden or arbitrary protocol
failure: the globals and mask alphabet were documented and validation remained
available. The fair classification is mixed:

- model failure: excessive exploration, importing a declared global, and
  repeatedly failing to apply explicit validation feedback;
- remaining interface burden: an ASCII bitmap constructor makes syntactic row
  discipline load-bearing even though the semantic decision is merely a set of
  cells.

Qwen's successful 9-call preflight established one-shot compatibility, not
reliability. The same model failed stochastically on the live initial frame.

## Muse: preflight budget failure

Muse was never given live-game authority because its required provider-only
preflight failed. Its first two attempts made straightforward API-use errors
(`inspect_evidence` imported as a module, then a nonexistent frame `id` key).
It recovered access and spent the remaining calls inspecting regions and
components, but never invoked `submit_grounding_claim`. Because the preflight
card bound failure to no live run, there is no Muse gameplay evidence in this
condition.

This is again primarily a bounded-decision failure, with a smaller residual
interface contribution from the model treating pre-bound functions as modules.
All 12 provider calls succeeded.

## What the comparison establishes

### Supported

- The v32 atomic submission path removed the earlier double-submit and
  unreadable-recovery confounds.
- DeepSeek can discover and prospectively use `Translate(-5, 0)` from LS20
  evidence, preserve the moving object's identity, and detect a no-op boundary.
- The verifier withheld promotion from action-insensitive or untested
  contrasts: none of the five supported DeepSeek effects was action-specific.
- Qwen was cheapest, but its single preflight pass did not predict dependable
  interface use.

### Not supported

- No model completed level 1.
- No model established `{1: up, 2: down, 3: left, 4: right}`.
- No model executed an alternative action or formed a tested objective-and-
  constraint model.
- The run does not rank Muse's gameplay ability because Muse never passed the
  live gate.

## Responsibility boundary and next change

Do not respond by adding more philosophical warnings to the prompt. The next
change should be a deeper interface deletion: replace textual mask bitmaps with
a semantic cell-set or rectangle constructor whose output is canonicalized by
the host. That removes row-width and character-alphabet work from the model
without choosing the region for it.

Keep these responsibilities model-owned:

- selecting the object or region;
- choosing an informative action;
- deciding between translation, replacement, and no-op;
- maintaining provisional object, objective, and constraint hypotheses;
- stopping inspection early enough to act within a disclosed budget.

Keep these responsibilities host-owned:

- canonical masks and IDs;
- schema and register conservation;
- enforcing the call cap;
- withholding action-specific credit until an alternative action is actually
  tested;
- exposing but not interpreting outside-region and automatic changes.

The next paid test should be shorter than 44 actions: first qualify deterministic
claim submission across several seeds, then require one four-action coverage
ladder. Only a candidate that survives that ladder should receive another
44-action budget.

## Evidence

Live roots:

- `artifacts/live-ls20/20260814T092200Z-deepseek-v4-flash-coreweave-matched-44-v32`
- `artifacts/live-ls20/20260814T092200Z-qwen37-flash-matched-44-v32`

Preflight roots:

- `artifacts/model-preflight/20260814T091432Z-deepseek-v4-flash`
- `artifacts/model-preflight/20260814T091432Z-qwen37-flash`
- `artifacts/model-preflight/20260814T091432Z-muse-glimmer-30b`

Key live hashes:

| Evidence | SHA-256 |
|---|---|
| DeepSeek summary | `379ec6f27eec58a0c07811b4c78816c9b6661dc3c6389e96a0841ceed4093807` |
| DeepSeek responses | `6871dd169e7dbc6532d55355f94028ca36d6145503fb63bd244ce1e9cf1b4182` |
| DeepSeek actions | `9dc42045d55b0a2568822e87414c0850399d02d924b461f408b1a4ae3e8d1b1a` |
| DeepSeek verdicts | `633b7414d5b5f9b24007b87a242e4f0a109517e12d747a795be64aea7281b3ff` |
| Qwen summary | `47b1ae1a057ca9ec31bc6099366af7f9b97c214bec4d028985a79cf6e5ac3399` |
| Qwen responses | `26e7f64b503627f5ed5ba3fa124c8084b3fa03a8b9ebe23f6ee3487c9ff1bc1b` |
| Muse preflight responses | `08cf644440d41a8c58ecec43c3b936d0e2861d7f4055e5ae891877c7f53120e0` |
