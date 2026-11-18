# C8 independent post-run audit

Verdict: **INVALID_RESULT**.

The paid launch was mechanically well formed, but the scientific comparison was not. Both arms were stopped by the same shared Actor facade defect before either could exercise the intended construction task and before treatment could reach its intervention. Equal failure under broken shared machinery is not a valid negative result.

## Launch and accounting

The candidate manifest and card match the required hashes. Both C6 reviews and the fresh user authorization bind that identity; authorization at `2026-09-17T17:56:41.984380+00:00` preceded launch start at `17:56:41.995757+00:00` and the first generation at `17:56:43+00:00`. There was one driver invocation, control first and treatment second.

I independently hashed all 168 files named by the sealed launch manifest: no mismatch. The launch tree is read-only, no file is newer than `ARTIFACT-MANIFEST.json`, both workers are reaped, no matching worker/driver remains, and all eleven candidate/predecessor snapshot inventories still match. Exact credential bytes occur in zero launch artifacts.

Raw request/response/receipt triples recompute to:

| Arm | Requests | Prompt | Completion | Reasoning | Cached prompt | Cost |
|---|---:|---:|---:|---:|---:|---:|
| control | 6 | 88,462 | 750 | 0 | 65,856 | $0.016628584 |
| treatment | 6 | 97,740 | 661 | 0 | 70,560 | $0.018086070 |
| total | 12 | 186,202 | 1,411 | 0 | 136,416 | $0.034714654 |

All twelve responses were HTTP 200 with `finish_reason=tool_calls`; unresolved requests and unresolved exposure are zero. Every raw request/response hash matches its receipt. Every response identifies `qwen/qwen3.8-27b`, provider `Reka`, direct attempt 1, selected endpoint model `qwen/qwen3.8-27b-20260814`; requests pin `reka/fp8`, disable fallback and reasoning, and request 4096 maximum output tokens.

All twelve non-message parameter sets are identical. The initial requests become identical after removing only the single declared 728-character treatment `<operational_cycle>` block. There was no retry, fallback, substitution, rerun, hidden consolidator, or executed game action.

## Raw endpoints

Control ended with zero proposals, theory versions, modes, evidence entries, revisions, bound actions, or replayable assets. It read the initial evidence and obtained one bounded `grid_summary`, but four grid-analysis cells errored and a later cell was refused by `CELL_STATE_REPAIR`; the six-call purse ended before any retained claim, decision, or submission.

Treatment also ended with zero proposals, theory versions, modes, evidence entries, revisions, bound actions, or replayable assets. It successfully called `workflow_skill()` and `operational_cycle('read')`, but `canonical/cycle.json` remained `phase=idle`; it never called observe, reconcile, revise, select, declare, decide, prepare, or result. Its six-call purse ended during the same grid inspection boundary.

Thus all eight predeclared common-rubric artifact fields are absent for both arms: entered observation in a retained asset, retained scoped claim, reconciliation, decision-relevant uncertainty/procedure, mode, prospective action record, post-observation revision, and reusable next-policy asset. No counterevidence opportunity occurred.

## First causal boundary

The common system prompt advertises `grid_array`, `model_grid`, `trajectory`, and `preflight_model` as public Actor methods and explicitly instructs NumPy use. But `ActorFacade` in frozen `cell_state.py` does two incompatible things:

1. `_sanitize` wraps every non-primitive return, including a NumPy ndarray, in `PublicObject`. Preserved traces show `g == c` becoming a scalar `bool`, `np.where`/`.any` failing, and `int(...)` plus `np.asarray(..., dtype=int)` rejecting `PublicObject`.
2. Its approved set omits the advertised `model_grid`, `trajectory`, and `preflight_model`; the treatment trace records `self.model_grid()` failing with `AttributeError: Actor capability is not public`.

This was shared across arms, but that does not rescue the comparison. It blocked the intended evidence-to-proposal task in both arms, and treatment never entered the operational-cycle intervention. The comparison is machinery-confounded and uninformative.

The card covers the defect through its persistence/private-name contract (public host APIs must remain usable) and its requirement that the original/full Actor tool surface remain present. It would nevertheless be safer for a successor card to state explicitly that cross-arm parity does not validate a pair when common machinery prevents both arms from exercising the task or prevents treatment from reaching the intervention.

## What this establishes

It establishes only that the authorized pair ran with correct routing, containment, accounting, and parity; both raw endpoints were incomplete; and the first observed boundary was the shared facade/public-API mismatch.

It does **not** establish treatment efficacy, lack of treatment advantage, arm equivalence, model incapacity, gameplay competence, gameplay utility, semantic truth, objective discovery, predictive correctness, or contradiction-revision ability.

Next step: stop. Do not repair or rerun under this release. A future, separately authorized identity must first qualify the exact facade with a deterministic fixture that exercises ndarray indexing/equality/conversion and every prompt-advertised Actor method. Only then should another paid pair be considered.
