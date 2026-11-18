# R10A-PACKAGE-REVIEW — HOLD

Independent reviewer: Karl Popper method — test the acceptance contract with counterexamples. Reviewed 2026-09-16. **HOLD for package readiness. No package approval, manifest freeze, authorization, or launch approval.** Gate machinery was explicitly outside qualification scope and is not the reason for this HOLD.

Two reproducible mechanical defects remain. The supplied tests pass; the additional probes demonstrate behavior those tests do not exclude. Fixes can remain within the isolated R10 package. No new user decision, semantic evaluator, compute authorization, or architecture change is needed.

## 1. HIGH — A candidate accepted by the real constructor can poison pending state and stop subsequent gameplay

Locations (relative to `analysis/r10-prep-20260916/r10a-package/`):
- `evaluate_visible.py:31-48`: replay accepts a branch whose state is `None`, provided assimilation returns a dictionary.
- `frozen_client.py:93-95`: that state becomes `_pending=None` while `_pending_key` is populated and the query is reported `predicted`.
- `frozen_client.py:110-111`: the matching successor cannot be assimilated because `None` is also the no-pending sentinel.
- `live_agent.py:78-86`: the next submit sees a pending key from the previous step and raises `RuntimeError: unassimilated executed prediction cannot be cancelled`.

**Reproduction:** `test_boundary_probes.py::test_admitted_null_state_breaks_next_live_submit`. The production service runs the actual isolated persist job, successful sealed prefix inspection, source transport, sealed visible replay and exclusive freeze publication. Candidate:

```python
def initialize(o): return {}
def predict(s, a): return [{"state": None, "commitments": {}}]
def assimilate(s, o, v): return {}
```

It returns `CANDIDATE_READY`, `frozen=true`, one successful inspection and two requests. The real live overlay and sealed RPC then query `CLICK 31 41`, execute its scripted successor, return `AMBIGUOUS_PENDING`, and throw on the next `DOWN`. The state writer/underlying action submission is a deterministic test replacement; this probe does not claim a public-game run. Admission, source execution, query and failing overlay are real package paths.

Evidence: `basetemp-boundary/test_admitted_null_state_break0/probe.json`, plus its `constructor-job/`, `host-events.jsonl` and `ipc/transitions.jsonl`.

**Why blocking:** R10a is intended to establish usable freeze/query/transition mechanics. This is an ABI-state bookkeeping contradiction, not inaccurate game prediction. The existing prompt's request for non-null state (`constructor_contract.py:32-34`) does not enforce compatibility, and the same disclosure says visible replay permits any JSON branch state. Although the underlying client defect is inherited, it lies directly on the acceptance path being qualified.

**Minimal resolution:** Give absence of a pending prediction a distinct representation (for example use `_pending_key` as the presence discriminator) so a valid JSON null state can be assimilated normally. Alternatively enforce and disclose a consistent non-null state contract at both admission and runtime, with truthful failure behavior that preserves ordinary play. Do not solve this only in the prompt. Regression must cover accepted source through freeze, correct executed successor and the next submission; include rejected/unexecuted pending predictions so the fix does not weaken transition binding.

## 2. MEDIUM — Preserved child artifacts are mistaken for an authoritative live freeze after hard stop

Locations in the same package:
- `construction_service.py:129-143,164`: hard timeout kills the job and returns `FAILED/frozen=false`; only a successful child is promoted to the session root.
- `audit.py:61-83`: `freeze_record()` recursively discovers child `FROZEN.json` without distinguishing promotion or failed parent status.
- `gameplay_play.py:166-177`: session `frozen` similarly accepts any descendant `FROZEN.json`.

**Reproduction:** `test_boundary_probes.py::test_failed_child_freeze_is_misclassified_as_live_freeze`. A test-only job wrapper invokes the real `persist_job.main()` (real inspection, candidate evaluation and child files), then delays exit. The real service's shortened five-second hard deadline kills it. Root freeze is absent, child freeze exists, service reports `FAILED`, `worker_alive=false`, yet the actual audit returns the child as the freeze record.

Evidence: `basetemp-boundary/test_failed_child_freeze_is_mi0/probe.json`, its failure marker and retained `constructor-job/`. The artificial delay targets the actual gap between child publication and parent promotion; it is a scheduling fixture, not an observed production timeout frequency.

**Why blocking:** Retaining immutable child evidence is correct, but calling it a live accepted freeze contradicts the service and Actor client, which only reads the root freeze. This is a telemetry defect introduced into the relevant path by retaining the job directory beneath the session. This probe does **not** show late promotion or an unauthorized action; root publication correctly remains absent. It also does not establish that the full audit would report overall success: missing use receipts can still prevent that.

**Minimal resolution:** Resolve authoritative freezes only at recorded session roots and require successful parent promotion/status. Retain child records separately as unpromoted/aborted evidence. Apply that distinction to both audit and session summaries; do not delete the forensic files. Keep an after-child-write/before-parent-promotion deadline regression.

The supplied `tests/test_constructor_deadline.py:23-26,66-77` kills the first request, whose response is a submit without the now-required inspection. That response could not have produced an accepted candidate even if left running. It usefully checks an early kill but is insufficient alone for the later publication boundary.

## Confirmed behavior and limits

| Area | Independently checked result |
|---|---|
| Exact package | All 62 files match inventory; all 60 notebook assets decode to exact source bytes and EXPECTED hashes. ASSET-HASHES and three card copies reconcile. No package input changed during review. |
| Source delta | Recomputed source diff exactly matches supplied patch: 10 changed Python files. All 63 snapshotted predecessor files remain unchanged. |
| Real entry and source transport | Supplied suite: **14 passed in 77.68 s** under mapped-user wrapper. Successful fixture uses generated notebook → official Agent.main → adapter/worker/relay → production constructor job → sealed inspection/replay. Literal backslash-n serialization failure, syntax and ABI failures are repaired via normal tool feedback; legitimate escaped literal remains unchanged. |
| Scheduling | Delayed real-entry fixture: **1 passed in 79.00 s**, six constructor calls with 11 s each; measured construction **66.612971869 s**. Confirms operation beyond the unchanged 60 s Actor-cell limit. Effective production limits are six requests, 240 s soft, 245 s hard; constructor is deferred to host context. Worker budget is 600+90+15 s, constructor POST 30 s/client <=35 s, launcher 1800 s. Card correctly discloses outer budgets can shorten the episode. No full 240 s exhaustion run was performed. |
| CLICK and state binding | Independent raw reconciliation checks two prospective queries (`CLICK 31 41`, `DOWN`), unique exact-action/step successors, source digest, and query pre-grid equality to the actual transition before-grid. One exact CLICK assimilation completes with `unscored`; final DOWN remains pending at termination. This final pending state is expressly permitted by the card, not a defect. |
| Truthful scoring | Successful freeze is ABI replay only; no dynamics accuracy claim. Existing typed scorer regression passes. The tested null-state case is lifecycle failure, not a semantic score. |
| Receipts/caps | Successful pair: 10 Actor +6 constructor calls, 0 summarizer/critic in these fixtures; raw request/response receipt reconciliation complete. Missing usage, wrong model, failed HTTP and per-arm call-cap fixtures pass. Six constructor calls share B's incumbent purse; total configured ceiling remains 258+1 warmup. No target-server warmup, real model/GPU or SDK attachment qualification is claimed. |
| Hard stop | Existing early-kill and outer-descendant tests pass. Reviewer late-child test confirms killed process and no root promotion; exposes the separate child-artifact classification defect above. |

## R10b documented resolutions

`REVIEW-RESOLUTION.md:4-5` and `r10b-preparation/EXPERIMENT-CARD.md:30,34` correctly record the two accepted readings: arm-local capability parity over divergent trajectories, and prospectively linked typed counterevidence without host certification of prose semantics. The current user message directly confirms that the submitted answers were accepted in the implementer terminal; this review verifies their representation in artifacts, not a separate replay of that terminal. No renewed approval is needed. No runnable R10b package is approved or evaluated here. R10b remains gated on Hermes verifying collected R10a mechanical acceptance.

## Exact-byte bindings

All paths below are within this reviewer directory unless otherwise stated. `input-hashes.json` records the reviewed package, supplied test sources and review documents by repository-relative path. `reconciliation.json` records the independently recomputed inventory, matching diff and exact reviewer fixture sessions. `R10A-PACKAGE-INVENTORY.json` is an unfrozen candidate inventory, not a launch manifest.

- Input hash-map SHA-256: `e0c450d363c17d89ab0012ce80d7111258874ddc1afff885105e1612329701ea`
- Supplied inventory SHA-256: `70d104f981fa9a495997548b8471246861cc1258bfdf6eb5c05572c1022ba0f2`
- Notebook SHA-256: `83feeebef0f63fcf0f101433ffaa3244acf5938deb3186266aab5f917ed7f22f`
- Packaged card SHA-256: `6425dc576370b0f68ea884ff5ea11cbf6989aa80bc08beaf0c86d3c8e4c72bc9`
- Reviewer `test_boundary_probes.py` SHA-256: `5b998c8e294845982369e6179f98d80e169b354641b0ce4f0ce6b1d0c4015622`
- Reviewer `reconcile.py` SHA-256: `bacdd7202d72ef5d0ca455a717f9d31b645eb3317259a7321d9304eb8ef09954`

Changed inputs invalidate applicability of this exact-byte review; any future approval requires binding the repaired bytes and regenerated notebook/inventory. No approval exists for these bytes.

## Reproduce

Run from repository root. These are the exact executed test commands; use fresh basetemp names and logs for another rerun rather than overwriting this evidence.

```bash
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/tests --basetemp=analysis/r10-prep-20260916/independent-package-review/basetemp-main
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 R10_FIXTURE_CONSTRUCTOR_DELAY=11 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -x -p no:cacheprovider 'analysis/r10-prep-20260916/tests/test_runtime.py::test_actual_supervisor_worker_http[None-False]' --basetemp=analysis/r10-prep-20260916/independent-package-review/basetemp-delayed
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/independent-package-review/test_boundary_probes.py --basetemp=analysis/r10-prep-20260916/independent-package-review/basetemp-boundary
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/independent-package-review/reconcile.py
```

Corresponding captured logs: `pytest-main.log`, `pytest-delayed.log`, `pytest-boundary.log`, `reconcile.log`. Reviewer boundary tests: **2 passed in 9.39 s**, meaning both defect reproductions succeeded; these are not acceptance passes. Full raw fixture evidence is retained under the three basetemps.

## Handoff

Implementer should repair the two mechanical contracts, regenerate the candidate inventory/notebook, and supply regression evidence for another independent exact-byte review. Do not freeze these bytes. Gate-machine qualification, package review, manifest freeze, frozen launch review, Hermes hashes/gates and SDK attachment verification remain separate subsequent steps. No implementation edits, Git/tracker mutation, remote calls or launches were performed by this reviewer. All new reviewer files and test evidence are in the permitted directory. Available for re-review.

SUPERSET_WORKER_DONE
 task: R10A-PACKAGE-REVIEW
 summary: HOLD; two reproducible mechanical defects despite 14 supplied tests and delayed real-entry test passing. Exact source/inventory/notebook reconciliation complete.
 files: analysis/r10-prep-20260916/independent-package-review/REPORT.md and referenced evidence
 checks: 14 supplied tests; 1 delayed integration; 2 successful defect reproductions; 62-file/60-asset reconciliation; unchanged input hashes and predecessor snapshot.
 handoff: Repair pending-state contract and authoritative freeze classification, regenerate bytes, then request independent re-review. No launch approval.
