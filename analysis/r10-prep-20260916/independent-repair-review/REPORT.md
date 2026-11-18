# R10A-DELTA-REVIEW — PASS for exact package; PASS for offline gate tooling

Independent reviewer, 2026-09-16. **The two named package HOLD findings are resolved for the bytes below. No blocking defect found in the bounded repair or newly active gate tooling review.** This is local package approval and offline tooling qualification, not a frozen launch review or launch approval. Production remains unfrozen and the live launch gate must remain HOLD.

The prior `independent-package-review/REPORT.md` and all 2384 snapshotted reviewer evidence files remain unchanged. That HOLD correctly describes the previous bytes; it is not erased or rewritten. This delta review supersedes those two findings only for the repaired identity below.

## Exact-byte bindings and bounded delta

Independently recomputed all REPAIR-HASHES and GATE-HASHES entries. All 62 package files match the candidate inventory; all 60 notebook assets decode to identical source bytes and embedded EXPECTED hashes. ASSET-HASHES and all three card copies reconcile. No reviewed input changed during this review; 63 snapshotted predecessor files also remain unchanged.

| Artifact | SHA-256 |
|---|---|
| `R10A-PACKAGE-INVENTORY.json` | `b2325fb93db5dfa16254a7d3d6c06b82a8b628bb61f4374276be6fe3e76e8aef` |
| `r10a-package/construction.ipynb` | `87afeeb324d33bafe4a7992abf1f5f17ba9ef83631d716b3c5287a061fa770a3` |
| Packaged card, unchanged | `6425dc576370b0f68ea884ff5ea11cbf6989aa80bc08beaf0c86d3c8e4c72bc9` |
| `R10A-SOURCE-DIFF.patch` | `89fe18a9b5fd3e0a7cdecf25d7f59af17138f9661810fe874f009bc5054708cd` |
| Reviewer `input-hashes.json` | `94357453b0bd62e5b705c8fa4d7a87c566f578c1c5aae3a2b31e78f3d53db11c` |
| `repair-20260916/GATE-HASHES.json` | `5012750ac8f31c47464cc70a3ecd29e21a63724e02d3409cda1f6a1bce5c0170` |

Compared current package hashes against my previous input snapshot, then rederived textual deltas from preserved prior integration materializations after verifying their old hashes. Only four Python files changed: `frozen_client.py`, `constructor_contract.py`, `audit.py`, `gameplay_play.py`. The other two changed files are regenerated `ASSET-HASHES.json` and `construction.ipynb`. `package-delta.patch` preserves this independently derived comparison. Budgets, card, game, Actor/chassis and constructor scheduling implementation are unchanged.

Changed candidate inputs invalidate package approval. Changed gate inputs invalidate this tooling review and require their own review/requalification; gate sources must subsequently be included in the frozen manifest.

## Repair acceptance

Locations below are relative to `analysis/r10-prep-20260916/r10a-package/`.

### Previous finding 1 resolved: JSON null is data, not the pending sentinel

`frozen_client.py:81-82,93-95,110-121` now uses the exact action/step key to distinguish pending presence. It permits an admitted null successor to reach sealed assimilation while retaining duplicate-query and wrong-transition refusal. `constructor_contract.py:32-34` describes that same contract.

Reran `tests/test_repair_boundaries.py::test_admitted_null_state_next_submit_and_unexecuted_binding` as part of the full required wrapper suite. Independently inspected and reconciled the generated trace:

- Real service → isolated persist job → actual prefix inspection → sealed evaluation → accepted null-state candidate → root freeze.
- An unexecuted `CLICK 0 0` leaves its original pending binding.
- A second query returns `PREDICT_WITHOUT_ASSIMILATE`; wrong action and wrong successor step return `REJECT_MISMATCH` without replacing the pending prediction.
- The live overlay cancels only the unexecuted original-step prediction, records the cancellation, executes `CLICK 31 41`, assimilates null with `unscored`, and admits the subsequent `DOWN`.
- Candidate assimilation explicitly asserts its input is None. The evidence therefore proves null transport, not just a successful return around a missing payload.

Evidence: `package-basetemp/test_admitted_null_state_next_0/probe.json`, its `constructor-job/`, host events and IPC transitions. As before, the overlay test uses a deterministic state writer and submission replacement; constructor admission and sealed execution are real. Separate official-loop fixtures cover actual CD82 transport.

### Previous finding 2 resolved: child evidence is not a live freeze

`audit.py:61-89` now requires digest-valid session-root FROZEN plus successful parent CONSTRUCTION and RESULT records, and only considers recorded session roots. `gameplay_play.py:164-180` shares that predicate and lists child forensic artifacts separately.

Reran `tests/test_repair_boundaries.py::test_post_child_write_hard_kill_has_forensics_but_no_live_freeze`. The real persist job finishes inspection/evaluation and writes its child freeze/result; a test wrapper delays its exit; the actual parent hard deadline kills it before promotion. Parent reports FAILED and worker dead; child bytes remain; root freeze is absent; actual audit returns no freeze; session summary reports `frozen=false` and retains the child-artifact path.

Evidence: `package-basetemp/test_post_child_write_hard_kil0/probe.json`, failure marker and `constructor-job/`. This is a controlled scheduling adversary, not a claim about production timeout frequency.

### Whole package regression

**16 tests passed in 91.03 s**, mapped-user wrapper, fresh reviewer basetemp. Covers both repairs plus generated notebook, real official-loop/relay/worker path, six-request ordinary inspection/source repair, escaped-literal preservation, CLICK and retained state, typed scoring, receipt failures, caps, early timeout and outer-descendant retirement.

Independent raw reconciliation of the successful CD82 pair confirms 3 actions per arm and official recording reconciliation; six constructor requests, two inspections and four submissions; two exact-coordinate/action/step/digest prospective queries with pre-grid equality to actual transition before-grid; one unscored assimilation and one final pending query. Final pending is permitted by the unchanged card and not mislabeled assimilated. No accuracy or model-quality conclusion follows.

The earlier delayed >60 s integration evidence remains applicable to byte-unchanged scheduling sources; it was not redundantly rerun here. No full 240 s exhaustion, real model/GPU, competition SDK attachment, or scientific outcome qualification is claimed.

## Newly active gate tooling — offline PASS

Read complete active `launch_control.py`, `make_manifest.py`, `verify_frozen.py`, README and adversarial tests. Exact source hashes:

| Active source | SHA-256 |
|---|---|
| `launch_control.py` | `ef632e432b61990cb567766418af9f1b6a1dbd5bad8ba515c0bf25f597fce923` |
| `make_manifest.py` | `4f7e34e796791721233aef304ae2d977083d1152b3df6d3b0e221b951aa2453b` |
| `verify_frozen.py` | `ed970bae6a74f680e4ae2215356ee07932a3547c35ad65a6b5aaa31e318ee752` |

**141 supplied offline adversarial tests passed in 1.19 s. Five additional reviewer tests passed in 0.07 s.** All provider subprocesses in these fixtures are intercepted; SDK status adapter uses fake clients. Synthetic fixture manifests, approvals, releases and consumption markers are exclusively under the new reviewer evidence directory and confer no production authority.

Verified boundaries:

- `launch_control.py:135-150`: exact inventory/notebook/package bytes and type-strict affirmative package decision, including boolean `launch_authorized=false`.
- `make_manifest.py:9-31`: approval before writes, card equality, source coverage, review/byte recheck, exclusive/fsynced manifest and authorization, refusal of existing or partial control state. Production freeze was not invoked.
- `launch_control.py:153-201`: complete frozen source coverage, metadata/limits/authority bindings; launch-ready mode additionally requires manifest-bound independent launch review and Hermes release.
- `launch_control.py:255-288`: preflight and post-preflight validation, exclusive/fsynced consumption before one push, persistent consumption on uncertain outcomes, no automatic push retry. Offline tests include revocation, source drift and competing consumption.
- `verify_frozen.py:7-19`: read-only frozen validation can support review without the launch review it is intended to inform; it cannot enable push. Reviewer test calls this actual main entry against disposable state, verifies no writes and no provider calls, then checks missing launch review/release and byte drift still hold.
- Exact-version monitoring is tested through executed status adapter code; local polling deadline does not claim cancellation. Real SDK attachment evidence remains a separate mandatory prelaunch human/PM review gate, as the README states.

No gate-tooling HOLD is reported. The production gate's expected HOLD on missing freeze/review/release is a prerequisite state, not a tooling failure.

## Package decision and authority boundary

The separately authorized `r10a-verification/package-review/DECISION.json` records `APPROVED_LOCAL_PACKAGE`, exact candidate inventory in the legacy `manifest_sha256` field, exact notebook SHA-256, and `launch_authorized: false`. Its only approval scope is this local repaired package. The actual active `validate_package()` predicate is run against the production decision and bytes, read-only; output is retained in `approval-predicate.json`. The production `launch_control.py check` is also run read-only, with its expected missing-manifest HOLD retained in `production-gate-check.json`.

No production manifest, authorization, frozen launch review, Hermes release, CONSUMED or launch receipt is created in this review. Existing user launch authority remains subject to the required sequence: review-gated freeze by its authorized owner → independent exact frozen launch review and SDK attachment evidence → Hermes verification/release → one private push. R10b RUN still requires Hermes verification of collected R10a mechanical acceptance. No new budget ruling is needed.

## Reproduce and inspect

Run from repository root; use fresh basetemp names/logs for another run. Logs here retain stdout/stderr from these exact commands:

```bash
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/tests --basetemp=analysis/r10-prep-20260916/independent-repair-review/package-basetemp
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/r10a-verification/test_launch_control.py analysis/r10-prep-20260916/r10a-verification/test_make_manifest.py --basetemp=analysis/r10-prep-20260916/independent-repair-review/gate-basetemp
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/independent-repair-review/test_gate_review_boundaries.py --basetemp=analysis/r10-prep-20260916/independent-repair-review/reviewer-gate-basetemp
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/independent-repair-review/reconcile.py
```

Evidence: `package-tests.log`, `gate-tests.log`, `reviewer-gate-tests.log`, `reconcile.log`, `reconciliation.json`, `input-hashes.json`, `prior-evidence-hashes.json`, `initial-reconciliation.json`, `package-delta.patch`, reviewer tests and all fresh fixture directories. Only the permitted reviewer directory and the expressly allowed package decision are written. No candidate/gate edits, Git/tracker mutation, real provider call, freeze or launch.

SUPERSET_WORKER_DONE
 task: R10A-DELTA-REVIEW
 summary: PASS exact repaired package; PASS offline gate tooling. Both prior HOLD defects resolved; no frozen launch approval.
 files: analysis/r10-prep-20260916/independent-repair-review/; analysis/r10-prep-20260916/r10a-verification/package-review/DECISION.json
 checks: 16 wrapper tests; 141 gate tests; 5 additional gate boundary tests; exact hash/delta/embedded-byte reconciliation; preserved previous evidence; actual package predicate and expected production gate HOLD.
 handoff: Package decision permits the next authorized review-gated freeze step. Frozen launch review, SDK attachment verification and Hermes release remain outstanding. Available for exact frozen review after a manifest exists.
