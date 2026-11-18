# R10A-REPAIR-AND-LAUNCH-PREP handoff

Repaired package is ready for independent re-review. Two defects reproduced before editing and both acceptance paths now pass. Gate tooling is prepared and locally qualified. **No independent approval, production manifest freeze, authority consumption, launch or submission was performed.** User's launch instruction and existing one-private-push authority remain subject to the mandatory gates.

## Exact package bytes

- Inventory: `b2325fb93db5dfa16254a7d3d6c06b82a8b628bb61f4374276be6fe3e76e8aef`
- Notebook: `87afeeb324d33bafe4a7992abf1f5f17ba9ef83631d716b3c5287a061fa770a3`
- Card, unchanged: `6425dc576370b0f68ea884ff5ea11cbf6989aa80bc08beaf0c86d3c8e4c72bc9`
- Source diff versus R9: `89fe18a9b5fd3e0a7cdecf25d7f59af17138f9661810fe874f009bc5054708cd`

Paths relative to `analysis/r10-prep-20260916/`: `r10a-package/`, `R10A-PACKAGE-INVENTORY.json`, `R10A-SOURCE-DIFF.patch`, `tests/test_repair_boundaries.py`. `repair-20260916/REPAIR-READY.md` describes both fixes. Package bytes have stayed unchanged since the repair-ready report; only verification/docs changed afterward. Budget/game/semantic authority unchanged. No unrelated files or frozen predecessor edits.

## Commands and retained evidence

Run from repository root. Do not overwrite these evidence directories when reproducing; choose fresh basetemp/log suffixes.

```bash
# Before-fix reproduction, executed before package edits. These passing tests prove defects.
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/independent-package-review/test_boundary_probes.py --basetemp=analysis/r10-prep-20260916/repair-20260916/before-probes
# Repair acceptance, real admission/inspection/sealed RPC/live overlay and late child kill.
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/tests/test_repair_boundaries.py --basetemp=analysis/r10-prep-20260916/repair-20260916/after-probes
# Regenerate notebook/inventory, then complete required wrapper suite.
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/build_notebook.py
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/tests --basetemp=analysis/r10-prep-20260916/repair-20260916/full-suite
# Offline adversarial gate/freeze fixtures, no remote admission.
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/r10a-verification/test_launch_control.py analysis/r10-prep-20260916/r10a-verification/test_make_manifest.py --basetemp=analysis/r10-prep-20260916/repair-20260916/gate-fixtures-final
# Actual read-only gate check; expected exit 2/HOLD, no provider calls.
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/r10a-verification/launch_control.py check --run-dir "$PWD/analysis/r10-prep-20260916/r10a-cd82-20260916"
```

Results in this directory: `BEFORE.stdout` 2 reproduced defects; `AFTER.stdout` 2 acceptance passes; `SUITE.stdout` 16 passes in 95.67s; `GATES-final.stdout` 141 passes. `GATE-HOLD.stdout` records missing independent package approval. `RECONCILIATION.json` checks 62 package files, 60 embedded assets, all 2384 reviewer-owned files unchanged, 63 predecessor files unchanged, stable repair hashes, and no production control files created. All observations are local fixtures, not real model/GPU qualification or scientific outcomes.

The null-state regression executes an accepted JSON-null candidate, preserves refusal for duplicate query/wrong action/wrong step, cancels only an unexecuted original-step pending prediction, assimilates the executed CLICK and admits the next DOWN. The late-kill regression executes the real persist job through child freeze creation, delays only its exit, then kills through the real service deadline before promotion. Child forensic bytes persist while audit and session summary both report no live freeze.

## Gate-tool state and remaining gates

Active `r10a-verification/launch_control.py`, `make_manifest.py`, `verify_frozen.py`, and their tests are ready for independent scrutiny. Exact gate hashes are in `GATE-HASHES.json`. See verification README for schemas and source coverage. The inherited preparation `.template` is historical, not the active entry.

1. Independent reviewer re-reviews the reported package bytes and writes affirmative package approval if warranted. Current HOLD is not cleared by implementer tests.
2. Only after that approval may review-gated freeze run. No freeze was invoked on the production identity in this task.
3. Independent frozen launch review, including exact SDK attachment evidence; read-only frozen byte checker does not grant approval.
4. Hermes independently verifies gates/hashes/attachment and explicitly releases the exact manifest. Launch gate requires that release before any provider read or consumption.
5. Only then one private push; zero retries and no submission. R10b run still requires Hermes verification of collected R10a mechanical acceptance. No changes to R10b scope or authority.

SUPERSET_WORKER_DONE
 task: R10A-REPAIR-AND-LAUNCH-PREP
 summary: Both mechanical defects reproduced and repaired; exact package bytes reported for independent re-review; freeze/launch tooling offline-qualified. No production freeze, consumption or remote admission.
 files: analysis/r10-prep-20260916/repair-20260916/; r10a-package/; tests/test_repair_boundaries.py; r10a-verification/
 checks: 2 before defect reproductions; 2 acceptance regressions; 16 required-wrapper passes; 141 offline gate tests; exact-byte and reviewer/predecessor preservation audit; actual gate HOLD.
 handoff: analysis/r10-prep-20260916/repair-20260916/HANDOFF.md
