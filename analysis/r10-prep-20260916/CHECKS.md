# R10 local checks

All commands ran from `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus`. These are local scripted HTTP/CPU/public-game fixtures, not target-GPU/model qualification or scientific outcomes. Preserve existing evidence directories. For a reviewer rerun, choose a fresh basetemp and log suffix instead of overwriting the paths below.

## Independent predecessor baseline

```bash
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-r9-gameplay-candidate/tests
```

`R9-baseline.stdout`: 105 passed in 105.47s. Some inherited tests import older packages; this result establishes the predecessor baseline, not R10 acceptance.

## Final R10a package tests

```bash
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/tests --basetemp=analysis/r10-prep-20260916/local-final
```

`R10A-final.stdout`: 14 passed in 75.08s. Covers generated notebook loading, actual official loop in both arms, six-request provider-shaped continuation/inspection/repair, CLICK binding, call-cap and malformed/failed HTTP controls, original s5i5 source reproduction, legitimate escaped-string sealed execution, deferred Actor-cell construction, typed scoring replay, R9 stale-state reproduction versus repaired overlay, and killed constructor/no late freeze.

## Constructor exceeds Actor cell limit through the real official loop

```bash
R9_PACKAGE_GAME=cd82 PYTHONDONTWRITEBYTECODE=1 R10_FIXTURE_CONSTRUCTOR_DELAY=11 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -x -p no:cacheprovider 'analysis/r10-prep-20260916/tests/test_runtime.py::test_actual_supervisor_worker_http[None-False]' --basetemp=analysis/r10-prep-20260916/local-long-constructor-v2
```

`R10A-long-constructor-v2.stdout`: 1 passed in 78.75s. Six HTTP responses each delayed 11s; constructor measured 66.684002s. Uses fixture Actor admission120s and fixture launcher900s so the probe can exercise the delay; production retains600s/1800s. Effective constructor6 requests/240s soft/245s hard are read from the real entry. This demonstrates operation beyond60s, not exhaustion testing of every possible240s path.

## R10b prepared schema

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/r10b-preparation/test_world_theory_schema.py
```

`R10B-schema.stdout`: 9 passed. Structural preparation only; the full R10b transport/context/deadline integration remains future build work.

## Artifact and raw evidence reconciliation

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/verify_local.py
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/r10a-verification/collect_plan.py analysis/r10-prep-20260916/local-final/test_actual_supervisor_worker_0/runtime-output/solver-workspaces/raw/RESULTS.json
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/r10a-verification/launch_control.py check --run-dir "$PWD/analysis/r10-prep-20260916/r10a-cd82-20260916"
```

`LOCAL-VERIFICATION.json`: 62 package files and60 embedded assets reconcile;63 predecessor files unchanged; exact requests, repairs, official counts, source digests, coordinates and transition bindings checked. Each success fixture has2 prospective executed queries,1 assimilation and1 final query left pending at the stop. Pending is not scored or mislabeled assimilated. `LOCAL-COLLECTION-PLAN.json` names exact session and partial constructor paths. `GATE-HOLD.stdout` exits2 on missing manifest using the exact absolute run directory. Zero remote calls.

## Retained failed local fixtures

- `local-official-loop/`: fixture classified constructor by presence of submit_candidate, which the new inspection-first phase omits. It sent Actor code during inspection. Corrected fixture classification uses request_evidence, matching production role classification.
- `local-official-loop-v2/`: freeze and CLICK query/assimilation succeeded, but inherited audit rejected coordinate-bearing action labels. Fixed the R10 audit while preserving coordinates.
- `local-regressions/`: exact s5i5 source has both literal backslash-n separators and trailing `"}`; newline decoding alone fails. Corrected fixture strips that known trailing fragment explicitly. Production does neither global replacement.
- `local-regressions-v2/`: fixture monkeypatch hit NOOA's dynamic-method guard; fixed fixture installation with explicit metaclass base assignment. No production guard bypass.
- `local-long-constructor/`: shortened outer fixture budget could not fund120s per-arm admission; refused before gameplay. Increased only fixture outer timeout.

These failures are preserved local qualification work, not retries of a frozen or remote experiment identity.
