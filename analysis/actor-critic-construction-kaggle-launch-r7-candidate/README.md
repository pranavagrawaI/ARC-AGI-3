# R7 leftover-descendant teardown repair — locally verified, not launched

## Outcome

R6's process guard observed a successful `donor_bootstrap` child (rc=0) then flipped
to failed because leftover pid 1707 was still live. R7 reaps those orphans inside
trusted bootstrap shutdown and leaves the leftover-descendant guard unchanged.
R5 constructor disclosure and R6 host-owned freeze use are preserved. Caps,
public LS20 path, and `launch_once.py` refusal are unchanged. No remote launch.

- Prep: `analysis/actor-critic-construction-kaggle-launch-r7-candidate`
- Package: `deployment/kaggle/actor-critic-construction-kaggle-r7`
- Predecessor R6 suite (86) remains passing, plus new guard/teardown cases: 94 total.
  `tests/test_teardown.py` reproduces the R6 failure shape (setsid orphan under a
  rc=0 child fails the unchanged guard), proves the reap for a direct child and
  for an orphan reparented after its intermediate exits, and proves the guard
  still fails an orphan bootstrap does not own.
- Ruff passes for owned R7 prep and the packaged contract module.
  Full-package Ruff retains the inherited R4 diagnostic baseline, with none
  introduced. See `PACKAGE-LINT-BASELINE.json`.

## Active path traced

1. Outer `process_guard.py` remains a dedicated Linux subreaper. After the child
   exits it still fails a passed status if any descendants remain (lines 82-84).
2. `donor_bootstrap.py` now becomes a child subreaper before launching the
   supervisor, then `reap_owned_descendants` records each owned leftover's pid/comm/cmdline,
   SIGKILLs and reaps it, and writes `SHUTDOWN-REAP.json` before returning 0;
   it raises after a 30-second bound if any remain.
3. A successful integration therefore returns 0 with zero live descendants. If a
   descendant cannot be reaped, bootstrap fails and the existing guard still fails.
4. R6 host-owned freeze use in `live_agent.py` and the R5 constructor contract
   are byte-identical to the frozen R6 package.

## Files and scope

Owned R7 edits are `donor_bootstrap.py`, this report, the card, builder identity
retargeting, and `tests/test_teardown.py`. Frozen R6 files are not edited.
`PREDECESSOR-HASHES.json` keeps historical paths; R6 bytes are appended, not
rewritten in place.

## Verification evidence

- `FINAL-TEST.stdout` / `FINAL-TEST.xml`: full successor suite.
- `TEST-INVENTORY.json`: all predecessor cases preserved plus teardown cases.
- `LOCAL-CHECK.json`: package/notebook/origin/protected/predecessor reconciliation.
- `LINT.stdout` / `PACKAGE-LINT-BASELINE.json`: clean owned source and unchanged
  packaged baseline diagnostics.

Reproduce from repository root (no provider calls):

```sh
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r7-candidate/build.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-launch-r7-candidate/tests
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r7-candidate/verify.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r7-candidate/contract_evidence.py
deployment/kaggle/.venv/bin/python -m ruff check --no-cache analysis/actor-critic-construction-kaggle-launch-r7-candidate deployment/kaggle/actor-critic-construction-kaggle-r7/constructor_contract.py
```

## Remaining limits

No launch, provider interaction, model-generated success, target GPU/image
qualification or gameplay improvement was tested. Existing local fixture
boundaries for GPU, loader and model serving remain explicit. R7 requires
separate exact-hash review and remote authorization if the coordinator wants
to run this teardown repair. The remote leftover pid 1707 was not named in
retained GPU-process evidence; local fixtures record synthetic leftover identity,
and the remote `SHUTDOWN-REAP.json` will name the R7 equivalent if one exists.
