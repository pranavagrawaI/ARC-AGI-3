# R10a freeze and launch gate tooling

Active tools are the `.py` files. `launch_control.py.template` is the retained previous preparation snapshot and must not be used. Tooling has offline adversarial qualification; it is not independent package approval or Hermes release. No production freeze or launch has occurred.

## Required order

1. Independent repaired-package review. Reviewer-owned affirmative record goes at `r10a-verification/package-review/DECISION.json`, using status `APPROVED_LOCAL_PACKAGE`, `manifest_sha256` equal to the exact candidate inventory hash, `notebook_sha256` equal to the reviewed notebook, and `launch_authorized: false`. The legacy field `manifest_sha256` here binds `R10A-PACKAGE-INVENTORY.json`, not the later run manifest. The implementer must not write this approval.
2. `make_manifest.py --run-dir EXACT_ABSOLUTE` checks that approval and every inventory byte before writing anything. It checks card equality, rechecks inputs, exclusively writes manifest then authorization, and refuses any repeat or partial preexisting control state. No remote calls. The user launch instruction is carried forward; the current missing independent approval prevents this step.
3. Independent frozen launch review. `verify_frozen.py --run-dir EXACT_ABSOLUTE` checks frozen source coverage, hashes, package approval and authorization without requiring the review it is supporting. It never writes approval or grants launch authority. Reviewer must also check the exact competition attachment through SDK evidence, limits, lifecycle and user authority. Expected `launch-review.json` contains `status: APPROVED` and the exact run `manifest_sha256`.
4. Hermes independently reruns hashes/gates and SDK attachment verification, then explicitly releases the exact identity by writing `hermes-release.json`: `status: RELEASED`, `authority: Hermes`, exact `identity`, `target`, `manifest_sha256`, `notebook_sha256`, `max_pushes: 1`, `retries: 0`. The implementer must not fabricate this release. It is excluded from manifest sources to permit release after frozen review.
5. `launch_control.py check --run-dir EXACT_ABSOLUTE` requires package approval, frozen manifest, frozen launch review, authorization and Hermes release. `push` performs the same checks before and after read-only quota/target checks, exclusively/fsyncs CONSUMED before one push, and retains the marker on every uncertainty. No automatic retry. `monitor` only checks the exact launched SDK version; its local deadline is not remote cancellation.

## Paths and commands

Exact production run directory:
`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/r10a-cd82-20260916`

Package and inventory remain outside the gate directory and must not change during independent review. Gate code/tests, inventory, independent package review, run card and all package files are covered by the future manifest. Exact package hashes are in `../repair-20260916/REPAIR-HASHES.json`.

Offline qualification:

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/r10-prep-20260916/r10a-verification/test_launch_control.py analysis/r10-prep-20260916/r10a-verification/test_make_manifest.py --basetemp=analysis/r10-prep-20260916/repair-20260916/gate-fixtures-final
```

All approval/manifest/release/consumption records in this suite are synthetic disposable fixtures; every provider subprocess is intercepted and status SDK calls use fake clients. They are not production authority records. Use fresh basetemp/log names for a rerun.

Adversaries cover absent/negative/malformed reviews and authority, source and inventory drift, metadata/game/machine restrictions, symlinks and bytecode, no pre-approval freeze writes, approval recheck, exclusive/partial freeze, source coverage, quota/target ambiguity, revocation during preflight, consumed-marker race, uncertain push with no retry, exact-version monitoring, and a frozen-byte check that cannot enable push without launch review/Hermes release.

SDK attachment verification is still a real prelaunch evidence requirement. Metadata alone does not establish remote attachment. This task made no SDK/provider call or admission. `collect_plan.py` remains a read-only exact-path planner for recorded sessions, IPC, child forensic artifacts and official recordings; collection later must be version-bound.
