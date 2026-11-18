# R5 constructor constraint disclosure — locally verified, not launched

## Outcome

The new constructor contract replaces the donor prompt's false semantic promises
and incomplete Python ABI. The live host binding, constructor request assembly,
local relay projection and actual HTTP fixture all deliver the disclosure.
No runtime, evaluator, budget, game source, Actor/critic policy or protected input
was widened or repaired. This is a prompt intervention, not an authorized remote
retry. `launch_once.py` still refuses unconditionally.

- Prep: `analysis/actor-critic-construction-kaggle-launch-r5-candidate`
- Package: `deployment/kaggle/actor-critic-construction-kaggle-r5`
- 86 tests passed; all 63 R4 cases retained, with 23 additional cases.
- 64 embedded assets / 57 compiled Python sources reconcile with notebook,
  package and origins. Zero drift in 422 protected source bindings, 515 predecessor
  file bindings, and five public-identity bindings.
- Ruff passes for the complete R5 prep and new packaged contract module.
  Full-package Ruff retains exactly R4's 124 inherited diagnostics, with none
  introduced. See `PACKAGE-LINT-BASELINE.json`; it is not a claim of a lint-clean
  donor package. Frozen inputs were not edited to clean historical lint.

## Active path traced

1. `kaggle_probe.py` configures the existing 45-second constructor soft budget,
   50-second hard deadline and two requests. Its relay shares 23 application calls
   beneath the 24-server-call ceiling (one warmup). Actor/summarizer cap is 20;
   the relay has no separate critic cap. Construction requests have their own
   episode limit, not a new purse.
2. `headless.py` / `persist.py` / `construction_service.py` / `construction_job.py`
   preserve the host-owned execution path, persisted progress and hard termination.
   `prefix.ActorPrefixCase.compact_view` supplies initial/current integer grids
   and chronological step/action summaries, not all intervening grids.
3. `host.bound_constructor` substitutes `evaluate_visible.evaluate_visible_abi`
   and restores its donor binding afterward. R5's builder replaces the copied
   donor `SYSTEM` assignment with `constructor_contract.SYSTEM`, then appends
   `runtime_contract(limits)` to the actual `run_construction` messages. No checker
   or admission code is changed; an AST comparison checks that exact delta.
4. `constructor.run_construction` enters commit immediately at this package's
   45 seconds/two requests. `execute_python` is absent and rejected. The old
   prefix path advertisement does not make inspection usable. The new prompt
   states this explicitly and tells the model to submit/repair or request evidence.
5. The generic evaluator requires object initialization and assimilation, exactly
   one prediction branch, dict commitments and optional list dependencies. It does
   not check dependency element types, informative commitments, transition truth,
   semantic ancestry or prediction accuracy; assimilation receives only
   `{"status":"unscored"}`. Empty commitments and arbitrary JSON branch state can
   pass. For the downstream frozen client, non-null dict state is recommended
   because `None` is its pending-state sentinel. This recommendation is not
   misrepresented as a generic evaluator type check.
6. `integration_boundary.SealedBoundary`, `containment_base.rootfs/bounded`,
   `qualified_runner` and `inspection_runner` enforce the unchanged chroot,
   uid/gid/capability/syscall seals, current-call-only candidate input, separate
   inspection prefix, clean stdin/fds/environment, fresh-process state, JSON/source
   bounds, deadlines and resource limits. Only explicit stdlib sources are copied;
   preloaded modules are not full-stdlib support. In sealed local replay `copy`,
   `random` and `numpy` are unavailable; `json`, `collections`, `re` and preloaded
   `math` are available. This is local runtime evidence, not a promise about every
   module or the unexercised target image.
7. `frozen_client.FrozenModelClient` queries do not execute actions; assimilation
   must match the executed action and next step. All model state travels through
   JSON, not process globals or writable files.

## Files and scope

New handwritten R5 files are `constructor_contract.py`, `contract_evidence.py`,
`tests/test_constructor_contract.py` and this report. `build.py` adds only the
contract module, replaces the donor SYSTEM and appends actual limit disclosure.
The card and package identity are fresh; inherited tests and verification code
are retargeted to R5. Historical R3/R4 scope checks remain historical; the generated
loader test adds R5 without dropping the R4 case. Existing real HTTP tests now
assert the constructor disclosure on received request bytes.

Relative to the frozen R4 deploy assets, only `constructor.py`,
`EXPERIMENT-CARD.md`, and `nooa-join/CHASSIS.json` change, plus the new
`constructor_contract.py`. Generated `ASSET-HASHES.json`, `MANIFEST.json`,
`kernel-metadata.json` and `construction.ipynb` carry the corresponding new
identity/content. Exact prep and deploy paths/digests are in `CHANGESET.json`;
`FREEZE.json` binds the complete local handoff. No game code or synthetic ABI
example is added to the model-visible package.

## Verification evidence

- `FINAL-TEST.stdout` / `FINAL-TEST.xml`: full successor suite, 86 passed.
- `TEST-INVENTORY.json`: all 63 predecessor cases preserved; 23 new cases.
- `CONTRACT-EVIDENCE.json`: actual bound-constructor outbound requests through
  scripted repair, real sealed valid replay and exact R4 captured-source replays.
  The source with unavailable `copy` still fails; the source with list commitments
  still fails. This is **local replay**, not an invented remote second verdict.
- `LOCAL-CHECK.json`: package/notebook/origin/protected/predecessor reconciliation.
- `LINT.stdout` / `PACKAGE-LINT-BASELINE.json`: clean owned source and unchanged
  packaged baseline diagnostics.

Reproduce from repository root (no provider calls):

```sh
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r5-candidate/build.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-launch-r5-candidate/tests
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r5-candidate/verify.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r5-candidate/contract_evidence.py
deployment/kaggle/.venv/bin/python -m ruff check --no-cache analysis/actor-critic-construction-kaggle-launch-r5-candidate deployment/kaggle/actor-critic-construction-kaggle-r5/constructor_contract.py
```

## Remaining limits

No launch, provider interaction, model-generated success, target GPU/image
qualification or gameplay improvement was tested. The model still cannot inspect
the full prefix in the two-request commit-only episode; no budget/phase change is
implied. The checker remains unscored and permissive about semantic content.
The larger honest prompt consumes existing prompt/time admission capacity; no
claim is made that disclosure alone yields a useful model or avoids future
construction failures. Existing local fixture boundaries for GPU, loader and
model serving remain explicit. R5 requires separate exact-hash review and remote
authorization if the coordinator wants to run this new prompt intervention.
