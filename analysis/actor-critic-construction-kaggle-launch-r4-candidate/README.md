# R4 exact public game discovery repair

Fresh prep/package only: `pranavagrawai/actor-critic-construction-kaggle-r4`, coordinator run `20260914T195518Z`. Frozen R3/earlier packages and `analysis/actor-critic-construction-kaggle-launch-r4/{HOLD.md,DIAGNOSIS.json}` are preserved. No provider calls, pushes, model requests or competition submissions were made by this implementation owner. `launch_once.py` still refuses; package `complete: false` / `push: false` is not launch authority.

## Changes

- `game_preflight.py`: stdlib-only exact public LS20 source/metadata size+SHA gate, strict directory/file types, no symlinks, no ambiguous versions/extra selected files, durable pass/failure receipt. Only known LS20 paths are examined; no broad input-root scan.
- `donor_bootstrap.py`: gate after existing GPU/seal qualification, before offline installer, model startup or warmup.
- `kaggle_probe.py`: corrected competition-mounted default, application revalidation before relay admissions, narrowly selected version-directory discovery, explicit OFFLINE mode and full `ls20-9607627b` make call, explicit duplicate-discovery/None rejection.
- `build.py`: ship gate and bind fresh identity/card. Game bytes are **not** shipped. Supervisor/installer, donor/core/v2/v11, policy, ABI, containment and all resource caps remain unchanged.

`SOURCE-IDENTITY.json` binds the coordinator's exact public files, retrieval receipts and public manifest. The mismatching checkout source remains unchanged at SHA `298c810da2850d557c95d92a2cbd846df29a45d7134e20888617bedf5dafcd92` and is rejected by the gate.

## Qualification

The complete R4 suite retains all 36 R3 case IDs and adds 27 cases (63 total): exact public provenance; 20 generated bootstrap/application invalid-source cases; valid pre-installer default-path positive control; 3 adjacent discovery/make/reset failures; R4 asset/cap scope; and an additional generated R4 compiled-loader run alongside preserved R2/R3. `TEST-INVENTORY.json` mechanically reconciles old/new test IDs.

All six actual application/supervisor HTTP cases now run the generated notebook's unchanged packaged app and real worker/sealed constructor against the exact public game in a **private read-only mount at the production default path**, without `--environments`. Scripted loopback HTTP is labeled synthetic upstream, not actual model inference. Positive cases exercise original/revised Actor actions, critic, two constructor turns including real inspection, frozen query/assimilation and retirement. Failure cases retain missing usage, wrong model, second-request transport failure and Actor cap. Seeded ambient ONLINE mode cannot override explicit OFFLINE.

The dynamic-loader, generated CLI/parser, real candidate+inspection inherited-input/loader isolation, constructor deadline and descendant cleanup regressions remain. See `STARTUP-AUDIT.md` for adjacent seams and limitations. `FINAL-TEST.stdout` / `FINAL-TEST.xml`, `LINT.stdout`, `LOCAL-CHECK.json` and `FREEZE.json` bind final results and bytes.

## Read-only recheck

```sh
deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r4-candidate/verify.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-launch-r4-candidate/tests
deployment/kaggle/.venv/bin/ruff check --no-cache analysis/actor-critic-construction-kaggle-launch-r4-candidate
```

Tests materialize/rebuild only disposable packages. Do not run the owner-only `build.py` on a frozen delivery. Provenance recomputes embedded assets, origins, 422 protected OpenRouter sources, predecessor/diagnosis/checkout bindings and public retrieval identities. FREEZE binds all final prep/package files except itself and generated caches.

## Limits

Local fixtures do not establish successful target model startup/inference, current Kaggle mount availability, target hardware/seal or provider quota/teardown. The prior RTX PRO 6000/warmup/shutdown observations belong to R3, not R4. The driver remains `test_join_qualification.build_agent` with a test Agent base, **not Agent.main**. Integration success is not prediction accuracy or gameplay improvement. Coordinator alone owns independent exact-hash review and any authorized sole launch. No more edits after freeze.
