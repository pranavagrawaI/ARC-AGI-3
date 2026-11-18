# R3 trusted driver-loader propagation repair

Fresh prep/package only; no predecessor, launched file, donor, application, policy, or sandbox change. Target `pranavagrawai/actor-critic-construction-kaggle-r3`; coordinator run `20260914T192100Z`. No provider calls or pushes. The package keeps `complete: false` and `push: false`; `launch_once.py` refuses even with a PASS marker.

## Cause and scope

`STARTUP-AUDIT.md` records the shared failed premise, preserved remote evidence and source-wide startup consumer audit. Bootstrap discovery reconstructed a validated driver environment; supervisor GPU/lifecycle queries and model-server launch instead inherited the notebook scrub. R3 adds only an explicit supervisor exec environment, reconstructing LD_LIBRARY_PATH using unchanged `accelerator.loader_environment()` defaults/ownership checks. It does not mutate bootstrap globals or restore ambient notebook variables. Supervisor, installer, application, CandidateRPC and inspection code remain byte-identical to R2.

## Executable evidence

- `tests/test_driver_loader.py` compiles a tiny C library and real `dlopen` executable. A disposable private mapped-root chroot provides genuine root-owned default allowlist paths without mocking trust checks or mounting host driver directories.
- Both preserved R2 and generated R3 notebook assets materialize and pass bootstrap hashes. Actual bootstrap main runs real initial GPU discovery, and actual generated argv enters the unchanged supervisor CLI/parser/system snapshot/GPU query. R2 initial discovery succeeds but supervisor fails with actual exit12/missing library. R3 query loads successfully and advances to the fixture's unavailable `/proc` lifecycle prerequisite.
- In the same inherited environment, the fixture executes the **exact generated production server-environment construction and Popen statements**, launching the compiled consumer. It observes a successful real dlopen plus owner/profile flags. The inherited-env compute-app query also executes. No model or CUDA success is fabricated.
- The dynamic-loader fixture explicitly substitutes unavailable nested seal/installer prerequisites; it does **not** claim an uninterrupted full model startup. Separate real CandidateRPC and inspection executions start with a genuinely loadable driver in the trusted parent and prove neither child sees LD_LIBRARY_PATH, reads the library nor loads it. Existing full generated-default notebook, failed seal, fd/stdin/environment, HTTP/worker/game, bounds, accounting and descendant cleanup tests remain.
- Trust checks reject writable and non-root-owned directories/files. Defaults ignore seeded ambient LD_LIBRARY_PATH/LIBRARY_PATH. Scope regression pins unchanged runtime assets, caps, driver, and metadata except the new identity/title.
- `LOADER-RED.stdout` is the genuine pre-fix failing regression, not synthesized output. `LOADER-GREEN.stdout` records the post-fix comparison. `FINAL-TEST.stdout`, `LINT.stdout`, `LOCAL-CHECK.json` and `FREEZE.json` bind final checks/bytes.

## Recheck without changing frozen files

From repository root:

```sh
deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r3/verify.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-launch-r3/tests
deployment/kaggle/.venv/bin/python -B -m ruff check --no-cache analysis/actor-critic-construction-kaggle-launch-r3
```

Tests rebuild only disposable packages. `build.py` is the explicit owner-only regeneration command; do not run it on frozen delivery. `verify.py` reconciles origins, notebook/source payloads, all 422 protected OpenRouter bindings and predecessor snapshots. FREEZE hashes cover final package/prep sources and evidence, excluding only the self-referential FREEZE file.

## Unresolved remote scope

Synthetic fixture GPU labels are not NVIDIA observations. Local success does not establish actual RTX PRO 6000/NVML/CUDA compatibility, pinned offline mounts/install, SGLang startup/warmup/inference, real target-image seal or lifecycle teardown. No inference was requested locally or remotely by this owner. Coordinator alone owns independent exact-hash review, provider quota/timeout/identity/terminal-state gates, sole push and subsequent real hardware evaluation. Local monitor/process cleanup is not provider cancellation.

Application driver remains `test_join_qualification.build_agent` with the test Agent base, **not Agent.main**; integration success is not prediction accuracy or gameplay improvement. Fixed caps are in the experiment card and unchanged manifest. No further edits after freeze; any additional repair requires a new owned revision.
