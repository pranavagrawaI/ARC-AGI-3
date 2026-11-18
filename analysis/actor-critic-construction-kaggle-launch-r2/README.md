# Kaggle package repair and local qualification

This directory owns packaging adaptations only. It does not modify the successful v2/OpenRouter implementation, constructor donor, core integration, or v11 runtime. `verify.py` reconciles the protected 422-file OpenRouter manifest as well as the generated notebook and package.

## Recheck frozen bytes

Run from the repository root:

```sh
deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-launch-r2/verify.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-launch-r2/tests
deployment/kaggle/.venv/bin/python -B -m ruff check --no-cache analysis/actor-critic-construction-kaggle-launch-r2
```

The tests rebuild only into disposable directories and compare against the frozen manifest. They do not rebuild the real package or edit predecessor files. `build.py` is the explicit owner-only regeneration command, not a launch command. `FREEZE.json` records final exact hashes and executed check results; it is not independent approval or launch authorization.

## R2 delta

`STARTUP-AUDIT.md` records the parser repair and subsequent guard audit. The new real-CLI regression reads the generated predecessor/successor notebook argv, proves rejection/acceptance through the same parser, and exercises the real supervisor to local accelerator preflight. The 180-second CLI setting is not the application's 30-second POST cap; runtime warmup observation uses the remaining startup budget. No launched bytes are edited.

## Repairs

| Boundary | Implementation and evidence |
|---|---|
| Notebook delivery | Actual notebook embeds every required source and hash. Clean and identical existing assets materialize without adjacent-file assumptions or `SameFileError`; drift, symlinks and unexpected capsule files fail closed. Bootstrap and guardian run in fresh `-I -S -B` interpreters, independent of notebook module caches. |
| Runtime closure | v11 worker, state delivery, wrapper solver and chassis are packaged under `nooa-join/`. Flat v2/constructor imports are adapted to capsule-relative paths. Extracted driver functions, test Agent base, and hash-pinned frame converter close the driver dependencies without requiring pytest in the remote venv. |
| Donor lifecycle | Builder decodes only the pinned production supervisor and model manifest. `runtime_install.py` preserves the donor offline install/venv/provenance producer, excluding its obsolete archive extraction. No old `remote_entry.py`, constructor case, sealed trace or nested zip is embedded. |
| Local transport | Explicit logical-to-SGLang projection strips gateway and response-only fields, preserves opaque tool IDs and rejects mismatched continuation. Local response model must equal the exact mounted Qwen path. Startup argv must use that exact `--model-path`; missing argv is never fabricated. Raw receipts remain unchanged, without CoreWeave/provider tags. |
| Accounting/acceptance | Audit consumes raw local receipts and identity evidence. The full production supervisor invokes the actual probe, checks successful integration, rereads persisted accounting and reconciles server access counts. It accepts `local_requests`, not invented legacy Solver sessions. One runtime warmup is reserved inside the total 24-call ceiling, leaving at most 23 application calls and at most 20 Actor/summarizer calls. |
| Stop/cleanup | The next model admission is refused once integration is complete. Proven v2 abort-retirement is carried into the probe: no extra action or fictitious finish. Failure stops admission and retains the first boundary. Outer notebook guardian is a dedicated Linux subreaper with parent-death signaling and an exact descendant sweep, including independently sessioned children. Unexpected survivors make success fail. |
| Pre-model containment | Real candidate initialize/predict/assimilate and inspection probes execute before installation/server warmup. Filesystem, stdin and environment checks must pass. Additional tests exercise inherited FD canaries. Failed sealing cannot fall through to model startup. |

## Local evidence boundaries

The HTTP integration test executes the **entire generated production invocation/acceptance/accounting branch**, including a real adapter subprocess, real v2 worker, offline public game, isolated constructor inspection, two-turn constructor continuation, freeze, preview, assimilation, critic and retirement. Only unavailable GPU/model-startup/log context is explicitly supplied by the fixture. Fixture counters and hardware context are labeled synthetic, not real SGLang/GPU observations.

Variants cover automatic action confirmation and revision, missing usage, wrong model identity, HTTP failure after a completed request, and Actor reserve exhaustion. They assert exact environment/host action agreement, same-digest action/step binding, pre-send qualification stop, retained known usage and no extra admission after failure. Constructor deadline tests wait beyond the released HTTP response to detect late freezes/results. The notebook-death test kills the outer parent and checks that actual separately sessioned descendant PIDs are gone.

The actual default notebook startup is also executed with all fixture-routing environment variables absent and conflicting cached modules present. On this non-GPU host it correctly produces durable failure evidence at GPU discovery and does not reach installation/model startup. Synthetic executable fixtures separately exercise trusted query execution under the scrubbed environment: missing binary, loader-style failure, malformed response, wrong device and a labeled positive control.

## Remaining remote/coordinator gates

- No real RTX PRO 6000, mounted model, offline wheel installation, SGLang CUDA startup or model-backed inference was executed here. The successful application fixture is not a successful cold-start Kaggle notebook run.
- The target-image seal and model-server shutdown observer must produce real receipts. Local containment and descendant cleanup tests do not establish GPU-image kernel behavior or provider shutdown.
- Provider timeout/job identity/quota/terminal-status readback, target absence, exact metadata/image attachments, sole-push authorization and affirmative exact-manifest independent review belong to the coordinator. Local monitor exit is not cancellation.
- The driver is the inherited `test_join_qualification.build_agent` with a test Agent base, **not `Agent.main`**. ABI replay/assimilation is unscored; integration does not establish prediction accuracy or gameplay improvement.
- `MANIFEST.json` intentionally keeps `complete: false` and `push: false`. The local repaired package is reviewable, not remotely qualified or authorized. `launch_once.py` still refuses even when given a PASS marker.
