# Independent repaired-package review

## Decision: APPROVED_LOCAL_PACKAGE

Exact manifest SHA-256: `234a62ec3727d6593043324f3f471877755da0fc146dfaef6d754375531213b3`

Exact notebook SHA-256: `0fc78e34ff45d82e95d815bcd8b39ddef7c6c8067cd2775446df50dea3053962`

No concrete remaining **local-package** blocker was established. This approves only the frozen package's bounded local qualification. It is **not launch authorization, successful Kaggle cold-start qualification, GPU/model qualification, or a scientific/gameplay result**. The packaging card remains zero-push; `launch_once.py` refuses; a coordinator-owned launch gate and remote prerequisite checks remain separate.

## Independent execution

- Ran the actual 25-test repaired suite in a mapped-root user namespace with Python 3.12.14, bytecode/cache writes disabled and disposable output under this review directory: **25 passed in 47.16 seconds**. This is independent execution, not the implementer's or coordinator's reported result. Commands and full stdout/stderr are in `checks.json` and `suite.stdout`/`suite.stderr`.
- Reconciled before and after execution and again before this decision: **62 assets, 55 compiled embedded Python sources, 422 protected OpenRouter source bindings; zero mismatches or protected drift**. All `FREEZE.json` analysis-source hashes also remained unchanged. Final reconciliation is `final-reconciliation.json`.
- Added `extra_reproductions.py`, executed in the same mapped-root namespace. These exercise generated entrypoints rather than only helper predicates:
  1. Materialized the actual notebook into a separate clean layout. Ran the unchanged generated bootstrap through fresh `-I -S -B` Python and the notebook's scrubbed environment, with fixture-routing variables absent and a network-denying audit hook. Supplied only a synthetic NVIDIA-query executable path. Real discovery/parsing and real sealed initialize/predict/assimilate/inspection ran successfully; the actual offline installer then rejected the absent hash-pinned wheel with `expected one exact provider-normalized patched wheel under /kaggle/input, found []`. The bootstrap retained that first failure. This establishes the local interpreter/import/prerequisite ordering **through installer entry**, not wheel installation or a GPU observation.
  2. Executed the complete generated production invocation/acceptance/accounting branch with the real probe subprocess, real worker, offline game and scripted loopback upstream, but deliberately inflated both synthetic server access counters by one. The actual application completed with 14 receipted application requests; the production final reconciler rejected it with `official-launcher request-count contract did not close`. Thus a passed probe is not blindly accepted when server accounting disagrees. Hardware/startup/liveness/log context remains explicitly synthetic, not observed SGLang behavior.

Extra receipts: `extra-results.json`, `cold-entry/stdout`, `cold-entry/stderr`, `cold-entry/actor-critic-construction/{GPU,SEAL,BOOTSTRAP-FAILURE}.json`, and `wrong-counter/rejection.txt` with the underlying application evidence below `wrong-counter/runtime-output/`.

## Prior HOLD findings and repaired behavior

| Prior defect | Observed repaired boundary |
|---|---|
| Missing notebook-delivered assets / SameFileError | Actual self-contained cell materializes clean and identical-existing layouts; hash drift fails. Bootstrap executes in a fresh interpreter rather than trusting cached notebook modules. |
| Repository-relative worker and missing driver closure | Generated capsule contains flat overlays plus `nooa-join` runtime/chassis and extracted driver/frame-converter dependencies. Actual supervisor/probe/worker/game fixtures execute from the materialized layout. |
| OpenRouter/CoreWeave-only success audit | Raw local responses remain local. Model/startup identity, raw-body reconciliation and local usage are checked without fabricated provider tags. Missing usage, wrong model and failure after one valid receipt are independently exercised. |
| Probe incompatible with supervisor acceptance/accounting | Successful real application fixtures are accepted by the generated production branch, while failed application fixtures and the additional deliberately wrong server-count fixture are rejected. No legacy Solver sessions are invented. |
| Forbidden nested donor case / obsolete fixture entry | Decoded notebook assets reconcile to the allowlist. The obsolete case/archive/remote entry is absent; bootstrap has no fixture-routing branch. GPU and actual containment precede installation/server warmup. |

## Containment, wire, stop and cleanup coverage

The independently executed suite covers real candidate and inspection subprocesses with inherited stdin, descriptor, environment and outside-file canaries plus positive controls. It also covers pre-model seal failure blocking installation, constructor hard timeout with a wait beyond the delayed response to detect late freeze/result publication, and outer-parent death with actual separately sessioned descendant disappearance. These are bounded local containment/cleanup checks, not comprehensive adversarial security or target-kernel certification.

The successful HTTP fixtures exercise constructor inspection and continuation, opaque tool-call correlation, response-only metadata removal, local reasoning-disabled projection, real freeze/use, critic, same-digest action/step assimilation, exact environment/host action agreement and pre-send retirement without an extra action. Reserve exhaustion stops at the documented Actor/summarizer allocation. Missing or invalid receipts and completed-response/transport failure paths do not become zero-usage success. The independent wrong-counter reproduction additionally checks production rejection after successful application execution.

## Explicit unqualified boundaries

- No attached runtime wheelhouse, actual offline dependency installation, target interpreter/driver stack, mounted model, SGLang CUDA startup, real server access logs, model inference or real model-server teardown was exercised. The installer uses the bootstrap interpreter for pip, installs the system runtime via uv, then creates and populates the separate Solver venv; source inspection found this setup before its supervisor/worker consumers. The local absent-wheel failure is an expected fixture-context limit, not proof that mounted production dependencies install successfully.
- No provider calls, real model requests, pushes, target-absence checks, remote quota accounting, cancellation, timeout semantics or terminal-state collection occurred. The repaired notebook's process controls do not establish provider shutdown.
- The driver is `test_join_qualification.build_agent` with a test Agent base, not `Agent.main`. ABI assimilation is unscored; scripted integration is not prediction accuracy or gameplay improvement.
- No fresh launch approval is inferred from this document. `MANIFEST.json` remains `complete: false`, `push: false`; final inspected FREEZE says `no_more_edits: true`, `pushes: 0`, and no consumption/authorization/launch-result marker was found in the two candidate directories. The coordinator remains sole launch owner.

All review scripts, logs, disposable reproductions and this decision were written under `analysis/actor-critic-construction-kaggle-verification/repaired-review/`. No candidate or protected source was modified.
