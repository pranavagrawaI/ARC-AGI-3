# R4 HOLD — public game identity prerequisite unresolved

No R4 deployment package, launch authorization, provider call, or launch was created. R1/R2/R3 and scientific-policy sources were not edited.

## Diagnosis

R3 `kaggle_probe.py:104` defaults to `/kaggle/input/arc-prize-2026-arc-agi-3/environment_files`. The bootstrap launches the production supervisor, whose application argv does not override `--environments`. Neighboring `deployment/kaggle/submission-20260912-v15/build.py:39,61` and `s0-public1-singular-action-v13-candidate/build_notebook.py:238-261` instead use `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files` and verify the public manifest.

Installed Arcade recursively scans metadata under the supplied directory. A missing directory silently produces an empty inventory. OFFLINE `make('ls20')` returns None when no matching game is discovered. R3 then calls `reset()` without checking the result. The retained terminal receipt reports that exact exception at application line 277. The complete remote mounted inventory is not available; the missing `/competitions/` component is a source-supported cause, reproduced locally, not a newly observed remote mount fact.

`diagnose.py` exercised real Arcade in a fresh `-I -B` interpreter, with an allowlisted environment and temporary Kaggle-shaped directory tree: R3 path => empty discovery, None, identical reset exception; established neighboring path => discovery and reset of `ls20-9607627b`. The positive control deliberately uses the **nonmatching local source** and does not establish exact Kaggle game parity or packaged worker startup.

## Identity blocker

Public manifest: `analysis/deployment-probes/20260901T054322Z-public25-kaggle-v1/public-environment-manifest.json`, SHA256 `842477b6336e4f556be999d6c9b76664238a5c7d991740263abf72fc1f864a35`.

- Public `ls20.py`: 104880 bytes, SHA256 `9607627b3d85bdb21ff28b40fefaa6a873691175981c6351b7e5a6637416f9d5`.
- Available checkout `ls20.py`: 105874 bytes, SHA256 `298c810da2850d557c95d92a2cbd846df29a45d7134e20888617bedf5dafcd92`.
- Public metadata SHA256 `181e2f8a540962461a1c799160bdd11963d7276d6c42a368324dd84f3573803f`; checkout metadata SHA256 `8e5c8d64a3c6537e9dd126c53b146c42fa9f89903ed22a7e2d30d4ca5b08782c`.

Repository source walk excluding caches/virtualenvs found only that nonmatching game source. `analysis/deployment-probes/20260905-q6-local-package-tests/ENVIRONMENT-CHECK.json` and RESULT.md already document the same mismatch; removing a license header did not establish equality. The local source is also bound by the consumed OpenRouter manifest. Neither silently replacing it nor treating its version label as byte identity is justified.

Required unblock: provide the existing public-manifest-matching LS20 source and metadata from an attributable mounted/archive source, without changing the pinned game or protected checkout source. No provider access was used to retrieve it under this task's prohibition.

## Adjacent seams audited

- Current fixture app wrapper explicitly injects `--environments` pointing at checkout (`tests/test_runtime.py:150`), hiding the actual default-path failure and source disparity.
- `make('ls20')` chooses latest `date_downloaded` when more than one version exists. A future mechanical resolver must reject missing, wrong, ambiguous, malformed and symlinked sources before game execution and verify the pinned public files; no broad scan of wheel/model datasets.
- Environment-source validation presently occurs only after costly model startup. A future repair should validate mounted files before installation/serving with stdlib-only code, then revalidate at application discovery/reset.
- Arcade's `OPERATION_MODE=competition` can override an explicit OFFLINE constructor. The scrubbed default environment is therefore an important invariant; do not restore ambient environment variables as a discovery workaround.
- Reset/import/discovery exceptions occur before worker creation; early failure evidence must remain durable and account for zero application requests. Existing early GPU/seal calls also precede the application's later failure-handler region.

## Recovered remote evidence and targeted collector request

Coordinator recovered exact-version-number1 `model-runtime/official-launcher-result.json`: failed reset, zero admitted/receipted application calls across actor/summarizer/constructor/critic. `model-runtime/result.json` confirms official-launcher failure and lifecycle passed, term-only true. This does **not** mean zero model calls: the runtime separately completed one warmup (coordinator's recovered startup evidence).

If another bounded diagnostic is useful, the producer at packaged `official_launcher_runtime.py:2175-2181` writes **`model-runtime/official-launcher-command.json`**, containing full application stdout/stderr and return code. Its stdout should distinguish empty Arcade inventory from a game import failure. The remote runtime did not produce a mounted game file/hash manifest, so this receipt alone cannot recover the exact missing public asset bytes for local packaged qualification.

## Executed verification and limits

- Diagnosis real-Arcade reproduction: passed its two assertions; positive control explicitly not public parity.
- Ruff on new diagnostic script: all checks passed.
- Existing R3 provenance verifier: passed, 62 assets, 55 compiled Python sources, 422 protected sources, 312 predecessor files, no drift.
- R3 manifest SHA256 `29f552e70dada5a70396cf2d701ab2e8faae050d83fab3f68e0f575a17823fc0`; notebook SHA256 `fefbf8a4a22781df03590280f9d3ec3703d4d435ed5569d2580374d46252dfb5`.
- Full R4 suite, generated loader regression, packaged-entry real public reset/worker startup, wrong/ambiguous-source repair tests, and R4 launch freeze: **NOT EXECUTED / NOT CREATED**, because the conditional identity prerequisite failed. Existing R3 baseline tests were not edited or rerun.

This is a frozen diagnosis/HOLD receipt, not an R4 implementation or approval. No scientific conclusion is supported by the failed R3 run or the local control.
