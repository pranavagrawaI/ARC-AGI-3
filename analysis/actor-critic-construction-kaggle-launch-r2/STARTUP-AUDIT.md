# R2 startup guard audit

## Minimal runtime delta

The notebook-delivered `donor_bootstrap.py` changes only the supervisor argument `--request-timeout-s 30` to `--request-timeout-s 180`. The supervisor itself is byte-identical to the failed package. The exact R8-L3B parser contract requires startup=1200, request=180; application HTTP=30 was mistakenly substituted for the latter. Package/title/source identities and explicit cap documentation are updated separately.

**Important precision:** 180 is the donor supervisor's required CLI request-timeout setting, not a newly imposed 180-second warmup timer. In this official-launcher branch the runtime-owned single warmup is observed under the remaining 1200-second startup budget (`wait_for_sglang_warmup`, supervisor lines 2055–2065). The direct `post_bytes(...args.request_timeout_s)` path is below the official-launcher branch's return (2224 versus 2254–2257), so it is not used by this integration. Do not describe this repair as extending application HTTP or changing warmup implementation.

## Next guards inspected

- Parser: exclusive mode flags, Triton/PyTorch backends, paired launcher/probe paths, R8-L3B ownership, port=30000, exact startup/request pair, and positive launcher timeout. Generated successor argv satisfies all; mutation regressions exercise startup, launcher, backend and port rejection. No second timeout-domain conflict found.
- Supervisor real main: output identity creation, actual system snapshot, accelerator preflight, lifecycle observer preflight, pinned runtime provenance/files, pinned mounted-model verification, server startup, warmup evidence, backend resolution, and request-count reconciliation. First unavailable prerequisite on this local host is accelerator discovery; later GPU/model/runtime predicates are inspected, not remotely qualified.
- Runtime identity verifies attached wheel/provenance bytes; no pin is weakened. Bootstrap still qualifies GPU and sealed execution before installer. No installer/model/provider operation is executed by the new CLI regression.
- Startup HTTP polling remains five seconds per GET; startup plus warmup observation remains under 1200 seconds. Notebook remaining-work alarm stays 3100 seconds; outer guardian stays 3500 plus eight seconds retirement. Positive launcher timeout stays 1020 with existing cleanup reserves; probe admission is `min(900, launcher_timeout - 60)`.
- Application relay keeps HTTP POST=30, constructor soft=45/hard=50, 24 total chat requests including one runtime warmup (23 application), Actor/summarizer=20, actions=8. Existing real-worker/loopback fixture tests and deadline tests cover these paths. No global bound increases.

## Regression evidence and limits

`tests/test_supervisor_cli.py` decodes the actual predecessor and successor notebook bootstrap bytes and evaluates their command assignment with the same materialized root. Their only argv difference is 30→180. Both are submitted to the same unmodified packaged `parse_args` in fresh `-I -S -B` interpreters: predecessor rejects with exit 2; successor parses successfully. This does not bypass the parser by calling `run_probe` with a hand-built namespace.

The real supervisor executable also runs with only its output directory redirected and isolated interpreter flags. It returns exit 1 with `result.json` status `invalid`, failure_stage `accelerator_preflight`; config records 1200/180, and real system-before evidence exists. This is successful local regression of boundary progression, not successful model startup. No synthetic GPU is needed to establish the first unavailable prerequisite.

Initial test assertions incorrectly expected receipt key `stage`, then status `failed`; inspection established the actual producer fields `failure_stage` and `invalid`. Only the new test assertions were corrected. The failed full-suite output is retained as `LOCAL-TEST-initial.stdout`; no production change was made to accommodate these assertion errors.

Failed predecessor prep/package/run bytes remain protected by `PREDECESSOR-HASHES.json`; `verify.py` checks them plus the 422-file OpenRouter binding and all fresh notebook/package/origin hashes. No push, provider call, real GPU/model inference, or remote-receipt claim is made. Launch gate, authorization, independent review and remote observation belong to the coordinator.
