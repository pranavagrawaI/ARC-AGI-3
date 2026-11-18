# READY — launch prohibited

Claim: **READY FOR READ-ONLY REVIEW AND A SEPARATELY AUTHORIZED ONE-RUN LAUNCH; NOT LAUNCHED.** r3 makes no Kaggle push, provider/model call, gameplay action, or competition submission.

## Closed contracts

- The actual r3 `remote_entry.py` ran through a three-request `ScriptedServer`, wrote `status=passed`, and was then accepted by the exact decoded v15 embedded supervisor status-and-reconciliation branch. `solver_sessions` maps the donor’s required Actor-call fields to three durable, successful construction-model HTTP request/response receipts; it explicitly says these are not gameplay Actor calls. `gameplay_result` remains null and `candidate_classification` is separate from launcher infrastructure status.
- Transport failure makes one attempt, persists failure evidence, exits nonzero, reports infrastructure `status=failed`, leaves `candidate_classification=null`, and emits no solver-session claim.
- System prompt, user packet, tool description, fixture inspection code, and sealed mount consistently use `/evidence/case.json`.
- Inspection preserves the original 32,000-byte source cap, 8-second timeout, and 20,000-byte aggregate stdout/stderr process cap. It returns the original status/returncode/stdout/stderr/truncation envelope; benign candidate stderr is retained without contaminating the trusted startup seal. The seal travels on a separate closed descriptor. Actual sealed tests cover stderr, nonzero exit, 17,000-character truncation, and rejection/killing of a 21,000-byte flood.
- Candidate calls remain on the qualified unchanged runner with a fresh 15-second shared boundary, 3-second per-call limit, 200,000-byte source/argument/result caps, and 1,000,000-byte process-output cap.
- `KeyboardInterrupt` and `SystemExit` at the host boundary now persist a terminal failure result before propagation. A regression proves a caught interrupt after `SUFFIX-ATTEMPT.json` gets `RESULT.json` plus a terminal event, while external SIGKILL remains marker-only.
- Exactly-once run ownership, opaque tool IDs, suffix-attempt marking, one suffix assessment, and local wire/raw-response/receipt durability remain enforced.

## Verification

- Original constructor suite: **46/46 passed** (`evidence/constructor46.stderr`).
- Qualified containment suite: **53/53 passed** under mapped root (`evidence/containment53.stderr`).
- r3 mapped-root integration/wire suite: **20/20 passed** (`evidence/mapped-root-suite.stderr`).
- Actual generated notebook passed in deterministic normal, cache-poison, and local-HTTP modes. Each made 3 model requests, executed 2 sealed inspections, reached `INTEGRATION_VALID_CANDIDATE_SUPPORTED`, and made exactly one suffix attempt. Local HTTP mode carries 3 reconciled solver requests with durable receipts (`evidence/notebook-*-fs/working/`).
- Prelaunch passed with the launch guard closed (`evidence/PRELAUNCH.json`). Explicit unauthorized launch exited 1 with `LAUNCH_DISABLED` (`evidence/launch-guard.*`).
- Consolidated checks and hashes: `evidence/FINAL-VERIFICATION.json`; reproducible source/package hashes: `MANIFEST.json` and package `BUILD.json`.

Canonical mapped-root test command:

```bash
unshare --map-users 0:1000:1 --map-users 1:100000:65536 \
  --map-groups 0:1000:1 --map-groups 1:100000:65536 \
  --setgroups allow --setuid 0 --setgid 0 \
  python3 -m unittest discover -s tests -v
```

## Launch control and remaining gap

`deployment/kaggle/construction-policy-integration-20260913T221412Z-r3/launch_once.py` remains a disabled placeholder. `AUTHORIZATION.json` is absent and no push command exists.

No target Kaggle execution, live model inference, live SGLang observation, accelerator observation, quota receipt, gameplay result, or remote integration receipt exists for r3. Local HTTP tests use real sockets and SGLang-shaped envelopes but are transport/harness qualification only.
