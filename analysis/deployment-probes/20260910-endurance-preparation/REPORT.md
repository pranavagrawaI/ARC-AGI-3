# Public endurance preparation — 10 September 2026

Subsequent live result: [endurance public v1](../20260910-endurance-public-v1/RESULT.md)
failed before model startup because the generated 600-second startup timeout was
rejected by the r8-l3b parser. Local checks below genuinely passed but missed the
exact deployed launch-argument combination. This candidate is not deployment-ready.

Status: implementation and local qualification complete; no Kaggle launch.
The next public-only run is blocked on Kaggle authentication, not authorized for
competition submission. Previous v5/runtime-v3 and its public-run evidence remain
unchanged. New candidate: deployment/kaggle/submission-20260910-v6, runtime-v4.

## Changes

1. Deadline hierarchy: work ends 3100 seconds after the first notebook code cell;
   outer hard watchdog at 3300; planned Kaggle session cap 3600. Readiness timeout
   600; launcher allowance at most 2700, reduced by actual startup elapsed time.
   Work-alarm exceptions enter cleanup; cleanup has a separate reserve. A hard
   kill is containment, never evidence of clean shutdown.
2. LS20, S5I5, CD82 sequentially, each with 600 seconds of model-call admission
   and at most 200 Solver actions. Fail rather than silently shorten a game's
   precommitted window when startup leaves insufficient time. Budget includes
   environment/worker startup and tools; it is not 600 seconds of GPU decoding.
3. Fail on the first model request exception. Persist the primary error even
   when cleanup also fails; prohibit another logical admission. Disable retries
   in NOOA and the LiteLLM/SDK layers, not merely in an environment variable.
4. Record request IDs in completed-call records, host-received action timestamps,
   and worker peak RSS at model admission/budget exhaustion. These support later
   time-window analysis, not exact GPU memory peaks or precise action execution
   timestamps. Notebook explicitly refuses competition rerun mode.

## Failure that changed the implementation

The original builder did not read ARC_LLM_RETRIES. Its RetryConfig retained
default endpoint retries, and the OpenAI-compatible transport also retried.
The strict real-entrypoint tests exposed multiple HTTP attempts inside a single
logical model admission. Local fake HTTP endpoints were used; no model or GPU.

| Injected fault | Before, offline / gateway HTTP attempts | After, offline / gateway |
| --- | --- | --- |
| Response slower than 2-second request timeout | 3 / 3 | 1 / 1 |
| HTTP 503 | 12 / 12 | 1 / 1 |

Reproduce counts with `jq length local-2/test_real_entrypoint_slow_http0/requests.json`
and the equivalent `slow_http1`, `http_erro0`, `http_erro1` paths. Corrected
targeted results are preserved under local-3 with the same relative paths.
The regression requires nonzero exit, persisted failure, no normal game-progress
record, exactly one HTTP attempt, and no surviving Solver agent PID.

This does not retroactively invalidate successful HTTP accounting in the earlier
public run; it does withdraw any inference that its zero-retry environment alone
proved failure-path behavior.

## Qualification history

- local-1: 31 passed in 240.33 seconds, before strict fail-fast assertions.
- local-2: 29 passed, 4 failed in 232.62 seconds. Preserved, not discarded as a
  recovered success. Failures were the two transport faults in both host modes.
- local-3: all 4 targeted failure checks passed in 42.46 seconds; 19 deselected.
- local-4: all 33 checks passed in 207.57 seconds; console output in
  local-4-pytest.log and full per-test artifacts in local-4/.
- Final notebook description now explicitly says public-only; after rebuilding,
  all 4 selected packaging/deadline checks passed in 0.03 seconds (23 deselected).
  Executable payloads were unchanged by this final documentation-only rebuild.

The local tests execute the official Agent entrypoint and actual packaged NOOA
worker against controlled model responses and local public game environments.
They cover successful actions, idle budget stops, continued next-game operation,
delayed HTTP, HTTP error, infinite Python tool execution, worker retirement,
request accounting, and an actual outer work alarm with a local fake server.
They do not establish target-GPU endurance, live solver progress, long-context
behavior, or nine-hour throughput. No new model or game experiments were launched
on Kaggle during preparation.

## External blocker and next action

`kaggle kernels status pranavagrawai/arc3-first-submission-20260910` returned a
permission error. `kaggle kernels list --mine --page-size 1` then reported
authentication required; see kaggle-access-check.log. The earlier quota reading
is not a current verified launch allowance. Do not push until authenticated
status and at least one GPU hour remaining are freshly checked.

Card and source are staged at ../20260910-endurance-public-v1/; not launched.
After authentication and a fresh status/quota check, push one public-only
3600-second run, record the actual returned version,
preserve all outputs in a new directory, classify operational validity before
reporting solver performance. No automatic retry and no competition submission.
