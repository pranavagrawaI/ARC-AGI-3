# R8-L3B — pinned SGLang orderly-shutdown runtime successor

**Preparation state:** Replacement local runtime artifact built; not live-qualified and not launchable.

## Question

Can the corrected SGLang runtime preserve the qualified Qwen request path while converting root-first TERM into clean application shutdown with no NCCL process-group warning, `SystemExit: 0` traceback, or `asyncio.exceptions.CancelledError` traceback—without running distributed destruction after an exception?

## Predecessors

Kaggle version 12 is permanently classified `teardown_log_unacceptable`; parent verdict SHA-256 `0f9977f3f47d8149ff515d7e0c20997f2f93fa8990dcfbf49ff9b903d2cd0ded`, independent review `c241e30adfaa1367028fddb01b069f182447d5ca49f4b2461d37f9f9ceceba14`.

The first local R8-L3 candidate is retired. Its implementation review `b45cd965c20df76a7fa369f363d0f3475166944c627e54109565c039678f4bb8` rejected cleanup gated only by mutable `scheduler.gracefully_exit`; its artifact review `c104d1008894e9ac1e10d5e7c0ab22f2e17c8c6b6a281869ab6d15bf65d32741` accepted packaging only.

## Fixed comparison

Preserve the version-12 model, exact 2,447-byte request, parsers, SGLang base commit and dependency lock, backend/cache flags, warmup and request counts, root-first observer, deadlines, ownership/resource checks, and strict log classifier.

Change runtime wheel:

- base: `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`;
- rejected R8-L3: `d374ab01ce9d3445a7e201fbd08a16d490ae1ef49e9e7139e58320c272dbfa03`;
- replacement R8-L3B: `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0`.

## Treatment

1. Tokenizer shutdown requests the owning Uvicorn `Server` to exit and returns instead of raising async `SystemExit`.
2. The default single-tokenizer HTTP route binds and later clears that server-exit callback.
3. A scheduler-local `scheduler_loop_returned_normally` witness starts false and becomes true only after `run_event_loop()` returns.
4. Graceful host cleanup and `cleanup_dist_env_and_memory()` require both that witness and `scheduler.gracefully_exit`.
5. Distributed cleanup remains in `finally` after host cleanup, so it still runs if host cleanup raises on the proven normal-return path.

Watchdog stop, `ShutdownReq`, scheduler wait, and child containment ordering are unchanged.

## Local evidence

- Exact upstream wheel and three source-entry hashes are enforced by the builder.
- The callback mechanism passed an independent exact-version Uvicorn `0.52.4`/uvloop `0.22.1`/Starlette `1.6.0` signal-override subprocess probe: root SIGTERM returned `0`, closed lifespan, and emitted no traceback, `SystemExit`, or `CancelledError`.
- Executable scheduler control: normal loop return performs metrics, host, and distributed cleanup; exception after `gracefully_exit=True` performs metrics cleanup only.
- 61 deployment tests passed; one unrelated test skipped.
- Ruff passed.
- Two wheel builds were byte-identical.
- Wheel changes are exactly three Python files plus regenerated `RECORD`; all 4,266 rows verify.
- Patched files parse; wheel installs and compiles under Python 3.12.

## Prediction for later qualification

The exact request path passes. Root-first TERM again produces root return `0`, no observer KILL, complete closure, and a valid quiet window. Teardown contains only classified ordinary shutdown notices and none of the NCCL warning, `SystemExit: 0`, `CancelledError`, traceback, or unclassified nonempty lines.

## Result map

- Request/path/count regression → reject.
- Async traceback remains → HTTP exit ownership incomplete.
- NCCL warning remains → distributed cleanup incomplete.
- Exception path enters host/distributed destruction → reject.
- Resource closure regresses or KILL is required → reject.
- Exact request PASS plus root `0`, TERM-only closure, valid quiet window, and strict log PASS → R8-L3B passes and authorizes only preparation of a separately frozen S0 candidate.

## Authority ceiling

This local candidate grants no dataset upload, Kaggle push, GPU execution, retry, ARC action, S0 run, submission, or promotion authority. Live qualification requires new dataset/notebook identities, exact-byte reviews, and separate user authorization.
