# R8-L3 — pinned SGLang orderly-shutdown runtime successor

**Preparation state:** Local runtime artifact built; not live-qualified and not launchable.

## Question

Can a three-file patch to the exact Kaggle-v12 SGLang wheel preserve the qualified Qwen request path while converting root-first TERM into clean application shutdown with no NCCL process-group warning, `SystemExit: 0` traceback, or `asyncio.exceptions.CancelledError` traceback?

## Predecessor

Kaggle version 12 is permanently classified `teardown_log_unacceptable` at `analysis/deployment-probes/20260830T212134Z-qwen38-r8l2-kaggle-v12/VERDICT.md`, SHA-256 `0f9977f3f47d8149ff515d7e0c20997f2f93fa8990dcfbf49ff9b903d2cd0ded`. Independent review `c241e30adfaa1367028fddb01b069f182447d5ca49f4b2461d37f9f9ceceba14` accepts that classification.

Version 12 established:

- exact request path PASS;
- one successful runtime warmup and one successful probe request;
- native `take_action({"action":"LEFT"})`;
- observer root-only TERM;
- root return code `0`;
- complete process/listener/GPU closure in `3.012s`;
- valid quiet window;
- lifecycle failure solely from forbidden teardown bytes.

## Fixed comparison

Preserve:

- model dataset, revision, files, and served alias;
- exact 2,447-byte request and response parser;
- SGLang upstream commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`;
- dependency lock SHA-256 `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`;
- Triton attention, PyTorch sampling, FP8 KV cache, memory/cache flags, Qwen parsers, warmup behavior, and request counts;
- root-first TERM observer, ownership model, completed-scan deadlines, KILL containment policy, port/GPU checks, and strict default-deny log classifier.

Change only the runtime wheel from upstream SHA-256 `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d` to local successor SHA-256 `d374ab01ce9d3445a7e201fbd08a16d490ae1ef49e9e7139e58320c272dbfa03`.

## Treatment

1. `tokenizer_manager.py`: after scheduler closure, invoke a callback owned by the HTTP server and return normally instead of raising `sys.exit(0)` through the async task.
2. `http_server.py`: instantiate the single-tokenizer Uvicorn `Server`, bind a callback that sets `server.should_exit = True`, and clear that callback after `server.run()` returns.
3. `scheduler.py`: on graceful scheduler exit, run `cleanup_dist_env_and_memory()` in `finally` after host-resource release so every distributed group is destroyed even if host cleanup raises.

Watchdog stop, `ShutdownReq` dispatch, scheduler wait, and child containment ordering remain unchanged.

## Local evidence

- Builder: `deployment/kaggle/scripts/build_sglang_r8l3_wheel.py`.
- Tests: `deployment/kaggle/tests/test_sglang_r8l3_wheel.py`.
- Runtime manifest: `deployment/kaggle/runtime-patches/r8l3/manifest.json`.
- Patched wheel: `deployment/kaggle/runtime-patches/r8l3/sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl`.
- Two builds were byte-identical.
- Wheel changes are limited to the three named Python files plus regenerated `RECORD`.
- All 4,266 `RECORD` rows verify.
- All three patched source files parse.
- The wheel installs and compiles in a clean Python 3.12 environment.
- A discriminating executable control proves the original watchdog raises `SystemExit` while the transformed method stops the watchdog, dispatches scheduler shutdown, contains remaining children, requests server exit, and returns normally.
- 25 focused R8-L3/R8-L2 tests pass; Ruff passes.

## Prediction for later qualification

The unchanged request path passes. Root-first TERM again achieves complete closure without observer KILL and root return code `0`. The teardown suffix contains ordinary Uvicorn shutdown/exit notices but contains none of:

- `destroy_process_group() was not called before program exit`;
- `SystemExit: 0`;
- `asyncio.exceptions.CancelledError`;
- any traceback or unclassified nonempty line.

## Result map

- Request/path/count regression → reject the runtime patch.
- Missing distributed warning but async traceback remains → HTTP exit ownership is incomplete.
- Async traceback absent but NCCL warning remains → scheduler distributed cleanup is incomplete.
- Resource closure regresses or KILL is required → reject the runtime patch.
- Exact request PASS plus root `0`, TERM-only complete closure, valid quiet window, and strict teardown-log PASS → R8-L3 passes and authorizes only preparation of the next separately frozen S0 candidate.

## Authority ceiling

This local artifact grants no dataset upload, Kaggle push, GPU execution, retry, ARC action, S0 run, competition submission, or promotion authority. A later live candidate requires a new private runtime dataset identity, notebook/probe bindings, local validation, independent exact-byte reviews, and separate user authorization.
