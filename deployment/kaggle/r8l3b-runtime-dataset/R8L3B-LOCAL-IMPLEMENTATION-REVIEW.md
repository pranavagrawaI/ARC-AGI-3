# Fresh independent implementation/lifecycle review — R8-L3B

## Decision

**ACCEPT for local credibility only.** The frozen R8-L3B wheel corrects the rejected R8-L3 exception-path defect: the scheduler normal-return witness is false before the `try`, is set true only on the statement after `run_event_loop()` returns, and host plus distributed cleanup require both that witness and `scheduler.gracefully_exit`. The frozen default single-tokenizer HTTP route has a credible Uvicorn-owned shutdown path and the tokenizer watchdog returns without `SystemExit`.

This acceptance does not qualify a dataset, notebook, GPU execution, request result, or Kaggle lifecycle. It grants no upload, launch, retry, ARC action, S0, submission, or promotion authority.

## Exact candidate identity

All five participants matched the requester-supplied digests before review. Their final hashes also matched after the full suite, dynamic probes, rebuilds, installation, and report creation.

| Participant | SHA-256 before | SHA-256 after |
|---|---|---|
| `deployment/kaggle/PROTOTYPE-r8l3b-sglang-orderly-shutdown-EXPERIMENT-CARD.md` | `ca77e4a1a4bf49c3a32b007ab7b36f0509d79d1e1729a2dffd22691c378ec4aa` | `ca77e4a1a4bf49c3a32b007ab7b36f0509d79d1e1729a2dffd22691c378ec4aa` |
| `deployment/kaggle/scripts/build_sglang_r8l3_wheel.py` | `3ce16555e2625ef72d10bc4e9096d8eed320a9be8f6ea33aa539505900220c6b` | `3ce16555e2625ef72d10bc4e9096d8eed320a9be8f6ea33aa539505900220c6b` |
| `deployment/kaggle/tests/test_sglang_r8l3_wheel.py` | `43f90de8c977248deb7eb8eb7913725437177075aa618eab86562b14f1da04ee` | `43f90de8c977248deb7eb8eb7913725437177075aa618eab86562b14f1da04ee` |
| `deployment/kaggle/runtime-patches/r8l3b/manifest.json` | `adb11a6608be04399bd903de99c96288ace6d69fd58520214de3dc9c98c2a630` | `adb11a6608be04399bd903de99c96288ace6d69fd58520214de3dc9c98c2a630` |
| `deployment/kaggle/runtime-patches/r8l3b/sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl` | `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0` | `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0` |

The supplied upstream wheel at `/tmp/sglang-r8l3-runtime/sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl` recomputed to `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`. The pinned checkout `/tmp/sglang-r8l3-source` was clean at commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`. Each of the three source entries in that wheel was byte-equal to its checkout counterpart.

## Prior blocker and correction

The rejected R8-L3 wheel allowed this sequence: `ShutdownReq` set `scheduler.gracefully_exit`, `run_event_loop()` then raised, and the `finally` block still entered host/distributed destruction. That contradicted the adjacent exception-path hang precaution.

The actual R8-L3B wheel source at `sglang/srt/managers/scheduler.py:5408-5452` now has this state transition and gate:

```python
scheduler = None
scheduler_loop_returned_normally = False
try:
    ...
    scheduler.run_event_loop()
    scheduler_loop_returned_normally = True
except Exception:
    ...
finally:
    if scheduler is not None:
        scheduler.metrics_reporter._shutdown_fpm()
        if scheduler.gracefully_exit and scheduler_loop_returned_normally:
            try:
                scheduler.release_host_resources()
            finally:
                cleanup_dist_env_and_memory()
```

The canonical executable control ran in the full test suite with `gracefully_exit=True` in both cases:

| Control | Observed ordered events |
|---|---|
| normal `run_event_loop()` return | `event_loop`, `metrics_shutdown`, `host_cleanup`, `distributed_cleanup` |
| exception after shutdown request | `event_loop`, `metrics_shutdown` |

Thus the rejected path no longer reaches host or distributed cleanup. The exact wheel control flow separately shows that a host-cleanup exception cannot skip distributed cleanup: `cleanup_dist_env_and_memory()` is the body of the `finally` attached directly to `scheduler.release_host_resources()`. Its exception still propagates after distributed cleanup, so cleanup does not overwrite that failure.

I attempted an additional disposable AST execution of the exact wheel function to inject all three paths, including host cleanup failure. The execution gate did not admit that command, so I do not claim a second dynamic result for the host-failure case. The canonical dynamic controls establish the disputed normal-return/exception distinction; the host-failure property is established by the exact `try/finally` bytes above and Python control-flow semantics.

## Uvicorn ownership and order

The exact wheel changes only the default one-tokenizer branch in `http_server.py:2657-2683`. It constructs `uvicorn.Config` with the same explicit arguments previously passed to `uvicorn.run`, constructs the owning `uvicorn.Server`, binds a synchronous closure that sets `server.should_exit = True`, calls `server.run()`, and clears the callback in `finally`.

The tokenizer installs its running-phase SIGTERM handler later, during ASGI startup via `auto_create_handle_loop()` at `tokenizer_manager.py:2141-2163`. Therefore the callback is bound before the signal handler and watchdog capable of invoking it exist. On shutdown, the preserved tokenizer order at lines 3162-3174 is:

1. stop `_subprocess_watchdog`;
2. dispatch `ShutdownReq()`;
3. wait up to 15 seconds for scheduler processes;
4. hard-contain remaining children with `kill_process_tree(..., include_parent=False, wait_timeout=60)`;
5. invoke the bound HTTP callback.

A fresh subprocess probe used Uvicorn `0.52.4`, uvloop `0.22.1`, and Starlette `1.6.0`. ASGI startup replaced Uvicorn's SIGTERM handler as SGLang does; one external SIGTERM produced return code `0` and this event order:

`callback_bound`, `lifespan_started`, `sigterm_handler`, `watchdog_callback`, `server_should_exit`, `lifespan_closed`, `callback_cleared`.

The server returned with `should_exit=True`. Output contained no `Traceback`, `SystemExit`, or `CancelledError` and ended `EXACT_VERSION_UVICORN_CALLBACK_ORDER_PROBE_OK`.

## Tokenizer no-SystemExit path

A fresh probe extracted the actual `TokenizerManager.sigterm_watchdog` AST from the candidate wheel and executed its already-drained graceful path. It returned `None` with ordered events:

`log_info`, `watchdog_stop`, `dispatch_shutdown`, `kill_children`, `server_exit`.

The extracted method contains neither `sys.exit` nor an explicit `SystemExit` raise. The probe ended `ACTUAL_WHEEL_TOKENIZER_NO_SYSTEMEXIT_PROBE_OK`. If no HTTP owner configured the callback, the new path raises `RuntimeError("HTTP server shutdown callback is not configured")` after child containment; direct `Engine`, HTTP/2/Granian, SSL-refresh, and multi-tokenizer ownership remain outside this frozen route.

## Actual wheel source and change scope

The candidate and upstream archives contain the same 4,266 entry names. Exactly four entries differ:

| Entry | Upstream SHA-256 | Candidate SHA-256 |
|---|---|---|
| `sglang/srt/entrypoints/http_server.py` | `056422af61f51573256560509d42ed4f23151364158474aff1c012cce95c880b` | `c47245e4f64744301dcd9669d3e76984a918b0d8a50b0d0611d8b4dc81ffddbb` |
| `sglang/srt/managers/scheduler.py` | `0d4190dcff3dcda16812fbe95f08ec9c04234a8f12e80a81ff1406c58d46f748` | `dbb04590d727493498ea537bb1b70dc57d9b636da3ed78361093277e6d244b77` |
| `sglang/srt/managers/tokenizer_manager.py` | `0d194294e7d56fe71cb7f13701d6021a0f7c41f5b6480319627ee1bbe8d94110` | `90bf218aec5cf24be7c822821bca6527a3bb1dc834824eae81a06ea862c30cdf` |
| `sglang-0.5.19.dev591+g8e04f66a7.dist-info/RECORD` | `65c901dfaf8940a8ef548b63f74ec8bcb850fc43fda35bcf09379a30a33167c3` | `28ddca28cef98fb4f8dacde4ed0dbee494197705566d0c438f4db2645c37538b` |

Timestamps, compression types, and external attributes remained equal for changed entries. All 4,266 `RECORD` rows independently verified, and all three patched Python files parsed.

The complete source diff contains only:

- HTTP default-server ownership and callback binding/clearing;
- scheduler distributed-cleanup import, normal-return witness, and two-factor cleanup gate;
- tokenizer callback state plus replacement of terminal `sys.exit(0)` with callback validation/invocation.

No request handler changed. No startup route outside the one default single-tokenizer server branch changed. Granian/HTTP2, SSL-refresh, and multi-tokenizer branches are byte-preserved.

## Test and build evidence

- `uv run --project deployment/kaggle --frozen pytest -q deployment/kaggle/tests` → `61 passed, 1 skipped in 9.17s`.
- `uv run --project deployment/kaggle --frozen ruff check deployment/kaggle/scripts/build_sglang_r8l3_wheel.py deployment/kaggle/tests/test_sglang_r8l3_wheel.py` → `All checks passed!`.
- Two independent builder executions and the frozen wheel were byte-identical at SHA-256 `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0`.
- Fresh Python `3.12.3` wheel installation succeeded; `compileall` succeeded with one unrelated existing `SyntaxWarning` in `sglang/srt/utils/rpd_utils.py:318`.
- `git diff --check` passed.

An initial install attempt used the ambient Python `3.11.15` and correctly rejected the `cp312` wheel. Repeating with the pinned Python `3.12.3` interpreter succeeded; the first result is an interpreter-selection control, not a candidate failure.

## Live-only uncertainties

Local evidence cannot determine:

- whether each real Qwen scheduler rank reaches distributed group destruction without NCCL/Gloo delay, exception, or deadlock;
- whether the model starts and the frozen 2,447-byte request still returns the required native tool call;
- whether real root-first TERM emits only classifier-allowed ordinary Uvicorn notices and no NCCL warning, traceback, cancellation, signal-exit, or hard-kill bytes;
- whether TERM-only process, listener, and GPU closure plus the quiet window satisfy the frozen deadlines;
- whether any future command using direct `Engine`, HTTP/2/Granian, SSL-refresh, or multiple tokenizer workers needs a separate shutdown owner.

Those uncertainties require separately authorized dataset/notebook exact-byte review and live qualification. They do not weaken the narrower conclusion that this frozen wheel is locally credible for that later work.

VERDICT: ACCEPT