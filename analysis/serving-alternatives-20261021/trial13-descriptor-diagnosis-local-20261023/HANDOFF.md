# Stable local descriptor/lifetime proposal — NO GPU PACKET

Selected correction: selected/phase_tools.py only. Close each completed owned phase's stdout/stderr when its process group is verified empty and removed. Keep active phases' streams open. On second-open/Popen/admission failure, close only the newly opened parent streams and rethrow the original error. No probe IPC, native engine, query ownership/filter, group membership, deadline, resource threshold, cache or scientific admission changes.

The poll-based IPC wait under poll/ is a preserved UNSELECTED alternative. It handles high descriptors but does not stop retained-stream accumulation. Direct-fd2048 read and blocked-write witnesses separate that backend limitation from source lifetime growth. Selected probe_ipc.py remains byte-exact original.

Observed CPU source-path evidence: CPython3.12.14 and3.14.8 both run524 real loopback503→200 requests through unchanged baseline PhaseTools.ready/start/cleanup. Baseline retains1050 parent log handles, then shutdown.observe raises ValueError at probe_ipc.read_exact→select.select (local fd1056). Lifecycle-only keeps2 live-server handles and9fds; full observation completes (CPU fixture still ineligible, no release success claimed). Poll-only completes with1050 retained handles. Current selected lifecycle startup-error cleanup also passes the same pressure case on3.12.14.

This is a causal reproduction of the local source mechanism, NOT exact attribution of the remote failure. Remote fd/callstack absent; remote3.12.13 vs local3.12.14 patch difference explicit. Original trial13 primary error is consistent with this mechanism; the separate UnsafeIdentity555 cleanup error is preserved/unexplained. Historical trial13 NOT_QUALIFIED remains, no release/candidate evidence or promotion.

Selected public contracts:7/7PASS onCPython3.12.14 — completed stdout/stderr preserved/closed; active-server streams remain open through another phase then close at group cleanup; repeated whole cleanup; nonzero exit retains error/logs; missing executable retains FileNotFoundError and releases streams; second-open failure preserves existing stderr and closes stdout; HTTP500 readiness retains status/body/error and active-server streams. Lifecycle-only before startup exception cleanup fails exactly2of7; retained under lifecycle-contract-red. No baseline65 retry or full65PASS claim.

Framing/schema/EOF/partial/stall/deadline tests: original7unchanged tests PASS for baseline/lifecycle/poll, plus7high-fd tests forpoll. Original protocol/filter/ownership code unchanged in selected tree. Every controlled frontend fixture retained pidfds and observed exit; final fds return to4 in pressure cases. Cleanup assertions concern CPU fixture ownership only.

All invocations<=600s; matrices137.163s(Python3.14) and122.745s(Python3.12). No network outside loopback, no GPU commands/model loads/API, no frozen run edits/new notebook/card/SDK/assignment/launch/stage/commit. Original1905bindings reverified unchanged. No controller redesign.

Reproduction (fresh outputs only):
- timeout600s python3 qualify.py --output FRESH_MATRIX_DIR
- timeout600s /home/pranav/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12 qualify.py --output FRESH_PY312_DIR
- timeout600s /home/pranav/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12 run_selected_case.py selected pressure FRESH_SELECTED_DIR
- selected_checks.py and framing_check.py preserve their exact logs; default named output dirs are exclusive, use a separate copied evidence directory to rerun rather than overwrite originals.
- python3 verify_evidence.py checks all prior bindings and declared outcomes without rerunning pressure.

Selected source SHA 42d45e48e115e7390bd3fee9b11623315bbed0887a89df4333ba743a5c32e8f5
Original IPC SHA cd13e64e526e1595fb95643c6250a6bd0ab7af2419a45ce7a5493e448d6afb45
Proof SHA 49cb26bcabd1f884dafabb23cd40bca747ca7be48d67789f6829995b8e618fd8
Evidence inventory SHA 518cdbdb5ecdc1a7564ea985224cf50900c422ecd0fbbd2a43609ffcaa9eee5b
111 source/behavior checks; raw logs and all variants in the inventory.
