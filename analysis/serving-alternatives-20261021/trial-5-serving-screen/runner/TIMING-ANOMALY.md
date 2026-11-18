# Unresolved independent CPU timing failure

Status: **UNRESOLVED**. This is additional evidence after `checks.log` / `QUALIFICATION.json`; it does not replace the earlier passing run. No runner source changed in response, and no GPU work was performed.

Independent reviewer `/root/paired_card_review` reported the exact command `python3 -B -m unittest -v test_build test_supervisor` from this directory. Its first run had 26 passing tests and one failure: `test_trickle_response_has_absolute_deadline`. The fixture reported elapsed **20.537394675 seconds** for an explicit total of **8 seconds**, failing the test's **8.3 second** allowance. The excess over the configured total was **12.537394675 seconds**. The full suite took **30.360 seconds**. This is a failed timing check, regardless of the cause.

The failed fixture's temporary `supervisor.jsonl` and raw request output were deleted by unittest `tearDown`. Therefore there is no retained per-event trace from that failed run with which to locate the delay. Reviewer tool-output evidence was requested; this note must not be mistaken for a recovered primary event trace.

Reviewer immediately repeated the isolated fixture: **0.66184 seconds**, terminal error `control: health absolute deadline`, no surviving groups. Reviewer then repeated all 27 tests on the unchanged files: **27 pass in 9.430 seconds**. The runner owner's prior 27-test run passed in 9.368 seconds. These observations show that the failure did not reproduce in those repeats; they do not explain or erase it.

Reviewed source SHA256: `supervisor.py` = `b0908c90f391ff7bcb74e8b4e2c048a2ddeda6ad755b5191c89fcd1d5010f841`; `http_once.py` = `67541bd2591a666414d9268be384e39bad713aaadc2183db42f44cb335e39c66`; `build.py` = `183063de9fe2e55f4c1e6c358e0e07d2a4c43c88af41ca801d0ac34521eef3cb`.

Bounded source inspection found no intended 20-second wait in the failing fixture's normal path: the health request ceiling is 0.3 seconds, its configured TERM grace is at most 0.5 seconds, subprocess wait is at most 0.2 seconds, and monitoring sleeps are 0.01/0.02 seconds. However, process creation, synchronous `/proc` scans, file writes/flushes and OS scheduling are not interruptible by the same Python monitoring loop. Missing failed-run timing evidence prevents attributing the result to any one of those mechanisms. No scheduling or resource explanation has been established, and no targeted repair is supported by the available evidence.

For the proposed real feasibility run, **5100 seconds is the supervisor's cooperative deadline**, with a **separate required 5400-second platform boundary**. The coordinator must review this unresolved timing observation before assignment. No claim of an independently proved 5100-second hard wall-clock guarantee is supported by these CPU results.
