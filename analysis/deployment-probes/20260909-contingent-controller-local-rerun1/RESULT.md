# Local controller qualification: valid integration failure

109 tests passed, 1 failed. The unexpected-observation subprocess case executed the fallback action, but never ran the controller stop check. Its second raw request records `return_result validation error: Method invariant failed: result kind WAIT, but no actions were submitted for turn 1`. The first action was submitted for turn 0; the host had already published turn 1 when the postcondition read the mutable latest state. The false rejection caused a second model call inside the old handle invocation and bypassed the dispatcher. This is a causal runtime failure, not a model-quality failure.

Repair: bind the WAIT postcondition to the turn entering the dispatcher call. Add a deterministic fast-successor regression and requalify the changed candidate under a fresh card. Original requests, responses and IPC are retained under pytest/raw/test_actual_contingent_worker_3.
