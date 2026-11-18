# Local qualification: Gregg revision

Status: **LOCALLY_QUALIFIED_TARGET_UNMEASURED**. No upload or GPU run was performed.

Package: `deployment/kaggle/baseline-cost-gregg-20260906`. Original package and production solver were not edited by this revision. Source snapshot hashes are retained in SOURCE-PIN.json.

## Revision

The bounded experiment separates natural replay from forced-output controls and engine warming from request-cache state. It retains the chronological continuation control; traces short and 95k contexts with graphs enabled; compares identical finite request sets at C1/C2/C4 for captured and mechanically extended 95k inputs; includes executor waiting in the common-ready deadline; and preserves incremental events for partial failures. Missing or invalid fixed-control decode timing aborts rather than producing an apparent speed result. Total planned requests: 72; runner cap: 5400 seconds.

## Verification

Package byte/hash check and all **10 local tests passed**. Tests include actual loopback streaming HTTP, partial-response and missing-timing rejection, real thread overlap, deterministic deadline accounting, C2 partial-failure evidence, admission stopping after failure, captured-body preservation, all experiment branches and exact embedded runner identity. Raw command results and artifact hashes are in LOCAL-QUALIFICATION.json.

An independent review found the partial-failure evidence gap; this revision closes it with per-batch JSONL events. These scripted checks establish local control flow and artifact integrity, not GPU performance or model behavior.

## Runner feasibility correction

`analysis/competitor-transfer-20260906/runner-overlap/REPORT.md` records an actual packaged-worker/NOOA loopback probe: a successor observation wakes an Actor outside the decide semaphore, permitting two outstanding model HTTP requests with separate ownership. A decide semaphore is therefore insufficient to assert global request serialization. This is local runtime evidence, not qualification of the pinned Kaggle environment, server batching, memory capacity or game deadlines.

## Still unqualified

Target SM120 startup, profiler availability, usable cache/queue/overlap telemetry, GPU resource limits, speed and natural gameplay quality require target execution. Captured prompts share same-game prefixes; the 95k set is mechanical. Natural replay stops at response-schema validation and does not execute tools. The 180-second common-ready measure is diagnostic, not a game deadline. No backend intervention or bottleneck diagnosis is selected. MTP remains parked.
