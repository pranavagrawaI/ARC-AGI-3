# R8-L2 — Qwen3.8 FP8 root-first lifecycle repair

**Preparation state:** Second replacement prepared, not yet accepted. The first frozen R8-L2 candidate was rejected by lifecycle review `5d8879db69af351d91b6f3fa2e5f6cbedee92940b653fe57feb11a216aa47ead` and authority review `59e34ba23e0f3c0deb0f1763acb289cd3cce5236bd337c2c26a2d6f7a50e4b4c`. The first replacement was then rejected by lifecycle review `ce56e65ce77bcb6ac95391c29dd72d9da701f7231957059e9c2d342e26993ec5`; its authority review `0bdb934ee80e7e649fbf4645dae4ee1a3378ae178b136368e25a607897a113ed` accepted only those now-retired exact bytes and is not reusable. This successor preserves both rejections, the accepted R8-L1 evidence, and the root-only TERM treatment while repairing completed-scan deadline adjudication and retaining the actual first completed closed scan as the quiet anchor. It remains unlaunchable until two fresh exact-byte reviews accept the complete second replacement and the user separately authorizes one bounded launch against its final hashes.

**Question:** After the unchanged R8-B2-qualified request path completes, does sending TERM only to the SGLang root let its built-in coordinated shutdown close the complete externally observed process tree, listener, GPU applications, and teardown log by the frozen deadline?

**Posture:** Lifecycle qualification. The R8-B2 request path is a non-regression precondition, not an experimental variable. Diagnose only after a classified failure.

**Predecessor:** R8-L1 Kaggle version 11 was a valid classified failure. The inherited request path passed and containment closed all processes, listeners, and GPU applications, but process-group TERM killed the detokenizer with `-15`; SGLang's watchdog interpreted that intentional child exit as a crash, sent SIGQUIT, and the root exited `-9`. Its verdict is `analysis/deployment-probes/20260830T122205Z-qwen38-r8l1-kaggle-v11/VERDICT.md`, SHA-256 `49ad572d8ce791379adbe4f7f3a539445b111376013b860a2482d3e68206f14e`. Its accepted post-run review hashes to `810a8dcad0c73c045ba8ff1e100ca26dd9de0637f62173d3520b871dcbc40c3c`; the complete manifest hashes to `73ec63d2afdf73203388d0b46da116542c3d3673d653756b611f7eeed9a09d3c`.

**Contrast:** Preserve the version-11 model revision and files, SGLang wheel and dependency lock, private/offline Kaggle RTX PRO 6000 substrate, Triton attention, PyTorch sampling, CUDA graph/cache settings, `qwen3` reasoning parser, `qwen3_coder` tool parser, automatic VLM warmup, exact 2,447-byte visual request, one warmup plus one probe request, response parser, server launch, ownership marker, process/resource observation, log taxonomy, port `30000`, timing budgets, and zero ARC actions. Relative to R8-L1, the treatment change remains only the initial graceful-shutdown signal target; the copied probe also deletes one inherited unused `socket` import. Relative to both rejected R8-L2 generations, this second replacement changes only observer/authority enforcement:

1. start the SGLang server in a new POSIX session and process group;
2. bind every server-side process to the fixed environment marker `R8_L2_OWNER=pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path:v12`;
3. before teardown, have the probe parent—outside the target session/process group—capture the owned process and resource set;
4. send TERM only to the PID/start-time-verified target root and wait at most 30 seconds;
5. admit raw target PGID/SID only in the pre-TERM ownership snapshot; later scans require recorded PID/start-time continuity, marker, descent, listener, or GPU attribution;
6. use group KILL only when a still-live pre-TERM PID/start-time identity proves that the numeric PGID remains the target generation, otherwise skip group KILL and contain only individually identity-verified survivors;
7. anchor the final five-second quiet interval at the first completed closed scan, not before that scan starts;
8. accept only exact R8-L2 startup/request timeouts of 1,200/180 seconds and bind the exact Kaggle CLI 2.2.4 server-side 1,800-second session deadline;
9. admit a TERM-grace closure only when the complete scan's recorded completion timestamp is no later than `T0 + 30 s`; a scan that crosses that boundary enters failed containment even when it returns closed;
10. retain that same first completed closed TERM or containment scan as the quiet-window anchor and persist its phase and completion time;
11. classify the complete teardown-log suffix against the frozen taxonomy below.

No model-serving, warmup, request construction, response parsing, scheduler, concurrency, gameplay, or submission behavior may change.

## Ownership boundary

The lifecycle observer is the probe parent. It must record its own PID, process-group ID (PGID), session ID (SID), and cgroup path and prove that its PGID and SID differ from the target root before sending any signal.

At server launch, record `pid`, `/proc/<pid>/stat` start time, PGID, SID, cgroup path, command bytes, and the fixed ownership marker. Immediately before TERM, write `lifecycle-owned-before.json` containing the union of:

- the target root identity;
- every live recursive descendant of the target root;
- every live process with the target PGID or SID;
- every same-user process whose `/proc/<pid>/environ` contains the exact ownership marker;
- every process owning a listening socket on TCP port `30000`;
- every process listed by `nvidia-smi --query-compute-apps=pid,used_memory` whose PID is already in one of the preceding ownership classes.

The pre-launch observer must find zero GPU compute applications. After launch, every GPU compute application that cannot be attributed to an owned process is recorded separately and prevents lifecycle closure; it may not disappear from adjudication merely because `/proc` exited before the GPU query completed.

Each member must include PID, parent PID, `/proc` start time, PGID, SID, cgroup path, command, ownership reasons, listener socket inodes/ports, and GPU allocation if present. PID plus `/proc` start time is the process identity; a reused PID is not the same process. Record the target cgroup even when Kaggle does not delegate a unique writable child cgroup. A shared cgroup may corroborate ownership but may not, by itself, add unrelated notebook processes to the owned set.

A process that reparents or changes its PGID/SID after the pre-TERM capture remains owned by its recorded PID/start-time identity. After TERM, matching the old numeric PGID or SID alone does not establish ownership because Linux may reuse those numbers after the original group/session disappears. A newly observed process is owned if it carries the marker, descends from a still-owned process, owns the recorded listener, or owns a GPU context attributable to an owned PID chain. The final scan must be from the external observer, never inferred from the target's return code or shutdown script.

## Bounded teardown policy

1. Flush the request and server logs and capture `lifecycle-owned-before.json`.
2. At `T0`, revalidate the target PID/start-time identity and send one `SIGTERM` only to that root PID with `os.kill`.
3. Poll process identities, the port-`30000` listener, and target GPU compute applications every 250 ms for at most 30 seconds. Adjudicate the grace boundary from each complete scan's completion timestamp, not when the scan began. Do not declare success early from the root process's exit alone.
4. If any owned process remains at `T0 + 30 s`, classify graceful teardown as failed and write `lifecycle-term-timeout.json`. Send one `SIGKILL` to the target PGID only if a current member's PID/start-time identity was already recorded in the pre-TERM snapshot; otherwise record that the group signal was not sent. Send `SIGKILL` individually to every current identity-verified survivor outside any safely signalled target group. KILL is containment, never a lifecycle PASS.
5. Poll for at most 10 additional seconds. The first complete closed external scan from either TERM grace or KILL containment is the final quiet-window anchor; do not discard it and substitute a later scan. Require at least one further closed scan at least 5 seconds after that anchor. Scan execution time before the anchor does not count toward the interval.
6. At or before `T0 + 45 s`, write the final process, listener, GPU, cgroup, and log classifications. Any observation command failure, permission denial, missing `/proc` identity, or unparsable output makes the lifecycle result `INVALID`, not PASS.

## Teardown-log taxonomy

Classify only the log suffix beginning at `T0`; preserve it byte-for-byte as `teardown.log` with its source byte offsets.

**Allowed normal events:**

- an optional timestamp only in exact `[YYYY-MM-DD HH:MM:SS]` form before an otherwise allowed ordinary line;
- the exact SGLang notices `SIGTERM received. ... Draining requests and shutting down...` and `Gracefully exiting... Remaining number of requests 0`;
- normal uvicorn shutdown/exit notices;
- SGLang `kill_process_tree called` notices when they are not accompanied by a surviving process or forbidden event;
- exact Python `import time:` diagnostic rows emitted by the frozen `--profile-imports` route;

**Forbidden events:**

- `destroy_process_group() was not called before program exit`;
- any NCCL, CUDA, distributed-process-group, shared-memory, semaphore, socket, or GPU-resource leak warning;
- any traceback or exception block, including `SystemExit: 0`, `asyncio.exceptions.CancelledError`, `KeyboardInterrupt`, and `GeneratorExit`; no multiline exception container is allowlisted;
- any nonzero `SystemExit`, segmentation fault, abort, core dump, or unhandled exception;
- any `ERROR`, `CRITICAL`, `Exception`, `Traceback`, or `Warning` event not mapped byte-for-byte to an allowed entry;
- any request still running, queued, or cancelled at TERM;
- any observer or cleanup error.

Write one entry per nonempty event to `teardown-log-classification.json`, including byte range, exact text, class, rule, and allowed boolean. Strip only the exact timestamp grammar above; bracketed arbitrary text is not a timestamp. Allowed rules must match the complete remaining line; an allowed prefix does not license a suffix. Every nonempty line outside the closed ordinary-line allowlist is forbidden by default. The raw log, not this classification, remains authoritative.

**Prediction:** The preserved R8-B2 route again completes exactly one warmup and one probe request. TERM reaches only the root, so the detokenizer remains alive until SGLang marks shutdown, drains requests, stops its subprocess watchdog, and coordinates child exit. Every owned process exits within 30 seconds without KILL; two external quiet-window scans find zero owned/reparented target processes, zero listeners on port `30000`, and zero target or unattributed GPU compute applications; the root return code is 0; and every nonempty teardown line belongs to the closed ordinary-line allowlist with no traceback or exception block. The first decision-reversing counterevidence would be a clean root exit with the already observed NCCL, `SystemExit: 0`, or `CancelledError` teardown events; that would reject R8-L2 without weakening the taxonomy.

**Measure:**

- inherited model: `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`;
- inherited SGLang: `0.5.19.dev591+g8e04f66a7`, commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`;
- inherited wheel SHA-256: `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`;
- inherited dependency-lock SHA-256: `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`;
- inherited model manifest SHA-256: `2c2c2f2a4344c33d6f54d9f065a4975b91822cd2836d4300ff50f425e0a322a4`;
- inherited request image SHA-256: `158e93fdce8151a7af1160873a5121d6886eacacc8de3fd3b73b19d35033b87d`;
- inherited request: 2,447 bytes, SHA-256 `f4108f8f6b05f32bd84dd4bfaad69968cff537d93a09d467d0f77393a6413cf0`;
- required request non-regression evidence: the R8-B2 model/readiness, warmup, backend, request-count, transport, response, parser, and raw request/response artifacts;
- required lifecycle evidence: `lifecycle-observer-preflight.json`, `lifecycle-target-launch.json`, `lifecycle-observer.json`, `lifecycle-owned-before.json`, `lifecycle-signals.json`, `lifecycle-polls.jsonl`, `lifecycle-term-timeout.json` when applicable, `lifecycle-owned-after.json`, `listeners-before.json`, `listeners-after.json`, `gpu-processes-before.json`, `gpu-processes-after.json`, `teardown.log`, `teardown-log-classification.json`, `shutdown.json`, `system-before.json`, `system-after.json`, `result.json`, and `server.log`. `shutdown.json` records the quiet-anchor phase and monotonic completion time, the first-closed-scan elapsed time, and the accepted TERM-closure elapsed time when one exists. Top-level `result.json.status` is `passed` only when its `r8_l2_gate` records request-path PASS, lifecycle PASS, and teardown-log acceptance as one conjunction.

## Candidate binding

The bindings below are one candidate. Any later byte change requires a new card and run identity.

- lifecycle-capable probe: `deployment/kaggle/PROTOTYPE-qwen3_8_r8_l2_probe.py`, SHA-256 `41ee8cf659a290884be897fc9b6da7947805588d9a38c69b011360ef9f711fa2`;
- notebook builder: `deployment/kaggle/scripts/build_r8l2_probe_notebook.py`, SHA-256 `964aa9cd25d18e953112048c326f35225bc48ad213ac6977c0397bbf93d2ec53`;
- frozen R8-L2 notebook: `deployment/kaggle/r8l2-notebook/r8l2_probe.ipynb`, SHA-256 `1aac5608cf60d68e1fa1a430cc4f8a38018f6dafc119c2fa2e7e697f8a785286`;
- kernel metadata: `deployment/kaggle/r8l2-notebook/kernel-metadata.json`, SHA-256 `eeb1e39ba7f71fe01f02690b1ebf5f775fc8fc084d225d88ab3860d9c4aba364`;
- local lifecycle regression tests: `deployment/kaggle/tests/test_r8l2_lifecycle.py`, SHA-256 `76e76d3a5acd7215f18f937e89cd8b1790a194ab67f93c1f9ee85f644fb25cac`;
- launch binding: `deployment/kaggle/r8l2-launch.json`, SHA-256 `cb62748b61bfc51d1f25286936c0c27d425d1fc57c1421bfb3906f67b6befe3c`;
- card SHA-256: computed only after all preceding bindings are final;
- expected remote predecessor: private kernel `pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path`, version 11 `ERROR`, with executable-cell identity matching the accepted R8-L1 readback;
- expected new immutable version: 12;
- CLI and metadata accelerator: `NvidiaRtxPro6000`;
- internet: disabled.

**Cap:** Exactly one private Kaggle kernel push and execution after separate explicit user authorization bound to the final hashes. The only permitted launcher is the exact `deployment/kaggle/r8l2-launch.json` command: `uvx --from kaggle==2.2.4 kaggle kernels push --path deployment/kaggle/r8l2-notebook --timeout 1800 --accelerator NvidiaRtxPro6000`. Kaggle CLI 2.2.4 maps that timeout to server-side `ApiSaveKernelRequest.session_timeout_seconds=1800`; a launch that cannot verify this projection is invalid before push. Batch timeout: exactly 1,800 seconds. GPU exposure: at most 0.50 GPU-hours. R8-L2 accepts only startup timeout 1,200 seconds and probe-request timeout 180 seconds. TERM grace: 30 seconds. KILL containment wait: 10 seconds. Final quiet window: 5 seconds measured between completed closed scans. Lifecycle deadline from TERM: 45 seconds. Runtime warmups: exactly 1 and successful. Probe requests: exactly 1 and successful. Total chat-completion accesses: exactly 2 and successful. TERM signals: at most 1 root-PID signal. KILL signals: at most 1 identity-guarded group signal plus one per escaped recorded survivor, only after TERM timeout. ARC actions: 0. Competition submissions: 0. Automatic retries: 0. Recheck quota immediately before any authorized push.

**Stop:**

- success: the persisted top-level result is the conjunction of inherited request-path PASS, lifecycle PASS, and strict teardown-log acceptance; pre-TERM ownership evidence is complete; the external observer is outside the target PGID/SID; one root-PID TERM lets the target coordinate all owned processes to exit inside 30 seconds without KILL; two quiet-window scans prove zero owned processes, port listeners, target GPU applications, and unattributed GPU applications; and every nonempty teardown line belongs to the closed allowlist;
- graceful-teardown failure: any owned process survives 30 seconds, whether or not KILL later contains it;
- containment failure: any owned process, listener, or target GPU application remains at 45 seconds;
- log failure: closure succeeds but a forbidden or unclassified teardown event exists;
- futility: the inherited request-path non-regression fails, the observer cannot establish ownership, or any required external observation is unavailable;
- abort before model work: GPU, privacy, network, model/runtime identity, quota, predecessor, output path, candidate digest, or control-plane identity differs from the reviewed candidate;
- after any failure or invalidity: preserve exact evidence and do not retry automatically.

**Invalid if:** Any candidate binding remains `TBD`; candidate, metadata, launch binding, Kaggle CLI version/request projection, runtime, model, manifest, request, parser, backend, graph/cache, hardware, network, or predecessor identity differs; observer and target share a PGID or SID; the server is not started in a new session/process group; the ownership marker differs or is absent from the target root; the pre-TERM inventory is missing any required ownership class or process identity field; TERM targets the process group or any process other than the PID/start-time-verified root; post-TERM ownership admits a process from numeric PGID/SID equality alone; any group signal lacks a currently revalidated pre-TERM PID/start-time member of that group; an escaped survivor is signalled without current PID/start-time revalidation; observation is performed only by target code; listener ownership is inferred without inspecting socket inodes; GPU closure is inferred only from aggregate free memory; PID reuse is not distinguished by `/proc` start time; a complete TERM scan finishing after `T0 + 30 s` is admitted as graceful closure; the first complete closed TERM/containment scan is not the quiet anchor; the quiet interval is measured before that anchor completes; startup/request timeouts differ from 1,200/180 seconds; the server-enforced session timeout differs from 1,800 seconds; any postcondition scan fails or occurs after the deadline; the request-path non-regression differs from R8-B2; evidence is overwritten; or an automatic retry occurs.

**Result map:**

- request-path or candidate identity differs before TERM → `R8-L2 INVALID`; do not interpret lifecycle;
- ownership cannot be established or external observation is incomplete → `R8-L2 INVALID: observer_contract_invalid`;
- a TERM-grace scan completes after 30 seconds, whether closed or not → graceful PASS is unavailable and bounded containment begins; eventual clean closure remains `R8-L2 FAIL: kill_required`;
- TERM closes the root but any descendant, reparented owner, listener, or GPU application remains at 30 seconds → `R8-L2 FAIL: graceful_shutdown_incomplete`; run bounded KILL containment, preserve evidence, no retry;
- KILL is used and all residue is gone by 45 seconds → retain `R8-L2 FAIL: kill_required`, plus `containment PASS`;
- anything remains at 45 seconds → `R8-L2 FAIL: containment_incomplete`;
- process/resource closure passes but a forbidden or unclassified log event exists → `R8-L2 FAIL: teardown_log_unacceptable`;
- request-path non-regression, TERM-only closure, both quiet-window scans, and strict log classification all pass conjunctively → `R8-L2 PASS`; authorize only preparation of a separately frozen one-cold-action S0 candidate;
- every branch preserves the independently accepted R8-B2 request verdict; R8-L2 cannot revoke or expand that earlier claim.

**Outputs:** The existing private Kaggle kernel at immutable version 12 created only by a separately authorized push; full pulled output and kernel log under a new timestamped `analysis/deployment-probes/` directory; an additive verdict separating request non-regression, graceful closure, containment, process absence, listener absence, GPU absence, and log acceptance. Preserve every raw poll and failed observation. This card becomes immutable when exact-byte review begins and never grants launch permission, S0 authority, gameplay authority, concurrency authority, throughput authority, submission authority, or QRM promotion.
