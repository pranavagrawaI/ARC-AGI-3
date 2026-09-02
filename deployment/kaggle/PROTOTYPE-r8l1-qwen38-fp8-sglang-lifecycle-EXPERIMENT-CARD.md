# R8-L1 — Qwen3.8 FP8 complete lifecycle qualification

**Preparation state:** Second repair candidate prepared, not yet accepted. The first rejected candidate is preserved at `analysis/deployment-probes/20260830T102306Z-qwen38-r8l1-freeze-review/VERDICT.md` (SHA-256 `12a855680005ccbf3cf9ae7074a2b7ac4631793b97a6bc54f7bef37dce880a3c`); the rejected replacement is preserved at `analysis/deployment-probes/20260830T105533Z-qwen38-r8l1-repair-review/VERDICT.md` (SHA-256 `e391c598672affab288465284705b9316b6c21b30c7d29447d15c089c258caa6`). Both are retired. This candidate keeps conjunctive PASS and fail-closed GPU attribution, deletes all multiline traceback allowance, and accepts timestamps only in the exact `[YYYY-MM-DD HH:MM:SS]` grammar. It remains unlaunchable until two fresh exact-byte reviews accept the complete replacement. The user's conditional launch authorization applies only after both ACCEPT the unchanged six hashes.

**Question:** After the R8-B2-qualified Qwen3.8 FP8 route completes its frozen warmup and probe request, does one bounded process-group teardown leave, by a fixed deadline, no owned process, listener, or GPU allocation and only classified teardown-log events?

**Posture:** Lifecycle qualification. The R8-B2 request path is a non-regression precondition, not an experimental variable. Diagnose only after a classified failure.

**Predecessor:** R8-B2 Kaggle version 10 passed request qualification. Its verdict is `analysis/deployment-probes/20260830T084353Z-qwen38-r8b2-kaggle-v10/VERDICT.md`, SHA-256 `a204be6c96790e98c5a7e5461864bc86a7aafaf370494dd6e254f4426c706441`. Its fresh-context post-run review is `POST-RUN-INDEPENDENT-REVIEW.md`, SHA-256 `bc6568da065570caa1d213172e5f3a9abdc2574645c5e91bf37af6ad2171c468`. The final 35-file manifest hashes to `5073e4a37f59cc1fbd9c3120d12b438a886e1d27c2a2214b4a10c8e41bb86c30`.

**Contrast:** Preserve the version-10 model revision and files, SGLang wheel and dependency lock, private/offline Kaggle RTX PRO 6000 substrate, Triton attention, PyTorch sampling, CUDA graph/cache settings, `qwen3` reasoning parser, `qwen3_coder` tool parser, automatic VLM warmup, exact 2,447-byte visual request, one warmup plus one probe request, response parser, port `30000`, startup/request budgets, and zero ARC actions. Change only lifecycle ownership, observation, and teardown:

1. start the SGLang server in a new POSIX session and process group;
2. bind every server-side process to the fixed environment marker `R8_L1_OWNER=pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path:v11`;
3. before teardown, have the probe parent—outside the target session/process group—capture the owned process and resource set;
4. signal the owned process group with TERM, wait at most 30 seconds, use KILL only as bounded containment, then check the postcondition externally by 45 seconds after TERM;
5. classify the complete teardown-log suffix against the frozen taxonomy below.

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

A process that reparents or changes its PGID/SID after the pre-TERM capture remains owned by its recorded PID/start-time identity. A newly observed process is owned if it carries the marker, descends from a still-owned process, owns the recorded listener, or owns a GPU context attributable to an owned PID chain. The final scan must be from the external observer, never inferred from the target's return code or shutdown script.

## Bounded teardown policy

1. Flush the request and server logs and capture `lifecycle-owned-before.json`.
2. At `T0`, send one `SIGTERM` to the target PGID with `os.killpg`.
3. Poll process identities, the port-`30000` listener, and target GPU compute applications every 250 ms for at most 30 seconds. Do not declare success early from the root process's exit alone.
4. If any owned process remains at `T0 + 30 s`, classify graceful teardown as failed, write `lifecycle-term-timeout.json`, send one `SIGKILL` to the target PGID, and send `SIGKILL` individually to every recorded survivor that escaped that PGID. KILL is containment, never a lifecycle PASS.
5. Poll for at most 10 additional seconds. During a final 5-second quiet window, perform at least two complete external scans separated by at least 1 second.
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

**Prediction:** The preserved R8-B2 route again completes exactly one warmup and one probe request. TERM reaches the complete target process group; every owned process exits within 30 seconds without KILL; two external quiet-window scans find zero owned/reparented target processes, zero listeners on port `30000`, and zero target or unattributed GPU compute applications; the root return code is 0; and every nonempty teardown line belongs to the closed ordinary-line allowlist with no traceback or exception block.

**Measure:**

- inherited model: `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`;
- inherited SGLang: `0.5.19.dev591+g8e04f66a7`, commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`;
- inherited wheel SHA-256: `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`;
- inherited dependency-lock SHA-256: `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`;
- inherited model manifest SHA-256: `2c2c2f2a4344c33d6f54d9f065a4975b91822cd2836d4300ff50f425e0a322a4`;
- inherited request image SHA-256: `158e93fdce8151a7af1160873a5121d6886eacacc8de3fd3b73b19d35033b87d`;
- inherited request: 2,447 bytes, SHA-256 `f4108f8f6b05f32bd84dd4bfaad69968cff537d93a09d467d0f77393a6413cf0`;
- required request non-regression evidence: the R8-B2 model/readiness, warmup, backend, request-count, transport, response, parser, and raw request/response artifacts;
- required lifecycle evidence: `lifecycle-observer-preflight.json`, `lifecycle-target-launch.json`, `lifecycle-observer.json`, `lifecycle-owned-before.json`, `lifecycle-signals.json`, `lifecycle-polls.jsonl`, `lifecycle-term-timeout.json` when applicable, `lifecycle-owned-after.json`, `listeners-before.json`, `listeners-after.json`, `gpu-processes-before.json`, `gpu-processes-after.json`, `teardown.log`, `teardown-log-classification.json`, `shutdown.json`, `system-before.json`, `system-after.json`, `result.json`, and `server.log`. Top-level `result.json.status` is `passed` only when its `r8_l1_gate` records request-path PASS, lifecycle PASS, and teardown-log acceptance as one conjunction.

## Candidate binding

The bindings below are one candidate. Any later byte change requires a new card and run identity.

- lifecycle-capable probe: `deployment/kaggle/PROTOTYPE-qwen3_8_r8_l1_probe.py`, SHA-256 `755b966234f01b35fbcb9fcfb4796b8206f085688ae73a38652f3e9e1e0473e3`;
- notebook builder: `deployment/kaggle/scripts/build_r8l1_probe_notebook.py`, SHA-256 `6e7ce94cef9b9f6046ddcb4499a32073231f3a44e78b988c2f0e1bf136182490`;
- frozen R8-L1 notebook: `deployment/kaggle/r8l1-notebook/r8l1_probe.ipynb`, SHA-256 `b5ed4614ea2b91362b80efb8edf146f4168eab974f2579cd12dbc344599b3e99`;
- kernel metadata: `deployment/kaggle/r8l1-notebook/kernel-metadata.json`, SHA-256 `a56b24cd70afd386907266ebf02682e295e2ecafe0f9123fe4ae00418f8c584a`;
- local lifecycle regression tests: `deployment/kaggle/tests/test_r8l1_lifecycle.py`, SHA-256 `c3f9fffd4799bba43a4831201098bc3d26109c5833664a61bf9207600637a3ea`;
- card SHA-256: computed only after all preceding bindings are final;
- expected remote predecessor: private kernel `pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path`, version 10 `COMPLETE`;
- expected new immutable version: 11;
- CLI and metadata accelerator: `NvidiaRtxPro6000`;
- internet: disabled.

**Cap:** Exactly one private Kaggle kernel push and execution after separate explicit user authorization bound to the final hashes. Batch timeout: 1,800 seconds. GPU exposure: at most 0.50 GPU-hours. Startup timeout: 1,200 seconds. Probe-request timeout: 180 seconds. TERM grace: 30 seconds. KILL containment wait: 10 seconds. Final quiet window: 5 seconds. Lifecycle deadline from TERM: 45 seconds. Runtime warmups: exactly 1 and successful. Probe requests: exactly 1 and successful. Total chat-completion accesses: exactly 2 and successful. TERM signals: at most 1 group signal. KILL signals: at most 1 group signal plus one per escaped recorded survivor, only after TERM timeout. ARC actions: 0. Competition submissions: 0. Automatic retries: 0. Recheck quota immediately before any authorized push.

**Stop:**

- success: the persisted top-level result is the conjunction of inherited request-path PASS, lifecycle PASS, and strict teardown-log acceptance; pre-TERM ownership evidence is complete; the external observer is outside the target PGID/SID; one group TERM causes all owned processes to exit inside 30 seconds without KILL; two quiet-window scans prove zero owned processes, port listeners, target GPU applications, and unattributed GPU applications; and every nonempty teardown line belongs to the closed allowlist;
- graceful-teardown failure: any owned process survives 30 seconds, whether or not KILL later contains it;
- containment failure: any owned process, listener, or target GPU application remains at 45 seconds;
- log failure: closure succeeds but a forbidden or unclassified teardown event exists;
- futility: the inherited request-path non-regression fails, the observer cannot establish ownership, or any required external observation is unavailable;
- abort before model work: GPU, privacy, network, model/runtime identity, quota, predecessor, output path, candidate digest, or control-plane identity differs from the reviewed candidate;
- after any failure or invalidity: preserve exact evidence and do not retry automatically.

**Invalid if:** Any candidate binding remains `TBD`; candidate, metadata, runtime, model, manifest, request, parser, backend, graph/cache, hardware, network, or predecessor identity differs; observer and target share a PGID or SID; the server is not started in a new session/process group; the ownership marker differs or is absent from the target root; the pre-TERM inventory is missing any required ownership class or process identity field; TERM or KILL targets only the known root PID instead of the bound group plus escaped recorded survivors; observation is performed only by target code; listener ownership is inferred without inspecting socket inodes; GPU closure is inferred only from aggregate free memory; PID reuse is not distinguished by `/proc` start time; any postcondition scan fails or occurs after the deadline; the request-path non-regression differs from R8-B2; evidence is overwritten; or an automatic retry occurs.

**Result map:**

- request-path or candidate identity differs before TERM → `R8-L1 INVALID`; do not interpret lifecycle;
- ownership cannot be established or external observation is incomplete → `R8-L1 INVALID: observer_contract_invalid`;
- TERM closes the root but any descendant, reparented owner, listener, or GPU application remains at 30 seconds → `R8-L1 FAIL: graceful_shutdown_incomplete`; run bounded KILL containment, preserve evidence, no retry;
- KILL is used and all residue is gone by 45 seconds → retain `R8-L1 FAIL: kill_required`, plus `containment PASS`;
- anything remains at 45 seconds → `R8-L1 FAIL: containment_incomplete`;
- process/resource closure passes but a forbidden or unclassified log event exists → `R8-L1 FAIL: teardown_log_unacceptable`;
- request-path non-regression, TERM-only closure, both quiet-window scans, and strict log classification all pass conjunctively → `R8-L1 PASS`; authorize only preparation of a separately frozen one-cold-action S0 candidate;
- every branch preserves the independently accepted R8-B2 request verdict; R8-L1 cannot revoke or expand that earlier claim.

**Outputs:** The existing private Kaggle kernel at immutable version 11 created only by a separately authorized push; full pulled output and kernel log under a new timestamped `analysis/deployment-probes/` directory; an additive verdict separating request non-regression, graceful closure, containment, process absence, listener absence, GPU absence, and log acceptance. Preserve every raw poll and failed observation. This card becomes immutable when exact-byte review begins and never grants launch permission, S0 authority, gameplay authority, concurrency authority, throughput authority, submission authority, or QRM promotion.
