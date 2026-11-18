# Trial10 release: bounded local diagnosis

Status: collection finished; CPU/source diagnosis finished. No repair, successor packet,
GPU call, remote API call, threshold change, or cache intervention was made here.
Trial10 remains unchanged and FEASIBILITY_ONLY, ratio null. Its standalone control
is valid for that run (310 measured completions), independently reproduced by root.
There is no candidate observation or qualified inter-arm release.

## Supported failure boundary

The primary `control-release.json` records an exact RTX PRO 6000 device with
95906 MiB used and an empty compute-applications string. The frozen validator
first rejects GPU memory above 2048 MiB. The subsequent host and port eligibility
checks were not reached remotely.

Changing **only** GPU usage to zero in a CPU counterfactual exposes the second
unchanged rejection. The host eligibility expression is 138308407296 bytes,
below the required 150323855360 bytes by 12015448064 bytes. This counterfactual
does not make the real GPU reading pass and does not identify a correction.

The original predicate is preserved. A synthetic exactly-at-boundary input passes;
one byte below fails. These are validator boundary checks, not measured hardware.

## Observation ordering

Frozen `release_check.run` executes the GPU-memory query, then a separate
compute-applications query, then reads host/cgroup/disk data. It saves the raw
receipt before validation. There are no individual query start/end clocks.
Therefore the high GPU reading and empty later process listing are not proven
simultaneous. No value may be discarded because another query appears inconsistent.

The primary supervisor shows control arm_pass at monotonic 3131.360114725,
server PGID551 cleanup empty at 3132.376315027, release spawn at
3132.377249756, and release exit1 at 3135.485968101. The raw GPU and applications
queries occurred inside that release phase; their exact times are unavailable.
No candidate spawn exists. Final owned groups are empty.

## Specific native shutdown interaction

The actual stderr ends with SIGTERM/draining, detokenizer649 exit -15, watchdog
SIGQUIT, and a five-second crash-diagnostic delay. This sequence has a direct
match in the pinned wheel:

- The supervisor sends SIGTERM to the entire owned process group, then permits
  at most one second of graceful time before SIGKILL escalation.
- `control_prepare.serve` launches the native server with `start_new_session=False`.
- The native signal handler sets `gracefully_exit`. Its asynchronous graceful
  watchdog can sleep five seconds before observing that flag; after draining it
  stops the child watchdog, dispatches `ShutdownReq`, and waits up to fifteen
  seconds for schedulers.
- The separate child watchdog treats a nonzero child exit as a crash and sends
  SIGQUIT. The observed detokenizer -15 and SIGQUIT match this path.

Thus group-wide SIGTERM interrupted the intended native graceful ordering in the
observed run. This is source-supported shutdown-path evidence, **not proof that
it caused the later 95906 MiB reading**, nor proof that a longer wait would pass.
The trace does not establish whether graceful scheduler resource release ran.

The actual console entrypoint is `sglang.cli.main:main` -> `sglang.cli.serve` ->
LLM `run_server` -> HTTP `launch_server` -> `Engine._launch_subprocesses`.
The CLI finalizer calls `kill_process_tree` without `wait_timeout`; its default is
fire-and-forget. `Engine.shutdown` has a separate sixty-second wait, but the
existence of that Python-API method does not establish execution by the CLI.

## Process visibility: actual CPU intervention

`diagnose.py` invokes the exact copied frozen `Supervisor.start/cleanup` with
CPU-only owned processes. The sole intervention is descendant session inheritance.
An inherited descendant is removed with its group. A descendant deliberately
created with `start_new_session=True` remains alive while the original PGID is
empty, and is then explicitly killed using its captured PID/starttime identity.
Both local fixtures are cleaned up.

This is a real counterexample to treating PGID emptiness as complete descendant
coverage. It does **not** show that native trial10 processes detached. The shipping
wrapper explicitly preserves the group; the inspected scheduler/detokenizer path
uses multiprocessing. The wheel's textual `start_new_session=True` occurrence in
the GPU-kernel test/benchmark helper is not evidence that this server used it.
PID-namespace and permission visibility were not captured sufficiently to exclude
other device holders. Empty apps and empty owned groups do not identify the cause.

## Cgroup accounting boundary

The captured memory.current is 65580773376 bytes. The file counter is
64808374272, anonymous memory 280051712, and kernel memory 486404096. The file
list counters include active_file 48823808000 and inactive_file 15984361472.
These overlapping/type-versus-list quantities must not be added as disjoint usage.

Linux documents `memory.current` as usage for the cgroup and descendants, `file`
as filesystem cache including tmpfs/shared memory, and active/inactive fields as
internal reclaim-list states. The counters do not prove immediate reclaimability
or identify which files/processes charged the pages. [Kernel cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)

The shipping prefetch reads model files and can populate filesystem cache; no
per-file charge provenance or matched cgroup-current baseline was captured.
The large file counter is consistent with cached data and is not evidence of a
live anonymous model allocation. It also does not prove sufficient reclaimable
headroom for the next load. Adding active_file, dropping caches, or relaxing140GiB
would change the original eligibility contract and is not justified by this run.

## What would distinguish the remaining explanations

Current evidence does not choose among a short-lived release/accounting transition,
unobserved device holder, or persistent driver/device memory state. NVIDIA's
compute-applications query reports active compute processes; FB memory and process
inventory are different observations. Conditional NUMA-GPU accounting caveats in
the documentation are not an explanation here: actual GPU NUMA status was not
recorded. [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

The missing discriminating observations are per-query timestamps and a bounded
release series, process identity/lineage/namespace coverage before and after
shutdown, and cgroup max/current/stat snapshots across prepare/load/replay/exit.
A series that crosses **both original predicates** would support a transient
eligibility delay; a stable failed series would not. A verified surviving device
holder would support a lifetime/visibility explanation; no such holder is observed
in the retained receipts. Native graceful completion would need its own evidence.
These are evidence requirements, not a prepared run or launch request.

## Reproduction and bindings

From this directory, use a fresh local output name:

```bash
python3 diagnose.py --output another-local-run
```

Canonical retained proof: `run2/PROOF.json` (six checks passed). It binds the
diagnostic script, all inspected copied sources, exact wheel member bytes, the
primary release/result/supervisor receipts, the trial10 freeze, and all908 original
freeze inputs. It records zero remote and GPU calls. `run1` remains preserved.
This is diagnosis qualification, not a repaired release or GPU qualification.

The existing raw reducer receipt used a relative input path. Its saved result used
an absolute path; only the missing-candidate error-path string differs. For exact
saved-result equality, from the unchanged trial10 directory use:

```bash
python3 -c 'import json,reduce_api; from pathlib import Path; print(json.dumps(reduce_api.compare(Path("/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-10-exact-wire-local/remote-readback/raw-view/output")),indent=2))'
```

Root independently reproduced exact saved-result equality with that absolute
invocation. The original raw receipt and inventory are preserved. The single ZIP
remains partial; the complete control raw joins do not qualify its central
directory or recover the absent candidate. Monitoring is already disabled.
