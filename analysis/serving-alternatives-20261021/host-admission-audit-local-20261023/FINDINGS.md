# Host admission audit — local evidence only

There is no retained derivation of the original140GiB gate from the next loader's
peak demand. This audit does not change that gate. Release12 remains NOT_QUALIFIED;
matched capacity remains blocked. The missing quantity is the exact candidate's
incremental host charge/peak in the post-control state.

The earlier HOST-THRESHOLD-TRACE.md already traced the numeric contract to trial8
release_check.py:21/card:49. Trial3/7 startup requires at least170GiB of TOTAL
capacity (host total capped by cgroup memory.max), not170GiB free. It never reads
memory.current. Trial7 ran with175GiB cgroup capacity. Passing that startup guard
therefore cannot justify remaining headroom or a new inter-arm admission threshold.
This is a retained-artifact search result, not exhaustive historical provenance.

Five local vLLM source files were hash-equal to trial7 installed-source receipts,
and the inspected model config matched trial7's exact6146-byte asset digest.
The compressed-tensors branch deliberately selects unquantized PLE storage despite
FP8 metadata. Actual trial7 logs confirm BF16, CPU, pinned=True. For TP1, config
ngram_size3/heads_per_ngram8 gives16 heads and160 columns per head; the pinned
source's prime layout gives320001536 padded rows. Pure source layout execution
(with no torch/model import) and independent trial division agree. This requests
102400491520 bytes for PLE plus1271398400 bytes for the input embedding: a total
103671889920 bytes (96.551971GiB) of named pinned CPU tensor storage. This is NOT
a measured incremental cgroup peak or a sufficient admission budget. It omits
other processes, metadata, loader temporaries, allocator overhead and cache charges.

The checkpoint contains97 weight files/167.740675GiB on disk. The pinned default
loader chooses safetensors iteration with safe_open/get_tensor, not whole-shard
f.read eager loading. Its auto-prefetch rule compares checkpoint size to90percent
of its AvailableRAM estimate; trial7 explicitly skipped auto-prefetch. The target
and MTP passes logged10.78GiB and3.42GiB AvailableRAM after construction had begun.
Those rounded, stage-local estimates are neither entry headroom nor a peak trace.
The pinned wrapper honors host availability and a cgroup-usage helper; this audit
has not recovered that helper's base-image implementation, so it assumes no
inactive-file accounting semantics for the log value.

The71.91GiB model-loading log is in GPU startup accounting, not a CPU peak; nearby
GPU-worker logs describe device capacity, GPU weights/non-torch memory, activations
and KV cache. Neither checkpoint bytes nor this GPU footprint is a host admission
budget. Trial7's supervisor preserved startup MemAvailable/cgroup maximum, but no
startup memory.current, continuous host series, or cgroup peak. Six diagnostic
full-image/29-token replies prove fit/functionality for that session, not S10
full-response host peak or post-control co-residence. The prior independent raw
receipt remains the scope authority; nothing is retroactively upgraded.

Release12 independently records85 failed host observations. Its last cgroup
maximum/current/inactive_file are187904819200/74794835968/6589726720 bytes, yielding
119699709952 bytes (111.479042GiB) under the unchanged formula. Host MemAvailable
was186985857024 bytes, so the cgroup expression is the binding term. The same last
stat has file74030149632, active_file67440373760 and anon233672704 bytes. These are
measured accounting categories, not charge attribution or reclaimability proof.
This audit adds no active_file credit, performs no cache operation, and does not
assume waiting will make those pages eligible. The apparent difference between
111.48GiB heuristic headroom and96.55GiB tensor storage is NOT a proven margin.

The smallest useful future experiment would answer only: can the exact candidate
load and reach actual health200 once from the actual post-control state, with
ordinary loader/cache behavior, while recording phase-bound current/stat/OOM events
and available process/peak observations? No replay or capacity verdict is needed
for that question. It requires a separate bounded risk judgment, card and authority;
it is not authorized or constructed here. Read-only memory.peak may include the
preceding control peak, process RSS can double-count shared pages, and sampling can
miss transients; the measurement contract must expose those limitations rather
than label a sampled maximum a hard upper bound. Actual successful load would
qualify one observed state, not a universal threshold.

For a numerical admission rule, the relevant quantity would be a defensible upper
bound for the NEXT load's incremental demand plus an independently chosen margin,
using compatible host/cgroup scope. Only a storage lower bound is known today.
Do not substitute96.55, checkpoint167.74, GPU71.91, or a number fitted below this
run's111.48. Keep140 and stop if the uncertainty/risk is not accepted. Changing
PLE dtype, loader/cache policy or host envelope instead changes the package or
experimental conditions and needs separate qualification. None is implemented.

Reproduce the audit with a fresh local output:

    timeout 600s python3 audit.py --output independent-FRESH

AUDIT.json binds24 primary local inputs before/after. No GPU, model allocation,
network/API, cache/controller/threshold change or successor packet was performed.

The explicit causal model is: prior control state influences initial file/anon
charge and reclaim behavior; candidate package/loader influences new pinned,
temporary and cache charges; those trajectories plus the cgroup limit determine
OOM/ready. The observed headroom expression is a proxy of initial counters, not
a measurement of the next trajectory. Two explanations remain compatible: the
conservative counter rule rejects a load that normal reclaim could accommodate,
or additional/pinned/transient demand really exceeds the permitted charge. The
existing data cannot uniquely select between them. No causal benefit of active
file reclaim or an altered threshold has been established.
