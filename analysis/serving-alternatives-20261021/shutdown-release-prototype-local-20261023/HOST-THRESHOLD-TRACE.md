# Host threshold: retained source trace and evidence gap

The inspected paired-screen source first specifies the140GiB remaining-headroom
gate in trial8 `release_check.py:21` and `EXPERIMENT-CARD.md:49`; trials9/10 preserve
the same validator and contract. This is provenance of the numeric contract,
not an independently derived memory requirement. No justification tying140GiB to
a measured next-package peak allocation was found in the inspected artifacts.
It remains the predeclared requirement until a separate explicit judgment.

The source computes headroom from host MemAvailable capped by cgroup maximum minus
current usage plus inactive_file, floored at zero. It does not credit active_file.
This is an eligibility heuristic/contract, not a guarantee of allocatable memory
or a cache-match requirement. `resources.cgroup` preserves that exact arithmetic.
Tests compare partitioned gates to the frozen validator and test original numeric
boundaries. No empirical adjustment was fitted to the failed run.

The startup guard (`candidate_prepare.py:87–92`, also called by control prepare)
uses a separate170GiB minimum of total host capacity and cgroup memory.max, plus
80GiB writable disk. It does **not** read cgroup memory.current or prove140GiB
remaining headroom. The release contract separately requires100GiB disk.
Passing startup capacity is therefore not evidence of passing release eligibility.

At trial10 release, the unchanged expression yields138308407296 bytes versus the
required150323855360. This host failure is exposed by changing only the GPU reading
to zero in a CPU counterfactual. It is independently observed input evidence,
although the first GPU failure prevented the remote validator reaching this line.

The captured cgroup file counter dominates memory.current; active/inactive fields
are reclaim-list state, not proof of immediate reclaimability. The shipping prefetch
reads mounted model files and can populate cache; the retained data does not bind
charges to particular files or prove those pages will become eligible within a
shutdown wait. Host-headroom startup receipts do not contain a matched cgroup-current
baseline. There is no CPU proof that native graceful exit fixes this host gap.

The useful missing evidence is a phase-bound cgroup series, its scope/identity and
the next loader's actual allocation demand. A stable failure of the original gate
must remain a failure. No active_file addition, cache clear, memory-reclaim command,
threshold reduction or further GPU work is implemented or requested here.

Source references and hashes are in `LOCAL-DELTA.json`. The retained local diagnosis
already links the kernel cgroup documentation; this preparation performed no new
network research. Search scope was the trial4 card, trial5/7 feasibility contract,
trial8 paired release/card, trial9/10 inherited release/card/startup source and
top-level local survey evidence. This does not claim exhaustive provenance outside
the retained workspace.
