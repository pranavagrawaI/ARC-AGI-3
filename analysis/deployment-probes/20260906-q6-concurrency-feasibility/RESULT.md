# Pinned q6 concurrency feasibility

The pinned q6 host serializes Solver decisions. The downloaded official Swarm creates one thread per game, but the shared LocalSolver holds a process-wide BoundedSemaphore(1) from sending a turn until receiving its decision. In q6, that interval can contain multiple Actor model/tool rounds. Parallel Swarm threads therefore do not establish useful cross-game model concurrency.

`check_pinned_host.py` loads the exact shared_local_solver.py named by the summary-boundary manifest, then runs the existing two-instance concurrent admission regression using real local worker subprocesses. Observed order is generation_begin, generation_end, generation_begin, generation_end. No GPU or network model calls are made. This proves the pinned admission behavior, not server-side scheduling or real-game result routing.

Reproduce:

```sh
deployment/kaggle/.venv/bin/python analysis/deployment-probes/20260906-q6-concurrency-feasibility/check_pinned_host.py
```

The inference agent's existing baseline-cost-20260906 card already distinguishes engine warming from prefix-cache warming, uses a cold/continuation pair, records server timing and cache telemetry, compares a finite set at concurrency 1/2/4, preserves production graphs, and separates natural completion from fixed decode controls. No inference files were changed during this check.

Consequences for the next decision:

1. An isolated serving throughput win does not transfer through the current host gate. Qualify an explicit host change only if the server experiment warrants it.
2. Real-game qualification must use distinct state/workspace ownership, correctly routed results, and request/decision/game deadlines, including time waiting for admission. Creating extra work from future or duplicate turns is invalid.
3. The replay set currently contains distinct requests from one game. Its cache sharing is not an independent-game distribution. Its natural completions do not establish time to a usable Actor decision.
4. A GPU timeline locates time; it does not alone establish memory versus arithmetic saturation. Keep hardware-counter diagnosis as a separate evidence requirement before a resource claim.
5. Prioritize completed levels and valid progress within the actual run deadline. Requests/s and tokens/s remain diagnostic measures.

No semaphore change, server launch, notebook push, or paid API call was performed.
