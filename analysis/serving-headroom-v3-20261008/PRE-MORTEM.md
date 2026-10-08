# Repair plan and pre-mortem

Outcome: reach actual GPU generation and profiling with trustworthy measurements; preserve incumbent serving behavior and failed runs.

Plan: extract the actual SGLang wheel; match all selected source files to installed v2 receipts; bind CPU native-method qualification and both notebook guards to those bytes; validate the actual notebook schema; freeze; perform one bounded S10 qualification and collect/audit all raw outputs.

Failure scene: another run stops before useful model evidence, or a successful-looking report certifies an unqualified implementation.

- Same version label masks changed source -> fixture and wheel differ -> prelaunch tests describe another implementation. Observed in v2. Prevent by whole-wheel/lock/source hashes and a regression that rejects the exact old allocator fixture locally; execute both actual generated guards against real package layouts.
- Builder writes invalid cell metadata -> schema warning or execution failure. Observed warning in v2. Remove code-only fields from Markdown and run nbformat.validate on generated notebook.
- Runtime hash guard removed to force progress -> incompatible code passes qualification -> false interpretation. Retain installed-source guard and test drift/missing members as failures.
- Concurrent writers change contract schema -> tests and notebook disagree or duplicate GPU work. Observed during preparation. One board-declared owner, other session read-only, freeze/hash/clean-tree checks and exclusive launch receipt; no duplicate run.
- CPU/native-control tests pass but CUDA profiler/cache/counter behavior fails -> invalid trace or throughput. Unknown until GPU. Keep mandatory real traces, occupancy, raw counter/response reconciliation and finite bounded run. No fallback settings or automatic retry.

Residual GPU feasibility and profiler perturbation are accepted for this one-hour qualification. All changes are local until freeze and launch; no scored efficacy or16-capacity conclusion is available from this run.
