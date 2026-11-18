# Narrow local handoff

The accepted direction is now implemented and locally verified (60tests/14.012s,
including27 unchanged core regressions): the release
controller owns query execution/pidfds, receipt-worker lifetime owns no queries,
and intended cgroup/observer/mount-derived paths plus a fresh-run native stderr
identity are validated. Sources/proof/previous evidence are bound in LOCAL-DELTA.

The causal intervention is ownership, not another filter: under the failed design,
worker death removed the only active cleanup owner when allowed thread exec had
cleared PDEATHSIG. Under this design the worker starts only after controller-owned
queries are cleaned. A separate actual cancellation test kills a worker while an
allowed thread-exec query is live; the still-live controller cleans the query by
its retained pidfd despite PDEATHSIG0. Neither test attributes this behavior to
nvidia-smi, and neither establishes real GPU release.

Exact runtime wiring still requires a later reviewed bundle, not a nominal retry:

1. Trusted runner creates fresh output lease, opens its empty stderr stream,
   starts/retains the shipping wrapper Popen under existing ownership, records
   wrapper identity, and passes the prepared run_id into registration. The runner
   retains ownership if publication/binding fails; the scope helper never spawns.
2. While the registered native frontend is alive, bind its cgroup/namespace/mount
   paths and fd2 to that lease. Unsupported scope/path/identity blocks execution.
3. The controller calls the new release seam. Queries and cleanup finish under
   the original per-observation end before receipt-worker creation. Raw clocks,
   errors, resources and unresolved query ownership remain distinct evidence.
   Fatal controller-query preparation receipts are preserved in the primary
   release result; clean query-phase expiry preserves the owned fallback.
4. Any actual nvidia-smi/seccomp incompatibility, failed scope binding or uncertain
   cleanup stops. Do not disable filter, substitute fixture paths, relax resource
   predicates or infer native context destruction from logs.

The earlier minimal feasibility question remains one shipping control load plus
actual health200 then shutdown/release, without replay/candidate/capacity claim.
The supported proposal remains3900s platform/3600s cooperative/300s cleanup;
this local work does not build/freeze that packet or authorize it. A pass would
only establish the tested post-load release, not post-replay/inter-arm release.
A host-headroom failure independently blocks qualification even if graceful GPU
release works. No empirical replacement for140GiB is proposed.

Next judgment: exact local review of this correction and runtime boundary code.
Actual shipping/Kaggle integration and the bounded feasibility experiment remain
separate preparation/authorization steps. Trials1–10 stay unchanged; no slot or
launch is requested by this handoff.
