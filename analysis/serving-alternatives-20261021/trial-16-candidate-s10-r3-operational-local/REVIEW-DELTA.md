# Trial16-r3 operational-only delta

Reviewed r2 freeze1a80f1b3 and all11180 bindings preserved. All42 runtime payloads, config/metadata, unassigned outer bytes, sources hash and candidate-only science/caps are identical. No runtime61/65 rerun is needed or performed; approved r2 source-bound qualification remains referenced. Original inherited64/65 limitation remains.

Only operational launch.py/reconciliation.py and collection_tools/readback.py/reduce.py change. Historical predecessor charge77433.978 stays bound to immutable closed trial15 receipts. Current preflight instead binds reviewed baseline0/reserved0/window ending2026-10-31T00:00:00Z/actual allowance21600/minimum21600. The installed SDK raw response and typed durations/refresh must agree. Missing/changed fields, moved usage, reservations, stale/future clocks or changed window fail closed. The5400s cap must fit actual available allowance and finish before refresh; no108000 UI assumption. Session terminal, scoped stub, exact/negative search and parsed no-runs404 predicates remain unchanged. Before/after receipts persist separately and are revalidated after final clearance on both branches.

Collection uses the same measured window/allowance contract, reports current baseline0 and historical77433.978 separately, and refuses old/missing/contradictory terminal quota. It does not reinterpret or subtract prior-window historical receipts. Live quota remains distinct from terminal spend.

Local installed SDK objects replay the coordinator's exact saved raw response through real fresh_checks with remote endpoints intercepted and network blocked. Focused regressions cover reset+old predecessor, field and allowance/window mutations, usage/reservations before/after, all-named branch, raw/typed and clock disagreement, and unchanged scoped-stub/predecessor rejection. Original failed preflight and a test-only expectation that SDK-deserialization failure returns an object are retained. Missing SDK object data stays missing; no receipt is invented.

No ownerAPI/GPU/assignment/launch/stage/commit. Requires new exact operational review and separately serialized assignment.
