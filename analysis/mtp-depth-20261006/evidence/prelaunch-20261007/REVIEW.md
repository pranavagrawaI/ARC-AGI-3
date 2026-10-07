# Prelaunch review

Candidate: inherited untracked experiment plus the 2026-10-07 completion edits, bounded to analysis/mtp-depth-20261006. Contract: final user message in T3 c07540ef-efa9-489a-9f6b-13cac5bad5d3, plus authorization to finish the next move in this thread.

Supported findings fixed before freeze:
- M3C could be dropped or absent while an adoption verdict used two anchors. It is now protected and mandatory.
- CARD.md and the uploaded payload still described the earlier design. Card/order/provenance are reconciled and upload readback is required before launch.
- Provenance changed the notebook call and silently disabled rehearsal time scaling. The substitution now asserts its anchor and scales the session deadline too.
- A missing quota record was treated as zero usage. Missing/invalid quota now blocks launch. A subsequent live check exposed the SDK serializer emitting 21783.604.0s and discarding duration days; admission now uses its parsed timedelta.total_seconds(), with the live result and day/fraction regression preserved.
- The mounted dataset could present its own stale manifest. The notebook now binds an expected manifest and code commit independently, and the local verifier checks code against git, uploaded readback, and the tag.
- A socket hang could prevent the watcher from recovering. SDK requests receive connect/read timeouts, polling faults stay in the log, and dry checks fail promptly.
- The owned-kernel list includes the exact redacted sentinel ref empty, title [Private Notebook], no author/version, fixed timestamp 2010-04-01. Querying its numeric id 0 produces a permanent unknown. The checker skips only that exact non-runnable sentinel, queries recent owned kernels with an explicit owner, and retains unknown blocking for every other unidentified entry.
- Merely writing a wave-2 launch decision could release its priority before its push. Accepted control receipt or hash-checked HOLD evidence is required.
- Startup exceptions escaped before launch.json was written; they are now recorded once.
- An anchor without warm-up or finite U/CV could appear usable. It now makes the comparison incomplete.

Review covered analyzer, planner, launcher, replay, payload, simulation/checkpoint construction, notebook factory, provenance, watcher, verifier, and scoped tests. Q7 source audit is preserved unchanged. The original recorded alpha and traffic are unchanged. No supported unresolved requirement gap remains in the completed local gates. CUDA graphs/nsys availability, real model load and runtime workload fidelity remain GPU questions; failure is retained and not retried. No score/throughput gain is established locally.

Risk measurement uses stdlib AST/trace with the explicit contract in tests/risk.py, no installed dependencies and no threshold. The same final test universe was run against preserved before-files and final files. The inherited candidate fails required contract tests and lacks added APIs, so that baseline is not a passing quality result and no CRAP improvement claim is made. Final unit coverage does not cover the GPU/notebook execution; a real generated-notebook mock run, payload readback and actual launch receipt are separate required gates. Higher legitimate boundary complexity is retained rather than split solely to lower the metric. Scores and outputs are in risk-{before,after}.json and their test logs.
