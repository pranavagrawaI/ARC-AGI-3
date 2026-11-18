# actor-critic-construction-integrated — interface / ownership

Implementer: this identity only.
Reviewer 015a: read-only.
Coordinator: analysis/actor-critic-construction-verification/ only.
Incumbent v11, live construction, prior analysis identities: do not edit.

## Seams

Prefix: worker `ipc/transitions.jsonl` as written by v11 `host_contract.py:187-199`.
Each row: `step`, `turn`, `action`, `before.grid`, `after.grid` (integer matrices).
Not `states.jsonl`, not `actions.jsonl`, not invented `executed_actions.jsonl`.

Constructor: live `constructor.run_construction` with injected `evaluate_visible_abi` and live `execute_inspection`.
Do not call `integration_boundary.observed()` / LS20 `grid[61]` bar.
Visible replay = sealed initialize/predict/assimilate ABI survival on the prefix.
Prospective suffix is `not_run`. ABI-valid source is not a supported model.

Freeze: host-owned `FROZEN.json` mode 0444. Actor and critic cannot write it.
Play path: v11 Actor still `submit_actions`. Actor may `query_frozen_predict` / `assimilate_frozen` which RPC through `CandidateRPC`/`SealedBoundary`.
Critic: v11 `solver_agent.py:902-973` unchanged; `tools=None`; no constructor diagnosis in the packet.

Containment: qualified SealedBoundary. Local tests wrap with mapped-root
`unshare --map-users 0:1000:1 ... --setuid 0 --setgid 0`. No unprivileged exec fallback.

Transport: `FixtureTransport` labeled scripted, or `HttpTransport` (real HTTP, not a live model).

Failed construction: no freeze; query returns `NO_FROZEN_MODEL`; Actor still plays; critic still runs.
