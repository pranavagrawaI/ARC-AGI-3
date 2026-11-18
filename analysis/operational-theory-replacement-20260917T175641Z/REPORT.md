# C7R1: the one authorized replacement pair ran; both constructions incomplete

Exact OpenRouter model `qwen/qwen3.8-27b`, singleton endpoint `reka/fp8` (Reka, fp8). Fresh public metadata
(2 unauthenticated GETs, `endpoints.raw` sha256 `4857d780f4c77d9a2489f258cffc77dbc577449bd157681d6b178ad6f5b51406`) matched the pinned
provider/model/quantization/context/max-completion/prices/parameters with zero drift, and the frozen
`run_pair.gates()` accepted C6R2-REVIEW.json (`0f3a8a53df820de5…`, task
`operational-theory-T1R7`) plus the manifest-bound user authorization written before dispatch
(`d21b579fa3399634e9fc47085eb2b9e9cd370de02df6b3f565cea4993fbdd4a5`). The frozen driver was invoked once, as its own CLI, in a
subprocess; no HERE relocation, monkeypatch, retry, fallback, prompt alteration or manual repair.

| Arm | Requests | Input tokens | Output tokens | USD | Accepted proposals | Construction complete |
|---|---:|---:|---:|---:|---:|---|
| Control | 6 | 88,462 | 750 | 0.016628584 | 0 | No |
| Treatment | 6 | 97,740 | 661 | 0.018086070 | 0 | No |

Total: **186,202 input + 1,411 output = 187,613 tokens**, **$0.034714654** of the $2 ceiling.
All twelve receipts are resolved (request/response hashes, usage and `usage.cost` match the raw bodies);
reasoning tokens and unresolved exposure are zero; every response routed `Reka` direct on attempt 1.
The pair took 53.6 s (26.8 s + 26.7 s of the 240 s/arm caps). Each arm ended at its
six-request ceiling with zero `submit_actions`, so the recorded second transition was never revealed.

What the raw bodies show (FIRST-BOUNDARY.json, pointers into `live/*-raw/006.request.json`): both
arms spent their budget trying to analyze the grid with numpy, as the common system prompt instructs,
and both hit the same wall. `self.grid_array()` comes back through the facade as an opaque
`PublicObject`, so `g == c` is a bare bool, `int(cell)` and `np.asarray(g, dtype=int)` raise
TypeError, and `.any` fails; `self.model_grid()` and `self.trajectory()`, both advertised in the
shared prompt, are not in the facade's approved set ("Actor capability is not public"). Control saw
4 errored cells and one CELL_STATE_REPAIR refusal; treatment read the workflow skill and
`operational_cycle('read')` (its canonical cycle stayed `idle`, 0 theory versions) and then errored on
the same grid path twice. This boundary is identical in both arms, so it is not one of the card's
invalidity conditions; it is the first missing workflow stage under the result-map row "both
incomplete". Whether a facade that contradicts the Actor's own advertised capabilities makes the
comparison uninterpretable is C8's call, not the operator's.

`PARITY.json`: exact first-request equality after removing only the declared `<operational_cycle>`
block (728 chars), identical non-message parameters on all twelve requests.
`INTEGRITY.json`: receipts, both workers reaped, no matching bwrap processes, every candidate /
predecessor / T3 / T4 / C7-HOLD inventory file unchanged, manifest `539540436f6805b1…` after.
`WORKFLOW-REVIEW.json` keeps the eight-item common rubric unresolved/absent for independent C8; the
treatment-only schema metrics are not a comparative statistic. No gameplay was executed; nothing here
speaks to gameplay competence or efficacy. No further launch or replication is authorized.
