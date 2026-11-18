# actor-prefix-construction-join-20260914T131345Z

Question: Can a host adapter project an Actor-executed IPC prefix onto the constructor-visible observation ABI, freeze one candidate, and load that freeze into GameplaySolver, without a constructor model, without Kaggle, and without treating SCRIBE notes or Actor helpers as the world model?

Posture: Execution. Feasibility of the first join seam (prefix → construction surface → freeze → GameplaySolver). Not construction competence, not gameplay utility, not a three-role solver.

Contrast: none

Prediction: A fixture IPC prefix of N+1 `states.jsonl` lines (`grid_rows` hex) plus N executed primitive actions projects to a constructor case whose `compact_view` / `inspection_view` pass `validate_constructor_surface`; constructor-visible observations are `{grid: list[list[int]]}` only; `construction_messages` for that case contain neither sealed suffix, `collision`, `knowledge/*.md`, nor `helpers/*.py`. A fixture candidate written to `FROZEN.json` loads through `GameplaySolver.start_game` / `decide` and returns a typed action. Hex `grid_rows` on the Actor surface is rejected if forwarded unprojected. `GameplaySolver` does not import `solver_agent`.

Measure: `analysis/actor-prefix-construction-join-20260914T131345Z/tests/test_prefix_adapter.py` — projection equality of decoded grids and actions; ABI rejection of unprojected `grid_rows`; absence of forbidden constructor-surface keys; freeze bytes hash; `GameplaySolver.decide.action` type and membership in the fixture's available actions. Raw artifacts: `fixtures/actor_ipc/states.jsonl`, `fixtures/actor_ipc/executed_actions.jsonl`, `evidence/case.json`, `FROZEN.json`, pytest stdout.

Cap: local CPU only. Zero constructor model requests. Zero critic calls. Zero Kaggle pushes. Zero edits to `deployment/kaggle/submission-runtime-20260912-v11/`, `analysis/construction-policy-live-20260914T093351Z/`, and `analysis/construction-policy-offline-20260914T115025Z/`. Wall 120 s for the pytest file. Spend not applicable.

Stop: success if the adapter tests pass. Futility if Actor IPC cannot be projected without inventing transitions or if freeze cannot load in `GameplaySolver`. Abort if a test would require OpenRouter, SGLang, a constructor completion, a critic completion, `submit_actions`, or a Kaggle kernel.

Invalid if: `solver_agent.py`, the critic, or either consumed construction identity is edited; constructor or critic LLM is called; suffix, evaluator payload, or verifier internals are visible on the constructor surface; `knowledge/*.md` or `helpers/*.py` is treated as candidate source; Actor or critic can write `FROZEN.json`; more than one freeze exists; play path (b) (Actor `predict_action` tool) is implemented; LS20 live steps are used as unseen evidence; host text tells the constructor `collision` unless that token is on the allowed observation surface.

Result map:
- positive → next identity is an offline constructor-from-actor-prefix card (still no Kaggle). Critic stays unchanged. Play path remains (a) `GameplaySolver.select_action` over frozen bytes. Do not stuff constructor prompts into `solver_agent`.
- negative → repair this adapter only; do not start constructor calls; do not open play path (b).
- ambiguous → preserve fixtures and pytest stdout; do not interpret as construction completion or gameplay support.

Outputs: `analysis/actor-prefix-construction-join-20260914T131345Z/` for this card, adapter, copied observation ABI pin, fixtures, freeze, and pytest evidence.

This card is not approval and not a Kaggle authorization. Implementation of this identity may start only against this card. Live Actor+critic and live construction remain separate packages.

## Frozen before adapter code

New identity. Not a flag on the consumed Kaggle kernel and not an edit of the offline construction-policy identity.

Roles for this unit:
- Actor IPC is input only (already-executed prefix). No live Actor. No `submit_actions`.
- Constructor is not called. Fixture candidate source is host-owned.
- Critic is absent.
- Host adapter is the only writer of constructor-visible `case.json` and of `FROZEN.json`.
- Play path after freeze is (a) `GameplaySolver` over frozen bytes. Path (b) is out of scope.

Actor prefix input (fixture, not a live episode):
- `ipc/states.jsonl` lines shaped as v11/s0 NOOA host state records: `turn`, `step`, `level`, `state`, `levels_completed`, `available_actions`, `grid_rows` (hex strings, alphabet `0123456789abcdef`).
- `ipc/executed_actions.jsonl`: one primitive action string per executed environment step, already drained. Do not parse batched `submit_actions` records in this unit.
- Pairing: state line i is `before` of step i+1; executed action i; state line i+1 is `after`. Require N+1 states and N actions. Reject length mismatch rather than invent a transition.

Constructor surface: copy `observation_abi.py` into this identity as a pin. Canonical observation remains `{"grid": list[list[int]]}`. Compact prompts may omit full grids; any observation they include uses that object. `grid_rows` is Actor-only and forbidden on the constructor-visible surface.

Freeze: write fixture candidate source to `FROZEN.json` (`schema`, `candidate_source`, source digest). Exactly one freeze. `GameplaySolver` loads those bytes in-process. No suffix assessment in this unit.

Out of scope: SCRIBE replacement in `solver_agent`, Actor tool `predict_action`, constructor policy comparison, sealed-suffix scoring, critic seeing a frozen candidate.

Ownership: implementer writes only this directory; reviewer binds exact hashes read-only; no launch owner because Kaggle is forbidden.
