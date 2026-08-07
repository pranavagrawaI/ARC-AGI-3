# 0003 live initial one-action E4B v1

- **Question:** Does the frozen replay certificate transfer to online LS20 seed 0?
- **Code:** `78971043f1f92cbeecfda9fa8c049c3dff483129`.
- **Contrast:** Feasibility gate only; no causal contrast.
- **Prediction:** Reset matches exactly; E4B selects `live-initial-action3`; one `ACTION3` gives the exact frozen successor.
- **Measure:** Exact current, decision, and successor checks; raw request, response, frames.
- **Cap:** 1 local E4B request; 1 non-reset environment action; $0 provider cost; 10 minutes.
- **Stop:** Abort before action on reset, endpoint, contract, legality, or decision mismatch. Never execute a suffix.
- **Invalid:** Model drift, reset mismatch, unavailable environment, incomplete record, or identity mismatch.
- **Map:** PASS → precommit a bounded route-ceiling episode; FAIL → stop and localize; INVALID → repair under a new card.
