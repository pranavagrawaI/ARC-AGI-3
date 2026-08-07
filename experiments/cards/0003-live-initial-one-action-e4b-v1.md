# 0003 live initial one-action E4B v1

- **Question:** Does the frozen historical replay certificate transfer to a fresh online LS20 seed-0 session?
- **Implementation:** commit `78971043f1f92cbeecfda9fa8c049c3dff483129`.
- **Contrast:** Feasibility gate only; no causal contrast.
- **Prediction:** Reset frame/public state match exactly; E4B selects `live-initial-action3`; one `ACTION3` yields the exact frozen successor frame and public fields.
- **Measure:** Exact current/decision/successor checks plus raw request, response, and frames.
- **Cap:** 1 local E4B request; 1 non-reset environment action; $0 provider cost; 10 minutes.
- **Stop:** Abort before action on any reset, endpoint, contract, legality, or decision mismatch. After action, never execute a suffix.
- **Invalid:** Model ID drift, online reset mismatch, unavailable environment, incomplete raw record, or runner/card identity mismatch.
- **Result → action:** PASS permits a separately precommitted bounded route-ceiling episode. FAIL stops live work and localizes the mismatch. INVALID gets a new card after repair.
