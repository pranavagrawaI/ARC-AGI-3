# 0003 live full-route ceiling E4B v1

- **Question:** Without a play cutoff, how many online LS20 levels can E4B execute at the route ceiling?
- **Code:** `da27394`; fixture `410cf80`.
- **Contrast:** None; terminal-or-failure performance run.
- **Prediction:** 404 exact decisions/successors reach `WIN`: 7 levels, zero probes.
- **Measure:** Levels, actions, probes, calls, completion indices, exact checks, raw trace/session.
- **Cap:** No play cap; stop at `WIN`. Certificate limits exposure to 404 local calls/actions; $0; 2 hours.
- **Stop:** `WIN`, first mismatch/abstention/error, or certificate exhaustion; no head after mismatch.
- **Invalid:** Model/session/reset/identity drift, unavailable environment, incomplete record.
- **Map:** WIN → report full execution ceiling; partial → report first failed level/action; invalid → repair/new card.
- **Boundary:** Historical route is privileged Machinery input; this does not test acquisition, discovery, or general ARC performance.
