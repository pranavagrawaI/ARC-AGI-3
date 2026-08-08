# 0003 live full-route ceiling E4B v2

- **Question:** How many online LS20 levels can E4B execute without a play cutoff?
- **Code:** `bc8feeb`; fixture `410cf80`.
- **Contrast:** Rerun after ARC's 10 s POST timeout invalidated v1 action 376.
- **Prediction:** 404 exact decisions/successors reach `WIN`: 7 levels, zero probes.
- **Measure:** Levels, submitted/confirmed actions, probes, calls, exact checks, raw trace.
- **Cap:** No play cap; stop at `WIN`. Certificate limits exposure to 404 local calls/actions; $0; 2 hours.
- **Stop:** `WIN`, mismatch/abstention/error, or certificate exhaustion; no action retry.
- **Invalid:** Model/session/reset/identity drift, unavailable environment, incomplete record, missing successor.
- **Map:** WIN → report full ceiling; partial → report first gameplay failure; invalid → preserve separately.
- **Boundary:** Historical route is privileged Machinery input; this does not test acquisition, discovery, or general ARC performance.
