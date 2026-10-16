# Offline follow-up, 2026-10-16: forced helper invocation has no upside here

Question: should the next GPU run force `bounded_search` invocation (FINDINGS "supported next question")?
Answer from the existing 16 traces (`offline_classify.py`, no GPU): **no**.

Every one of the 15 failed episodes already wrote its own search (BFS/heap/A*). None failed for lack of search:

- **Search exhausted under the model's simulator** (goal unreachable in its rules): e.g. `b1-2903-control`
  explores 1,151,128 nodes and then 150,000 states with no route; `control-2903-control` returns EXH/CAP.
- **Route found in simulation, first native step mismatches**: `control-1701-*`, `control-2903-both`
  ("WIN at action 38" in sim, then `N+` MISMATCH on execution). Several simulators had 0 historical
  mismatches and still failed forward: history-consistent, wrong on the untested mechanic.
- **Token exhaustion**: 17 of ~150 responses hit the 12,288-token cap (about 11%, versus 1.1% in production
  runs, per constraint-audit-20261013), and the 32,768-token episode budget absorbs three of those.

`bounded_search` called on the same wrong simulator returns the same EXH or the same wrong route. The
only success (`control-1701-control`) came from repairing a rule after a native mismatch, not from search.
This matches oracle-hint (route 26/26, goal 0.50): the bottleneck is rule induction, not search.

Side finding (small-win candidate, not a bet): 39 retention rejections ("unavailable dependencies",
"defaults must be literals") produce 14 later NameErrors in 123 tool results (11%), 12 of them at the
B1 checkpoint. The mechanism is real friction; its level effect is unmeasured.

No GPU run follows from this experiment.
