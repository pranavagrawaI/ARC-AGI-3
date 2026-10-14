# Bounded search implementation and test plan

Intent: give the model one reusable search operation over its own successor program, key, goal and optional heuristic. Unit-cost edges only; no game-specific rules or coordinates in the helper. The host manages traversal, predecessor records, reopening and stopping reasons. Returned paths are predictions, not evidence that a game is solved.

Implementation seam: inference/utils/bounded_search.py, embedded into the Python sandbox and exposed as bounded_search. Pure Python and standard library only. It accepts two actual adapter families from existing frozen models: unified sim_step, and separate growth/shrink functions. Test-only adapters supply their archived game interpretation; the solver does not inherit those interpretations.

Failure scene: the new helper appears healthy but returns a false plan, consumes game actions during speculative search, loses callbacks at the next tool call, or turns a bounded negative search into goal rejection.

Mitigations: native prefix/route verification; a sandbox action guard during callbacks; real subprocess retention tests; explicit FOUND/HORIZON_EXHAUSTED/GRAPH_EXHAUSTED/RESOURCE_LIMIT/CALLBACK_ERROR statuses; caps before/after callback returns and between edges; the existing outer tool deadline kills non-returning callbacks. Model callbacks remain responsible for pure behavior and correct state equivalence. Arbitrary heuristics do not guarantee shortest paths.

Memory: stored predecessor nodes, including reopens, are capped. Stable node indices preserve the actual path even if an equivalent state later gets a shorter predecessor. Heap tie breakers never compare model state objects. Callback errors preserve diagnostics and release the sandbox action guard.

Sequence: independent graph/property tests; real ToolAgent/subprocess and saved-function tests; packaged-installer checks; frozen paired native screen; inspect each failure and record next discriminating step. Do not silently replace the incumbent. The bugfix parent's complete hash inventory is checked throughout.

Risk measurement: this new utility has no preexisting function baseline. The inherited parent's qualification is the integration baseline. Use the same explicit stdlib-trace/AST statement-line estimate established for B1 fixes, with a fixed bounded-search test universe, no new dependency/threshold/CI gate, and final new-function scores. Review the full delta rather than splitting responsibilities merely to reduce a metric.
