# Invalid comparison: volatile seed was evicted before the arm ended

The compact seed was present in every paid Actor request through baseline call59.
The baseline made no environment action and no observable correction in that prefix.
At call60, context truncation removed the volatile seed. The v2 fail-closed wrapper
blocked the request before OpenRouter, satisfying its safety purpose but triggering
the card's input invalidator. Because reservation occurred before the wrapper, the
ledger retained an unresolved reservation. The critic arm and pair B did not start.

This prefix demonstrates prolonged decision non-closure under the supplied evidence;
it does not provide an arm comparison. All requests, responses, snapshots, accounting,
and the failure are preserved. V3 places the same compact seed in the cacheable prefix
and checks its presence before reserving request spend.
