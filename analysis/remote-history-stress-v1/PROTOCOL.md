# Protocol

The frozen `remote-failure-composition` packet is evaluated through three
answer-key-independent scripted paths over the same checkpoint and exact source
pages:

1. Native requests every bounded page, computes, then returns.
2. CodeAct iterates over bounded queries in one persistent Python cell, then
   returns.
3. Native calls the generic `evidence_acquire` selector once, computes over its
   immutable granted handle, then returns.

`evidence_acquire` accepts only generic evidence selectors. It performs no
filtering, comparison, aggregation, or task-specific interpretation. The proof
makes no provider calls and takes no ARC actions.
