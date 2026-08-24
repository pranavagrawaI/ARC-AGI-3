# V12 compatibility protocol

V12 changes one burden from V11: G2A has no evidence capability. Both arms receive the same immutable task value containing a record `ref=L0:T1`, `frontier=L1:T2`, and `next_after=L0:T1` and must return only the record ref.

1. G0 adds typed termination.
2. G1 adds one metadata disclosure and returns a disclosed record `ref`.
3. G2A tests identifier binding under interference from immutable task data.
4. G2B supplies immutable RLE rows and tests only encoded-value interpretation.
5. G3 composes disclosure, RLE interpretation, record binding, and typed return.

G2A classifications: `L0:T1` correct; `L1:T2` frontier substitution; `L0:T2` identifier recombination; anything else generic grounding error. Metrics retain Python cells, provider calls, and terminal-return call index so unnecessary computation remains visible. Both arms in a started gate run before advancement is decided. Raw grids, decoder, fourth calls, ARC action, Builder, goals, planner, and theory mutation remain unavailable.
