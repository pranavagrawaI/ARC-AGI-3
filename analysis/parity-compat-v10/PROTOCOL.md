# V10 compatibility protocol

The ladder is sequential and raw grids are not an alternative under this resource contract.

1. G0 adds typed termination.
2. G1 adds one metadata disclosure.
3. G2A reproduces the V9 identifier topology: record `L0:T1`, frontier `L1:T2`, `next_after=L0:T1`.
4. G2B supplies two immutable RLE rows directly and adds only encoded-value interpretation.
5. G3 composes one disclosure, RLE interpretation, record binding, and typed return.

G2A classifications are exact: `L0:T1` correct binding; `L1:T2` frontier substitution; `L0:T2` turn-number leakage; anything else generic grounding error. Both arms in a started gate run before advancement is decided. No ARC action, Builder, goals, planner, theory mutation, decoder, or raw-grid fallback is authorized.
