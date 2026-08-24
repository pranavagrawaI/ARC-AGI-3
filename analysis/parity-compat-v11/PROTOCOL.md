# V11 compatibility protocol

V11 changes one model-visible value from V10: every evidence record now includes its canonical `ref` (for example, `L0:T1`). Cursor objects remain separate level/turn values. The evidence implementation is itself hash-bound.

1. G0 adds typed termination.
2. G1 adds one metadata disclosure and returns a disclosed record `ref`.
3. G2A reproduces the interference topology: record `ref=L0:T1`, frontier `L1:T2`, and `next_after=L0:T1`.
4. G2B supplies immutable RLE rows and adds only encoded-value interpretation.
5. G3 composes disclosure, RLE interpretation, record binding, and typed return.

G2A classifications: `L0:T1` correct; `L1:T2` frontier substitution; `L0:T2` turn-number leakage; anything else generic grounding error. Both arms in a started gate run before advancement is decided. Raw grids, decoder, ARC action, Builder, goals, planner, and theory mutation remain unavailable.
