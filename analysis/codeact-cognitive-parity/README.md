# Frozen cognitive parity benchmark

This benchmark is defined independently of native tools and CodeAct. Each
packet binds one repository-relative checkpoint tree, an evidence frontier, a
task instruction, treatment-visible inputs, and an evaluator-only answer key.

The four task kinds intentionally make different claims:

| Kind | Strongly scored | Not inferred |
|---|---|---|
| `retrieve` | evidence precision and recall | prose similarity |
| `discriminate` | evidence retrieval and accepted verdict | hidden reasoning |
| `recover_invariant` | evidence, invariant identity, and scope | universal transfer |
| `propose_action` | legality, grounding, and native-action agreement | action quality |

`CognitiveParityPacket.treatment_payload()` is the only packet view a runner
may place in model context. It mechanically excludes `expected`. Loading a
packet re-hashes every relative file path and byte in its checkpoint tree.

The first three kinds establish cognitive parity. Action proposals are a later
difference test: disagreement with a historical native action is parity data,
not evidence that either decision is better. No packet executes an ARC action.

Checkpoint curation remains the next benchmark task. Packets should be selected
adversarially from known Tycho successes and failures rather than sampled at
random, and a channel beyond ordinary frames/transitions should be admitted
only when a selected packet demonstrably requires it.
