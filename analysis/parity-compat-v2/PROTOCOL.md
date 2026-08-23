# Parity compatibility v2

## Question

Can the same model complete both the austere Native and sterile CodeAct protocols before either arm is compared on cognition?

This is an eligibility ladder, not an ARC or reasoning experiment. It has no environment action, mutation, Builder, planner, goals, `wmlib`, `symmetry`, filesystem, or network capability.

## Ladder

Both arms must pass each reached gate within three provider calls:

1. **Termination:** typed return; answer supplied; no evidence.
2. **Capability:** one exact metadata query; no grid decoding.
3. **Representation:** one exact full query using RLE.
4. **Raw fallback:** run only if RLE fails; same task and evidence, raw grids.
5. **Easy cognition:** use the representation selected by gate 3/4.

Native receives `evidence_query` when enabled plus `return_result`. CodeAct receives `execute_python` and `return_result`; its sterile executor receives `evidence.query` only when enabled. Both arms receive the same treatment payload, expected answer schema, checkpoint, query contract, call ceiling, and output-token ceiling. The evaluator-only key is never rendered.

## Interpretation and stopping

- Termination failure: reject the model/protocol pairing.
- Metadata failure: reject the model/capability pairing.
- RLE and raw failure: reject the representation pairing.
- Easy-positive failure: reject the model for cognitive parity.
- All prerequisites pass: eligible for the separately authorized 12-packet shadow screen, using the selected encoding.

Run both arms for a reached gate even if one fails, then stop before the next gate unless both passed. Maximum path: five gates, two arms, three calls = 30 provider calls. Protocol success, exact conclusion/reference match, and required capability use are all necessary. Preserve reports, provenance, provider traces, and partial failures.

No result from this ladder establishes CodeAct reasoning quality or action quality.
