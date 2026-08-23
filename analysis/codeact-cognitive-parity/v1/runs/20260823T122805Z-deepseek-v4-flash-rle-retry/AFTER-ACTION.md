# Invalid RLE retry

## Observation

Exact RLE transport kept the native evidence request within context. The native easy-positive arm completed with evidence precision and recall 1.0 after five calls. Its three rejected returns were schema repairs, recorded rather than silently accepted. The first CodeAct response then had `finish_reason="stop"`, blank content, malformed reasoning, and zero tool calls. The controller stopped before any Python cell or evidence query. Six completed calls cost USD 0.008192905. No ARC action occurred.

## Causal diagnosis

The RLE intervention solved the measured context-overflow cause. The remaining failure is asymmetric provider-turn recovery: native returns receive validation feedback, while CodeAct treats a zero/multiple/unknown tool-call turn as fatal. This makes one stochastic malformed response terminate the entire treatment.

## Disposition

Preserve all artifacts. Add a T1-authored `ProviderTurnRejected` provenance event, render its bounded instruction into the next model turn, and count it as an invalid return. Do not infer CodeAct incapability from this attempt. Any retry gets a new identity and cumulative call/cost caps.
