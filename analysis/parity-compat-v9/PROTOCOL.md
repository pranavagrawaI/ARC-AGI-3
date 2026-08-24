# Parity compatibility v9

## Intervention

V9 changes only the compatibility ladder's terminal boundary. Instead of exposing a wrapper containing a four-way discriminated cognitive-answer union, each Native and CodeAct call exposes one retrieval-only tool schema with two required top-level fields:

```text
return_result(conclusion: string, evidence_refs: EvidenceRef[])
```

T1 validates those fields and constructs the canonical `RetrievalAnswer`; T2 and the model receive no additional authority. General cognitive-parity tasks retain the original union contract.

## Frozen comparison

Gemma V8 failed termination in both arms while progressively approximating the nested union schema. V9 retains the same model-independent gates, evidence, three-call ceiling, executor, authority, and evaluation. The G0 instruction is made congruent with the flat tool contract. Passing G0 tests schema compatibility only; later gates retain their previous meanings.
