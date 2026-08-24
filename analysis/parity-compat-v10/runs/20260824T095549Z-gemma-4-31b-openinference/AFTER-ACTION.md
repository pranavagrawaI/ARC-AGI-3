# Gemma 4 31B V10 factored compatibility: after action

## Judgment

The run is valid, but it did **not** test the intended identifier-interference or RLE hypotheses. Both arms passed termination, successfully obtained the correct G1 evidence, and identified the LEFT record at level 0 turn 3. Both then failed because the disclosure did not contain the model-facing value required by the terminal schema: `evidence_ref="L0:T3"`.

Native returned the visible bare turn value `"3"`. CodeAct returned the visible disclosure `payload_sha256`. Neither answer is irrational given the available representation. The record itself contained `level=0` and `turn=3`, but no `ref` field and no stated construction rule from those fields to `L0:T3`.

Therefore the machine disposition `rejected_capability_pairing` is too broad scientifically. Capability retrieval succeeded in both arms. The failure is **representation/terminal-contract incongruence after successful disclosure**.

## Observed sequence

```text
G0 termination
    Native    valid ok=true, one call       pass
    CodeAct   valid ok=true, one call       pass

G1 capability retrieval
    Native    rejected underspecified query
              repaired query
              disclosed LEFT at level 0 turn 3
              returned evidence_ref="3"      fail

    CodeAct   rejected empty query
              repaired query
              disclosed LEFT at level 0 turn 3
              returned disclosure hash         fail
```

Both arms used two query attempts and one successful disclosure. The ladder stopped correctly before G2A, G2B, and G3.

## Integrity and cost

- Eight calls were dispatched and accounted.
- Cost: USD 0.000913740.
- Tokens: 4,973 prompt and 1,474 completion.
- Every response used OpenInference and `google/gemma-4-31b-it`.
- No provider, budget, input-cap, or authority violation occurred.

## Consequence

Do not interpret this as evidence that Gemma cannot retrieve evidence or bind identifiers. The intended G2A example assumed a disclosed `ref=L0:T1`, but the actual evidence DTO exposes only separate `level` and `turn` fields. The next correction is smaller than another model accommodation: make record identity explicit in the evidence value, identically for Native and CodeAct, then rerun the ladder under a new frozen package. Do not rerun this authorization.
