# Gemma 4 31B V11 explicit-ref compatibility: after action

## Judgment

V11 repaired the V10 evidence/output mismatch exactly: after every disclosed evidence record gained its canonical `ref`, both Native and CodeAct passed G1 with `L0:T3`.

G2A does **not** establish an identifier-binding failure in CodeAct. Native selected `L0:T1` correctly. CodeAct spent two calls repairing query syntax, used its third and final call to disclose the correct record—including `ref=L0:T1` beside `frontier=L1:T2` and `next_after=L0:T1`—and then had no call left for `return_result`.

The machine disposition `rejected_identifier_binding` correctly names the failed gate, but it is too broad as a scientific diagnosis. The observed CodeAct failure is **correction-path exhaustion after successful disclosure**. Identifier selection was never observed, so identifier interference remains unresolved for that arm.

## Observed sequence

```text
G0 termination
    Native    valid ok=true, one call                         pass
    CodeAct   valid ok=true, one call                         pass

G1 capability retrieval
    Native    disclosed LEFT record ref L0:T3
              returned L0:T3                                 pass

    CodeAct   repaired query, disclosed LEFT record ref L0:T3
              returned L0:T3                                 pass

G2A identifier binding under interference
    Native    disclosed record ref L0:T1
              selected L0:T1                                 pass

    CodeAct   call 1 omitted limit=1                          reject
              call 2 passed the query as a positional dict   TypeError
              call 3 used exact keyword arguments
                     disclosed ref=L0:T1                     success
              no call remained for return_result             fail
```

The ladder stopped correctly before G2B and G3.

## Integrity and cost

- Twelve calls were dispatched and accounted.
- Cost: USD 0.001279240.
- Tokens: 7,678 prompt and 1,900 completion.
- Every response used OpenInference and `google/gemma-4-31b-it`.
- No provider, route, budget, input-cap, or authority violation occurred.

## Consequence

Do not add an RLE decoder: G2B was never reached, so V11 provides no new evidence about representation competence.

Do not interpret the overall gate failure as identifier substitution. The clean next probe should isolate identifier binding from capability syntax by supplying the exact G2A disclosure topology as immutable task data and asking only for the record ref. G1 has already established capability retrieval. This is more discriminating than merely buying a fourth call, which would preserve the confound between query repair and identifier binding.
