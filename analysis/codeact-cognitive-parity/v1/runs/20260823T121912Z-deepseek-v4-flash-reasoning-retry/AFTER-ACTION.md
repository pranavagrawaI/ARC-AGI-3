# Invalid repaired-interface retry

## Observation

The repaired native arm accepted the provider's first `evidence_query(kind="transitions")`. That result serialized 16 transitions with two raw 64x64 integer grids each. The next request was 263,423 tokens, above Sail Research's 262,131-token context budget, and was rejected before a second response. One completed call cost USD 0.00007091; the rejected request has no observed cost record. No CodeAct call and no ARC action occurred.

## Causal diagnosis

The authoritative value-copy interface is exact but not context-safe. A 16-transition raw JSON page is about 277 KB before provider tokenization. The failure is caused by the evidence wire representation, not by retrieval or reasoning. The first invalid attempt's parallel-call repair worked as intended.

## Disposition

Preserve all raw artifacts. Add an exact run-length grid encoding as the default for both model-facing evidence lanes while retaining explicit raw encoding for internal parity checks. Any further retry receives a new run identity. Its call and monetary caps must include both invalid attempts against the original 192-call and USD 3 authorization.
