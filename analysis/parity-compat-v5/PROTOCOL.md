# Parity compatibility v5

## Intervention

V5 repairs the meaning and accounting of evidence queries. All raw Native and CodeAct requests pass through the same semantic normalizer before authorization and execution.

For `projection="metadata"`, `grid_encoding` cannot affect the returned value and is removed from the semantic query. Omitted, `raw`, and `rle` spellings therefore authorize the same request, share one canonical hash, and use one cache entry. Fields that can change the metadata result remain required and exact. Full projections retain `grid_encoding` as a semantic dimension for Gate 2.

Every capability invocation consumes one bounded query attempt. Only a valid request whose result crosses into the model domain consumes a successful-disclosure allowance. Evidence gates permit at most two attempts and one successful disclosure. A rejected request can therefore be corrected once, while a second successful disclosure remains impossible.

## Provenance

T1 records the raw arguments in `CapabilityRequested`, the canonical query and hash in `CapabilityNormalized`, and actual information release in `CapabilityDisclosed`. Rejection never implies disclosure. The model-visible evidence contract names `max_query_attempts` and `max_successful_disclosures`; metadata contracts omit grid encoding.

## Ladder and interpretation

The ladder remains: termination; metadata capability; RLE representation; raw fallback; easy cognition. Both arms finish a reached gate before advancement is judged, with three provider calls per arm. No evidence, authority, terminal schema, transport, evaluator key, filesystem, network, Builder, planner, goal, or ARC mutation capability changes.

V5 tests whether the cross-model V4 failure was caused by query semantics and retry accounting. Passing Gate 1 establishes bounded metadata capability use only. It does not establish RLE competence or cognitive superiority.

Paid execution requires a new manifest-bound authorization, exact route, frozen pricing, worst-case reservation, append-only accounting, and fail-closed provider handling.
