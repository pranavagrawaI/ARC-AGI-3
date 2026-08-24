# Matched Actor V2 Pilot — zero-cost repair after-action

## Judgment

The resource-safety repair passes offline. The cognitive comparison has **not**
been rerun and remains unauthorized. V1 remains an immutable, inconclusive
partial observation.

## Two containment rings

The evidence service now limits both record count and compact serialized bytes.
Records remain atomic; the preserved call-40 query returns two records (6,605
bytes), `truncated=true`, `truncation_reason="payload_limit"`, and continuation
cursor `L0:T2` under an 8,192-byte ceiling.

The provider adapter assembles messages, tools, output schema, model, and
non-secret provider parameters before admission. With no certified route
tokenizer, it uses the conservative bound `UTF-8 bytes + 512 protocol tokens`,
reserves the requested output ceiling, and persists `CallPlanned` only after
input, total, call, and dollar checks pass.

For the pinned NOOA `CompletionClient`, admission uses the same cache-control,
tool conversion, structured-output, dummy-tool, and tool-choice preparation as
the provider client. Transport credentials and client objects are excluded
because they cannot change prompt tokens. Endpoint and rate-limit retries are
forced to zero, so one durable dispatch permits exactly one provider attempt.

## Preserved call-40 replay

The exact V1 request serializes to 74,870 bytes. V2 bounds it at 75,382 input
tokens against the unchanged 16,384 ceiling, records `RequestRejected`, and
makes zero provider calls. See `call-40-replay.json`.

## Financial semantics

V2 reconstructs three disjoint views from its journal:

- `actual_provider_spend_usd`: every response with priced usage, including a
  violating response;
- `authorized_accounted_spend_usd`: accounted calls whose observed route and
  usage satisfy authorization;
- `violating_spend_usd`: route or usage violations that nevertheless cost
  money.

Applied conceptually to preserved V1 evidence:

```text
actual_provider_spend       $0.012034884
authorized_accounted_spend  $0.008180996
violating_spend             $0.003853888
```

A provider-reported token count above the conservative preflight bound is now a
persisted integrity violation. Recovery reconstructs the paid violation and
never retries it. Malformed or unpriceable usage also becomes terminal; its
unknown financial exposure remains reserved rather than disappearing.

## What remains invariant

V2 reuses the exact V1 six-packet sample and semantic hash, Native and CodeAct
prompts, DeepSeek route, counterbalanced schedule, per-session/global caps, and
advancement rule. Both arms use the same V2 evidence service. No canonical or
ARC mutation capability was added.

## Next decision

The disabled V2 manifest may be enabled only by a fresh hash-bound authorization.
A future paid rerun must change no cognitive treatment input at the same time.
