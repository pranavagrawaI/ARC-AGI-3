# Matched Actor V2 Pilot Protocol

## Question

Can the unchanged six-packet Matched Actor pilot run without an evidence disclosure or assembled provider request exceeding its frozen resource contract?

## Sole intervention

V2 preserves V1's sample, prompts, model route, cognitive controllers, authority, caps, counterbalanced order, and advancement rule. It changes only the trusted resource boundary:

1. Evidence records are atomic and pages stop at both record and serialized-byte limits. A continuation cursor and explicit truncation reason preserve the remaining sequence.
2. The launcher measures the exact normalized request after messages, tool schemas, output schema, and provider parameters are assembled. A conservative UTF-8-byte upper bound plus protocol reserve must fit the input and total-token ceilings before `CallPlanned` and dispatch.
3. Reconciliation records actual provider spend independently from valid authorized spend and violating spend.
4. The pinned CompletionClient preparation path is mirrored for admission and endpoint retries are forced to zero. One ledger dispatch permits one provider attempt.

No provider call is authorized by this disabled manifest.

## Zero-cost acceptance

- The preserved V1 call-40 request is rejected before `CallPlanned` and dispatch.
- Its triggering evidence query returns an atomic payload-bounded page with a continuation cursor.
- Pagination has no duplicate or omitted records.
- Native and CodeAct use the same bounded evidence implementation.
- Request admission catches accumulated history even when one disclosure is safe.
- Provider-reported usage beyond the preflight bound becomes a persisted accounting violation; recovery never retries it.
- Malformed usage becomes a terminal accounting violation with its worst-case reserved exposure retained.

## Future paid rerun

A fresh authorization may bind an enabled copy only after the zero-cost proof passes. It must retain the V1 six-packet sample, DeepSeek route, prompts, caps, and advancement rule. The V1 partial result remains inconclusive and is not pooled with V2.
