# 0003 — Temporal executable Game Judgment

Frozen, zero-environment-action experiment implementing the matched evidence ladder in
`../0002-decision-relevant-reconstruction/NEXT-LOOP-PLAN.md`.

This directory now contains two authority-separated paths:

1. `fixtures/` plus `game_judgment/provider.py` assemble credential-free provider requests.
   Images are delivered as PNG data URLs, not path metadata. This path never reads `scorer/`.
2. `scorer/` is opened only after a provider response exists. It owns protected successors and
   the complete set of accepted decision signatures.

`freeze_batch.py` passes each case compiler only the public prefix through its declared frontier.
The compiler validates grounded candidates, replays executable transition programs, rejects
contradicted or uncovered plan dependencies, performs bounded route search, and emits a
discriminator when unresolved programs predict different public outcomes.

## Frozen cases

| Case | Frontier | Isolated burden | Prefix-derived decision artifact |
|---|---:|---|---|
| execution-control | 3 | serialize a verified route head | one replay-covered left action |
| planning-control | 11 | construct a route to supplied state/objective | three replay-covered up actions |
| game-judgment | 13 | preserve competing latent programs | one public-outcome discriminator; no executable plan |

## Commands

Regenerate fixtures and run tests:

```bash
uv run python experiments/hypothesis/0003-temporal-game-judgment/freeze_batch.py
uv run python -m unittest discover \
  -s experiments/hypothesis/0003-temporal-game-judgment/tests -v
```

Prepare all 24 exact provider requests without credentials or scorer access:

```bash
uv run python experiments/hypothesis/0003-temporal-game-judgment/run_ladder.py \
  --prepare-only --output /tmp/0003-prepared
```

Run only the paid 31B evidence ladder:

```bash
uv run python experiments/hypothesis/0003-temporal-game-judgment/run_ladder.py \
  --phase ladder --authorize-provider-calls --output runs/<run-id>
```

Recheck only the repeated execution/serialization ceiling after a protocol repair:

```bash
uv run python experiments/hypothesis/0003-temporal-game-judgment/run_ladder.py \
  --phase ladder --only-arm H --authorize-provider-calls --output runs/<run-id>
```

Run the complete gate, including thinking layout, content perturbation, and E4B transfer:

```bash
export GEMMA_E4B_ENDPOINT=http://127.0.0.1:8000
export GEMMA_E4B_API_KEY=...  # optional when the local endpoint has no authentication
uv run python experiments/hypothesis/0003-temporal-game-judgment/run_ladder.py \
  --phase full --authorize-provider-calls --output runs/<run-id>
```

The primary OpenRouter key is read from `ARC_LLM_API_KEY` or `OPENROUTER_API_KEY`. Live calls are
refused without `--authorize-provider-calls`. Full mode preflights the required E4B endpoint before
the first paid 31B request.

## Frozen runtime and gates

`runtime-config.json` pins the 31B model revision, Novita BF16 endpoint, decoding, visual-token
budget, two repeats, model-request cap, cost cap, and stop rules. Endpoint metadata is fetched and
validated before a live run. Provider/model drift, missing parameters, an unhealthy endpoint, or a
cap violation fails closed.

The transport sends at most one request every five seconds and retries only HTTP 429/502/503/504,
at most three attempts with a frozen five-second linear backoff. Every HTTP attempt is retained in
the transport audit and counted separately from logical model requests.

The ladder stops at the first arm where every case passes twice with one stable decision signature.
Full mode additionally requires:

- identical decision signatures on the two thinking-off bookends of off→on→off;
- the predeclared signature change on intact→perturbed→intact; and
- repeated stable success on `google/gemma-4-E4B-it` through a separately audited
  OpenAI-compatible endpoint.

Generated thought text is recorded only as a diagnostic. It is not treated as causal telemetry.

## Visual provenance

Every request audit binds source and delivered PNG hashes, source and delivered pixel sizes,
whether the `STEP N` overlay was rendered, the requested Gemma 4 visual-token budget, and the
reference-realized square token grid. The frozen budget is 280; the published Gemma 4 reference
processor realizes a 16×16 grid (256 visual tokens) for a square image. Provider-reported visual
grid remains a separate nullable field rather than being inferred from billing tokens.

References:

- <https://ai.google.dev/gemma/docs/capabilities/vision/image>
- <https://ai.google.dev/gemma/docs/core/model_card_4>
- <https://openrouter.ai/docs/guides/overview/multimodal/image-understanding>
- <https://openrouter.ai/docs/guides/routing/provider-selection>
