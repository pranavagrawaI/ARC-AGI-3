# Local qualification

Runtime contract: `8532c0a7d18fbbf7d4fa4afd583a868cbfe347d03977053533864d2cb9e105d9`.

- Full runtime suite: 159 passed.
- Frozen qualification subset: 103 passed.
- Exact notebook and entrypoint suite: 35 passed.
- Contingent-worker suite: 4 passed.

The live worker test proves that an invalid critic response leaves the
eight-Actor-call boundary pending, a later valid question binds to an
Actor-authored event, the critic receives no tools, and the question reaches the
same pending Actor continuation. Packaging tests cover three-action cadence and
per-mismatch cadence through the real subprocess entrypoint.

Evidence: `analysis/deployment-probes/20260912-epistemic-questioner-local-v1/`.

These are deterministic and loopback HTTP checks. They do not establish Qwen
question quality, Actor adoption, gameplay benefit, GPU behavior, or Kaggle
performance.
