# Epistemic questioner v8 local qualification

Result: PASS for implementation and local integration. No live-model or gameplay
claim is licensed.

The v7 two-call/two-step critic with a 20-review cap was replaced by an
event-driven questioner. It reviews on level advance, prediction-check failure,
an observed multi-action sequence, three observed actions, or eight Actor calls
since the last valid review. Concurrent triggers coalesce, there is no review
count cap, and invalid output leaves the boundary pending.

The critic returns one three-field question or an all-empty result. `claim_ref`
must identify an entry extracted solely from Actor assistant content or Actor
tool-call arguments. Full context remains visible, but host, system, user, tool,
and earlier critic material cannot be source-bound as Actor reasoning.

Verification:

- 159 tests passed in the complete runtime suite.
- 103 tests passed in the frozen runtime qualification subset.
- 35 notebook and real-entrypoint tests passed.
- 4 contingent-worker tests passed.
- Runtime contract: `8532c0a7d18fbbf7d4fa4afd583a868cbfe347d03977053533864d2cb9e105d9`.
- Notebook SHA-256: `276c0434267e4848a65e3e33a4b06eb19c8b4482a067da1280e27134d47ca4b4`.

The loopback worker proves an invalid first review is retried, the valid second
review binds to Actor-authored material, the critic has no tools, and the question
enters the same pending Actor continuation. The entrypoint fixture proves reviews
at steps 3, 6, and 9 under single-action play and at every new prediction mismatch
under an intentionally false executable model.

Unresolved: Qwen may still ask generic, misleading, or ignored questions. Prompt
presence, schema validity, and delivery do not establish epistemic benefit. A
live trace test is required before another Kaggle run.
