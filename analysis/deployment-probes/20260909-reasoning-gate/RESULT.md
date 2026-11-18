# Reasoning gate repair

Implemented in `deployment/kaggle/designer-player-discovery-20260908-v2`.

Routine client construction rejects global reasoning-on settings. Both synchronous
and asynchronous request entry points validate configuration and per-call
overrides. Ordinary requests explicitly disable both reasoning switches.
`deep_investigate()` supplies an internal identity capability on its own request;
the capability is removed before provider transport. No ambient context or shared
configuration is switched on, so concurrent summarization cannot inherit it.
The existing default one-call deep budget remains enforced.

The notebook builder rejects global thinking-on settings. The historical
`probe.ipynb` is unchanged. A separate `reasoning-gated.ipynb` has been generated
locally; it has not been uploaded or run on Kaggle and is not a new launch card.

Verification: 85 candidate tests passed, plus seven packaging tests. Coverage
includes real worker subprocesses against a local HTTP fixture, scoped deep
invocation and exhausted budget, rejection of ordinary reasoning overrides,
and an actual NOOA summarizer running concurrently with a held deep request.
The observed reasoning switches in the concurrency test were on/off/off for
deep/summarizer/next ordinary call. Packaging tests extract freshly generated
notebook bytes, verify manifest hashes, and run the embedded worker.

The first summarizer fixture used the wrong response schema; it was corrected
to the real Predict response field `value`. Two lifecycle fixtures depended on
global thinking-on; they now use off while retaining their original length-error
and signal-retirement assertions.

Contract: `f19e8d3c96d934a876c0be3db7b698918d064448b7c4d89f8121914c0bb36bae`.
Notebook SHA-256: `974d9ce34a77c8aa08b4601dc21b530c72a69453cf2bc9cfe3be9b877e6bf503`.

Limits: local request-path qualification is not target-GPU validation or evidence
of improved solving. The model still judges when an abstraction problem merits
the deep tool. This guard enforces the call route and budget, not the semantic
quality of that judgment, and is not a security boundary against arbitrary Python.
