# Experiment evidence checks

`experiment_checks.py` is an offline audit of the preserved Flash-Next and
contingent-controller evidence formats. It calls no model, executes no saved
helper, and does not change the frozen candidate or card. Run it before using
a probe result to select the next experiment.

Exit codes: **0** supports only the report's named claim; **1** finds a
contradiction/invalidator; **2** leaves the claim unresolved. An invalidator wins
even when other evidence is missing. `verdict.json` and notebook COMPLETE are
never inputs to a validity decision. JSON outputs retain evidence SHA-256s and
line references. `--output` refuses an existing report file.

```bash
python3 tools/experiment_checks.py --output /new/path/audit.json flash RUN/raw
python3 tools/experiment_checks.py batching RUN/solver-workspace/ipc
python3 tools/experiment_checks.py coverage RUN/raw OFF/ipc/model_calls.jsonl ON/ipc/model_calls.jsonl --ceiling 32768 --output-tokens 1024
python3 tools/experiment_checks.py preserve DOWNLOADED_ATTEMPT NEW_ATTEMPT_DIRECTORY
```

The output parent directory must exist. `preserve` exclusively creates its
destination, rejects symlinks, special filesystem entries and nested destinations, and hashes the copied
files. It refuses a preexisting destination, including an empty one. A failed
copy remains for inspection and has no completed manifest. Download each
platform version into a new directory **before** preservation; copying cannot
restore artifacts a downloader already overwrote. The snapshot manifest records
bytes, not proof of platform/version identity: retain the platform execution ID
and frozen candidate/card manifest alongside the evidence.

## Rule coverage and explicit limits

| Card condition or claim | Check | Remaining judgment |
| --- | --- | --- |
| OOM invalidates | Detect allocation-failure/exception signatures with source lines, even if recovered | Other unrecognized log formats require inspection; never infer absence from an empty/missing log |
| Missing terminal stream/usage | Read raw SSE DONE, finish and token usage; compare results | Full response semantics remain a separate check |
| Request accounting | Inventory issued raw requests/responses/streams, including failed cases absent from results index; compare saved hashes and index fields | Separate unissued planned cases from lost issued responses; review unexpected/missing records |
| Failed cleanup | stopped, survivors, port and ownership match | Missing fields remain unresolved |
| Timeout | Inspect recorded failure tracebacks | All caps, including platform wall time, require review |
| Wrong model/runtime/GPU | Remains unresolved | Compare complete identity/provenance with frozen pins |
| Allowed prose/whitespace | Regression tests execute the existing probe's `generate` against synthetic streams | Fixtures are tests, not observations of the real server |
| Malformed/unoffered tools | Regression tests execute the actual probe with invalid tool names, argument JSON and schemas | Test additional response formats when the card changes |
| Steady-state memory | Split GPU samples using readiness epoch; report each serving sample | Instantaneous serving minimum remains unmeasured |
| Setup cost | Read setup, readiness and extraction separately | No comparison between configurations without a contrast |
| Batch eligibility | Compare three prerequisites against the actual runtime gate in tests; validate row types before counting | Legitimate no-model feedback can have no helper hash; historical inventory cannot reconstruct an atomic decision-time gate snapshot |
| Stop cause | Carry recorded stop reason and skipped count into submission ledger | Do not infer wall collisions or usefulness from action names |
| Context coverage | Compare fixture/live distributions, include output reservation, retain tail source lines | Retokenize exact messages/tools using candidate tokenizer/template; provider counts alone do not prove candidate compatibility |

The Flash audit deliberately cannot issue an overall feasibility PASS: identity,
semantics and cap checks still require evidence review. A clean scan returns
unresolved with those obligations listed. Its rule mapping targets the historical
card's exact Invalid-if clause and does not interpret arbitrary Markdown cards.
The batching command can support an **inventory**; it never reports a count of
declined opportunities. True permission rows need a committed matching proposal
and decision-time ordering before they become executable opportunities.

## Review before spending compute and before publishing

Before a successor run, freeze a new run ID, candidate and card. For every Stop
and Invalid-if rule, identify its executable check or reviewer evidence field.
Inspect each executable stop for a corresponding card clause. Exercise a passing
case and a discriminating failure, including completed-with-OOM, missing evidence,
allowed prose, permission false, and a changed helper hash. Select fixtures using
the live size distribution and the candidate context boundary. Save exact request
hashes, tokenizer/template identity and output reservations. Size coverage alone
does not establish behavioral or workload representativeness.

Before publishing, have a fresh reviewer derive the verdict from the card and raw
artifacts before reading the author's conclusion. The review must resolve the
audit's explicit pending obligations or keep the verdict unresolved. A correction
belongs at the top of the report, with superseded text visibly labeled. Human
review is required here; this tool does not claim to automate independence or
understand every card condition.

The independent September 9 review and its remaining evidence limits are recorded
in [the review disposition](../analysis/deployment-probes/20260909-experiment-checks/REVIEW.md).
It reconciles all 11 issued Flash responses, including the v2 failed exact-output
case, and verifies recorded identity metadata and observed caps. Those historical
review findings do not automatically discharge obligations for another run.

The historical probe also retains a nonempty-reasoning abort at `probe.py:84`
that the frozen card does not explicitly list as a Stop. Whitespace tests do not
prove full card/implementation parity. A successor must decide that condition
prospectively; the frozen probe is preserved as evidence.

## Tests

```bash
uv run --no-project --with pytest --with numpy --with jsonschema python -m pytest tools/tests/test_experiment_checks.py -q
```

Synthetic tests cover invalidation, missing/malformed nested evidence, no-overwrite
preservation, context/output accounting, temporal ambiguity, and the real probe
and runtime gate functions. Historical regression tests additionally read the
retained September 9 artifacts when available; they skip explicitly when that
corpus is absent. No GPU, API, game actions or helper execution is involved.
