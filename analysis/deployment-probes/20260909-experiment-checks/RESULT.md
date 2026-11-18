# Offline experiment checks

> **Current reviewed result:** three independent review passes completed; supported
> defects fixed; 49 tests pass. Use [review disposition](REVIEW.md) and reports in
> `reviewed-final/`. The original outputs below and intermediate `reviewed/`
> outputs are retained as development evidence. The original v2 response count
> omitted its preserved failed JSON response; corrected accounting is 3 issued,
> 3 preserved, 2 successful index rows.

Implemented in [experiment_checks.py](../../../tools/experiment_checks.py).
Commands, rule mappings and remaining review obligations are documented in
[EXPERIMENT-CHECKS.md](../../../tools/EXPERIMENT-CHECKS.md).

| Evidence | Checked result |
| --- | --- |
| [Flash-Next v2](flash-v2.json) | Invalid: server OOM; missing/unclassified evidence remains listed |
| [Flash-Next v3](flash-v3.json) | Invalid: server OOM, despite all eight responses |
| [Off arm](batching-off.json) | 13 feedback rows, permission true at start steps 6 and 14; two batch submissions, seven distinct helper hashes |
| [On arm](batching-on.json) | 19 feedback rows, permission false throughout; zero batch submissions, one distinct helper hash |
| [Context coverage](coverage.json) | Fixture maximum 21,257 vs reported live maximum 38,644; 8/136 prompt counts above 32,768, 11/136 with 1,024 output tokens reserved |

The v3 resource audit separates the all-run sampled minimum (1,259 MiB free)
from the three post-readiness samples (5,187 / 3,027 / 3,027 MiB free). It reads
514.107 seconds of setup separately from 410.056 seconds of readiness and
103.938 seconds of extraction. It makes no instantaneous serving-minimum claim.

Batch reports retain submission, matching prediction records and logged stop
reasons with file/line provenance. They do not invent a decision-time join where
the logs lack atomic proposal/feedback/hash snapshots. `declined_opportunities`
is null, not zero. Helper-version counts do not prove successful repairs.
The inventory's `supported` status supports only these recorded facts.

Context figures are provider-reported counts. Candidate tokenizer/template
verification remains unresolved; this report establishes a coverage defect in
the reported distributions, not exact cross-model token parity.

Verification: 28 regression tests pass, including the historical v2/v3 artifacts,
both live arms, direct calls to the existing response validator and batch gate,
and adversarial synthetic cases. Reproduce:

```bash
uv run --no-project --with pytest --with numpy --with jsonschema python -m pytest tools/tests/test_experiment_checks.py -q
```

These are offline commands with nonzero exit codes for invalid/unresolved claims.
They are not installed into a future launcher or publication pipeline. Frozen
cards, probes, raw evidence and solver runtime are unchanged. Current verdict
banners were added to the old v2/v3 reports; original accounts remain visible.
No GPU run, model request or game action was performed.
