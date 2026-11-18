# Recovered automatic-animation control

The control is recovered and passes the frozen public-screen health gate. The pair remains incomplete: the peak treatment was never launched, so animation benefit and the paired no-harm threshold remain untested.

## Observed result

- All 25 expected games, 74 levels, mean score 29.29355623, 3 wins and 22 time cancellations; 2743 recorded actions.
- Expected RTX PRO 6000 Blackwell hardware, readiness at 758 seconds, and 3696.43 seconds from benchmark start to final export.
- All 25 request traces present: 1409 requests, 1396 recorded responses, 2724152 completion tokens, zero automatic-image attachments.
- Serving audit: no logged non-200 chat responses, no trace error events, no detected fatal CUDA/OOM errors. Two pre-ready /health 503 responses precede benchmark start.
- Original kernel ID 141075118, version 1; current source matches the frozen control. All 263 downloaded files (954749950 bytes) had their recorded SHA256 hashes independently recomputed with no mismatches.

## Historical comparison and limits

The October 20 D16 screen used the identical parent notebook and game IDs: 80 levels, mean score 31.45792341, 3 wins, 3061 actions. This recovered run differs by -6 levels and -2.16436718 mean-score points. Six games gained levels, eight lost levels, and eleven were unchanged. The same three games were won. Largest level changes include tn36 4 to 0, sc25 3 to 0, and sp80 1 to 4.

These are unmatched trajectories on different workers/dates. The mixed per-game changes describe two runs; they do not identify a cause or estimate ordinary run variance. The public repeat confirms that the unchanged solver can still produce a substantial one-hour run, without validating hidden execution or diagnosing D16's hidden score.

## Recovery and disposition

The original controller stopped before collection because its two-hour submission-based observation deadline included queue time. The original STOPPED state and closure remain preserved. Following the user's separate recovery authorization, the existing collection-only subcommand downloaded the completed control, and local readers recomputed its results. No GPU launch, retry, treatment, source edit, hidden submission, queue action or promotion occurred. The follow-up schedule remains disabled.

Actual GPU consumption remains unknown within the accepted 6000-second control cap. No terminal per-run usage receipt was recovered; benchmark or log wall time is not substituted for GPU accounting.

Evidence: READOUT.json; HEALTH.json; SERVING-AUDIT.json; source-readback/identity.json; collection-execution.json; ../control/collection.json; ../control/request-receipts.json; ../control/output/benchmark.json; ../control/output/serve.log. read_control.py reproduces the local calculations into new files; it intentionally refuses to overwrite existing readouts.
