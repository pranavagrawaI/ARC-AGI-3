# 0158 LS20 long-run attempt v18

## Result

The fresh autonomous-continuation run executed one ARC action, recorded its
contradicted probe, and reset the grounding cycle without human mediation. It
then stopped before action 2 because Qwen exhausted the 8,192-token generation
allowance while deliberating over the second claim.

- Run: `artifacts/live-ls20/20260812T140729Z-qwen36-autonomous-44`
- Run ID: `fd587cf27b5f45659a6564d80caf1594`
- Actions: 1 of 44
- Levels completed: 0
- Provider calls: 16 successful calls; no transport failure
- Counted usage: 186,034 provider-reported input plus output tokens
- Scorecard: terminal, non-resumable

## Causal result

The autonomous policy behaved as designed. The first claim and commitment were
accepted, action 1 executed, the verifier returned
`PRIMARY_REPLACEMENT_MISMATCH`, and the complete claim/commitment/verdict was
stored as immutable `probe_result` evidence. The gate then reset to `EMPTY`
with `human_mediation: false` and no new API.

During the second claim, Qwen read that transition and probe result, made four
successful inspection calls, and then used all 8,192 output tokens comparing
candidate regions without submitting. The provider response was `incomplete`
with reason `max_output_tokens`. Thus the proximate failure is model generation
exhaustion; accumulated full-frame exposure and an unbounded exploration
convention materially contributed.

## Next candidate

The next-run interface retains only the current full grid, while preserving
metadata for all frames and exact transition/probe history. It tells Qwen not
to print or restate the grid, permits at most three exploratory calls before
submission, and frames action as the way to resolve remaining uncertainty.
The output allowance doubles conditionally to 16,384 tokens. No scored
trajectory is resumed.

The executed 20-file candidate was hash-checked and snapshotted. Failure-packet
checksums pass, and `classification-correction.json` preserves the detailed
classification without editing immutable packet evidence.
