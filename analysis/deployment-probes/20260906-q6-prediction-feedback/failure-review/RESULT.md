# q6 notebook failure, 2026-09-06

Live Kaggle status for `pranavagrawai/q6-prediction-feedback-20260906` is ERROR. The worker aborted with `visible note event differs at final request`; launcher returned 71. The downstream missing stop records are consequences of that abort. The assessor correctly marks the run INVALID.

## Confirmed cause

The first attempted context summarization reached the shared request interceptor in `solver_agent.py:735`. It calls the same `IntegrityGuard.check_messages` for Actor and summarizer requests. That guard snapshots an Actor PythonOutput rendering and searches for that exact string in any user/tool message containing its tool ID (`integrity_guard.py:100–116`).

The summarizer receives history as a quoted `history_markdown` parameter inside a Task prompt. Its escaping preserves the tool ID but changes the literal representation of quotes and newlines. This makes the Actor-format substring check fail on intact history.

The saved failed request identifies TokenBudgetSummarizer. Safely parsing its history string with `ast.literal_eval` recovers the exact 1,731-character expected event inside 232,342 characters of history. That same event passed 41 prior render checks. Deployed solver and guard files match the frozen candidate byte for byte. This is a demonstrated representation false positive, not evidence of note corruption.

Reproduce without network access or raw mutation:

```sh
python3 analysis/deployment-probes/20260906-q6-prediction-feedback/failure-review/reproduce.py
```

`REPRODUCTION.json` retains assertions and deployed source hashes. The reproduction diagnoses the saved request; it does not qualify a repair.

## What survived

- 19 actions, 12 Actor batches, zero completed levels.
- 60 completed Actor requests; 1,256.3 seconds of recorded model time.
- All requests reconcile: 60 solver requests plus one warmup, 61 successful completions.
- Zero completed summarizer calls: the interceptor rejected the first attempt before it reached the model. Zero here does not mean compression was never attempted.
- Prediction feedback: six mismatches, two matches, three unavailable records; two revised helper versions were followed by a match. These are model consistency observations, not independent proof of semantic correction or improved gameplay.

## Repair requirement

Make note validation aware of the request boundary: validate the exact history passed into summarization and verify its framework serialization at the final request. Retain summarizer request recording and reasoning-token rejection. Do not merely exempt all summarizer traffic or accept arbitrary escaped substrings anywhere in a request.

The missing regression is a real summarizer invocation over history containing a complete printed note. Test intact history, altered note content, and subsequent Actor requests after compression, in addition to the existing accounting and reasoning checks. Existing fake-model coverage used history without a recorded note and therefore missed this interaction.

The launched candidate and raw evidence remain unchanged. No repair or replacement GPU run was performed during this failure review. A replacement must use a new identity after local qualification; the failed run does not establish longer-run solver performance.
