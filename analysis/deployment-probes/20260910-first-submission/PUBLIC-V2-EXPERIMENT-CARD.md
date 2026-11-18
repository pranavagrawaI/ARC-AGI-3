# First submission public qualification v2
Question: Can the corrected all-game-budget notebook execute the live frozen 27B Solver on three public games and clean up within the quota-preserving run cap?
Posture: Integration qualification.
Contrast: none; not a controller or performance comparison.
Prediction: LS20, S5I5, CD82 each execute at least one Solver action and stop at the 10-action, 75-second admission, or win boundary; exact request and cleanup checks pass.
Measure: qualification/result.json, official launcher result, game-progress.jsonl, worker IPC including model calls and observations, server stdout/stderr and lifecycle files, notebook execution log, submission.parquet presence only after qualification. Inspect raw logs for OOM even if aggregate status says passed.
Cap: One private Kaggle NvidiaRtxPro6000 notebook session, 3600s platform cap (1 GPU-hour maximum); 3 public games, at most 10 Solver actions/game, 75s admission/game plus 120s last worker decision and 30s cleanup reservation; 90s client request timeout, retries 0, output tokens 16384/request. No paid external inference. Platform cap supersedes larger defensive subprocess timeouts in the packaged driver. Zero competition submissions under this card.
Stop: Normal completion of all games and cleanup; abort on error, timeout, OOM, failed identity/accounting or uncontained process. No automatic relaunch.
Invalid if: Wrong model/GPU/package, OOM, dropped game, lost request/action accounting, worker/server timeout, missing terminal records or failed cleanup. Score and win count are not pass conditions.
Result map: Pass allows a separate card and one competition submission of this exact successful version; failure requires diagnosis and a new card before another run; ambiguous evidence withholds submission. Do not launch until terminal public-v1 outputs have been archived without overwrite.
Outputs: analysis/deployment-probes/20260910-first-submission/kaggle-public-v2/ (fresh directory).
Candidate: deployment/kaggle/submission-20260910-v2/submission.ipynb SHA256 15c6ab29cc97f74a317f01060c0e0d3bc68e210426d391aef5cdce0743c3a056; frozen Solver contract 2c1e13ef4ce25e59121bcd6c637a869a8cb09c49fc58712fc9212234b8c262c9. Local entrypoint-v4 passed 9 tests; these were controlled responses, not live model evidence.
