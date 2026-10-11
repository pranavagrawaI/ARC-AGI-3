# prune-retain-probe-20261011
Question: Can the retained REAP448 mask load faithfully on the exact current Intel/SGLang stack and serve requests at 14 and 18 concurrency?
Posture: transfer qualification, before comparative efficacy.
Contrast: none. Target 448 experts, unchanged 512-expert INT4-g32 drafter, original atomic reduction. Current c14 context/sampling preserved; capacity raised to 18, Mamba slots 108, graphs include 14/16/18.
Prediction: all 48 target routers and expert banks load as 448; draft remains 512; both request groups complete. Target expert storage drops by approximately one eighth; no speed or gameplay claim.
Measure: pruning/install.json source hashes, pruning/target.json counts and live shapes, pruning/draft.json, server-info-start.json effective capacity, SMOKE.json and all 32 raw request/response pairs. Server log exposes model memory, KV allocation, retractions and errors.
Cap: one RTX PRO 6000, one launch, platform hard timeout 3600 s; request worker timeout 600 s and each HTTP call 180 s. Whole investigation maximum 21600 GPU-seconds (6 h), including all failed attempts. No submission, checkpoint upload, or model training.
Stop: end after both batches and process-group teardown. Any source, config, tensor inventory, router hash, live shape, startup, HTTP or finite-output failure aborts; preserve the error and do not retry.
Invalid if: missing raw files, wrong GPU, remote notebook mismatch, missing full target/draft receipts, capacity below 18, nonfinite/empty/failed responses, or non-COMPLETE kernel status.
Result map: valid qualification -> launch the frozen control/candidate gameplay pair once, only if two slots and full remaining quota are available. Invalid/negative -> retain c14, no gameplay launches, diagnose failure. Slot/quota conflict -> preserve prepared pair without launching. No repair or retry is authorized by this card.
Outputs: analysis/pruning-transfer-20261011/probe/{launch-attempt.json,launch-response.json,status-history.jsonl,output/,collection.json}; in-kernel /kaggle/working/pruning/.
Authority: Pranav, 2026-10-11: "Then let's see what we can do with this, let's do a few tests with retention." Agent-set conservative total cap 6 GPU-hours; retained c14 remains fallback.
