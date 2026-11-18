# Verdict — 20260905T-nooa-join-public3-reasoning

Kernel `pranavagrawai/prototype-nooa-join-public3-reasoning` version 1. Status ERROR. Elapsed ~32 s. No play. No retry of this card.

## Invalid

`--r8-l3b requires --startup-timeout-s 1200 --request-timeout-s 180`. The notebook passed `--request-timeout-s 300` so thinking completions could run longer. The frozen R8-L3B launcher rejects that at argparse. SGLang never started. No `result.json`, no recordings, no IPC.

GAME_OVER resync and thinking-on were not exercised.

Result map: **invalid** — do not interpret. Successor is a new card with `--request-timeout-s 180` and the same thinking-on extra_body.

Outputs: `analysis/deployment-probes/20260905T-nooa-join-public3-reasoning/kaggle-output/prototype-nooa-join-public3-reasoning.log`.
