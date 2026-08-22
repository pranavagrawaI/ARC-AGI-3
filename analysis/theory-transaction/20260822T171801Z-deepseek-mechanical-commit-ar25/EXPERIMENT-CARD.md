# DeepSeek mechanical-commit gate
Question: Did an advisory commit notice let investigation consume the pass despite DeepSeek tool use?
Posture: matched zero-ARC mechanism test at commit 4ee0158.
Contrast: control vs transaction; same 10/18 frontier, DeepSeek V4 Flash 0731, Sail FP4, reasoning, 4096 tokens, 12 steps, 900 s.
Mechanism: truncation continues construction; commit removes investigation tools and requires mutation/checkout.
Prediction: both arms leave investigation; transaction assesses an edit and preserves obligations by acceptance or restore.
Measure: offered/chosen tools, finish reasons, typed deltas, bytes, ledger, route, latency, cost.
Cap: 14 calls/arm, 28 total, $0.20; reserve $0.01/call; 0 ARC.
Stop: report/timeout, cap, missing cost, provider/tool error, or route drift.
Invalid if: commitments drift, no treatment edit is assessed, route/cost unavailable, or boundary exceeded.
Result map: pass -> short ARC prefix; otherwise no ARC.
Outputs: this directory.
