# 20260822-matched-builder-continuation
Question: Does a mechanics Builder continue better from retained 8/16 than canonical 0/16?
Posture: workbench
Contrast: same turn-18 workspace; only starting `world_model.py` differs.
Prediction: retained ends higher or reaches a shared score in fewer calls.
Measure: exact matched/evaluated before, after edits, and final; final accuracy and first-call-to-score.
Cap: seed 20260822 order retained then canonical; 12 tool steps, 14 provider calls and 240s/arm; 28 calls total; $0.42 total, reserving $0.015 before each call; 4096 output tokens/call; 0 ARC calls.
Stop: report, timeout, call cap, missing cost, provider error, or insufficient reserve.
Invalid if: hashes, frontier, provider context/route, task, tools, caps, or verifier differ.
Result map: retained better/faster -> add continuation selection; canonical better -> investigate anchoring; equal -> advantage inconclusive; invalid -> repair under a new card.
Outputs: `analysis/builder-continuation/20260822-matched/`
