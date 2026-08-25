# production-actor-paid-shadow-v1
Question: Can the production Native Actor complete a bounded off-policy shadow without ARC actions or resource/integrity failure, and does that feasibility persist over a longer prefix?
Posture: feasibility then conditional stress
Contrast: none
Prediction: the six-turn prefix completes within 30 calls; if so, the twenty-turn prefix also completes while provider context remains bounded.
Measure: phase assessment plus durable provider ledger; historical action agreement is descriptive, not correctness.
Cap: short 6 turns/30 calls/$0.50; conditional long 20 turns/30 calls/$0.50; 65,536 conservatively measured input units and 4,096 output tokens per call; zero ARC actions.
Stop: stop on any route/provider/token/cost/integrity failure; run long only when short assessment has passed=true and advance=true.
Invalid if: committed transcript/controller/launcher/runtime drift; sandbox runtime mismatch or missing bubblewrap; route drift; missing usage; any ARC action; unaccounted provider call.
Result map: short pass -> run long; short fail -> stop and diagnose; long pass -> proceed to real decision-quality shadow design; long fail -> diagnose the first recurring production seam.
Outputs: analysis/production-actor-paid-shadow-v1/runs/<authorization_id>/
