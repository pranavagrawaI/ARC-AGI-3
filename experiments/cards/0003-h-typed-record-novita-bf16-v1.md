# 0003-h-typed-record-novita-bf16-v1
Question: Can Novita BF16 repeat H decisions with closed typed PROBE records?
Posture: interface feasibility
Contrast: none
Prediction: all 6 H decisions pass; both PROBE calls contain integer `levels_completed` records and decode to the frozen canonical outcomes.
Measure: output `results.json` transport decisions/scores and `ladder-summary.json` stability.
Cap: 10 minutes; 6 logical requests; 18 HTTP attempts; $0.02; 0 environment actions.
Stop: finish after 6 requests; abort on provider/model drift, provider error, invalid transport, or cap breach.
Invalid if: endpoint audit fails, producer/request changes after start, or fewer than 6 decisions are scored.
Result map: 6/6 stable → return to A–H ladder; valid failure → stop and diagnose interface; invalid → no claim and use a new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/h-typed-record-novita-bf16-20260807T165222Z/`
