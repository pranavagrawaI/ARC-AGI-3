# 0003-ladder-typed-record-novita-bf16-v1
Question: What is the minimal repeated-success arm after the typed-record repair?
Posture: matched offline ladder
Contrast: sequential A–H evidence packages
Prediction: G is the first 6/6 stable arm; A–F fail their frozen gates.
Measure: output `results.json` per-row scores/signatures and `ladder-summary.json` minimal arm.
Cap: 10 minutes; 48 logical requests; 144 HTTP attempts; $0.05; 0 environment actions.
Stop: stop at first stable arm or after H; abort on drift, provider error, invalid transport, or cap breach.
Invalid if: endpoint audit fails, requests/producer change after start, or an arm is incompletely scored.
Result map: G pass/F fail → keep planning in Machinery; H pass/G fail → narrow verified plan choices; no pass → stop on interface; invalid → no claim/new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/ladder-typed-record-novita-bf16-20260807T165557Z/`
