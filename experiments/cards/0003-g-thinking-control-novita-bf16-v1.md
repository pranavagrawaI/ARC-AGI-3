# 0003-g-thinking-control-novita-bf16-v1
Question: Does native thinking change or destabilize the minimal G interface?
Posture: bounded offline protocol control
Contrast: thinking off → on → off on three frozen G cases after repeated G baseline.
Prediction: all decisions pass; first and restored off signatures match per case.
Measure: `thinking-summary.json` and 15 rows in `results.json`.
Cap: 15 model requests; 45 HTTP attempts; $0.02; 20 min; 0 environment actions.
Stop: pass after sequence; fail on invalid decision or off instability; abort on drift or cap.
Invalid if: changes beyond reasoning controls, drift, parse failure, or incomplete sequence.
Result map: stable → keep cheaper condition and test layout; unstable → fix protocol; invalid → retry with new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-g-thinking-control-novita-bf16-v1-20260807T201457Z/`
