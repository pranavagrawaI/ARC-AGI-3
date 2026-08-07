# 0003-g-context-layout-novita-bf16-v1
Question: Does candidate-local evidence layout improve or destabilize G under reasoning-off?
Posture: bounded offline layout control
Contrast: separated → candidate-local → separated; semantic artifacts and runtime stay fixed.
Prediction: all decisions pass and separated bookends match; retain the cheaper successful layout.
Measure: `layout-summary.json` and 15 rows in `results.json`.
Cap: 15 model requests; 45 HTTP attempts; $0.02; 20 min; 0 environment actions.
Stop: pass after sequence; fail on invalid decision/bookend drift; abort on provider/model drift or cap.
Invalid if: semantic artifacts differ, drift, parse failure, or incomplete sequence.
Result map: local passes → compare tokens then choose; local fails → retain separated; invalid → retry with new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-g-layout-v1-20260807T202123Z/`
