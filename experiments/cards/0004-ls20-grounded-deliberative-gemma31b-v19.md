# 0004 LS20 grounded deliberative Gemma 31B v19
Question: Does public grounding make the deliberative planner reach the trigger and solve?
Posture: offline grounded-planning ceiling
Contrast: v18; add controlled cell, displacement/block, resource change, and neutral overlay only.
Prediction: planner stops advancing imagined coordinates; at least 1/3 visits the trigger.
Measure: replies, grounding, completion; scorer-only trigger, target, reset, block counts.
Cap: 60 paid calls; 129 offline actions; 0 live; 30 min; $0.50.
Stop: each episode at completion, terminal, or 20 actions; run at 3 episodes or first error; no call after cap.
Invalid if: private/false-certain grounding, drift, wrong route, fallback, missing marker, provider error, or cap breach.
Result map: 3/3 -> remove grounding fields; 1-2 -> replicate; 0 with route repair -> add map; 0 without repair -> inspect grounding use; invalid -> repair only.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
