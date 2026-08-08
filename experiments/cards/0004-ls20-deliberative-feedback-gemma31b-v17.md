# 0004 LS20 deliberative feedback Gemma 31B v17
Question: Can deliberation let Gemma 4 31B solve the checkpoint?
Posture: offline full-capability ceiling
Contrast: v16 had no call; audit only parameters actually sent, not absent tools.
Prediction: at least one episode visits the trigger; hidden reset semantics may still block completion.
Measure: raw reply/reasoning, actions, completion; scorer-only trigger, target, reset, block counts.
Cap: 60 paid calls; 129 offline actions; 0 live; 30 min; $0.50.
Stop: each episode at completion, terminal, or 20 actions; run at 3 episodes or first error; no call after reported cost reaches cap.
Invalid if: leakage, drift, wrong route, fallback, missing marker, truncation, provider error, or cost breach.
Result map: 3/3 -> remove reply memory; 1-2 -> replicate; 0 with trigger gain -> expose budget semantics; 0 without -> neutral overlay ceiling; invalid -> preserve and classify.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
