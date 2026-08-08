# 0004 LS20 feedback planning Gemma 31B v12
Question: Can Gemma 4 31B complete the prefix-24 checkpoint E4B failed?
Posture: offline closed-loop model-size ceiling
Contrast: v11; only model/provider changes: E4B to Gemma 4 31B Novita bf16.
Prediction: 31B visits the supplied trigger more, but hidden reset semantics prevent 3/3.
Measure: `trace.jsonl` actions/outcomes; `episodes.json` completion; scorer-only trigger, target, reset, block counts.
Cap: 60 paid calls; 129 offline actions; 0 live; 20 min; $0.15.
Stop: each episode at completion, terminal, or 20 actions; run at 3 episodes or first error; no call after reported cost reaches cap.
Invalid if: leakage, drift, wrong model/provider/precision, fallback, missing/non-digit response, or cost breach.
Result map: 3/3 -> overlay ablation; 1-2 -> replicate; 0 with trigger gain -> expose budget semantics; 0 without -> overlay ceiling; invalid -> harness repair.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
