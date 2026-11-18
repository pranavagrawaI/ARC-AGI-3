# Model-selected discriminating experiment
Question: Can the model choose and interpret an informative action from a real UP transition without being supplied a particular predicate?
Posture: Bounded capability diagnosis.
Contrast: none; two fresh independent conversations on identical input, no causal comparison with earlier CodeAct runs.
Prediction: At least one conversation commits two observation-grounded explanations with distinct checkable outcomes for one action, then correctly evaluates those original predictions.
Measure: Manually audit COMMITTED.json against INPUT.json before interpreting the review; independently compute the committed relation from TRANSITION.json. Score grounding, outcome separation, measurement correctness and calibrated conclusion separately. A plausible story or correct binary verdict alone does not pass.
Cap: Two starts, two API calls per start, one model-selected environment action per start after replaying UP; 4096 output tokens/call, 80000 request bytes/call, 120 seconds HTTP/call, 600 seconds total wall time, $0.50 conservative total reservation with provider price ceiling. No retry.
Stop: Stop after both reviews. Abort on malformed plan, illegal action, input replay mismatch, HTTP failure, or any cap. Preserve partial outputs.
Invalid if: Supplied frames differ from replay; future state leaks into planning; selected action differs from committed action; missing response or raw transition. Protocol failure remains a diagnostic result, not evidence against causal reasoning.
Result map: Grounded discriminating test plus correct interpretation -> retain this prompting candidate for later actor integration; good test but faulty measurement -> consider a narrow measurement helper; weak explanations or overlapping outcomes -> work on test selection, do not add a restoration predicate; invalid -> fix the demonstrated harness failure in a new run.
Outputs: INPUT.json, requests.jsonl, responses.jsonl, a/b-COMMITTED.json, a/b-TRANSITION.json, plan/review contents, SOURCE-PIN.json, COST.json and RESULT.md in this directory. No model-generated code execution and no queued-package changes.
