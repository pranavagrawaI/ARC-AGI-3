# Epistemic reasoning critic v5

Question: Can the critic identify poor use of correct observations while leaving
provisional, justified reasoning alone?
Posture: Controlled diagnostic of epistemic reasoning, no factual-verification task.
Contrast: Four concern/control pairs share identical stipulated observations.
Pairs cover assumed objective plus misplaced parameter uncertainty; confirmation
that fails to discriminate; policy insulated from revision; and overgeneralization.
Reasoning is authored for the test, grounded in saved S5I5 step3 and CD82 step1
observations. These are not eight original Actor outputs or held-out natural traces.

Prediction: Four specific concerns with correct inferential explanations and minimal
revisions, four no-concern responses allowing bounded exploration. Expected results
and pair rubrics in CASES.json are withheld from requests. Facts are stipulated true;
critic is not asked to count, verify coordinates or solve the game.

Measure: Manual inference-gap validity, consequence, minimal-revision quality and
false-objection count. Status alone is insufficient. Check quotations, brevity and
whether the critic invents facts or demands exhaustive evidence. Eight raw responses,
request timings, provider identity, tokens and cost preserve the evidence.

Cap: Eight independent calls, 1024 output tokens each, temperature0, reasoning off,
45-second HTTP timeout each, zero retries, $0.25 spending cap. Conservative reserve
$0.04687086. Same qwen/qwen3.8-27b via coreweave/fp8 only; no fallback, gameplay or Kaggle.

Stop: Complete eight requests or stop on transport/provider/reasoning/cost/unfinished
generation failure. Format deviations (including JSON code fences) are recorded
and decoded without semantic repair for offline grading; no extra model calls.

Invalid if: Evidence differs within a pair; desired labels leak into prompts;
future trace outcomes are supplied; model route or input changes mid-run; replies
missing. Authored contrasts and explicit fault examples limit claims of transfer.

Result map: All four faulty-reasoning cases identified with sound explanations and
all controls preserved -> supports capability on this screen; test unedited saved
reasoning next. Any missed error -> identify the specific epistemic gap. Any false
objection -> no corrective authority. Do not infer Actor uptake or gameplay efficacy.

Outputs: This directory, including CASES.json, PREPARED.json, eight requests/replies,
STARTED.json, EXECUTION.json, timings, manual grades and RESULT.md. Seventeen local
harness checks passed before inference; only one new check qualifies pair construction.
Authorization: User requested this revised test using the existing endpoint.
