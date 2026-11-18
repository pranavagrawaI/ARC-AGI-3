# Exact-difference feasibility follow-up
Question: With exact pixel differences supplied, can the model select and interpret a discriminating experiment?
Posture: Diagnose reasoning after removing observed raw-string coordinate errors.
Contrast: none formally; exploratory follow-up to two raw-frame starts that invented changes. Not a randomized efficacy claim.
Prediction: At least one of two fresh starts grounds both explanations in actual changes, commits distinct checkable predictions, and interprets the selected action correctly without asserting a unique cause.
Measure: Original committed predictions, raw before/after frames and independently checked pixel values. Separately assess grounding, prediction separation, measurement and conclusion. Exact pixel deltas carry no supplied hypothesis or verdict.
Cap: Two starts, four API calls maximum, one selected action/start; 4096 output tokens, 80000 request bytes, 120s/request, 600s wall. Reserve $0.25 for all earlier attempts and at most $0.25 for this follow-up, keeping the task cap $0.50. No retries.
Stop: Both reviews complete or malformed plan, illegal action, replay mismatch, network failure or cap.
Invalid if: Wrong replay frames, leaked future state, executed action mismatch or missing raw evidence. Harness failures are not reasoning failures.
Result map: Correct grounded selection and review -> retain exact-difference evidence as candidate assistance, defer special predicates; grounded but weak predictions -> target selection next; sound predictions but faulty review -> target measurement; invalid -> report blocked evidence.
Outputs: INPUT.json, requests.jsonl, responses.jsonl, committed plans, raw transitions, reviews, source pins, costs and RESULT.md. Prompts unchanged from raw-frame run; only exact pixel-change records and explicit shape added to data. Queued Kaggle package untouched.
