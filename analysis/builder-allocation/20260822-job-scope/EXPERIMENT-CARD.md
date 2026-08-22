# 20260822-job-scope
Question: Did mixed Builder jobs cause the selected turn-18 timeout?
Posture: retrospective discriminating replay
Contrast: mixed-job cause vs within-mechanics search
Prediction: mixed-job cause requires at least 2 jobs and no job above 80% of calls; otherwise job allocation is not the next repair.
Measure: classify all 28 preserved Builder calls from arguments, report job counts, and extract verifier score by call cap from `game/builder-tool-trace.jsonl`.
Cap: 5 seconds; 1 preserved pass; 0 model calls; 0 ARC calls; $0.
Stop: finish after all 28 calls and the 8/16 score are bound; abort on source drift.
Invalid if: deadline/tool counts disagree, call 27 is not 8/16, or any call cannot be classified without tool-result boilerplate.
Result map: mixed -> implement typed job allocation; single-job -> do not implement it and test continuation from retained candidates; ambiguous -> label calls manually before action.
Outputs: `analysis/builder-allocation/20260822-job-scope/result.json`
