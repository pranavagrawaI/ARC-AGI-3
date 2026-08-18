# tycho-nooa-tb-first-live-20260818T103742Z
Question: One LS20 episode: trigger fire? builder change artifact? actor show uptake?
Posture: verification.
Contrast: none.
Prediction: trigger fires on same-level transition; if fired, builder changes >=1 artifact, report fresh/forced; uptake unpredicted.
Measure: trigger-builder-trace.jsonl fired/cause, artifact_hashes.*.changed, tested_report/followed_recommended_action (in order); builder-tool-trace.jsonl calls.
Cap: ls20 seed0; 20 actions; stop-after-levels 1; one live model; no retries.
Stop: action cap, level solved, terminal, or transport/protocol failure.
Invalid: turn_in_level never >0; crash pre-first-action; missing/malformed trace.
Result: fired+changed+engaged -> works, plan bigger run; never fires -> diagnose trigger; fires/no change -> diagnose builder; constructs/no engagement -> diagnose exposure; invalid -> rerun bigger budget.
Outputs: runs/tycho-nooa-tb-first-live-20260818T103742Z
