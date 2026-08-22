# 20260822-matched-budget-awareness
Question: Does fresh time/call telemetry make the reasoning-on Builder commit earlier and improve transition accuracy?
Posture: matched shadow treatment.
Contrast: Prior reasoning-on CoreWeave run vs same two turn-18 arms, model, task, workspace, tools, order, 12 steps, 4096 tokens,; 240 s; only treatment is the budget notice.
Prediction: At least one arm edits before timeout and improves: retained >8/16 or canonical >0/16.
Measure: Final matched/evaluated, first edit call/time, score trajectory, notices, and reasoning.
Cap: 14 calls/arm, 28 total, 240 s/arm, $0.42 with $0.015 reserve/call, 0 ARC.
Stop: Timeout/report, cap, provider error/route drift, missing cost, or reserve failure.
Invalid if: Source/model/provider/prompt/tools/caps drift beyond notice; missing notices; or no reasoning.
Result map: Improved -> retain and replicate; unchanged/lower -> reject; invalid -> repair evidence only.
Outputs: `analysis/builder-continuation/20260822-matched-budget-awareness/`.
