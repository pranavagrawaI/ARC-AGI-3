# Matched continuation, reasoning on
Question: Does reasoning raise transition accuracy versus the pinned reasoning-off run?
Posture: matched shadow.
Contrast: Same call-491 frontier, CoreWeave endpoint, 8/16-retained and 0/16-canonical arms, task, prompt, tools, order, and caps; reasoning request is on. Control: `20260822-matched-coreweave/result.json`.
Prediction: Retained finishes above 8/16; secondarily canonical finishes above 0/16.
Measure: Before/final matched/evaluated, score trajectory, reasoning tokens and nonempty reasoning/call.
Cap: 12 tool steps, 14 calls/arm, 240 s/arm, 28 calls, $0.42 with $0.015 reserved/call, 4096 tokens/call, 0 ARC.
Stop: Report, timeout, cap, missing cost, provider error, reserve failure, or non-CoreWeave route.
Invalid if: Any condition besides reasoning or stochastic response differs from the pinned control.
Result map: Higher -> performance prediction supported locally; equal/lower -> reject performance claim; no reasoning -> endpoint unsuitable for Builder diagnosis.
