# 32k reasoning-budget live probe

Status: **INCOMPLETE, lifecycle clean; action gate incomplete**.

The current candidate contract was held fixed:
`6db629026e9681f27cd91a56463a8f16bf2ef1dbf92d49b1d852a9ec8d07f3a6`.
Reasoning stayed enabled. The only treatment change from the preceding 16k
probe was total completion budget: **16,384 → 32,768 tokens**.

## Result

- 17 provider responses completed; 16 model calls were recorded by the worker.
  The extra completed response was in flight when the bounded arm closed.
- Reasoning was present on every response: **33,806 reasoning tokens** across
  **59,462 completion tokens**.
- Maximum per-response reasoning usage was **6,627 tokens**. No response
  exhausted the 32k budget and no `GenerationError` occurred.
- One action was reconciled; zero levels completed. The arm then reached its
  900-second cap. The mindset arm was not started under the stop rule.
- Lifecycle evidence is clean: 17 admissions, no post-close/deadline
  admissions, clean relay cleanup, and local worker retirement.
- Reported provider cost was **$0.2791744**; conservative reservation was
  **$2.36171348**, below the $3 cap.

## Comparison with the 16k probe

The 16k run failed after 16 responses when one response spent the entire
available completion budget on reasoning and emitted no tool call
(`GenerationError`). Its maximum observed reasoning count was 11,502.

At 32k, the model continued to emit tool calls after that range and never
hit the output-budget failure. This supports increasing the budget for a
reasoning-enabled trial, but it does **not** show that 32k is enough for a
complete decision: the run still reached its wall-clock cap after only one
action. The remaining bottleneck is now decision time/trajectory progress,
not immediate reasoning-token exhaustion.

This probe does not establish solver quality, mindset-arm behavior, Kaggle
qualification, or GPU performance. It also does not justify changing the
competition notebook yet; the next useful inspection is the retained reasoning
and tool trace to determine whether the extra budget is productive or merely
slow deliberation.

## Evidence

Raw evidence is retained here: `STARTED.json`, `PREFLIGHT.json`,
`requests.jsonl`, `responses.jsonl`, `reasoning.jsonl`, `lifecycle.jsonl`,
`RESULTS.json`, `QUALIFICATION.json`, `ASSESS-OUTPUT.txt`, and the worker
workspace. No Kaggle upload or launch was performed; no automatic retry was
performed.
