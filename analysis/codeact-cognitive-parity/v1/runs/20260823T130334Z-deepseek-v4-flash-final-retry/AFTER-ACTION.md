# After action: DeepSeek cognitive-parity screen

## Outcome

**Completed as a bounded failure screen; the CodeAct advancement rule was not met.** The global 131-call cap stopped the run after eight of twelve packets were fully attempted. Those eight yielded no matched report: native completed one easy-positive retrieval and exhausted seven sessions; CodeAct exhausted all eight.

This does not show that CodeAct is generally worse. It does show that **DeepSeek V4 Flash plus the current austere RLE evidence interface is not viable for the planned cognitive comparison**. Interface/tool reliability dominated reasoning quality.

## Decisive evidence

- The only scored answer was native on `remote-success-old-up`: evidence precision 1.0, recall 1.0, seven calls, five invalid returns.
- CodeAct did not call `return_result` once across 64 calls and eight invocations. All eight invocations exhausted their eight-call limit.
- CodeAct provenance recorded 30 rejected provider turns, 34 dispatched cells, six failed cells, seven evidence requests, six capability rejections, one successful evidence return, and zero accepted returns.
- Native also performed poorly: 26 of 67 turns had no tool call; only one of eight completed packets returned an answer.
- All 131 final-attempt calls reported the frozen Sail Research / `deepseek/deepseek-v4-flash-0731` route. No integrity violation, canonical mutation, or ARC action occurred.
- Final-attempt spend was 0.170032985 USD. Across all preserved attempts, 192 calls cost 0.2106296 USD, below the authorized 3 USD.

## What changed while earning the result

The failed attempts exposed controller faults rather than being discarded. The committed treatment now:

1. RLE-encodes model-facing grids exactly and reversibly to stay inside context limits.
2. Records malformed CodeAct turns as T1-owned rejected events and retries them.
3. Gives the native arm symmetric recovery for zero-tool turns, invalid evidence queries, and malformed return arguments.
4. Preserves a completed matched arm when the other exhausts its call budget, and continues the corpus.
5. Keeps route validation, cost reservation, timeouts, provenance, sterile execution, and the no-ARC-action boundary intact.

## Problems and resolutions

The preserved sequence was: invalid native schema use, context overflow, zero-tool CodeAct response, session timeout, zero-tool native response, CodeAct call exhaustion aborting the corpus, invalid native query arguments, malformed CodeAct JSON, and malformed native return JSON. Each was reproduced from its trace, repaired behind the same authority boundary, tested, verified, and committed before the next authorization. The final stop was not another harness defect: it was the precommitted global call cap.

## Decision

Do **not** advance CodeAct to a larger shadow or live ARC treatment on this route. Retrieval parity was not established, successful-case regression exceeded the frozen allowance, and the cost reduction is not meaningful because CodeAct produced no answer.

Do not add `wmlib` or `symmetry` yet. The treatment failed earlier: the model could not reliably consume evidence and complete the typed protocol.

## What comes next

Create a v2 real-model compatibility gate before another corpus run:

1. One easy-positive packet only; same facts and authority in both arms.
2. Require each arm to retrieve the required reference and return a valid typed answer within three calls.
3. Replace giant grid-bearing pages only if that packet proves representation is the blocker—for example, a matched metadata-only transition view or an explicitly documented pure decoder.
4. Advance to the twelve-packet corpus only if both arms pass. Otherwise kill this DeepSeek route for the comparison or choose a more tool-faithful model.

Another paid run requires a new explicit authorization; the frozen call budget is exhausted.
