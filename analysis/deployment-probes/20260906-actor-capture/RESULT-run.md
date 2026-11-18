# actor-capture-20260906 — the recorder worked; the Actor failed on undeclared tool names

`pranavagrawai/actor-capture-20260906`, pushed 08:47:01Z, queued ~2 h 53 m, RUNNING 11:20:14Z,
terminal **ERROR** by 11:43Z. Run `elapsed_s` 1218.4, well inside the 3600 s cap. Genuine
RTX PRO 6000, Internet off, ARC-AGI-3 attached.

**The instrumentation qualified. The gameplay did not.** These are separable and the distinction is
the whole result: the capture is valid evidence, the run is not a feasibility claim.

## The capture is valid against every gate in the card

| gate | result |
| --- | --- |
| body hashes unchanged before/after forwarding | all 14 request files match `recorded_before_forward` sha256 |
| paired responses intact | all 14 match `upstream_response_complete` sha256 |
| terminal response per attempt | 14/14, every one HTTP 200 |
| request provenance | 14 recorded = 14 `model_calls.jsonl` = 15 server accesses − 1 warmup |
| warmup counted separately | `warmup-facts.json` `chat_completion_accesses: 1`, bypasses recorder as designed |
| secrets excluded | no Authorization header saved; the only `api_key` hit is SGLang's own `null` server arg |
| lifecycle | `containment_passed true`, `root_returncode 0`, `term_only true`, teardown accepted |

`capture-summary.json` is `CAPTURED_REVIEW_REQUIRED`, 14 of a permitted 96 attempts. Reconciliation
balances exactly. Reasoning-off confirmed in every request (`chat_template_kwargs.enable_thinking:
false`), consistent with the q6 donor.

## Why the run died: three calls to tools that were never declared

The launcher failed with `TimeoutError: worker timed out waiting for decision` — the 900 s worker
wait — and `0` actions executed. That reads like a stall, but the captures show it was not. Every
request got a fast 200; the worker stopped because the NOOA codegen path gave up:

```
[headless] handle() raised GenerationError('Generation failed after 3 errors
(max_retries=3). Unable to generate valid code for `handle`.') (consecutive=1) — retrying in 2s
```

Every request declared exactly two tools, `execute_python` and `return_result`, with
`tool_choice: auto`. Three of the fourteen responses called something else:

| attempt | tool called | declared? |
| --- | --- | --- |
| 4 | `write_helper` | **no** |
| 12 | `submit_actions` | **no** |
| 13 | `submit_actions` | **no** |

The other eleven called `execute_python` correctly. The two consecutive `submit_actions` calls
exhausted `max_retries=3`, and the worker never produced a decision inside its 900 s window.

The arguments were also malformed in shape, not just in name — nested JSON passed as strings:

```json
{"actions": "[\"UP\"]", "predicted_path": "[{\"expect\": \"move\"}]", "rationale": "predict: move — ..."}
```

**This is not a capability failure.** The Actor's reasoning was sound. Its `step_log.md` note parses
the grid correctly, locates the player at (11,35) inside a hollow 8x8 box, identifies the bottom
opening, forms a testable ice-slide hypothesis with a discriminating UP-twice test, and commits to a
prediction. It knew what to do and could not express it through the declared interface. It invented
a plausible-sounding tool (`submit_actions`) for exactly the action it wanted to take.

## What this means for the corpus

**14 usable pairs, honestly labelled.** Clean monotonic growth — prompt 10,581 → 21,279 tokens,
messages 9 → 46 — all with `tool_result_present: true`. That is genuine early and tool-continuation
coverage, which is what the card predicted a short run would yield.

Per the card's result map, coverage gaps are reported, not filled:

- **No long-history coverage.** 21k tokens is nowhere near the ~102k compression threshold. The
  summarizer stayed unexercised here exactly as it did in the q6 quick-host run.
- **No forced-action coverage.** Every request exposed both tools under `tool_choice: auto`; none
  was forced. Per the README, exposing one tool would not have counted as forced either.
- **Three of fourteen pairs are invalid-tool responses.** They are worth keeping — they are the
  only direct evidence of this failure mode — but they must be labelled as such and excluded from
  any corpus meant to represent well-formed Actor traffic.

## The honest reading

The undeclared-tool failure is a real finding about the reasoning-off q6 workload, independent of
anything the capture changed. Only the launcher's model URL was routed through the recorder; the
solver ZIP, engine setup, action cap and controls were unchanged from the qualified q6 donor, and
the recorder forwarded byte-identical requests. Nothing in the capture path can produce a
hallucinated tool name.

What I cannot rule out from one run is whether this is reliably reproducible or a low-probability
sampling excursion that this run happened to hit twice in a row. The q6 quick-host run completed 10
actions over 53 calls without it, which is evidence that it is intermittent rather than systematic —
but that run was not instrumented to notice a recovered single occurrence, so its silence is weak.
Attempt 4's `write_helper` recovered; only the consecutive pair was fatal.

Per the card this is neither a recording failure (recording succeeded) nor a licence to relaunch to
fill categories. No automatic retry was performed.
