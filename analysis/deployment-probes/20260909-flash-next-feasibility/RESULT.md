# Flash-Next feasibility — result

> **Current verdict: operationally invalid.** The server log contains an OOM,
> which the frozen card invalidates without a recovery exception. The original
> validity and ambiguous-verdict claims below are superseded. Observations remain
> preserved; they do not license feasibility or speed conclusions.
> Reproducible check: [v2 audit](../20260909-experiment-checks/reviewed-final/flash-v2.json).
> The failed JSON response is fully accounted for: three requests were issued,
> five were unissued after the authorized exact-output stop. Assertions below
> that prompt ambiguity *caused* the failure are superseded by the
> [independent evidence review](../20260909-experiment-checks/reviews/flash-evidence/REVIEW.md).

## Original account (superseded where contradicted above)

Kernel `pranavagrawai/flash-next-feasibility-20260909` version 2,
`KernelWorkerStatus.ERROR` at 12:09:03Z after 562 s. **The run is valid under the
card.** None of the Invalid-if conditions occurred: model, runtime and GPU
identity are correct, request/response accounting is complete, there was no
timeout and no OOM, cleanup succeeded, and every issued request carried a
terminal stream and usage block.

## Verdict: ambiguous, not a compatibility failure

The card's Result map resolves to **Ambiguous → retain 27B and identify missing
evidence before a new run.** The prediction ("server ready within 1800 s and
completes all eight requests") is half confirmed and half falsified, and the
falsifying half is our prompt, not Keith's server.

## Serving feasibility is demonstrated, and survives the abort

Read this section independently of the failed request below. The infrastructure
result is established by the server reaching `phase: ready` and answering
requests at all; it does not depend on what request 3 returned, and it is not
weakened by the abort. Nothing here would change if the `json` case had been
worded correctly.


| Fact | Evidence |
| --- | --- |
| GPU is the intended one | `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`, 112 resource samples |
| Model identity | `Qwen/Qwen3.8-Flash-Next-NVFP4`, served from the pinned Kaggle model path, `max_model_len` 32768 |
| Runtime identity | `vllm-0.1.dev20073+g8e685d198-bf09729e` (SSE `system_fingerprint`) |
| Server became ready | `VLLM_SETUP_COMPLETE ready_seconds=450.07` against an 1800 s budget |
| PLE FP8 CPU-offload path worked | `PLE weight loading complete`, `Worker ready - 1 PleOffloadLayer(s)` |
| Native MTP is live | `spec_decode_num_drafts_total` 0 → 11, `num_draft_tokens_total` 0 → 32, `num_accepted_tokens_total` 0 → 32 |
| Cleanup was clean | `stopped: true`, `survivors: []`, `port_open: false`, identity matched |

Three of eight requests were issued. Two passed outright:

| Request | TTFT | Latency | Completion tokens | Result |
| --- | --- | --- | --- | --- |
| `text` | 0.631 s | 0.650 s | 2 | content exactly `4`, pass |
| `tool` | 0.820 s | 0.918 s | 26 | `submit_number{"value": 4}`, `finish_reason` `tool_calls`, schema-valid, pass |
| `json` | 0.457 s | 0.515 s | 11 | valid JSON `{"key": 4}`, **exact-match assert failed** |

MTP acceptance was 32/32 draft tokens, but over three trivial deterministic
completions totalling 39 tokens. That is proof the speculative path is
*engaged*, not a throughput measurement, and must not be quoted as an
acceptance rate.

## The abort was our prompt, not the server

`probe.py:98` asserts `json.loads(content) == {'value': 4}`. The model returned
`{"key": 4}`. The prompt was:

> Return JSON with the single key value equal to integer 4.

That sentence is genuinely ambiguous — "the single key value" reads as "the
single key, `value`" or as "the single key/value pair" — and the model resolved
it by naming the key `key`. It is a defect in our synthetic case, written by us
and never exercised against the real model.

**This is not a structured-output failure**, and must not be entered under the
card's clause "structured-output failure is evidence against compatibility".
`response_format: {"type": "json_object"}` did exactly its job: syntactically
valid JSON, `finish_reason` `stop`, `done` true, usage present, no reasoning
leakage. Only our value assertion failed. The card authorized this abort under
Stop ("failed synthetic exact-output check"), so the probe behaved correctly —
but the check does not measure what the card's Result map assumes it measures.

## The card's actual question is still unanswered

The card asks whether the profile serves "our retained Actor messages". The five
retained Actor cases (0, 1, 5, 12, 13) are requests 4-8 and were never issued.
No claim about Actor compatibility, tool offering under real Actor schemas,
long-context behaviour, prefix reuse, MTP-off performance, or ARC quality is
supported by this run.

## The reasoning-off risk did not fire

`STATUS.md` records a defect found while v2 was in flight: `probe.py:78` asserts
`not reasoning` on raw deltas, which rejects a valid response whose
`reasoning_content` is whitespace. On this stack it did not fire —
`reasoning_tokens` is 0 and `reasoning` is `""` on all three responses, so
`chat_template_kwargs: {enable_thinking: false}` genuinely suppresses the
channel here. The defect is real in code and now empirically dormant on this
configuration; it stays worth fixing because nothing guarantees that across
sampling or longer Actor prompts.

One correction to that note: it cited the donor's own preflight as bounding
evidence, but `vllm-server-preflight.json` shows `TAAF_KAGGLE_FAST_START=1`
skipped it (`skipped: true`, including "synthetic text" and "synthetic tool
call"). Our own `text` and `tool` cases were the first real exercise of this
stack, and they passed — stronger evidence than the donor path would have given.

## Next run

Two one-line fixes, both aligning code to the frozen card rather than amending
it:

1. `probe.py:98` — disambiguate the prompt to name the key explicitly, e.g.
   `Return a JSON object whose only key is named "value", with integer value 4.`
2. `probe.py:78` — `assert not reasoning.strip()`, since reasoning presence is
   not among the card's Stop conditions and its Measure clause treats reasoning
   as a legitimate TTFT channel.

Setup cost 450 s of a 2700 s session, so a corrected run has ample room for all
eight requests. Not launched: the card forbids automatic relaunch, and a second
concurrent RTX PRO 6000 session is Pranav's call.

## Full card audit of every abort in the probe

Applying "does the card state this rule?" to all fifteen abort sites, not just
line 78. The card's Stop list is: setup failure, timeout, HTTP/stream error,
empty output, malformed or unoffered tool, failed synthetic exact-output check.
Its Invalid-if list is: wrong model/runtime/GPU identity, missing
request/response accounting, timeout, OOM, failed cleanup, missing terminal
stream/usage.

| Site | Abort | Card authority |
| --- | --- | --- |
| `probe.py:24` | deadline `TimeoutError` | Stop: timeout |
| `probe.py:66` | stream `error` event | Stop: HTTP/stream error |
| `probe.py:82` | missing `done`/`usage`/`finish` | Invalid-if: missing terminal stream/usage |
| `probe.py:83` | empty `content` and no `calls` | Stop: empty output |
| **`probe.py:84`** | **any `reasoning` text** | **not stated — over-implemented** |
| `probe.py:87` | tool name not offered | Stop: unoffered tool |
| `probe.py:88-90` | arguments fail the offered schema | Stop: malformed tool |
| `probe.py:92` | `text` content is not `4` | Stop: failed synthetic exact-output check |
| `probe.py:94` | `tool` not exactly one `tool_calls` finish | Stop: failed synthetic exact-output check |
| `probe.py:96` | `tool` arguments are not `{"value": 4}` | Stop: failed synthetic exact-output check |
| `probe.py:98` | `json` content is not `{"value": 4}` | Stop: failed synthetic exact-output check |
| **`probe.py:100`** | **an Actor case answered without a tool call** | **not stated — over-implemented** |
| `probe.py:113` | donor `serving_setup.py` hash mismatch | Invalid-if: wrong runtime identity |
| `probe.py:141` | case count is not 8 | Cap: eight generation requests |
| `probe.py:160` | `cleanup['stopped']` false | Invalid-if: failed cleanup |

Thirteen of fifteen are card-stated. Two are not, and the second is worse than
the first.

### `probe.py:100` is the dangerous one

`assert calls and finish == 'tool_calls', 'Actor did not return an offered tool'`
fires on every `actor-*` case that answers in prose. It is not on the Stop list —
"malformed or unoffered tool" governs a tool that *was* returned, not the
absence of one.

The retained Actor requests settle it. All five send **`tool_choice: "auto"`**:

| Case | Tools | `tool_choice` | Messages | Prompt chars |
| --- | --- | --- | --- | --- |
| case-0 | 2 | `auto` | 9 | 33,713 |
| case-1 | 2 | `auto` | 12 | 34,683 |
| case-5 | 2 | `auto` | 23 | 44,500 |
| case-12 | 2 | `auto` | 44 | 46,771 |
| case-13 | 2 | `auto` | 46 | 46,848 |

Under `auto` the request explicitly permits a prose answer. The probe demands
`required` semantics while sending `auto`, so it would abort on behaviour its
own request authorised. The two synthetic tool checks at lines 94-96 are
different: that request sends `tool_choice: "required"`, so demanding a tool
call there is exact-output checking, and it passed.

The cost is not one lost case, it is the card's whole question. Cases run in
order, so a prose answer on case-0 forfeits cases 1, 5, 12 and 13 — the exact
evidence the card exists to gather — and writes `failure.json` where the Result
map would then read "compatibility failure". A prose answer is a *finding about
Actor compatibility* and belongs in `results.jsonl` as a measurement, exactly
where the run's tool-call accounting already goes. It is not a stop.

Every prompt also fits: 46,848 characters is roughly 12k tokens against
`max_model_len` 32768, leaving room for the 1024-token cap.

### Revised v3 change list

1. `probe.py:98` — disambiguate the `json` prompt (the gate is card-stated; the
   case was written ambiguously).
2. `probe.py:84` — `assert not reasoning.strip()`, or drop the assert and record
   reasoning as a measurement.
3. `probe.py:100` — record `calls`/`finish_reason` per Actor case instead of
   aborting, since `tool_choice` is `auto`. Keep lines 87-90: an *offered,
   well-formed* tool is card-stated and still worth enforcing.
