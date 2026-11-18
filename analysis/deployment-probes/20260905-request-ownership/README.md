# Request ownership diagnosis

Worker follow-up: [retained answer and redundant state wakeup](WORKER-RESULT.md).
The component results below remain the earlier evidence, not the worker verdict.

The cleanup/abort ordering defect is reproduced in unchanged methods from the
downloaded, manifest-verified SGLang wheel. It is **not established as the cause
of the four-hour Kaggle run**. The real-client loopback probe establishes a
separate cancellation survivor in NOOA and locates the SDK retry multiplier.
No serving code, wheel, model manifest, sampling configuration, or reasoning
allowance was changed. No Kaggle job or model inference was launched.

## Evidence and identities

- `reproduction.json`: CPU execution of wheel methods and counts from the saved
  reasoning-v2 server log. The scheduler, tokenization and IPC are fakes.
- `client-results.json`: real NOOA CompletionClient → LiteLLM → OpenAI SDK → HTTP
  requests to a delayed loopback endpoint, including monotonic event timestamps.
  Outbound connections to non-loopback addresses are rejected by an audit hook.
- SGLang wheel SHA256:
  `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0`.
  Matches `deployment/kaggle/runtime-patches/r8l3b/manifest.json`.
- Tokenizer-manager entry SHA256:
  `90bf218aec5cf24be7c822821bca6527a3bb1dc834824eae81a06ea862c30cdf`.
  The probe asserts the manifest source projection matches before executing it.
- NOOA wheel SHA256:
  `f2217a8d95d2239ca830a0d47ec7973123f618253a6c219fa64e1e43f278cf0c`.
  Matches wheelhouse v11, version `0.0.1.dev95`. Loaded directly from the verified
  cached wheel, rather than assuming the editable checkout is the deployed code.
- HTTP probe core versions: LiteLLM `1.97.0`, OpenAI `2.54.0`, HTTPX `0.28.1`.
  LiteLLM comes from its cached 1.97.0 installation because the local deployment
  venv currently has 1.98.0. This is not a full locked-environment rebuild; these
  core versions are asserted, but the entire transitive environment is not bound.

## SGLang controls

`reproduce.py` extracts five complete methods with AST, without editing their
bodies: `generate_request`, `_wait_one_response`, `_discard_pending_req_states`,
`abort_request`, and `create_abort_task`. Dependency initialization is stubbed.
The polling timeout is shortened to 50 ms; the real delayed-abort sleep remains
two seconds. Admission is witnessed before injecting cancellation.

| Case | Observed abort dispatches | Ordering |
|---|---:|---|
| Cancel while awaiting response | 0 | State removed; delayed abort returns |
| Close streaming generator after a yielded result | 0 | State removed; delayed abort returns |
| Non-streaming disconnect while waiting | 1 | Abort dispatched while state exists, then cleanup |
| Non-streaming disconnect after progress | 1 | Abort dispatched while state exists, then cleanup |
| Normal completion | 0 | Normal response-state removal is simulated |
| Input failure before dispatch | 0 | State removed; nothing admitted |

All six cases pass assertions describing the current behavior, including the
defective ordering. This proves loss of abort dispatch for the injected ordering;
it does not run ASGI response cancellation, the scheduler, or GPU generation.
It does not demonstrate that a particular deployed HTTP request encounters it.

The wheel's OpenAI chat non-streaming path calls `generate_request().__anext__()`
in `serving_chat.py:1817`. Its streaming response attaches `create_abort_task`
at line 1540. Both paths therefore require their own real-server qualification.
The non-streaming controls prevent treating a streaming reproduction as proof
about the Actor. Direct cancellation can also reach exceptional cleanup; a
disconnect detected by the response-wait loop has different ordering.

The saved reasoning-v2 log contains **39 deleted-state messages across 39 request
IDs, exactly one message per ID**. This lacks the sustained per-ID signature
described in [SGLang PR #35936](https://github.com/sgl-project/sglang/pull/35936).
Sparse output does not prove retirement either, particularly for non-streaming
requests. Attribution still requires request-specific scheduler evidence.

## Client controls

All observed loopback HTTP requests were non-streaming. These are artificial
messages through the actual client, not captured Actor requests.

| Case | Actual HTTP attempts |
|---|---:|
| Immediate normal response | 1 |
| Delayed response; all automatic retries disabled | 1 |
| Delayed response; SDK defaults, NOOA retries disabled | 3 |
| Delayed response; one NOOA retry allowed, SDK retries disabled | 1 |
| One explicit retry after fake work is known to have completed | 2 |
| Caller cancelled after HTTP admission | 1; provider later finishes and discards result |

The default-SDK case records `x-stainless-retry-count` values 0, 1, 2 at the
endpoint. This establishes that layer's multiplier for the tested timeout.

The NOOA retry-eligibility case initially falsified the probe's expectation of
two attempts. LiteLLM Timeout carries structured status **408**. The pinned
`nooa/unifiedllm/retry.py:155` classifier treats a present status not in its retry
set as terminal before considering timeout names. Raising NOOA's retry count
alone therefore does not retry this error. The probe retains that observation;
no production classifier was changed.

The phrase `LLM API error after 3 retries` is constructed by **CodeAct**
(`nooa/strategies/codeact.py:891`), above the CompletionClient and SDK. The join's
`headless.py:60` catches that failure and loops again after backoff. Therefore
the saved 541.34-second inner exception must not be interpreted as the total
duration or exact physical-attempt count of the whole decision. Three 180-second
SDK attempts explain its scale, but mapping the complete run needs identities.

For cancellation, the wheel's `_litellm_acompletion` uses `asyncio.shield` and
registers a callback to consume a late result. The actual-client probe observes
caller cancellation, subsequent fake work completion, and the provider's
successful result being discarded. Its observer wraps the existing discard
callback and preserves that callback's behavior. This is a real local instance
of the provider task outliving the caller, not evidence that a Kaggle worker
actually cancelled at this boundary. A successful socket write is explicitly
recorded as delivery-unknown, not client receipt.

## Run and limitations

From the repository root:

```bash
python3 analysis/deployment-probes/20260905-request-ownership/reproduce.py
deployment/kaggle/.venv/bin/python analysis/deployment-probes/20260905-request-ownership/client_probe.py
```

Both scripts assert observations and emit JSON. Paths to the cached wheels and
LiteLLM installation are explicit in the scripts. Both pass Ruff checks; the
control-flow and loopback probes completed successfully. They do not modify
production behavior or count fake response tokens as model-generated tokens.

Outstanding qualification:

1. Exercise the actual outer worker/CodeAct path for repeated WAIT, supersession,
   stale action authority and lost response. The current loopback probe stops at
   CompletionClient; it does not qualify the worker or decision deadline.
2. On the target server, correlate each transport attempt with a scheduler RID
   before inducing disconnect/timeout. Observe terminal status before releasing
   admission. Bound cleanup time and stop the trial if retirement is unknown.
3. Keep isolated FP8 and shallow-MTP comparisons independent. Re-run lifecycle
   cases on any candidate before making integrated performance claims.

The local GPU is an RTX 5060 Laptop with 8151 MiB. It cannot host the unchanged
27B FP8 target in the intended RTX PRO 6000 configuration. Real-server retirement,
FP8 gains, MTP correctness/performance, and integrated promotion remain untested.
No repair is selected by this diagnosis: removing the missing-ID guard would
require separate scrutiny of scheduler prefix matching and could affect other
requests. The next discriminating evidence is on the real request path.
