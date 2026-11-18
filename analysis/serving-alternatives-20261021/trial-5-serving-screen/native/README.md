# Native target trace preparation — NOT QUALIFIED FOR GPU ACTIVATION

This optional scheduler overlay is preparation for a future matched capacity
screen. Parent's first candidate feasibility packet must **not activate it**.
No GPU, runtime installation, model load, slot request, stage or commit was made.

## Implemented evidence

`overlay.py` verifies five exact source SHA256s, makes narrow anchored patches,
compiles outputs and writes a separately identified destination plus manifest.
It leaves original files untouched and refuses an existing destination.
Control sources are Pennyroyal SGLang d00d88efc8d6281b12be4f4073126aec95038c55;
candidate scheduler and async wrapper are vLLM
e9757321527ca1ecd514c07c1418dd2c53da3d19. Async wrapper and Request source remain byte-identical.
The installed distribution must independently match input source hashes before
any activation, and installed overlay must be read back against its manifest.
No source match is inferred from package version labels.

Run local checks without importing an engine:

```
python3 -B -m unittest discover -s analysis/serving-alternatives-20261021/trial-5-serving-screen/native -p test_capture.py -v
python3 -B analysis/serving-alternatives-20261021/trial-5-serving-screen/native/overlay.py /tmp/serving-native-overlay-unique-id
```

`capture_runtime.py` is a stdlib-only event sink; it must be importable in the
future engine process. Startup requires `SERVING_NATIVE_DIR` and
`SERVING_SESSION_ID`. It queries **actual** nvidia-smi GPU UUID once and requires
one reported physical GPU. No default fixture identity exists in production
startup. Unit tests explicitly instantiate a separate fixture sink instead.
It emits one JSONL file per PID, exclusive creation, CLOCK_MONOTONIC timestamps,
actual /proc boot ID, session ID, GPU UUID, contiguous sequence, startup/end,
and one-second heartbeats. Writes flush before returning; heartbeat failure
makes later event emission fail and missing-heartbeat validation must reject
an arm. Native source installation is deliberately not automated here.

## Raw native protocol (not trial4's already-joined protocol)

- `native_accept(nid,prompt_tokens,prompt_ids_sha256)`: engine's own ID and actual
  postprocessing prompt token sequence, canonical JSON integer-list hash.
- `target_first(nid,output_tokens)`: first observed positive **committed target**
  output count, once for the logical request; never draft placeholders.
- `native_finish(nid,output_tokens,reason)`: once, at finish-state update for
  SGLang and immediately upon entry to vLLM's `_free_request` after its finished
  assertion. These are distinct boundaries: runtime qualification must quantify
  their possible finish-time bias. Do not claim identical clocks/boundaries just
  because timestamps share CLOCK_MONOTONIC.
- `abort`, `abort_signal(nid_prefix,abort_all)`, `retract`, `preempt`,
  `pause_signal(mode)`: retain raw
  failure signals; ANY such event invalidates a future comparative arm. Abort
  prefix is not relabelled as an exact request ID. Waiting aborts may have zero
  target output and no `target_first`.
- `capture_end(unfinished)`: keep unfinished IDs; never fabricate a finish.

SGLang hooks occur after generation prefill/decode update_finish_state and
before release-side handling. Finished speculative blocks use `finished_len`,
not untrimmed accepted candidates. The embedding path has no target-output hook.
SGLang queue admission hooks retain retraction separately. vLLM's hook follows
the real stop-trim loop; finished IDs are sampled by `_free_request`. Preemption
and externally aborted finish are visible. Unsupported modes (streaming input,
pooling, prompt embeddings, distributed/multi-GPU/dedicated encoder) are not
qualified by this trace contract.

## Local qualification and its limits

Tests execute the **complete AST-extracted real upstream**
`process_batch_result_decode` for SGLang and `_update_request_with_output` plus
`Request.append_output_token_ids` for vLLM using provider-shaped CPU stubs.
They verify speculative-block trim, output-before-finish handling, overlap
skip/deduplication, empty/aborted target output, counter regression and unknown
ID failures. GPU token verification itself is stubbed, not tested. SGLang
prefill hook compiles but its full real method is not CPU exercised. No claim
of end-to-end installed-provider qualification follows from these tests.

## Remaining functional qualification (blocking activation)

1. Frontend capture must independently record incoming HTTP `X-Request-Id`,
   generated response ID and exact engine ID **at the real dispatch boundary**.
   Capture actual rendered UTF-8 bytes, actual token IDs, input decoded PNG hashes,
   processed image grid metadata and frontend/processor source identity. Never
   copy these from the local catalog or infer request IDs from response spelling.
2. Preserve separate frontend and engine PID/sequence provenance; validate common
   boot/session/GPU, coverage and heartbeat, then prove a bijective join. The
   existing trial4 reducer's single-process native envelope is insufficient.
   No merger/adapter that invents a single native PID is supplied here.
3. Reconcile actual prompt-ID hash/count and completed token count against real
   processor capture plus raw HTTP response usage, including a stop inside an MTP
   block and a short completion entirely in prefill. Check long full-image request.
   Prove abort/preempt/retract rejection and censored unfinished handling.
4. Verify real installed sources/overlays, normal and error process exit,
   heartbeat continuity, overhead, finish-boundary bias, and UUID/session identity
   in a separately reviewed bounded functional phase. A killed process may lack
   capture_end; preserve partial raw trace and classify invalid, no silent retry.

Exact vLLM frontend/renderer source bytes were retrieved read-only into
`source-vllm/`; `manifest.json` records URLs/hashes and the one 404. These are
source evidence only; no installed-runtime identity is implied. Parent and prompt
worker own the frontend/processor continuation. This native overlay does not
block the smaller candidate-only feasibility packet, which makes no capacity,
rate, production or adoption claim.
