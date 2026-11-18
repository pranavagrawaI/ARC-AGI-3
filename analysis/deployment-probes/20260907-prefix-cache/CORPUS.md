# Frozen Actor successor corpus

The corpus contains two actual consecutive request pairs, plus four batch cases drawn from the same capture. It preserves parsed request bodies exactly; it does not reconstruct prompts from model outputs or input hashes.

| Pair | Capture attempts | Original provider prompt tokens |
|---|---|---|
| normal_short_0_1 | 0 → 1 | 10,581 → 11,250 |
| normal_long_12_13 | 12 → 13 | 21,152 → 21,279 |

Batch cases are attempts 0, 5, 12, and 13; their prompt tokens are 10,581, 17,600, 21,152, and 21,279. The union is five unique requests. “Long” means longer within this available corpus, not a 100k-token case.

Source: `analysis/deployment-probes/20260906-actor-capture/raw/actor-capture-20260906/`. Each request and response hash is checked against the proxy's `events.jsonl` (`recorded_before_forward` and HTTP-200 `upstream_response_complete`). Every frozen record includes both source paths, both SHA256 values, attempt number, and original usage. Only chat request bodies are copied; no HTTP authentication headers are included. Model and sampling fields are retained. The captured requests omit `max_tokens`; replay output caps therefore constitute deliberate diagnostic interventions, not original natural settings.

Rebuild with `python deployment/kaggle/prefix-cache-20260907/build_corpus.py`; add `--check` to verify without writing. Unknown arguments are rejected. Model generation defaults come from the existing `actor-mtp-20260906/corpus.json`, whose path and hash are retained.

The capture run ended in error. HTTP-200 responses establish transport completion, not successful Actor behavior: attempts 12 and 13 returned `submit_actions`, which was not an offered tool. Records include `captured_quality` with returned and unoffered tool names. These remain valid prompt-cache inputs but cannot support claims about valid action generation.

## What actually changes first

All fourteen captured requests have a byte-identical first system message, 21,564 characters long. Across each selected pair, the first unequal JSON message is a previously cache-marked tool result:

- 0 → 1: message index 6, `tool_call_id=prefill_6cdeaf91`.
- 12 → 13: message index 41, `tool_call_id=call_408dd30400bf4e16a270a95e`.

Its content changes from a one-item text-block list containing `status: complete` plus `cache_control: {type: ephemeral}` to the plain string `status: complete`. The cache marker shifts to a more recent result. Later differences include appended history. This is a JSON representation difference; it does **not** prove a difference in model-visible tokens. The notebook must compare the server-rendered input IDs to identify the actual prefix boundary.

No volatile-system-context relocation is justified by this corpus: the system block is unchanged. Altering message ordering before measuring actual token prefixes would test an unsupported intervention. The proposed notebook therefore measures actual reuse and chunk size rather than reorders the solver prompt.

## Summary boundary is unavailable

The saved `20260906-q6-summary-boundary` run proves two real summary calls and resumed play, but the available raw artifacts contain model responses and integrity hashes rather than full pre-summary, summary, and post-summary request bodies. Neither those hashes nor response text can establish exact original requests. The corpus explicitly sets `coverage.summary_boundary=false` and supplies no fabricated pair or intermediary.

Consequently this run cannot establish summary-boundary cache disruption, behavior at roughly 100k tokens, or cross-turn volatile-state behavior. Those need an additional instrumented capture with complete requests around a real compression. The present pairwise measurements are valid normal-successor diagnostics only; they do not establish improved gameplay.
