# Unified live contact result

Status: **passed within the one-action claim ceiling**.

The current unified `trigger/control` Solver completed the real path:

```text
ARC authorization and reset
→ arc-host
→ live worker
→ OpenRouter Qwen 3.8 27B
→ native Solver tools
→ legal LEFT action
→ host-recorded successor
→ worker observation closure
→ clean action-budget finish
```

## Observed result

- Zero-action preflight: passed; no provider call or ARC action.
- One-action contact: launcher exit 0; `actions=1`; `stop_reason=action_budget`.
- Worker metadata: live client, trigger mode, control treatment, host sandbox.
- Decision: `LEFT`, present in the host-advertised action vocabulary.
- Successor: `NOT_FINISHED`, zero level increase, 109 changed cells in bbox `[0,15]–[23,63]`.
- Provider: 8 Actor calls, 143,543 prompt tokens, 13,312 completion tokens, 8,019 reasoning tokens, 21,168 cached tokens, recorded cost `$0.08395400`.
- Tools: 10 rows; 9 complete. One malformed `run_python` request omitted `code`; validation rejected it before execution, and the Actor recovered.
- Evidence closure: 8 checkpoint requests and 8 responses; empty worker stderr; exact host transcript, ARC recording, tool trace, and LLM trace retained.
- Credential scan: zero credential-value matches in the retained `analysis/unified-live-v2` artifacts. `.env` remains ignored and mode `0600`.

## Judgment

This repairs the immediate live-run blockers: ARC authorization works, the OpenRouter route constructs and responds, the worker starts, the Solver acts, the host accepts the action, and the successor closes back into the worker.

It does **not** show that the Solver understands `ar25`, that `LEFT` was strategically good, that a level can be completed, or that the 70 retained research-package failures are repaired.

## Evidence

- Card: `analysis/unified-live-v2/contact-20260828T170100Z/EXPERIMENT-CARD.md`
- Machine result: `analysis/unified-live-v2/contact-20260828T170100Z/RESULT.json`
- Run: `analysis/unified-live-v2/runs/contact-20260828T170100Z-ar25-s0-1a/`
- Preflight: `analysis/unified-live-v2/runs/preflight-20260828T165910Z-ar25-s0-0a/`
