# V13 terminal result

## Result

The run did **not** qualify the eight-action public-game path. It ended after six reconciled `ls20` actions when Actor call 9 exhausted all 16,000 completion tokens as reasoning and emitted no tool call. The official launcher therefore returned failure and the outer probe marked the request path invalid.

## What V13 established

- Every Actor request exposed `take_actions` as its only action primitive; no model-facing or completed legacy `take_action` occurred.
- Qwen submitted three batches: `UP`, `RIGHT`, and `UP, UP, LEFT, UP`. The host released six actions one at a time, each after the previous public successor was observed.
- The four-action batch avoided three Actor calls. The six releases match six official recording results and the launcher stdout sequence `ACTION1, ACTION4, ACTION1, ACTION1, ACTION3, ACTION1`.
- Containment and process lifecycle checks passed. No level was completed.
- The immediate and terminal Kaggle sources are byte-identical. Their notebook cells and metadata are identical to the frozen local candidate; Kaggle changed only JSON serialization bytes.

## Cause

Removing the second action primitive removed choice ambiguity but did not guarantee commitment. Calls 6, 7, and 8 were required `take_actions` calls and succeeded; call 9 had the same required single-tool surface but used its entire 16,000-token allowance as reasoning and ended with `finish_reason=length`. Therefore “two action tools caused the failure” is not sufficient.

## Next probe

Keep the 16,000-token exploratory budget. Change only the forced action-only request: send `chat_template_kwargs.enable_thinking=false` and cap it at 2,048 output tokens. This directly tests whether the exact SGLang/OpenAI-compatible route honors no-thinking commitment generation.

Machine-readable evidence: `TERMINAL-RECONCILIATION.json`.
