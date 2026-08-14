# Required-tool decision — 2026-08-12

## Evidence selected

Run `20260812T123422Z-gemma31b-inline-result` exposed only `execute_python`, yet
Gemma made no tool call. With `tool_choice=auto`, it narrated a purported grid
inspection, serialized rows, and exhausted 8,192 output tokens.

## One changed mechanism

At the audited provider boundary, override pinned NOOA CodeAct's `auto` default
with the frozen profile's `tool_choice=required`. The sole provider-visible tool
remains `execute_python`; `return_result(candidate)` remains sandbox-inline.
The fixture boundary applies the same requirement.

This changes action-point entry, not evidence, semantics, action authority,
provider, token cap, CodeAct budgets, or G0-G2 gates.

## Prediction and stop

Gemma must enter `execute_python`, enabling real `frames()` inspection. A tool
execution can still fail or return an invalid artifact; the first consequential
failure closes the new scored run. No prior trajectory is resumed.
