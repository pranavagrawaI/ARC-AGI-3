# Protocol

## Contrast

Both arms receive the same checkpoint, task, evidence service, immutable cache,
scoped read grants, pure computation, flat answer schema, authority, and numeric
caps. Native uses typed evidence tools plus computation-only `run_python`;
CodeAct uses Python capability proxies and persistent cells. The contrast is the
control language.

## Context/evidence contract

`evidence.query` acquires reality, freezes its exact result, and returns a compact
receipt. A handle is only a name. T1 grants read/view permission to one session,
invocation, role, frontier, and content hash. Native Python and CodeAct can read
only through the same broker. Neither can enumerate handles, mutate evidence,
write canonical state, or act in ARC. Provider history receives bounded receipts,
views, outputs, and deterministic collapsed history, never the bulk cached value.

## Pilot

The V1 six-packet sample and counterbalanced arm order are unchanged. Each arm
has six provider calls, 16,384 input tokens per call, 4,096 output tokens per
call, and 120 seconds. The global ceilings remain 72 calls and $3. Native is the
incumbent. The pilot is invalid on drift, unmatched semantics/authority, answer
key disclosure, missing trace/usage, route failure, or canonical/ARC mutation.
