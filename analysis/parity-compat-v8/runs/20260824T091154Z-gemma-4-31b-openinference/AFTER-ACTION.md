# Gemma 4 31B V8 compatibility: after action

## Judgment

Gemma 4 31B is not eligible for the matched Native-versus-CodeAct study under the frozen V8 protocol. Both arms received three clean responses on the exact OpenInference BF16 route and both failed the trivial termination gate. This is a typed-return compatibility failure, not a retrieval, representation, CodeAct-computation, or ARC-reasoning result.

Six calls cost USD 0.000505640. Authorization, provider route, accounting, call ceilings, authority, and trace integrity were valid.

## What happened

```text
Native
  1  result = "READY"
  2  result = {conclusion: "READY"}
  3  result = {conclusion: "READY", task_kind: "retrieve"}
     missing evidence_refs -> exhausted

CodeAct
  1  result = "READY"
  2  result = "READY"
  3  result = {conclusion: "READY", evidence: []}
     missing task_kind and used evidence rather than evidence_refs -> exhausted
```

The model moved toward the contract after validation feedback but did not reach the complete `RetrievalAnswer` within three calls. No evidence capability was exposed or used.

## Consequence

Do not advance this model to metadata or representation gates and do not rerun this authorization. The comparison with DeepSeek is decisive at the compatibility level: DeepSeek cleared termination and metadata under V8; Gemma 4 31B did not clear termination.

If the system must support this capability tier, the next useful intervention is not more discoverability. It is to test a shallower terminal boundary—for example, one gate-specific flat return schema rather than a discriminated union containing four answer types—while keeping T1 validation and authority unchanged. That would test whether schema depth, rather than tool use itself, is the blocker.
