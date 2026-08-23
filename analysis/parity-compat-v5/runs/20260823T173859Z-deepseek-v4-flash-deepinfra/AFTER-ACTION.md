# DeepSeek V4 Flash V5 compatibility: after action

## Judgment

DeepSeek V4 Flash did not clear V5 Gate 1. Both Native and CodeAct passed the termination gate in one call. Native then used the metadata capability once and returned the correct answer in two calls. CodeAct preserved structured transport across all three turns but never invoked `evidence.query`: it spent the entire allowance inspecting the proxy's type, public members, absent docstring, and generic `(**arguments: Any)` signature. It therefore made zero query attempts, received zero disclosures, and never returned a terminal answer.

Seven valid calls through the exact DeepInfra FP8 route cost USD 0.000644200. Route, provider, authority, accounting, and trace integrity passed.

## What this run settled

V5's semantic normalization and disclosure accounting were not tested by the CodeAct arm because no capability request crossed the membrane. The preserved V4 failure—rejected incomplete query followed by a blocked correction—did not recur. The new failure lies earlier in the causal chain:

```text
visible semantic contract
        ↓
model seeks runtime affordance details
        ↓
proxy exposes no docstring and a generic signature
        ↓
three introspection cells
        ↓
no query, no disclosure, no terminal return
```

This is not evidence about RLE, retrieval reasoning, or Native-versus-CodeAct cognitive quality. It is evidence that the current CodeAct capability surface is not self-describing enough for this smaller model under a three-call compatibility gate, even though the required semantic arguments were present in the user payload.

## Consequence

Do not advance DeepSeek to representation or cognitive parity under this authorization. Do not rerun this completed authorization. Before another paid compatibility run, make the executable proxy congruent with the visible contract: give `evidence.query` a concrete keyword-only signature and useful docstring generated from the same manifest that defines validation. Keep semantic normalization, two-attempt accounting, one-disclosure scarcity, authority, packets, and call ceilings unchanged. A scripted regression should prove that `help`, `__doc__`, and `inspect.signature` all communicate the admitted query shape without granting new authority.

That intervention tests capability discoverability, not a looser parser or a larger budget.
