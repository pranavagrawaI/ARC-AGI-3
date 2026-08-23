# DeepSeek V4 Flash V6 compatibility: after action

## Judgment

V6 did not make DeepSeek eligible, but it sharply localized the remaining interaction failure. Both arms passed termination. Native queried metadata once and returned the correct answer in two calls. CodeAct used structured `execute_python` transport on all three turns, made zero query attempts, received zero disclosures, and produced no terminal return.

The manifest-derived contract was present in T2, and the concrete signature worked. DeepSeek reached it through this sequence:

```text
print(evidence.query), print(type(evidence))
    -> opaque object repr, proxy type
print(dir(evidence.query))
    -> discovers __doc__ and __signature__
print(evidence.query.__signature__)
    -> (*, kind, level, limit, projection, grid_encoding=None)
call ceiling exhausted
```

Seven valid calls through the exact DeepInfra FP8 route cost USD 0.000693164. Route, provider, authorization, accounting, worker commitment, authority, and trace integrity passed.

## What changed relative to V5

V5 ended after inspecting type/`dir`, an absent docstring, and a generic signature. V6 supplied the exact contract and DeepSeek successfully observed the concrete signature. The intervention therefore worked mechanically, but the useful information arrived on the third and final cell because the object's default representation remained opaque. Neither V5 semantic normalization nor evidence reasoning was exercised.

## Consequence

Do not increase the call ceiling and do not advance to representation testing. The cheapest discriminating next intervention is to make `repr(evidence.query)` itself show the exact admitted call and budget, generated from the same manifest contract. That targets the first operation DeepSeek actually chose, preserves all T1 validation and scarcity, and permits the predicted three-turn path: inspect, query, return.

Any further paid run requires a new frozen package and authorization; this completed authorization must not be reused.
