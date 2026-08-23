# Kimi K3 explicit-schema compatibility: after action

## Judgment

V3 successfully removed the hidden-schema confound. Both Native and sterile CodeAct passed termination on their first call with zero invalid interactions. Under v2, Native required all three calls and CodeAct failed. The explicit shared schema therefore caused the predicted protocol improvement.

The valid ladder then stopped at Gate 1. Native queried once and returned the correct answer in two calls. CodeAct also issued the exact permitted evidence query successfully on its first call, and the result contained the required LEFT transition. It nevertheless exhausted its remaining two calls without returning.

Seven calls cost USD 0.03684936. Route, model, provider, authority, and trace integrity all passed.

## Gate 1 failure mechanism

After receiving the successful cell projection, Kimi twice emitted Python as plain assistant content in NOOA-like `# CodeAct cell 2` form rather than calling `execute_python` or `return_result`. T1 rejected both turns because the frozen controller requires exactly one tool call. No second evidence query, decoding error, or reasoning error occurred.

The model's first post-query reasoning explicitly said it would inspect the already-local `query_result`; the emitted code iterated the records and selected LEFT. The second emitted `query_result`. The failure is therefore the transport representation of continued CodeAct cells, not evidence capability use.

## Causal comparison

The v2-to-v3 intervention worked at its intended target:

- v2 Native: pass on call 3 after two schema errors
- v2 CodeAct: fail after three calls
- v3 Native: pass on call 1
- v3 CodeAct: pass on call 1

The next live uncertainty moved from terminal schema visibility to how T1 renders prior cells and accepts subsequent generated cells. The current projection renders prior CodeAct cells as assistant text, while the controller accepts new computation only through a tool call. Kimi imitated the visible cell form instead of the accepted transport form.

## Consequence

Do not advance to the cognitive screen. Do not relabel this as retrieval failure: CodeAct retrieved the exact evidence successfully. A V4 treatment should align the model-visible cell history with the accepted cell transport, preferably by preserving tool-call/tool-result structure rather than accepting arbitrary prose as executable Python. That intervention needs its own frozen package and authorization.
