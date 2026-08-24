# Qwen 3.8 27B V12 factored compatibility: after action

## Judgment

Qwen reproduces the Gemma result under the same frozen V12 ladder. Both arms passed identifier binding and isolated RLE interpretation. Native then completed the integrated G3 task in two calls, while CodeAct exhausted three calls before decoding or returning.

Unlike Gemma, Qwen made no malformed capability calls in G3. It spent two valid Python cells inspecting `evidence` and `evidence.query`, then used its final cell to issue the exact query and retrieve the correct `L0:T1` RLE record. Thus the failure is not query repair and not cognitive composition after disclosure. It is **introspection overhead before composition**.

Together, the Gemma and Qwen runs provide cross-model evidence that the present CodeAct interface is a worse fit than Native for this constrained, smaller-model regime. Both models possess the component abilities. Both Native arms compose them within the ceiling. Both CodeAct arms consume the ceiling operating the protocol before the final reasoning step begins.

## Causal chain

```text
G0 termination               Native pass    CodeAct pass
G1 capability retrieval      Native pass    CodeAct pass
G2A identifier binding       Native L0:T1   CodeAct L0:T1
G2B RLE interpretation       Native 5       CodeAct 5

G3 Native
    exact query
    return 11->5 + L0:T1                         pass

G3 CodeAct
    inspect globals / evidence / dir(evidence)
    inspect evidence.query docs and signature
    exact query -> disclose correct L0:T1 RLE
    no call remains for decoding or return                  fail
```

CodeAct's G3 path had zero invalid interactions and only one query attempt. The cost was entirely valid but unnecessary protocol discovery.

## Integrity and cost

- Eighteen calls were dispatched and accounted.
- Cost: USD 0.020901900.
- Tokens: 19,224 prompt and 5,154 completion.
- Every response used Chutes and `qwen/qwen3.8-27b`.
- No provider, route, budget, input-cap, or authority violation occurred.

## Consequence

This is enough evidence to stop treating another discoverability refinement as the default next move. Qwen inspected documentation that already contained the exact admitted call; additional documentation would not remove its choice to inspect before acting.

For the target architecture, retain Native as the smaller-model Actor substrate and preserve programmable computation as a bounded capability such as `run_python`, rather than making CodeAct the mandatory control language. A larger call budget could make CodeAct finish, but that is a different cost treatment—and Native has already demonstrated the required composition more cheaply.
