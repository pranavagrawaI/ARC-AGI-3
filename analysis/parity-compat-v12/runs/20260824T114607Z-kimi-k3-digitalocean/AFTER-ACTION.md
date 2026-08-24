# Kimi K3 V12 factored compatibility: after action

## Judgment

Kimi K3 performed worse than both smaller models on the constrained CodeAct treatment. Native passed termination and capability retrieval. CodeAct passed termination, then spent two G1 calls inspecting the execution namespace and capability object. Its third call issued the exact admitted query and retrieved the correct `L0:T3` record, but no call remained for `return_result`.

This is not a capability-retrieval failure in the ordinary sense: disclosure succeeded, with zero invalid interactions and one valid query attempt. It is **protocol-efficiency failure after excessive introspection**. The machine disposition `rejected_capability_pairing` correctly stops the ladder, but the exact mechanism is failure to terminate after successful capability use.

The larger model did not rescue CodeAct. It intensified the behavior harming CodeAct under a scarce-call regime: exploration of an already documented interface before acting.

## Causal chain

```text
G0 termination
    Native    repaired once, returned ok=true       pass
    CodeAct   returned ok=true                      pass

G1 Native
    exact query
    return L0:T3                                    pass

G1 CodeAct
    inspect names and types
    inspect evidence, dir(evidence), vars(evidence)
        exact admitted call is visible
    exact query -> disclose correct L0:T3
    no call remains for return                      fail
```

G2A, G2B, and G3 were correctly skipped.

## Integrity and cost

- Eight calls were dispatched and accounted.
- Cost: USD 0.035197500.
- Tokens: 5,006 prompt and 1,584 completion.
- Every response used DigitalOcean and `moonshotai/kimi-k3`.
- No provider, route, budget, input-cap, or authority violation occurred.

## Consequence

The model-size hypothesis is contradicted for this treatment: Kimi did not make CodeAct more efficient than Gemma or Qwen. Across all three V12 runs, Native completed every reached gate, while CodeAct repeatedly consumed its fixed budget operating the programming interface before terminal judgment.

Stop searching for a model that makes this CodeAct treatment win under the present constraints. The evidence now supports the production choice: use the Native Actor interface for the smaller-model system and retain Python as an explicit bounded computational tool. CodeAct may still be useful for a fresh Builder with a different budget, but that is a separate experiment and should not be inferred from these Actor compatibility results.
