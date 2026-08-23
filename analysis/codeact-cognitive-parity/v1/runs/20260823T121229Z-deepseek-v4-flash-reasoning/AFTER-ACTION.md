# Invalid first screen attempt

## Observation

The first native provider response was route-valid and cost USD 0.00007476. It returned two parallel `evidence_query` calls with `kind="transition"` and `kind="frame"`. The controller required exactly one tool call, so it stopped before executing either query. No CodeAct call and no ARC action occurred.

## Causal diagnosis

Two interface defects jointly explain the failure:

1. `NativeEvidenceArguments.kind` exposed an unconstrained string instead of the authoritative `frames | transitions` enum, licensing the singular spellings.
2. `NativeParityController` rejected parallel read calls even though the provider legitimately emitted multiple structured tool calls.

The easy positive control broke the treatment before cognitive comparison. This attempt is invalid for native-versus-CodeAct interpretation.

## Disposition

Preserve all raw artifacts. Revise the native schema and allow a batch of read-only evidence calls, while keeping typed return exclusive. Any retry receives a new run identity and retains this call's cost against the user's USD 3 total authorization.
