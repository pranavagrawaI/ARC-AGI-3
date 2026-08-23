# Parity compatibility v3

## Intervention

V3 changes exactly one model-facing contract from v2: both Native and sterile CodeAct receive the complete, identical Pydantic JSON schema for `return_result.result`. The schema exposes all four answer variants and the `task_kind` discriminator. T1 still validates Native returns; T2 still validates CodeAct returns. No authority, evidence, gate, evaluator, call ceiling, or ladder rule changes.

## Question

Can the same model complete both protocols when the terminal type is visible rather than recoverable only through validation errors?

## Ladder

Both arms must pass each reached gate within three provider calls: termination; metadata capability; RLE representation; raw fallback only after RLE failure; easy cognition using the selected representation. Both arms complete a reached gate before advancement is judged.

Native receives `evidence_query` when enabled plus `return_result`. CodeAct receives `execute_python` plus `return_result`, with `evidence.query` inside its sterile executor only when enabled. Neither arm receives mutation, filesystem, network, Builder, planner, goal, or environment authority.

## Binding and interpretation

The manifest binds both controller source files and the canonical SHA-256 of the shared return schema in addition to the v2 commitments. A Gate 0 failure now means failure under an explicit terminal contract, not hidden-schema recovery. All later interpretation and stopping rules remain those of v2: failure stops advancement; passing all prerequisites only authorizes consideration of a separately bounded cognitive screen.

Paid execution requires a manifest-bound authorization, exact route, frozen pricing, worst-case reservation, append-only call accounting, and fail-closed provider/integrity handling. A stopped or completed authorization is never silently retried.
