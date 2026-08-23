# Parity compatibility v4

## Intervention

V4 changes exactly one model-facing representation from v3: a CodeAct cell that actually executed is projected from T1 provenance as the original structured assistant `execute_python` tool call followed by its associated structured tool result. V3 flattened the prior action into assistant prose, making the demonstrated history grammar differ from the grammar required for the next action.

Assistant prose remains cognition only and is never executed. Rejected provider turns do not become executed cells. Failed or timed-out cells retain their actual tool call and an error result; they are never projected as successful execution. Packets, prompts, terminal schema, evidence API, executor authority, evaluators, thresholds, and three-call ceiling are unchanged.

## Question

Can Kimi K3 continue through the sterile CodeAct protocol after successful evidence retrieval when the history shows the same structured action grammar required on the next turn?

## Ladder

Both arms must pass each reached gate within three provider calls: termination; metadata capability; RLE representation; raw fallback only after RLE failure; easy cognition using the selected representation. Both arms complete a reached gate before advancement is judged.

Native receives `evidence_query` when enabled plus `return_result`. CodeAct receives `execute_python` plus `return_result`, with `evidence.query` inside its sterile executor only when enabled. Neither arm receives mutation, filesystem, network, Builder, planner, goal, or environment authority.

## Binding and interpretation

The schema-v3 manifest binds the execution projection and sterile executor in addition to both controllers and the canonical return schema. The CodeAct result records transport adherence as valid `execute_python`, valid `return_result`, plain prose, malformed tool call, or wrong tool.

A Gate 1 CodeAct failure after a successful query is interpreted from these transport classes, not as an evidence-retrieval failure. If structured history still produces prose, stop changing transport and reject this model/protocol pairing. Passing all prerequisites only authorizes consideration of a separately bounded cognitive screen.

Paid execution requires a manifest-bound authorization, exact route, frozen pricing, worst-case reservation, append-only call accounting, and fail-closed provider/integrity handling. A stopped or completed authorization is never silently retried.
