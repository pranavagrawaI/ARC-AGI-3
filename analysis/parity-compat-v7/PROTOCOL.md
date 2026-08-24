# Parity compatibility v7

## Intervention

V7 makes the cheapest ordinary inspection of a granted capability sufficient for correct use. `repr(evidence.query)` is generated from the same data-only capability contract that supplies the prompt payload, concrete signature, and help documentation. It displays exactly one admitted invocation and separate live `attempts_remaining` and `disclosures_remaining` values. It contains no ungranted capability or alternative enum value.

The T2 proxy decrements its advisory attempt count when it sends a new capability request and decrements its disclosure count only when T1 returns evidence. Cached local reads consume neither. These displays do not authorize or meter anything: T1 retains unchanged V5 semantic normalization, authoritative attempt/disclosure accounting, evidence ownership, and rejection authority.

## Frozen comparison and falsification

DeepSeek V6 used three cells for opaque `repr`, `dir`, and the now-concrete signature, then exhausted the call ceiling without querying. V7 changes only the first representation. Gates, evidence, prompts, return schema, signature, help, semantic validation, authority, and three-call ceiling remain fixed. Scripted controls establish both `inspect -> query -> return` and direct `query -> return` paths.

If DeepSeek sees the complete V7 representation and still spends its next turn on further introspection, discoverability is no longer the live explanation and must not be expanded again. Passing Gate 1 establishes only bounded metadata capability use; it does not establish reasoning quality or CodeAct superiority.
