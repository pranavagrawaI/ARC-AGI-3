# Parity compatibility v6

## Intervention

V6 changes only executable capability discoverability. The T1 treatment payload that defines the admitted evidence query is copied into the sterile execution manifest as a data-only capability contract. T2 uses that same manifest value to render a concrete keyword-only `inspect.signature`, an invocation-specific `__doc__`, and bounded `help(evidence.query)` output. T1 remains the sole validator, meter, evidence owner, and disclosure authority.

For the metadata gate, inspection names `kind`, `level`, `limit`, and `projection`, shows the exact admitted values, explains that `grid_encoding` is optional and semantically ignored, and reports the two-attempt/one-disclosure budget. Full projections still require the committed encoding. No omitted argument is filled by T2; generated code must submit a request and T1 applies unchanged V5 normalization and accounting.

## Frozen comparison

DeepSeek V5 passed termination and Native metadata use, but CodeAct spent all three cells on `type`/`dir`, `__doc__`, and `inspect.signature`. The proxy exposed no docstring and only `(**arguments: Any)`, so it made zero query attempts. V6 retains the same gates, evidence, prompts, return schema, three-call ceiling, route requirements, authority, and result-to-action map. Schema v4 additionally commits the exec'd `_sterile_worker.py`, making the intervention part of the authorized package.

Passing Gate 1 establishes only protocol-compatible bounded metadata use. It does not establish representation competence, reasoning quality, or CodeAct superiority. Paid execution requires a new manifest-bound authorization and append-only cost/route accounting.
