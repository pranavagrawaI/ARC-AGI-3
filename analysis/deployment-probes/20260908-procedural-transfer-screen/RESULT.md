# Procedural transfer screen result

Verdict: **ERROR — zero paid/provider calls; no model-quality evidence.**

The local relay refused the first rendered Actor request before forwarding it because the required diagnostic instruction was absent. The runner recorded `calls=0`, `reserved=0.0`, no actions, no experiments, no model-call records, no transitions, no helpers, and no procedures.

The cause is source-local and reproduced by inspection: `headless.py` installed the diagnostic `decision_workflow`, then unconditionally replaced the same context key with the ordinary discovery workflow before the first request. The literal two-cell demonstration remained intact, but the relay correctly treated the missing diagnostic instruction as prompt drift.

The original card, `STARTED.json`, `RESULTS.json`, worker logs, and empty IPC evidence are retained unchanged. This run cannot answer whether Qwen performs procedural transfer. No retry or paid relaunch was attempted.

The candidate repair keeps the diagnostic workflow from being overwritten and records every received request before validation, so a future rejected prompt remains inspectable. A future screen requires a new immutable experiment card and output directory.
