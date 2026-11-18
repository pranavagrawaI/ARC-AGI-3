# Repaired tractability probe: automated outcome audit

This report reconciles resource admission and engine outcomes. Mechanistic interpretation of model reasoning still requires reading REASONING-INDEX.json and the raw tool traces. No model rerun or promotion is performed by this process.

Audit valid: True; retained cases: 24; collected files hash-checked: 343.

| Arm | Stalled actions | Stalled levels completed | Final admission failures | Easy control solves |
|---|---:|---:|---:|---:|
| baseline | 4/4 | 0 | 0 | 2/2 |
| terminal | 4/4 | 0 | 0 | 2/2 |
| time | 4/4 | 0 | 0 | 2/2 |
| both | 4/4 | 0 | 0 | 2/2 |

Audit failures: none

Interpretation: a verified lack of final admission failures establishes the corrected resource path, not useful solving. Actions without completed stalled levels demonstrate commitment only. A failure to emit an action must be traced to final provider receipts, deadline/refusal or tool validation. Historical totals are descriptive because all arms now reserve final tokens; compare contemporaneous arms. First committed batches and two replicas do not decide general solvability.

Evidence: AUDIT.json, REASONING-INDEX.json, summary.json, output/tractability-output and frozen EXPERIMENT-CARD.md. Notebook duration is not exact billed GPU allocation; preserve the full cap until an unconfounded receipt exists.
