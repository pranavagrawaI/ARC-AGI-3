# Scoped implementation review

Candidate: only `solvers/m2_planning_tools` and `analysis/planning-transfer-20261014`, compared with the hash-pinned B1 bugfix parent and the frozen m2_search contribution. Unrelated tracked and untracked workspace changes are excluded.

Supported findings resolved:

- Native single-action receipts omit the batch `executed_count`. The probe initially classified actual execution as unconfirmed. The boundary parser now accepts the known single-action form and rejects inconsistent flags/counts/action lists. Pure and real-native regressions pass.
- The archived raw control lacks per-episode saved-data ownership. A common dispatch adapter now supplies one host-owned store to all four variants, after restoration. The adaptation is declared; baseline instructions and capability exposure remain the named control. Native control fixtures exercise persistence.
- Python boolean usage was accepted as integer usage by the archived transport. The subclass rejects it, preserves reserved/unknown exposure, logs failure, and stops after exactly one call. The auditor also rejects it.
- Self-reported completion and restoration flags were insufficient evidence. The auditor now checks nonempty historical request comparisons, first wire/request identity, action-prefix fidelity, and independently replays completion through the native engine.
- A source could change between episodes while each individual episode remained internally consistent. The matrix now freezes both checkpoint source trees, input tapes, common harness code, the native engine identity, and intervention source; each episode independently protects the same relevant inputs.
- PREDICTED after interruption does not mean that no action occurred. The interface/docs explicitly preserve that uncertainty and never recommend automatic re-execution.

Checks: 26 helper/probe tests, 29 inherited tests, five audit/transport/native replay tests, eight actual ToolAgent/native fixtures, 12 bounded frozen-model search arms and six frozen-model action probes. The model census used the existing CPython 3.13 diagnostic environment; package and native continuation qualification uses CPython 3.12 matching the notebook harness. No general timing speedup is inferred across environments.

Risk measurement: existing utility comparison retains the same 13-test suite and statement/CC definitions. The conservative maximum including embedded subprocess functions rises from 992 to 1122 as the sandbox wiring gains the required guards and probe operation. Embedded coverage is not measured by stdlib trace and is conservatively zero; these figures are not a failing repository gate or a maintainability claim. The new receipt parser has CC 15 and measured estimate 15.04; the probe pipeline has CC/estimate 10 with complete statement coverage in behavioral tests. New orchestration functions have a conservative maximum 342, with subprocess-only paths unmeasured. No threshold, exclusion or CI gate was added, and no function was split to game the scores.

No unresolved supported defect remains in the reviewed local candidate. Model adoption, independent-game transfer, end-to-end serving configuration and score efficacy remain empirical gaps. The current local result is a qualified interface and retrospective diagnosis. It does not deploy a solver or qualify a hidden submission.
