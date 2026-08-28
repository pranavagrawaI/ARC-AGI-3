# Applicability of the GLM 5.3 Flash v4 interface findings to the unified Solver

## Decision

Three of the five implementation findings survived in the unified Solver at the
baseline fixed point. The two report-gate findings did not: their causal
mechanism was experiment-owned policy in the old core worktree and was removed
from the unified production path.

## Implementation closure

The three applicable findings are now closed without restoring the obsolete
interface-audit report subsystem:

- **X-R3:** generated Python source is created outside the actor workspace and
  mounted read-only at `/tycho-generated` inside the sandbox. The launcher keeps
  the canonical workspace as Python's import root, so `wmlib` imports still work.
- **X-R4:** the request model rejects non-click coordinates and incomplete
  clicks. The provider-facing request carries two object branches: `CLICK`
  requires bounded integer `row` and `col`; non-click actions expose only the
  current-turn action enum. A local `Tool` adapter preserves this union across
  NOOA's generic schema-cleaning boundary.
- **X-R5:** the durable key was not renamed. `available` is Tycho's existing
  workspace-v1 field and historical workspaces already contain it. The mapping
  is now named `tycho-workspace-turn-v1`, centralized in `models.py`, and
  consumed as `available_actions` everywhere inside the Agent and Planner.

`analysis/unified-interface-audit-applicability/post-fix-result.json` binds the
fixed source bytes and records all five post-fix assertions as `true`. The
original `probe-result.json` remains unchanged as the negative baseline.

The table below records the **baseline applicability verdict**, not the current
state of the repaired code.

| Old finding | Unified verdict | What distinguishes the implementations |
|---|---|---|
| X-R1: fresh hashes do not enforce report closure | **Exact defect absent** | Unified `WorkspaceToolSurface` has no interface-audit report gate, evidence revision, or report revision. It does not admit actions on the strength of a cumulative Markdown report. |
| X-R2: coarse freshness revisions create false preconditions | **Exact defect absent** | The unified surface has no audit evidence/report revision counters to invalidate. |
| X-R3: `.tycho-run-*.py` leaks into actor workspace | **Confirmed** | Unified still calls Tycho `PythonSandbox.run_source`, which creates the source file inside the workspace before execution. Read-only mounting prevents mutation but does not hide the file. |
| X-R4: action coordinate constraints are prose-only | **Confirmed** | The current-turn schema restricts the action enum but independently permits `row` and `col` for every action. The returned `ActionDecision` rejects the combination later. |
| X-R5: live and durable action vocabularies use different field names | **Confirmed** | Live `GameTurn` uses `available_actions`; `GameWorkspace.record` receives `available` and writes that key to `turn_NNN.json`. |

The old treatment result—interface-first prompting produced more useful contact but worse final synthesis—does **not** transfer to the unified Solver. The unified request, context, sandbox, planner, and error interfaces changed. A new matched run is required to estimate behavior. It should not run until X-R3 and X-R4 are fixed and any future experiment-owned audit report has closure rather than freshness alone.

## Evidence

### X-R1 and X-R2: removed with the old experiment policy

Board message 104 fixes the unified runtime boundary and says on-policy/interface-audit launch policy remains outside production. The source comparison agrees:

- old core `tools.py` imported `interface_audit_enabled`, initialized `_audit_evidence_revision` and `_audit_report_revision`, advanced evidence revision after non-report tools, and rejected `take_action` with `audit_report_stale`;
- unified `tools.py` removes that import, all audit state, all revision updates, and the action precondition;
- a scoped search of `solvers/tycho_nooa/src/tycho_nooa/*.py` found no `audit_report_stale`, `interface_audit_enabled`, or `_audit_evidence_revision`;
- the live probe found neither `_audit_enabled` nor `_audit_evidence_revision` on the actual `WorkspaceToolSurface`.

This clears the **exact** defects, not the general epistemic warning. Unified still has model-authored `world_model_report.md`, `actor_beliefs.md`, and level-insight notes. Their freshness or existence does not establish truth or cumulative coverage. That is not presently an action-admission claim: builder reports are advisory and exact tool/checkpoint traces remain separate. If a later experiment treats one of these files as cumulative authority, X-R1 returns unless the experiment owns an append-only incident ledger and independently checks coverage.

### X-R3: transient source remains model-visible

The probe invoked the real unified `WorkspaceToolSurface.run_python` and asked the child to enumerate `.tycho-run-*.py`. It observed:

```text
['.tycho-run-ctd2jiyw.py']
```

After `run_python` returned, the workspace contained no matching file. This reproduces the old observe-then-`FileNotFoundError` mechanism: the transient is visible during execution and deleted afterward.

The implementation path is:

- `solvers/tycho_nooa/src/tycho_nooa/tools.py:683-692` delegates ordinary Python to the pinned executor;
- installed Tycho `workspace/sandbox.py:277-298` creates a named temporary file with `prefix=".tycho-run-"`, `dir=ws`, then unlinks it in `finally`;
- unified read-only sandbox wrapping changes mount permissions, not directory membership.

**Safety fact, evidence level 5:** reproduced through the shipped tool surface and sandbox.

### X-R4: conditional action validation still occurs at execution stage

The generated request schema for actions `UP` and `DOWN` contained:

- `action.enum = ["UP", "DOWN"]`;
- independent nullable integer schemas for `row` and `col`.

The probe submitted `{"action":"DOWN","row":10,"col":10}`. Request-schema validation completed. `ActionDecision` then raised `DOWN does not accept row or col`, and the durable tool row recorded:

```json
{
  "validation_status": "complete",
  "execution_status": "error",
  "error_stage": "execution"
}
```

The live behavior follows the source:

- `solvers/tycho_nooa/src/tycho_nooa/tools.py:93-105` creates the enum plus unconditional optional coordinate fields;
- `solvers/tycho_nooa/src/tycho_nooa/models.py:34-40` enforces the cross-field rule only when constructing the returned decision;
- `solvers/tycho_nooa/src/tycho_nooa/tools.py:403-457` therefore classifies the later Pydantic failure as execution.

**Safety fact, evidence level 5:** reproduced through current schema generation, dispatch, returned decision construction, and durable trace.

### X-R5: field-name drift survives actual serialization

For the same turn, the live object used `available_actions=["UP", "DOWN"]`. Its durable `level_0/turn_000.json` contained `"available": ["UP", "DOWN"]` and no `available_actions` key.

The source path is `solvers/tycho_nooa/src/tycho_nooa/agent.py:383-392`; the planner then reads `metadata["available"]` at `solvers/tycho_nooa/src/tycho_nooa/planner.py:272-301`.

**Safety fact, evidence level 5:** reproduced through the actual unified decision-start recording path.

## Consequence and repair ownership

1. **Do not port X-R1/X-R2 into production.** Their future repair belongs to an experiment-owned audit runner: host incident IDs, append-only coverage, trace-derived counters, and rejection of disappearing findings. Reintroducing report revision state in `WorkspaceToolSurface` would reverse the consolidation boundary from message 104.
2. **Fix X-R3 in the execution substrate.** Place generated source outside the actor evidence namespace, or mount/execute it at a reserved path the workspace cannot enumerate. Cleanup alone is insufficient.
3. **Fix X-R4 in the model-facing schema.** Emit a conditional/discriminated schema whose non-click branches forbid coordinates and whose click branch requires them. Keep the returned `ActionDecision` check as defense in depth.
4. **Treat X-R5 as a low-priority contract decision.** Canonicalize the name if compatibility allows; otherwise document/version the durable projection and give adapters one explicit conversion point.
5. **Do not infer treatment behavior from the old runs.** After X-R3/X-R4 and experiment-owned report closure are fixed, use a small matched contact before authorizing another 24-action pair.

## Reproduction and verification

Probe artifacts:

- `analysis/unified-interface-audit-applicability/probe.py`
- `analysis/unified-interface-audit-applicability/probe-result.json`

Probe command:

```bash
cd solvers/tycho_nooa
TYCHO_SANDBOX_RUNTIME=host uv run --frozen python \
  ../../analysis/unified-interface-audit-applicability/probe.py
```

The result binds the imported dirty-worktree source bytes for `agent.py`, `models.py`, `tools.py`, and the installed Tycho sandbox. Base commit: `fa190780ac20dd50b3b45a9ec7b63abbfda8bd79`.

Regression slice:

```text
12 passed in 11.75s
```

It covered current action-schema/error tracing, per-turn action enum generation, disposable planner isolation, and the full trusted-runtime test module. Ruff and `git diff --check` are required again after this report is written.

Post-fix probe artifacts:

- `analysis/unified-interface-audit-applicability/post_fix_probe.py`
- `analysis/unified-interface-audit-applicability/post-fix-result.json`

Post-fix command:

```bash
cd solvers/tycho_nooa
TYCHO_SANDBOX_RUNTIME=host uv run --frozen python \
  ../../analysis/unified-interface-audit-applicability/post_fix_probe.py
```

The final integrated fixed-candidate slice covered native flow, the trusted
runtime, planner behavior, and the production import boundary:

```text
92 passed in 5.47s
```

The post-fix result separately establishes that generated source was invisible
during execution, invalid non-click coordinates stopped at validation, the
provider schema retained its conditional branches, and the durable action-field
projection resolved through the named workspace-v1 adapter.

The repository-wide Solver suite is not an all-green gate because research
fixtures and retained corpora are absent in this checkout. It returned:

```text
70 failed, 650 passed in 51.96s
```

The fixed-point baseline was 70 failed and 649 passed. The one additional pass
is the new workspace-v1 projection test; the failing count did not increase.
The focused production slice above remains the completion gate for these fixes.
