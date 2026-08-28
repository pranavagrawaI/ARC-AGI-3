# Dirty main research disposition

Source worktree: `/home/hipra/projects/arc-agi-3`
Recorded HEAD: `f12fca0c1ed8a6fecb22760563fdde7400a9a438`
Recovery record: `/home/hipra/archives/one-solver-consolidation-20260828/main/RECORD.json`
Untracked manifest SHA-256: `75be66d09f61ff52e96992d1dd8e2b13984ec0429bca14cf2f5a7d7eafe7e0e5`
Untracked archive SHA-256: `cb2106414c2a626193b129e41a2e3f2df3bda8c94716d631cf6a516b8188b2e6`

This record classifies the dirty main corpus for the unified Solver. It does not
rewrite, delete, commit, or grant authority to the original files.

## Imported

### Behavioral Resolution Band

`CONTEXT.md` now carries the descriptive noise threshold needed to decide
whether a stochastic challenger is markedly better than an unchanged control.
The imported definition states its invalidation conditions and says explicitly
that it is not a significance test.

## Preserved in the recovery archive

### `experiments/METHODOLOGY.md`

Archive as historical research method. Much of it remains sound, especially the
separation of diagnostic probes, mechanism tests, capability builds, uptake,
behavior, and task outcome. Current `CONTEXT.md`, ADR 0004, and ADR 0005 already
state the live subset. The document also freezes Gemma and provider assumptions
that no longer define the Tycho Nooa incumbent, so copying it as current policy
would create two authorities.

### `experiments/SOLVER_LINEAGE.md`

Archive as the exact G0/L0-era candidate ledger. It is a 493-line historical
record tied to earlier Gemma and Luna experiments. It does not name the current
Tycho Nooa candidate stack. Candidate lineage now belongs in candidate manifests
or architecture history, not in a mutable global deployment ledger.

### `README.md`

Archive. Its navigation is useful for the old corpus, but its current invariant
says Gemma-4-31B is the contestant. That is not the current Solver contract.

### `docs/adr/0003-evidence-bearing-solver-search.md`

Archive as an accepted decision in the old main line. Its live principles are
already represented by the current domain language and the challenger selection
rule in ADR 0005. Importing it under a new number would preserve duplicate
policy without adding a current decision.

### `docs/adr/0004-model-owned-plan-compute-commit.md`

Archive as the decision behind the older Evidence Operator program. The unified
Solver instead uses native model tools, a read-only computation boundary, and
host-owned validated planner publication. The old operator contract remains
useful provenance but is not the current runtime design.

### Experiment cards, reports, manifests, analysis, and verifiers

The full 14,755-path corpus remains in the checksummed local recovery archive.
A future claim may import a card, report, or tested verifier only after naming
its required inputs, confirming those inputs are retained, and stating whether
it is current tooling or historical evidence. File existence alone is not a
reason to copy it into the unified branch.

## Rejected from the unified branch

- `environment_files/ls20/9607627b/metadata.json`: local absolute-path and date
  churn from another checkout.
- `.cache/`, `tmp/`, raw recordings, complete run workspaces, and ignored run
  directories: computational bulk with no active-branch authority role.
- Older 0159 runtime files where the later Tycho Nooa lineage owns the successor
  implementation.
- Bulk import of all experiment cards or reports. The recovery archive preserves
  them without making the active repository depend on 1.57 GiB of unselected
  working-tree state.

## Deferred concepts

`Balanced Cognitive Unit` remains archived. It may return only if a current
model repeatedly forms a correct substantial plan but fails at a smaller
commitment boundary, and a visible verbatim handoff separates that explanation
from alternatives. The current decision-discipline challenger does not require
it.

## Verification

The preservation tool verified all 14,755 untracked archive members against
their recorded content identities, preserved 46 symlinks without following
them, and confirmed that the 8,006-byte tracked patch applies to recorded HEAD.
No source or evidence file in the original main worktree was changed.
