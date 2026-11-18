# Independent implementation review

Candidate: current untracked `tools/experiment_checks.py` and
`tools/EXPERIMENT-CHECKS.md`; read relevant tests and historical off-arm IPC.
Review-only; no implementation edits, paid calls, helper execution, or board posts.

## Findings

1. **P2 — A malformed resource observation suppresses an already detected OOM verdict.**
   `tools/experiment_checks.py:153-168` accepts an unchecked resource epoch and
   later compares it with readiness. With an OOM log, `ready_epoch:10`, and a GPU
   sample whose `epoch:null`, `flash()` raises `TypeError` at line 161 instead of
   returning the invalid verdict and preserved source evidence. Other nearby
   unchecked shapes include `runtime:null` at line 168 and unhashable result
   names at line 106. The documented rule is that an invalidator wins even when
   evidence is missing. Validate field types before deriving observations,
   record unusable fields as unresolved, and still return the known invalidator.
   This does not produce a false PASS, but loses the audit report and its reason.

2. **P2 — Malformed action records receive supported batching statistics.**
   `tools/experiment_checks.py:201-215` does not validate `actions` as an action
   list. A record with `actions:"UP"` is counted as one batch because the string
   has length two, and the command returns `status:"supported"`. `actions:null`
   crashes instead. A malformed/older record should produce an explicit source
   diagnostic and unresolved inventory, with no invented batch count. Validate
   required action fields before joining or counting records. Missing turn keys
   also currently join on `None`, so apply the same validation before temporal
   associations. The historical inspected off-arm turns are unique and complete;
   this review found no incorrect join in that preserved run.

3. **P2 — Coverage either crashes or silently drops malformed Actor fixtures.**
   `tools/experiment_checks.py:233-235` calls `.get()` on `usage` without checking
   its shape. `usage:null` raises `AttributeError`. A non-integer prompt count is
   silently excluded, so two Actor rows can become a one-fixture distribution
   without any missing-evidence diagnostic; booleans and negative integers are
   accepted. Live counts already receive stricter validation at lines 241-245.
   Apply equivalent validation to every Actor fixture and preserve excluded-row
   source/reason diagnostics. A generic tokenizer caveat does not identify this
   separate defect or tell a reviewer that the reported sample was incomplete.

## Reproduction

Run from the repository root:

```bash
python3 analysis/deployment-probes/20260909-experiment-checks/reviews/implementation/reproduce.py
```

The script creates temporary evidence and reproduces all three findings. It
prints the exceptions, the erroneous supported batch count, and the missing
fixture diagnostic. Temporary evidence is cleaned automatically.

## Coverage and limits

Read all of `experiment_checks.py`, the documented claim/exit contract and tests;
inspected historical off-arm action/stop records and the executable submission
path. Checked SSE termination/accounting, exact card-clause restriction,
permission versus opportunity distinctions, temporal uncertainty, distribution
calculation and output reservation. Broad identity/semantic/cap review remains
explicitly unresolved in the checker; this review does not treat those disclosed
limits as implementation defects. No supported false overall Flash PASS found.
Preservation received a source read only; detailed filesystem and test-adequacy
review belongs to the parallel assigned reviewer. This report is not proof of
candidate tokenizer parity or experimental feasibility.
