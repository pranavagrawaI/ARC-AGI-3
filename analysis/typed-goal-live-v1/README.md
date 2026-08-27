# Typed-goal live evidence v1

This active-branch package preserves the minimum evidence needed for one bounded mechanism claim:

> In the selected Muse/Glimmer run, the Actor proposed a falsifiable candidate goal over host-registered evidence, requested an independent Builder review, explicitly selected the candidate, and received the selected ledger entry in its next decision request.

It does **not** show that typed goals improve ARC score, identify the true objective, or transfer to another game or model. The run stopped at its two-action cap with zero completed levels.

## Why this package is partial

The complete run is not tracked here. It remains in the local preservation archive:

- archive: `/home/hipra/archives/tycho-nooa-pre-split-20260827/typed-goal-live-v1.tar.gz`
- SHA-256: `b20ecd3ca0b1ca2cbe85885cfdb6ea1b1b38bd2441a1f9db3c772a7660146fa1`

That archive retains all 34 checkpoints, the raw provider recording, generated workspace, logs, and caches. This package excludes those broad families and keeps only five claim-bearing checkpoints, their referenced content-addressed blobs, the typed ledger and review, the host evidence cited by the goal, and the causal trace rows needed to join the lifecycle to the next Actor request.

Because it is deliberately partial, this directory is **not** a substitute input for a whole-run Hermetic Gate audit. The full preserved dossier passes its trace and checkpoint contracts but fails the current Gate on `legacy_authoritative_evidence_index`: its path-only evidence index predates content-bound typed-goal authority. It remains historical mechanism evidence and is not promotable. `SELECTED-EVIDENCE.sha256` verifies only the bytes retained here.

## Evidence map

`EXPERIMENT-CARD.md` is the precommitted feasibility question. `SOURCE-GENERATION.json` retrospectively binds the protected generating worktree's complete Solver and host source inventories, dependency locks, base revision, archived dirty-worktree patch, and run configuration. It is explicitly non-authorizing: retrospective byte binding does not manufacture contemporaneous approval or reusable execution authority.

The retained checkpoints are:

| Call | Source | Why retained |
|---:|---|---|
| 18 | Actor | `propose_goal`; the candidate does not yet exist in the request |
| 19 | Actor | `request_builder`; the proposed goal is now present |
| 30 | Builder | final independent objective-review response |
| 31 | Actor | `update_goal`; explicit selection after review |
| 34 | Actor | first decision request of sequence 1; contains the selected ledger entry |

The selected `llm-trace.jsonl`, `tool-trace.jsonl`, `trigger-builder-trace.jsonl`, and full nine-row host `transcript.jsonl` preserve the joins around those calls. The retained `workspace/ar25/epistemic/` files preserve proposal, review, update, and selected state. `workspace/ar25/level_0/turn_000.txt` is the exact host evidence cited by the goal.

The DeepSeek ten-action run remains diagnostic only and is absent from this branch.
