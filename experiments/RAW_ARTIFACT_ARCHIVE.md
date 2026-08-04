# Local raw artifact archive

Before the 0002/0006 branch integration, runtime-shaped evidence from the dirty
`main` worktree was moved to:

`/home/hipra/projects/arc-agi-3-archives/main-premerge-20260804/raw-evidence.tar.gz`

The archive contains iteration-0005 request and response fleets, process and trace
logs, workspace snapshots; NOOA reproduction and epistemic-context result trees;
checkpoint request/evidence state; and CD82 raw request/response bundles. Compact
summaries, audits, plans, manifests, reviews, and derived evidence remain tracked.

Archive SHA-256 and per-file bindings are stored beside it in
`raw-evidence.tar.gz.sha256` and `RAW_EVIDENCE.sha256`. A full extraction verified
all 1,305 retained files before their working-tree copies were removed.

Retiring the merged experiment worktrees exposed two additional ignored stores.
They were archived and fully extracted-file verified before those worktrees were
removed:

| Experiment | Archive | Retained files | SHA-256 |
|---|---|---:|---|
| 0002 provider smoke | `/home/hipra/projects/arc-agi-3-archives/0002-nooa/provider-smoke.tar.gz` | 870 | `dfc972b6217d7b42cb40a52db3acfcaf935898b71d197ba89ed2b62b4f329b9c` |
| early 0006 provider smoke, excluding 20 copied virtual environments | `/home/hipra/projects/arc-agi-3-archives/0006-nooa/provider-smoke-without-venvs.tar.gz` | 2,255 | `a1a1a17f05518b5f77dc01a8f40734e91cd0c57da7a2c5728ab009f7894b087d` |

The completed 0006 run archive is separately bound in
`experiments/agents/0006-nooa/ARTIFACT_ARCHIVE.md`. Virtual environments and
bytecode caches were discarded as reconstructable runtime copies. Worktree `.env`
files were not archived; they were byte-identical to the copy retained on `main`.
