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
