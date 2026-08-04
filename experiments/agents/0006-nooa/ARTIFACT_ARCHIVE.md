# Local artifact archive

The ignored experiment artifacts were removed from this worktree after a verified,
lossless archive of the non-runtime evidence was created at:

`/home/hipra/projects/arc-agi-3-archives/0006-nooa/artifacts-without-venvs.tar.gz`

Archive SHA-256:

`2d03003101d8ff8ebed968be0e0c139ee0448a92b2344801f5d889e6d2e7600b`

`ARTIFACTS.sha256` beside the archive binds all 7,441 retained files. Verification
extracted the archive and checked every file against that manifest before the local
artifact tree was removed. The 36 copied `provider-runtime/.venv` trees were
deliberately excluded; they accounted for about 12 GB of the 13 GB tree and are
reconstructable from the recorded environment bindings. Approvals, manifests,
source snapshots, raw recordings, provider records, worker IPC and process logs,
images, traces, summaries, and workspaces are retained.
