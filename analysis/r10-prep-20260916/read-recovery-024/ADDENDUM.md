# Request 024 read recovery addendum

SUPERSET_WORKER_DONE
 task: R10A-READ-RECOVERY-024
 summary: One separately authorized read-only request returned HTTP 200. Missing raw request 024 is recovered and verified; original archive and reports remain unchanged.
 files: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/read-recovery-024
 checks: Exact actual target, integer version_number=1, one request, 135824 bytes; raw reconstructed bytes and relay message digest match. 991 prior archive files checked unchanged.
 handoff: Independent outcome reviewer may read this supplemental artifact. Identity-mismatch HOLD, failed remote freeze/use criterion, current-only terminal-status limitation and R10b gates are unchanged.

## Observation

Exactly one ApiDownloadKernelOutputRequest was issued with owner_slug `pranavagrawai`, kernel_slug `r10a-cd82-constructor-repair`, version_number `1` as an integer, and file_path `model-runtime/solver-workspaces/raw/raw-requests/024.wire.json`. No latest lookup, listing, additional file download or experiment mutation occurred.

HTTP 200 returned **135,824 bytes**, retained byte-for-byte in `024.wire.json`.

Raw SHA-256: `b1ebac1bba2b697e4363c98362a1a1a4915f6a4b368801de84e161c06adfc163`.
Message SHA-256: `d33bf96b4506da4f33905465a652b8713cdb2e027c4ae5fedd952a952f0d8cdd`.

The message digest, message byte count, message count and non-message fields equal the preserved call-24 relay ledger. Reconstructing the wire object from independently retained IPC messages plus ledger metadata with frozen local_wire.project_local gives exactly the downloaded bytes and the same raw SHA-256. The ledger did not contain a whole-file hash; raw-file agreement is against that recorded reconstruction, not an invented prior raw hash.

The request occurred 8226.7 seconds after the original 429. That old receipt did not retain Retry-After; no original header value is inferred. The successful new response contained no Retry-After. No further attempt was made.

## Scope of gap resolution

The raw request archive is now complete across **two separate evidence locations**: 104 original wire files plus this separately recovered file, alongside the original 105 raw responses. The old missing-file result remains an accurate historical report and was not rewritten. The original 429 receipt and all original evidence hashes remain unchanged. Its previously omitted 82-byte error-body content is still unavailable; this recovery does not recreate it.

This closes only the request-024 raw-byte collection gap. It does not change the descriptive outcome: B exhausted six constructor calls with three failed submissions and no freeze. Actual target mismatch remains HOLD; provider terminal COMPLETE remains current-only. No gameplay efficacy claim, acceptance document or R10b clearance is issued.

## Reproducibility and preserved evidence

Executed once from repository root:

```bash
/home/pranav/.local/share/uv/tools/kaggle/bin/python -B analysis/r10-prep-20260916/read-recovery-024/recover_once.py
```

Do not rerun the collector. `REQUEST.json` and `RESPONSE.json` retain selector, cap, timestamp, HTTP result and hash. `VERIFICATION.json` retains exact comparisons and input hashes. `PRIOR-ARCHIVE-HASHES.json` and `PRIOR-ARCHIVE-UNCHANGED.json` bind 991 prior files. `EVIDENCE-HASHES.json` binds this addendum and the new artifacts. All writes are confined to this new directory; no old archive, frozen source, control or reviewer artifact was modified.
