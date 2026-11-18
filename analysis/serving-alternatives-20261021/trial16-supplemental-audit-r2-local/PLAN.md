# Separate trial16 supplemental-audit output finalization repair

LOCAL audit-only successor. No collection execution, full supplemental audit or scientific checker execution is authorized by this packet. The approved collector trial16-missing-raw-local/sourcepins7d34303027ebd87a174a734f3619659c0bf7933fd7f1b74bae2476dcb1a9f946 remains byte-exact and is running separately under its root-issued authority. No runtime61/65 rerun, GPU/monitor/SDK/auth/launch/stage/commit action.

The supported defect is at the output boundary: approved audit.py assigned the positive scientific status/rate before writing ORIGINS.json, and exception handling only appended an issue. A subsequent origins/report persistence error could therefore retain positive top-level fields.

## Exact correction

- All existing source, quota, raw-resource, OOM, replay/PID, publication and scientific-checker functions/calls remain unchanged. AUDIT-DELTA.diff shows only local module wiring and the output-finalization tail.
- The scientific candidate proof stays local until ORIGINS.json is written, flushed, fsynced and its directory flushed.
- SUPPLEMENTAL-AUDIT.json is then written/flushed/fsynced with output_completion_required=true. The report alone is never an accepted output.
- A hash-bound completion record joins the exact origins and report bytes. Its pending file is flushed before exclusive publication as AUDIT-COMPLETION.json and directory flush.
- The returned/read verdict comes through publication.read_completed, requiring the completion marker, exact origin/report hashes, complete fields and absence of AUDIT-FAILURE.json.
- Every exception resets status=INVALID_INCONCLUSIVE, U_resp=null and supplemental_complete=false. A final report or completion persistence failure returns invalid even if positive candidate bytes reached disk. Any exposed completion marker is quarantined on a failed final step; failures and pending bytes are preserved. If a failure receipt itself cannot persist, the function still returns invalid. Filesystem failure cannot be repaired or retried by this adapter.
- A failed scientific/provenance audit uses the same invalid-output path; it never exposes a candidate positive result.

## Focused qualification

The RED fixture reproduces only the approved output-finalization order: it supplies a literal candidate result to that boundary and injects an origins write error. It does not mock or execute a full scientific pass. GREEN cases exercise real temporary-file writes and fsync, report/flush/completion errors, stale or tampered reports, failure-receipt failure and final-directory flush quarantine. AST comparisons verify that all pre-existing helper functions and the unchanged scientific checker/publication verifier calls remain identical.

`verify.py FRESH_EXCLUSIVE_DESTINATION` honors its required destination argument and never overwrites approved mutable qualification outputs. The original owner's exact PROOF remains1bfbc63511c68028af883684cccee7fa1f1b5a86d8b6a75b51ba97282791a1b1. Root disclosed that original pre-rerun TESTS/RISK bytes were not retained; no exact copies were found in the bounded owner-artifact/temporary/cache check. Current root-rerun copies are preserved and labeled, not reconstructed as original evidence.

## Future audit interface, pending exact review

After separate explicit root audit authorization, use the separate audit.py with completed collection and a fresh audit destination. It imports the original binding/manifest and validation helpers from the approved adapter, verifies its exact source references and uses publication.py only for final output. Root and downstream consumers must use publication.read_completed (or the returned CLI verdict), never accept a bare positive SUPPLEMENTAL-AUDIT.json lacking its completion/hash join.

No audit is run during preparation. Original partial ZIP and frozen/overlay INVALID/null remain unchanged; individual-file supplementation never becomes a complete ZIP/CD claim. Collection failure is not candidate-negative. No comparison, ratio, adoption or new GPU authority follows.

Durable-output semantics use successful POSIX flush/fsync acknowledgements, not hostile storage attestation or a hard syscall bound. If the filesystem cannot preserve a failure marker/quarantine, the current call still returns invalid; raw orphaned candidate bytes cannot be treated as accepted without the full completion contract and successful caller result.
