# Trial16-r2 local superseding handoff — UNASSIGNED

Both demonstrated defects repaired locally. Coordinator exact review and a new serialized assignment are still required; no API, GPU, slot or launch authority exists.

Freeze `1a80f1b33b0d7c7a4bf2d27788c487ffb5fac684ddcef1d3377b76eebfebe61e` binds 1185 local files and 9995 preserved references. Sources `b2100140facb2283d0a32e1ac957245006a5341b7fd641a59701a1b7d74d1c30`; config `ceeaf91c1abb6dbfa8a3351e3b372c58fd64ff6cc9db7ba97c953c78996f98f4`; policy `e0043d559f1184a389292050e5e1f8dc5e20b67f42c29501587e08d3463f5e76`. All original trial16 9989 and trial15 6303 bindings verified unchanged. Full details and exact evidence hashes are in HANDOFF.json and HANDOFF-INTEGRITY.json.

Three of42 runtime payloads change: candidate_runner.py, observation.py, primary_bundle.py. The operational collection reducer changes only its publication join. Other39 payloads, config/metadata, candidate-only science, loader/cache, resource/identity/ownership gates and caps remain exact.

OOM still stops workload immediately and retains the original primary error. Bounded cleanup-only finalization now preserves final resource samples and orderly host-series completion. A coherent complete raw OOM fixture reaches CLOSE_TESTED_BOUNDED_CONFIGURATION through the unchanged independent checker; malformed/late evidence remains INVALID. No workload resumes after OOM.

Publication checks actual receipt-write completion, helper return and final caller completion-record persistence. The offline V2 receipt/caller hash-clock join and actual/bundled result equality reject stale passing snapshots. A delayed publication invalidates status and rate. This is cooperative lateness detection, not a hard syscall bound.

Exact canonical RED witnesses reproduced against original sources and GREENs retain raw receipts. Nine focused regressions passed. Source-bound qualification passed in 33.647s:61 discovered tests with2 SDK-dependent skips covered by installed SDK runs,1 SDK-primary and4 SDK-operational tests, exact1369 bodies, SDK serialization with zero network, and real new-outer foreign-cwd bootstrap checks. Proof SHA `2229047de76348dd0f9c87bd3215d5010d9501ce547b8250e5dafcbead5b02c0`. The inherited64/65 baseline is disclosed and was not rerun.

Read CARD.md, REVIEW-DELTA.md, identity-delta.json, repair-evidence/RED-GREEN.json and qualification-handoff/PROOF.json. Reproduction commands and source pins are in HANDOFF.json. Superseded snapshot, exact witnesses and intermediate test/assembly failures remain preserved. Actual GPU/real OOM/platform collection/full production-window feasibility remain unproved. Full raw collection is still required before any descriptive rate is accepted.
