# Prepared q6 quick host probe

Package: deployment/kaggle/q6-quick-probe-20260905. Kernel identity: pranavagrawai/q6-quick-host-20260905. Prepared only; no launch or paid inference.

20 local behavioral tests passed in 19.80s. Two inherited set_dynamic deprecation warnings. Tests import the extracted notebook payload; worker tests use the extracted adapter and assert the shared runtime matches the extracted bytes. Real NOOA actor, summarizer and dispatcher paths use fake model replies; this is not an FP8 capability measurement. Initial in-sandbox async tests completed assertions but hung in Python default-executor shutdown. The full suite passed outside that sandbox. A dispatcher assertion was corrected to ignore intentionally suppressed stale-only notifications.

Request construction: installed Kaggle 2.2.4, correct private RTX PRO 6000 route, 3600s session cap, zero transport calls. prepare.py --check verifies deterministic rebuild, source and launch hashes, outer source blobs and every join ZIP entry. Original speed/preflight bytes verified unchanged.

Included: NOOA v4 accounting and summarizer guards from the fixing agent's snapshot; conservative exact-grid host classification and full-grid inert memory; transition-backed before/after history; latest-state delivery with duplicate suppression; separate batch/action/request/skip/exposure counters. Full source provenance is DONOR-PIN.json in the package. No experiment protocol or detector threshold changes.

The new quick assessor rejects the old speed probe. The next live run has 10 individual actions and cannot estimate q6 stall incidence; there is no 15-turn or three-stall acceptance bar. The package README and EXPERIMENT-CARD.md contain bounds, validation commands and post-run assessment. LAUNCH.json records a concrete argv but grants no launch permission.

Contract: 537f2d6555de812bfcc7d1b7b56aaae361941afc3c5eaecf30f39de51def84d8
