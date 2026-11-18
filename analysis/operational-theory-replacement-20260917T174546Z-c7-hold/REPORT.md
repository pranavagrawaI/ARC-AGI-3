# C7: HOLD at preflight station 1 — frozen gate refuses the C6R1 review's task label

Status: **BLOCKED**, generation_calls = 0, network_calls = 0, credential never loaded,
no authorization written, no claim created, no release consumed, source identity unchanged
(manifest `539540436f6805b17a05ee92a7eda22b9eee85dc8a56ae254b9591bee7417c62`).

What passed before the block: card/manifest/review hashes exact; C6R1 decision PASS with
`findings: []` and `handoff.c7_authorized: true`; predecessor chain t1r6→…→t1 resolves (7 nodes,
0 errors); candidate read-only with no pycache and no non-manifest files beyond the four review/
manifest records; no prior replacement claim/output/CONSUMED for this manifest; offline release
and chain tests `8 passed in 0.07s`; bwrap/unshare present.

The block (C7-B1): `run_pair.gates()` requires the review record's `task` to equal
`operational-theory-T1R7` (EXPERIMENT-CARD.md: "Review input has role independent-reviewer,
decision PASS, task operational-theory-T1R7, and this manifest hash"). C6R1-REVIEW.json carries
`task: "C6R1"`. Reproduced locally against the frozen driver with preserved metadata bytes and a
throwaway gate-isolation probe (no network, no credential, no claim): the exact-byte review is
`REFUSED: missing exact-byte T6 review or replacement-pair authorization`; the same bytes with
only `/task` rewritten are accepted. The probe copies were hashed and discarded so no look-alike
review persists; `GATE-REPRO.json` and `operator/gate_repro.py` preserve the reproduction.

Why the operator did not fix it: the brief forbids modifying the frozen candidate/review and any
manual repair or bypass, and the card assigns review records to the independent reviewer. A
launch operator rewriting the reviewer's task label — or patching the gate — would be authoring
approval. This is a form defect in the review record, not a machinery defect and not a
substantive review failure.

Not done, deliberately: no fresh OpenRouter metadata GET. The brief orders the identity/scope
gate before the metadata lookup, and any metadata fetched now could not be the fresh bytes a
future authorization must hash-bind.

Handoff: independent C8 audit of this hold; then the C6R1 reviewer re-issues the PASS with the
gate's task label (a new excluded record such as C6R2-REVIEW.json with identical role/decision/
manifest/card hashes), after which a fresh C7-style launch runs station 1 again, the metadata
lookup, a new manifest-bound user authorization, and one pair. Nothing here is reusable as
authority. No construction or gameplay result exists.
