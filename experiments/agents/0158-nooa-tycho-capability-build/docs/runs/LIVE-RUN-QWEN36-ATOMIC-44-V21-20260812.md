# 0158 Qwen3.6 atomic 44-action v21 — 2026-08-12

## Outcome

The atomic-commitment experiment is complete. It fixed the v20 commitment
closure failure, but Qwen did **not** complete LS20 level 1 or reach 44 actions.

| Measure | Observed |
|---|---:|
| Confirmed actions / successors | 8 / 8 |
| Levels completed | 0 |
| Provider calls | 116 |
| Reported input + output tokens | 1,688,851 |
| Verdicts | 5 supported, 3 contradicted |
| Complete G2 cycles | 1 |
| Retryable transport failures | 0 |

Actions were `1, 1, 1, 1, 2, 1, 1, 1`. Every supported verdict was
`PRIMARY_REPLACEMENT_EXACT`; every contradiction was
`PRIMARY_REPLACEMENT_MISMATCH`. The scorecard is terminal
`PAUSED_FOR_REVIEW`, non-resumable, with no uncertain action.

Raw evidence:
`artifacts/live-ls20/20260812T151432Z-qwen36-atomic-44`.

## What changed and what it established

The old effect-constructor plus `submit_probe` sequence was physically absent.
All eight commitment artifacts closed through `commit_replace`, in
`6, 8, 6, 8, 10, 5, 6, 3` calls. Commitment iteration exhaustion did not
recur. This licenses the narrow conclusion that atomic commitment corrected
the observed v20 procedural closure failure. It does not establish gameplay
success or that commitments are efficient.

The exact hosted-worker qualification had already returned a typed claim and
atomic commitment in 7 calls / 23,126 tokens without ARC contact. Its preserved
output is `artifacts/provider-qualification/20260812T150857Z-qwen36-atomic-commitment`.

## Terminal failure

Action 8 had a confirmed successor and supported verdict. The next artifact was
the ninth **claim**, not a commitment. Its exact NOOA failure was:

```text
Generation failed after 12 iterations (max_iterations=12).
Unable to complete `analyze`.
```

The redacted error's 88-character length and SHA-256 exactly match that string.
Provider calls 1–3 were successful explorations. From call 4 onward, the hard
gate correctly rejected non-direct cells before execution. Calls 6–12 tried
`submit_grounding_claim`, but every cell first assigned `primary`; none was the
single AST expression the gate required. Qwen therefore understood the
semantic candidate and submitter but never adopted the host's unnecessarily
narrow submission grammar.

This is a new **claim-submission grammar iteration exhaustion**, not a 429,
transport timeout, output truncation, missing tool, schema mismatch, stale IPC,
environment failure, semantic contradiction, or recurrence of v20's
commitment closure failure. The immutable packet retains its generic label; the
evidence-backed correction is `failure-classification-correction.json`.

## Decision

Do not increase iterations. The smallest next change is to let submission-only
cells contain constructor assignments (`make_mask`, and for commitments
`cell_after`) plus one final phase submitter, while still physically rejecting
evidence reads, imports, control flow, and non-submission cells. This preserves
the three-exploration bound without confusing deterministic construction with
semantic exploration. Semantic handoff is not selected by this failure.

## Verification

- Offline: 36 host tests and 15 runtime tests passed; both Ruff checks and the
  provenance audit passed.
- Provider qualification: accepted claim and commitment; pinned Qwen/CoreWeave
  route; zero ARC actions.
- Live authority: 8 `authorized` records pair with 8 `consumed` records.
- Evidence: 8 issued actions, 8 successors, and 8 verdicts.
- Integrity: all 17 failure-packet checksums and all 20 candidate snapshot
  hashes pass; no configured credential value occurs in the run tree.
- Approval `0158-ls20-live-v20` is consumed and forbids resumption.

## Resume

Inspect `failure-classification-correction.json` and the final response at IPC
sequence 16. Implement a constructor-only submission grammar, qualify the
falsifying case where `primary = make_mask(...); return_result(...)` follows
three explorations, then use a new card and approval for any fresh run.
