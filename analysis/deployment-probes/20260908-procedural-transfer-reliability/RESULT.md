# Procedural transfer reliability screen

Verdict: **FAIL — 4/5 cases passed, but the mandatory competing-regions case failed.**

The matrix is technically valid. All five cases used fresh workers and completed two actions within exactly two Actor calls per decision. Twenty relay receipts equal twenty forwarded requests, twenty provider responses, and twenty worker model-call records. All responses report zero reasoning tokens. There were ten real four-argument experiment records, no summarizer requests, helpers, procedures, provider errors, or worker errors.

| Case | Result | Calls | Decisive observation |
| --- | --- | ---: | --- |
| changed action mapping | PASS | 2 + 2 | UP was a no-op; the second experiment explicitly tested DOWN after rejecting the assumed UP movement |
| competing regions | **FAIL** | 2 + 2 | only one alternative named coordinates; the other was `no change / other region`, so the pre-action test did not discriminate two predicted regions |
| reshaped target | PASS | 2 + 2 | the second decision explicitly reasoned from the measured reshaped 2x2 state |
| translated color/position | PASS | 2 + 2 | the second experiment used the measured upward translation on the changed grid |
| salient no-op | PASS | 2 + 2 | the second experiment and rationale explicitly treated UP as inert |

The competing-regions trajectory did adapt after contact: it measured that UP moved the color-7 cell left and then proposed testing whether DOWN moved it right. That does not rescue the precommitted failure. Before the first action, its supposed alternatives were not two executable regional predictions. The model still uses `other_region` as a bucket rather than doing the procedural work of specifying the second region.

This separates two abilities. Qwen reliably consumed returned evidence and kept claims narrow in these five cases. It did not reliably turn both hypotheses into discriminating, observable predictions before acting. All five trajectories also chose UP then DOWN, matching the demonstration's action order. Their evidence-dependent rationales argue against pure blind copying, but the invariant action sequence means this screen cannot establish flexible action selection.

Provider-reported cost was USD 0.052323; cumulative reservation was USD 1.54165448 under the USD 3 cap. Actor model time was 60.7505 seconds. Card and frozen contract identities remained valid after execution.

Per the result map, stop polishing prompt demonstrations at this checkpoint. Do not advance this primitive to a production comparison candidate on a reliability claim. A materially different interface would need to make alternative predictions executable or host-validated before action submission; another wording tweak would merely repeat the failed approach.

Evidence is preserved in `cases/<case>/raw/`, each case's `RESULTS.json`, the eight final preflight directories, `PRECOMMIT.json`, and aggregate `RESULTS.json`.
