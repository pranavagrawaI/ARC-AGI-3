# Read-only independent reducer review

Reviewer: GPT-6-Astra, /root/paired_card_review. First review found four demonstrated defects in the unfrozen reducer, while its original 12 tests passed:

- Duplicate candidate response rows produced a fabricated ratio of 2.0 and QUALITY_INVESTIGATION_ONLY.
- A request_errors=1 field did not invalidate the comparison.
- Changed boot identity or interior stop/start trace boundaries did not invalidate the comparison.
- A NaN response completion timestamp disappeared from the measured set without invalidation.

Repairs now require unique request/native IDs, response-to-attempt and native-finish reconciliation, explicit zero request errors, an accounted terminal/cutoff-pending receipt for every send, constant boot identity, outer trace boundaries only, and finite timestamps. Native streams without HTTP attempts and residence above S10 are also rejected. Nineteen CPU tests pass; local-checks.json preserves the command and output. The reviewer independently rechecked all four original findings and confirmed that every original probe now yields FEASIBILITY_ONLY. It found one additional send/native-first-output ordering gap; that was repaired with a nineteenth regression. The final ordering repair has local regression evidence, without a separate independent signoff.

Reviewer supported the draft distinction between decode residence and the native gauges, the newly declared residence bands, separate legacy control gate, unequal reached-prefix caveat, bounded verdicts and pending actual runtime processor qualification. Processor-fit wording in proposed-config.json was refreshed after the all-corpus audit completed.

An earlier T3 request to Claude Opus5.5 failed immediately with “Claude API rate limit reached. Try again later.” It supplied no review evidence. No GPU or experiment launch was involved.

Later automatic resume of the initially rate-limited Claude review completed. Its independent findings and design hold are in review-resumed.json and REVIEW-RESPONSE.md; the current CPU suite has20 methods.
