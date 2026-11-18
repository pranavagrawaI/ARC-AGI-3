# Heterogeneous typed self-test screen

Verdict: **FAIL — 1/4 cases passed; the interface is not reliable across epistemic types.**

All four cases are operationally valid. The frozen candidate and preflights matched; provider accounting is complete; reasoning tokens were zero; there were no provider errors or unresolved workers. Behavioral call-cap failures remain interpretable failures, not technical invalidity.

| Case | Required test | Result | Earliest boundary |
| --- | --- | --- | --- |
| competing regions | `region` / `changed_bbox` | PASS | Two non-null regional predictions were recorded; the unexpected left bbox contradicted both, and the next action changed to LEFT. |
| shape transformation | `shape` / `after_shape` or `shape_changed` | FAIL | The model constructed a valid shape test, but the lexical alignment guard mistook the action label `UP` for a directional claim and rejected it before submission. |
| color transition | `color` / `color_changes` | FAIL | The model gave identical color-change signatures to both hypotheses, so the host correctly rejected the non-discriminating experiment. |
| effect magnitude | `change_count` / `changed_cells` | FAIL | The model constructed distinct counts `1` versus `2`, but the lexical guard again rejected the action label `UP` as if the claim were regional. |

The gate required at least 3/4 passes including a non-spatial case. It achieved 1/4. This is not evidence that Qwen cannot reason about shape or magnitude: those two failures expose an over-broad host validator. The validator scans the whole question and updates for words such as `UP`, even when `UP` merely names the tested action. Color is the clean model-side failure: its two supposedly competing hypotheses predicted the same observable signature.

The screen therefore succeeded epistemically despite failing its gate. It broke the LS20-centered framing and separated two causes:

1. **Host overconstraint:** lexical semantic policing rejects valid non-spatial tests that mention directional action names.
2. **Model discrimination failure:** the color case did not notice that its predictions were identical.

Provider-reported cost was USD 0.0305782. Actor model time was 40.4384 seconds. No further run was launched.

Next action should be forced by these results: delete the lexical natural-language guard and prevent false knowledge structurally instead (host knowledge should assert only the typed observed prediction, never certify the hypothesis label). Retain target-to-metric binding and distinct-value validation. Then rerun shape, magnitude, and color plus the regional control under a new card.

Raw case evidence is under `cases/<case>/raw/`; each case has its own `RESULTS.json`.
