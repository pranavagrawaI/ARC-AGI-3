# packet-live-probe-v3

Question: Does adding the last five commitments' rationales to the packet as
citable claims (`recent_commitments`) restore the S5I5 objective-proxy diagnosis
that packet-live-probe-v2 lost (3 of 12 objective-gap questions at critic:3, :5,
:7 against a predicted 8)?

Posture: evaluation of one packet change prescribed by v2's result map.

Contrast: condition A only — the enriched packet at critic:3, :5, :7 with an
empty ledger, four samples each, 12 calls. v2's condition A, same boundaries and
rule, is the comparison arm (same model, same endpoint, one day apart; not
randomised, so a difference is suggestive rather than causal).

Prediction: at least 8 of 12 samples are objective_gap by the unchanged keyword
rule; at most 2 mechanics; at least 10 of 12 validate and bind; at least 1 of 12
cites an `actor-commitment-turn-*` ref.

Measure: `raw/calls.jsonl`, same schema; questions printed verbatim in
`RESULT.md`; per-boundary counts beside v2's.

Cap: 12 calls, max_tokens 192, one attempt each, 60s timeout, 6 minutes wall,
spend under $1.00 (qwen/qwen3.8-27b on OpenRouter).

Stop: success at 12 completed; abort on the 4th transport failure or on any
call reporting another model.

Invalid if: more than 3 calls fail; any reasoning tokens or tool calls; any
packet with a non-null pending objection; any packet sha256 differing from the
current `s5i5_fixture.py` output for that boundary.

Result map:
  positive → the enriched packet preserves the diagnosis; next is a controlled
    Kaggle card for v9 versus v8.
  negative but improved over v2 (4-7 of 12) → keep the change; widen to the
    Actor's notes files under a new card before any Kaggle run.
  negative and not improved (≤3) → rationales are not where the claim lives;
    revert to v2's packet and reconsider the window rule before any further
    spend.
  ambiguous → report per boundary; no promotion.

Outputs: `analysis/critic-closure-contract-20260912/packet-live-probe-v3/raw/`.
