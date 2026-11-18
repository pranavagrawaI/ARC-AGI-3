# packet-live-probe-v4

Question: Can the ten-item Actor code window be removed from the critic packet
without losing the S5I5 objective-proxy diagnosis? packet-live-probe-v3 showed
the critic citing a commitment rationale 9 of 9 times and the code window never,
while the window is about two thirds of the packet's tokens.

Posture: delete-centric evaluation — overdelete (window 0), with a partial
restore (window 3) in the same run so one card resolves both.

Contrast: two conditions, same three boundaries, empty ledger, five commitment
rationales in both. v3 (window 10) is the comparison arm: same model, endpoint,
rule and boundaries, one day apart, not randomised.
  W0. code window 0 — critic:3, :5, :7, four samples each, 12 calls.
  W3. code window 3 — the same, 12 calls.

Prediction: neither condition falls below v3's 6 of 12 objective_gap by the
unchanged keyword rule; W0's reported prompt tokens are under 2,200 at every
boundary (v3: 4,705-5,613); at least 10 of 12 validate and bind in each
condition; mechanics at most 2 in each.

Measure: `raw/calls.jsonl`, same schema; per-boundary counts beside v3's;
questions verbatim in `RESULT.md`; reported prompt tokens per condition.

Cap: 24 calls, max_tokens 192, one attempt each, 60s timeout, 10 minutes wall,
spend under $1.00 (qwen/qwen3.8-27b on OpenRouter).

Stop: success at 24 completed; abort on the 4th transport failure or on any
call reporting another model.

Invalid if: more than 3 calls fail; any reasoning tokens or tool calls; any
packet with a non-null pending objection; any W0 packet with a non-empty
`actor_claims`; any packet sha256 differing from the fixture output for that
boundary and window.

Result map:
  W0 ≥ 6 → set PACKET_CLAIMS to 0 in v9 (the window is ballast); next is a
    controlled Kaggle card for v9 versus v8.
  W0 < 6 and W3 ≥ 6 → set PACKET_CLAIMS to 3; same next step.
  both < 6 → the window carries something rationales do not; keep 10 and test
    the notes-file addition under a new card before any Kaggle run.
  any condition ≥ 8 → the original v2/v3 prediction is met at that window;
    adopt it and say so in CONTRACT.md.
  W0 tokens ≥ 2,200 → restate the size claim with measured numbers.

Outputs: `analysis/critic-closure-contract-20260912/packet-live-probe-v4/raw/`.
