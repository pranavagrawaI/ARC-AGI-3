# packet-live-probe-v2

Question: With the fixture corrected (empty ledger, system prompt counted), does
a reasoning-off critic reading the v9 compact packet locate the S5I5
objective-proxy gap at the three boundaries where v8's critic did, and does a
coherent `probe` closure in the ledger stop it re-raising that objection?

Posture: evaluation; a corrected retry of packet-live-probe-v1, whose
condition A was contaminated by a fixture bug at critic:4-7.

Contrast: two packet conditions, live, same model and endpoint.
  A. empty ledger at critic:3, critic:5, critic:7 — four samples each, 12 calls.
  B. critic:7 packet with the ledger showing critic:3 closed as `probe`
     (actions_budgeted 46, step 14). The real trace's newest claims say the
     Actor is filling the bar as a test, so this ledger state is coherent with
     the claims — four samples, 4 calls.
  No `withdraw` condition: the real trace's claims contradict it and a
  fabricated claim window would not be a fixture.
  v8's recorded questions at the same boundaries remain a reference only.

Prediction:
  A. At least 8 of 12 samples are objective_gap by the keyword rule in
     `probe.py`; at most 2 are mechanics; at least 10 of 12 validate and bind.
  B. At most 1 of 4 re-raises the bar-as-objective claim (objective_gap class
     whose question names bar, fill, row 63, or 64); at least 1 of 4 is empty
     or an affordability question.

Measure: `raw/calls.jsonl`, same schema as v1; every question printed verbatim
in `RESULT.md`. Reported prompt tokens compared with the corrected fixture
estimate (`packet_tokens_estimate`, now including the system prompt).

Cap: 16 calls, max_tokens 192, one attempt each, 60s timeout per call, 8 minutes
wall, spend under $1.00 (qwen/qwen3.8-27b on OpenRouter).

Stop: success at 16 completed calls; abort on the 4th transport failure or on
any call reporting a model other than qwen/qwen3.8-27b.

Invalid if: more than 3 calls fail; any reasoning tokens or tool calls; any
condition-A packet whose `objections.pending` is not null; any packet sha256
differing from the corrected `s5i5_fixture.py` output for that boundary.

Result map:
  A positive and B positive → the packet preserves the diagnosis and prose
    suffices after a coherent closure; next is a controlled Kaggle card for v9.
  A positive, B negative → keep the packet; add a mechanical reopen cooldown
    on a closed claim_ref to ObjectionLedger and retest B under a new card.
  A negative → the packet is too thin; add the last three commitments'
    rationales to the packet and retest under a new card before any Kaggle run.
  Ambiguous → report per boundary; no promotion.
  Reported tokens outside 0.8-1.3x of the corrected estimate → restate the
    size claim in CONTRACT.md with the measured divisor.

Outputs: `analysis/critic-closure-contract-20260912/packet-live-probe-v2/raw/`.
