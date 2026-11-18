# packet-live-probe-v1

Question: Does a reasoning-off critic reading the v9 compact packet, instead of
the v8 full transcript, still locate the S5I5 objective-proxy gap (that filling
the bottom bar is an untested inference about level completion), and does it
refrain from re-raising that objection once the ledger shows it closed?

Posture: evaluation of the v9 packet; falsification of the assumption that
"closed claims are not restated" can be left to prompt text.

Contrast: two packet conditions, live, same model and endpoint.
  A. packet with an empty ledger at each of the seven recorded S5I5 review
     boundaries (critic:1-7), two samples each — 14 calls.
  B. packet at the critic:7 boundary with the ledger showing critic:3 closed as
     `probe` (actions_budgeted 46, step 14), three samples — 3 calls.
  C. packet at the critic:7 boundary with the ledger showing critic:3 closed as
     `withdraw` retiring CLICK 32 32, three samples — 3 calls.
  The v8 recorded questions at the same boundaries are a reference, not a
  controlled arm: they came from Kaggle-local vLLM, not this endpoint.

Prediction:
  A. At critic:3, :5, :7 at least 4 of 6 samples ask an objective-gap question
     (links bar/fill/click to level completion, win, goal, or objective). At
     most 2 of 14 samples are mechanics questions (which cell, why a delta
     differed). At least 12 of 14 responses validate and bind.
  B and C. At most 1 of 6 samples re-raises the bar-as-objective claim.
  Token calibration: endpoint-reported prompt tokens fall within 0.7-1.4x of
  the fixture's chars/3.6 estimate for every call.

Measure: `raw/calls.jsonl` — per call: condition, boundary, sample, packet
sha256, reported prompt/completion tokens, wall seconds, raw response, validate
and bind outcome, and the keyword class {objective_gap, mechanics, other, empty}.
The keyword rule is in `probe.py` and is coarse; every question is printed
verbatim in `RESULT.md` for a reader to reclassify.

Cap: 20 calls, max_tokens 192 each, one attempt per call, 60s timeout per call,
10 minutes wall, spend under $1.00 (qwen/qwen3.8-27b on OpenRouter).

Stop: success when 20 calls complete; futility none (all calls are informative);
abort on the 4th HTTP or transport failure, or if the endpoint reports a model
other than qwen/qwen3.8-27b.

Invalid if: more than 3 calls fail; more than 30% of nonempty responses fail to
bind a claim_ref; any call used reasoning (reasoning tokens > 0) or tools; a
packet's sha256 differs from the one `s5i5_fixture.py` produces for that
boundary and ledger state.

Result map:
  A positive → the packet preserves the diagnosis; the next step is a controlled
    Kaggle card for v9 versus v8, not another local iteration.
  A negative (objective gap lost or mechanics ≥ 3) → packet too thin or too
    grid-shaped; add the last three commitments' rationales, drop changed_bbox,
    and retest under a new card before any Kaggle run.
  B/C negative (re-raises ≥ 2 of 6) → prose does not hold; add a mechanical
    cooldown in ObjectionLedger refusing to reopen a closed claim_ref for N
    steps, and retest under a new card.
  Ambiguous (predictions split) → report per boundary; do not promote v9 until
    a new card resolves the split.
  Token calibration outside 0.7-1.4x → rerun `s5i5_fixture.py` with the
    calibrated divisor and restate the size claim in CONTRACT.md.

Outputs: `analysis/critic-closure-contract-20260912/packet-live-probe-v1/raw/`
  (`calls.jsonl`, `packets/*.json`, `STARTED.json`, `FINISHED.json`).
