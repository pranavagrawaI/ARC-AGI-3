# model-serving wave-2: B1 + c14 qualifying pass (2026-10-11)
Question: does the stack B1 (m2_b1 overlay) + c14 (wave-1 E2 serving settings) run valid on the public-25, and where
do its levels land? This pass qualifies the exact kernel version for hidden draw D3 (2026-10-12 00:30Z).
Posture: stacking two levers with positive mechanism evidence; the hidden draw, not this pass, is the instrument.
Design: `build_wave2.py` = the frozen B1 notebook (sha-checked against the one that ran in m2-b1 pair-1) plus exactly
the wave-1 E2 lines and the startup fail-fast. The builder asserts there are no other differences.
References: B1 pair-1 102/117 (mean 109.5, score 51.05); E2 107 / 45.77; control mean 101.5 / 46.30.
Prediction: 105–125 levels; gen tok/s ≈ 815 (as E2); acceptance ≈ 2.67.
Validity (gates D3): COMPLETE; 25 games; log markers "M2 B1 installed", "priority gate active: 14 concurrent
streams", "READY after"; no startup-exit marker; ≥ 95 levels (a floor against a silently broken stack, not a
benefit test).
Result map: valid → `d3_gate.py` writes evidence/GO-d3.json and the timer submits this version as D3 (once);
invalid → no D3; the slot falls back to nothing (no substitute arm without a new decision).
Cap: 1 session, 1 RTX PRO 6000, hard timeout 9,000 s; experiment ceiling raised to 8.0 GPU-h (wave-1 used ≈ 4.5 h).
Authority: Pranav, 2026-10-11 ("I agree with the next step": build B1 + c14 and arm tomorrow's submission).
