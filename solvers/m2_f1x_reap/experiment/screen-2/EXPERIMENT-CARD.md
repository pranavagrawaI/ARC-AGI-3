# m2-f1x-reap screen-2: c16 (2026-10-25)

**Question:** On the F1x+REAP448 stack, do 16 streams at the same 116K input raise aggregate decode without overflowing KV, and play no worse than the 14-stream screen?

**Why:** Throughput is the only lever with a measured slope (levels ≈ 0.35 × speedup). Screen-1 at 14 streams ran KV p90 0.80. The hidden run keeps every slot busy (110 games / 14 slots), so aggregate tok/s is what pays there. Pranav asked for a GPU run today (2026-10-25: "We should run something, let's not waste another day"). This one is a ticket and a mechanism read, not a medal lever.

**Arm:** `screen-2/c16` = `m2-f1x-reap16-screen2` v1, built by `build_c16.py` from the screened and drawn screen-1 v1 (sha 9d634d6a…). It changes exactly MAXREQ, CUDAGRAPH_MAXBS, and ARC3_MAX_ACTIVE_STREAMS 14→16, and MAMBA_CACHE 84→96 (6 slots per stream). Everything else is byte-identical.

**Reference:** screen-1 merge, 80 levels, 823 tok/s, KV p90 0.80, 0 retractions, READY 676 s. There is no paired control: same-code runs show no run effect (F = 1.08), and one 60-min replicate sd is about 9 levels.

**Prediction (fixed before launch):** pool about 1.42M tokens (12 more Mamba slots × ~4.2K taken from the 1.47M pool). KV p90 0.88–0.95. Aggregate about 850–880 tok/s (+3–7%); per-stream about 54–56 tok/s. Levels: anything in 70–90 is noise.

**Pre-registered rules (`analyze_c16.py`):**
- Mechanism checks:
  - 448/512 pruning receipts.
  - Kernel log shows "priority gate active: 16 concurrent streams", and running requests reach 16.
  - Aggregate tok/s over minutes 0–60 ≥ 1.04 × 823 = 856.
  - KV p90 ≤ 0.95.
  - Retractions ≤ 5.
  - Preload NameErrors = 0.
  - out16k applied.
  - 25 games, none crashed.
- No-harm: levels ≥ 80 − 7 = 73.
- **PASS:** every mechanism check holds and no-harm holds. Queue c16 v1 for the first free day from 2026-10-26, without `requires`. It is an extra ticket in rotation with F1x+REAP14 and the control, not a replacement. Finals are picked by top public draw.
- **MECHANISM_FAIL (KV or retractions):** 16 streams overflow at 116K, so c14 is the ceiling for this stack. Nothing is queued.
- **MECHANISM_FAIL (throughput < 856):** the step time does not scale. The stream lever on this stack is closed. Nothing is queued.
- **NO_HARM_FAIL:** nothing is queued; a decision is needed. Below 73 is about 0.8 sd under screen-1, so it is weak evidence of harm.

**Cap:** one launch, hard cap 6,000 s, one GPU slot, about 1.3–1.7 GPU-h of the fresh 30 h week. No retry except for a proven infrastructure failure. The experiment makes no submission.
