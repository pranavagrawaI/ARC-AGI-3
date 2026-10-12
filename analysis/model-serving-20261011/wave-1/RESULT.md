# wave-1 result (2026-10-11)

Both sessions ran COMPLETE on one RTX PRO 6000 each (E1 8,095 s, E2 7,975 s charged, about 4.5 GPU-h in total).
`analyze.py` produced `readout.json` in each run directory. The analyzer reproduces the known totals on m2_evidence's
treatment run (105 / 45.99; c1 107, pc 96).

| | E1 Swift | E2 c14 | c1 | pc |
|---|---|---|---|---|
| marker | "Swift view: 95 files from 2 parts", READY 574 s | "priority gate active: 14", READY 502 s | gate 10 | gate 10 |
| time-weighted gen tok/s (min 30–90) | 741.6 | **815.1 (+19.8%)** | 680.6 | 679.4 |
| accept len (median) | 3.08 | 2.67 | 2.70 | 2.71 |
| running (median) | 9 | 13 | 9 | 9 |
| KV usage median / p90 | 0.52 / 0.63 | 0.80 / 0.89 | 0.72 / 0.86 | 0.72 / 0.85 |
| retractions | 0 | 0 | 0 | 0 |
| total generated tokens | 4.35M | **5.20M (+20%)** | 4.34M | 4.40M |
| **levels** | **47** | **107** | 107 | 96 |
| mean final score | 10.48 | 45.77 | 52.48 | 40.11 |
| per game vs control mean (+/−/0) | 1 / 21 / 3 | 10 / 7 / 8 | | |
| tokens per co-solved level vs c1 (median ratio) | 1.43 (n = 45) | 1.02 (n = 96) | | |
| levels by minute 20/40/60/90/121 | 15/34/40/45/47 | 32/57/73/97/107 | 28/58/79/98/107 | |

The card's control throughput figure (704) came from a different estimator. With the analyzer's estimator, the two
controls read 680.6 and 679.4, so E2's gain is +19.8%.

## E1: NEGATIVE (−54.5 levels vs the control mean; rule ≤ −10)
Swift loaded and served cleanly: no repetition replacements (0 vs control 1), tool calls 2,459 vs 2,319, tracebacks
191 vs 182. But it fell behind from the first minutes (15 vs 28 levels by minute 20) and spent **more** tokens per level
solved by both (×1.43), the opposite of its model-card claim (−63% thinking tokens on GPQA). The transcripts are
coherent but slow: single thoughts of 28–30K characters, and wrong mechanics held longer (sb26 L2: 35K tokens vs 2K for c1).

Confound, unresolved: this checkpoint also quantizes linear_attn and self_attn to INT4. Intel's base checkpoint keeps
them BF16. The run therefore tests "Swift post-train + INT4 attention" against "base + BF16 attention". Separating them
would need a Swift checkpoint with BF16 attention in W4A16 format (none exists; it would mean a re-quant from the 360 GB
BF16 release) or the Swift NVFP4 build on a vLLM/modelopt stack. Prior: a 2× collapse from attention quantization alone
is unlikely, and the Aider evidence already showed Swift losing on the hardest cases. Drop Swift on this stack; do not
spend more GPU on it unless the board shows Swift submissions moving up.

## E2: ADOPT (rule: ≥ +15% tok/s, no KV thrash, levels ≥ control mean − 5)
The throughput mechanism held exactly as designed: +19.8% aggregate decode at equal acceptance, KV p90 0.89, no
retractions, and the extra tokens were spent (+20% generated). Levels 107 = c1, +5.5 over the control mean, inside
the ~11-level identical-run sd. That fits the 0.35 throughput elasticity prior (+20% tokens → about +7%). Tokens per
co-solved level are unchanged (×1.02), so the shorter 92K context did not make the model less efficient per level. The
score is flat (45.77 vs 46.30): more levels, slightly lower per-level efficiency.

## Next
- Fold concurrency 14 (with the 92K context) into the candidate stack (B1 / m2_evidence) rather than submitting E2 alone.
- Throughput still has headroom: KV p90 0.89 at 14 streams; sirikilohit ran 16 at a 69K context. A c16 arm is
  the natural follow-up only if stacked; alone it is below instrument resolution.

Hidden draws (same day): D2 unchanged control = **29.44** vs 23.87 for the byte-identical 2026-10-01 draw (+23%).
D1 m2_evidence 26.24 lies inside the control's own spread.
