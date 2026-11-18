# m2-evidence-pair1 result (2026-10-09)

Invalidity: none. Both arms COMPLETE, 25 games each, settings matched (7254 s/game, concurrency 25, priority gate 10).
"M2 evidence installed" appears in the treatment log only. Raw: analysis.json (analyze_pair.py, written before data existed).

Verdict per card: INCONCLUSIVE.
| | control | treatment | Δ |
|---|---|---|---|
| mean final_score | 40.11 | 45.99 | +5.88 |
| levels | 96 | 105 | +9 |
| per-game score sign | | | 10 better / 8 worse / 7 tie |

Context: control r1 (2026-10-01, same notebook) scored 52.48 / 107. This control drew low. Against the mean of the
two full-cap controls (46.3 / 101.5), treatment is ~0 on score and +3.5 on levels. Per-game swings of 3–5 levels
(tr87 1→5, tu93 3→8, s5i5 2→6, tn36 2→5; cd82 6→4, g50t 3→0, ls20 4→2) match the swings between identical controls
(tr87 6 vs 1, tu93 9 vs 3), so they are not attributable.

Mechanism (explanatory only):
- Delivered: entries in 25/25 games (145 total), 12 revisions, re-shown 1–27 times per game. The keyword-only fix held; no positional misuse.
- Tokens on final unsolved levels: 1.3M (29%) treatment vs 1.8M (41%) control; this control r1 was 1.4M (33%).
- NameErrors in transcripts: 37 vs 45.
- ~40% of entries (58/145) describe an earlier level, i.e. level-up lessons; the rest were written mid-level.

Charge: 7,886 + 7,987 s session time = 4.41 h. Experiment total 4.86 of 6.6 GPU-h; 1.74 h remain.
The quota delta over the window (25,466 s) includes claude-hillclimb's likelihood-probe kernels launched in the freed slots.

## Cost of the treatment (2026-10-09, throughput/serve_stats.py, throughput/stats-15min.txt)
| | control | treatment |
|---|---|---|
| server decode, mean gen tok/s (SGLang Decode batch lines) | 686 | 679 |
| mean running requests / accept length | 9.00 / 2.69 | 8.98 / 2.67 |
| generated tokens (benchmark solver_note, authoritative) | 4.58M | 4.48M |
| generated tokens per level | 47.7K | 42.7K |
| uncached prefill tokens (server) | 10.9M (93% cached) | 12.7M (92% cached) |
| GPU session time | 7,886 s | 7,987 s |
No serving cost: decode throughput, concurrency and speculative acceptance match within 1%.
The extra uncached prefill (+1.8M) is mostly one burst at minutes 15–30 (2.86M vs ~1.1M), with no effect on decode rate.
Prompt overhead from re-shown records is ~105K tokens (0.07%).
CORRECTION: an earlier same-day reading of "12% fewer generated tokens" (4.03M vs 4.58M) summed requests.jsonl from a
treatment output truncated by the interrupted downloads (1.26 GB vs 1.40 GB). Use benchmark.json solver_note for token
totals; request logs are complete only after a verified full download.
