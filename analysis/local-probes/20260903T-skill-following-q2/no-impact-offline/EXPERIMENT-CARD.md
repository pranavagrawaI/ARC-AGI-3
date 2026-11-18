# 20260903T-no-impact-offline-q2d
Question: on the recorded q2d ls20 trajectory, does a HUD-band no-impact predicate — discovered from change frequency, not from colour 11 or row 61 — separate blocked/timer-only steps from playfield steps?
Posture: verification of a zero-token host predicate; diagnosis of wasted batches
Contrast: none (one trajectory, one game). Feasibility, not a causal claim about RHAE.
Prediction: rows that change on ≥80% of pairs are a thin band; HUD-only ∪ blocked covers the n∈{0,2,4} steps; at least one HUD-only step is GAME_OVER (so HUD-only ≠ safe no-op)
Measure: `score.py` against `../results/nemo_solver/20260903_160733_ls20_mdfiles_q2d/` env_step grids. Counts in RESULT.md.
Cap: this trajectory only; 0 API calls; not applicable spend
Stop: after one pass over 123 pairs
Invalid if: fewer than 120 consecutive grids, or grids not 64×64
Result map: clean split + death exception → host-side predicate is writable as a later card (do not ship from this file); split needs ls20-specific rows/colours → do not ship, contestant prior; offset still binding in this trace → spend on the pipe format; offset not binding → drop it; WikiSkill is not implied by either result
Outputs: this directory (`score.py` stdout, RESULT.md)
