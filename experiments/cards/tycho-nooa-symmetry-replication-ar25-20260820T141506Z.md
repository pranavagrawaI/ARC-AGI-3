# tycho-nooa-symmetry-replication-ar25-20260820T141506Z
Question: Does ar25 seed-0 level 0 replicate after fixes?
Posture: replicate.
Contrast: `symmetry-uptake-ar25-20260820T101248Z`; same game, seed, model, prompts, reasoning, 16k output, 20 actions; reviewed fixes plus advisory parser repair. `135428Z` was invalid at action 3 when bare builder CLICK crashed before actor control.
Prediction: complete by action 17 after actor symmetry reach; builder resolves the mandatory completion checkpoint.
Measure: summary; successful symmetry calls; actions; completion evidence; checkpoint required/resolved; final `outcome()`.
Cap: 20 actions; 60m; 1 attempt/model; <=$4.96.
Stop: cap, terminal, 60 min, provider failure, or spend refusal.
Invalid if: unavailable; pre-builder crash; symmetry absent; every reach errors.
Result map: completion+mechanism -> local replication; completion only -> behavioral; no completion -> diagnose; invalid -> repair.
Outputs: `runs/tycho-nooa-symmetry-replication-ar25-20260820T141506Z/`.
