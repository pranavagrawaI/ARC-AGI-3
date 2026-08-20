# tycho-nooa-symmetry-replication-ar25-20260820T133152Z
Question: Does ar25 seed-0 level 0 replicate after the reviewed fixes?
Posture: replication.
Contrast: `symmetry-uptake-ar25-20260820T101248Z`; same game, seed, model, 20 actions, prompts, reasoning and 16k output; reviewed fixes only. `123220Z` was invalid at action 0.
Prediction: complete by action 17 after actor symmetry reach; completion becomes a mandatory builder checkpoint.
Measure: summary; uptake; actions; `level_0/level_complete.json`; checkpoint required/resolved; final `outcome()`.
Cap: 20 actions, 60 min, one attempt/model, <=$5.67 remaining credit.
Stop: cap, terminal, 60 min, transport/provider failure, or spend refusal.
Invalid if: unavailable; pre-builder crash; symmetry absent; all reaches error.
Result map: completion+mechanism -> local replication; completion alone -> behavioral only; no completion -> diagnose divergence; invalid -> repair infrastructure.
Outputs: `runs/tycho-nooa-symmetry-replication-ar25-20260820T133152Z/`.
