# tycho-nooa-symmetry-replication-ar25-20260820T123220Z
Question: Does ar25 seed-0 level 0 replicate after the reviewed fixes?
Posture: replication and instrumentation audit.
Contrast: `tycho-nooa-symmetry-uptake-ar25-20260820T101248Z`; same game, seed, model, 20 actions, prompts, and reasoning; reviewed fixes only.
Prediction: complete by action 17 after actor symmetry reach; completion becomes a mandatory builder checkpoint.
Measure: summary; symmetry uptake; transcript actions; `level_0/level_complete.json`; trigger checkpoint required/resolved; final `outcome()`.
Cap: 20 actions, 60 min, one attempt/model, <=$10.
Stop: cap, terminal, 60 min, transport/provider failure, or spend refusal.
Invalid if: unavailable; pre-builder crash; symmetry absent; all reaches error.
Result map: completion+mechanism -> local replication; completion alone -> behavioral replication; no completion -> diagnose divergence; invalid -> repair infrastructure.
Outputs: `runs/tycho-nooa-symmetry-replication-ar25-20260820T123220Z/`.
