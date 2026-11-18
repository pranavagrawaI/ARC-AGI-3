# Frozen solver + shallow MTP: integrated pilot

Question: Can the current prediction-feedback solver complete attributable actions
using shallow MTP in the real offline competition launcher/environment?

Candidate: exact embedded solver from the frozen q6-prediction-feedback longer-run
notebook (contract e9b0cae6…), unchanged model manifest, wheels, isolation, reasoning,
tools, retry behavior and lifecycle patch. Only engine change: EAGLE steps1/topk1/
drafttokens2 and metrics. Session/action limits are reduced for this pilot, not the
per-request reasoning allowance. The donor's ongoing baseline run is not modified.

Cap: one user-launched private RTX PRO 6000 session <=4800s (1.34 GPU-hours), one
ls20 game, <=10 individual actions,1800s gameplay admission after first action,
900s pending-worker window,3000s launcher envelope. Inherited engine startup and
installation fit within the session cap. No automatic relaunch or competition
submission. Monetary price unknown; no external model API is used.

Required evidence: actual MTP selection and speculative work in server artifacts;
81-file model verification; embedded worker/dispatcher identities matching the
manifest; exact model-call reconciliation including any summarizer calls; actual
action/transition records; host errors, timeouts and containment; orderly teardown.
Retain invalid tool outputs and recovery attempts. Do not call them useful actions.

Success: model requests reconcile, at least one valid committed environment action,
no integrity/lifecycle failure, and actual MTP work established. This selects a
candidate for a longer integrated comparison, not deployment or a speed claim.
Reaching the action cap with zero solved levels does not prove gameplay quality.

Stop/fail: manifest or hardware mismatch, worker/launcher timeout, engine crash,
invalid action authority, containment failure, unaccounted requests, or unresolved
teardown. Normal admission/action-budget exits are distinct from hard timeouts.
If no action is reached, do not relaunch automatically. Save and inspect artifacts.

Interpretation: this combines a locally tested solver candidate with an isolated
MTP winner; neither local tests nor the synthetic engine result qualify this
combination. No extra unknown-tool recovery patch is included: the captured host
already returned available-tool feedback. Tool reliability remains measured.
Short pilot does not guarantee summarizer coverage. Use the separate 14-request
replay for engine attribution; comparing this pilot with a100-action donor run
cannot establish a matched gameplay speedup.
