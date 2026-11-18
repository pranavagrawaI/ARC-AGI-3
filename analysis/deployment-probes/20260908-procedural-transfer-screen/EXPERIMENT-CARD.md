# Procedural transfer screen
Question: Can the deployed Qwen 3.8 27B follow the literal two-decision CodeAct procedure on a changed action mapping within two Actor calls per decision?
Posture: bounded feasibility screen after deterministic fixture and rendering fidelity pass.
Contrast: none. One synthetic target has the demonstrated appearance but a changed mapping: RIGHT moves the marked cell upward, UP is a no-op, LEFT moves it downward, and DOWN moves it right.
Prediction: the Actor records two observable alternatives and submits action one within two calls; after the host returns the actual mapped successor, it measures that transition, narrows its update, and submits a non-blind second action within two more calls.
Measure: raw relay requests/responses and worker ipc/model_calls.jsonl, experiments.jsonl, actions.jsonl, transitions.jsonl, helpers directory, procedures.jsonl, plus RESULTS.json call counts and lifecycle outcome.
Cap: one fresh directory; at most four Actor requests, zero summarizer requests, two submitted actions, 240 seconds per request, 600 seconds total, and USD 1.00 reserved; explicit reasoning off; no fallback, forwarding retry, relaunch, full game, Kaggle run, or submission.
Stop: stop after action two; stop immediately on a fifth Actor admission, any summarizer request, reasoning token, provider/transport/worker failure, timeout, helper/procedure creation, malformed experiment signature, or cap exhaustion.
Invalid if: candidate/card identity drifts, either admitted request lacks a terminal response, reasoning-off is not verified, host transition provenance is missing, or request/worker records do not reconcile.
Result map: pass advances the compact primitive into a fresh production candidate for a controlled baseline comparison; blind copying or more than two calls in either decision ends demonstration polishing at this checkpoint; technical failure preserves the run and requires a new card before retry.
Outputs: immutable card, STARTED.json, raw requests/responses/errors, and worker artifacts under analysis/deployment-probes/20260908-procedural-transfer-screen/. No automatic promotion.
