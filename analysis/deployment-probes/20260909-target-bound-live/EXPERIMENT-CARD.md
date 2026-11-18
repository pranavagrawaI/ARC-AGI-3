# Target-bound regional self-test live gate

Question: Does the target-bound host contract force Qwen 3.8 27B to formulate and use a genuinely regional discriminator on the previously failed `competing_regions` case?

Posture: live feasibility gate after local regression and packaged-path qualification.

Contrast: the same synthetic `competing_regions` case and model settings as the valid 2026-09-08 behavioral failure, with the changed contract requiring an epistemic target compatible with its metric and rejecting directional/regional language outside `target=region`, `metric=changed_bbox`. This is a targeted before/after probe, not a general reliability estimate.

Prediction: action one occurs within two Actor calls after a host-accepted experiment with `target=region`, `metric=changed_bbox`, and distinct coordinate predictions. The host scores the exact bbox. Within two further calls, the model inspects that result, records another valid target-bound experiment, and submits an evidence-dependent second action.

Measure: raw relay receipts, requests, responses/errors, model calls, experiments, results, actions, transitions, reasoning usage, cost, wall time, and cleanup. The first target, metric, predictions, and score are decisive.

Cap: one case; four Actor admissions maximum, two per decision; two actions; zero summarizer/deep calls; 240 seconds per request and 600 seconds total; USD 0.60 reservation. Qwen `qwen/qwen3.8-27b` via `coreweave/fp8`; temperature 0.2, top-p 0.95, top-k 20, 16384 completion tokens; reasoning disabled. No fallback or retry.

Stop: finish after action two; stop on call breach, drift, reasoning drift, provider/transport failure, unresolved admission, spend/time cap, or failed retirement. Preserve partial evidence; no automatic relaunch.

Invalid if: frozen real-worker success, call-limit, deadline, and rejection preflights are incomplete; card/candidate/runner changes after start; provenance is ambiguous; admitted calls lack terminal accounting; or cleanup is unresolved.

Result map: valid pass -> create and execute a fresh five-case broader reliability card; valid behavioral failure -> stop without the broader run and diagnose the earliest boundary; technical invalidity -> preserve evidence and require a new card.

Outputs: artifacts under `analysis/deployment-probes/20260909-target-bound-live/`. No Kaggle run, upload, submission, or production promotion.
