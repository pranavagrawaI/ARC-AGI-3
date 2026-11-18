# Procedural transfer reliability screen

Question: Does Qwen 3.8 27B reliably reuse the literal two-decision CodeAct procedure across five predeclared variations without copying its conclusion?

Posture: bounded reliability screen after one valid no-op target passed the feasibility gate.

Contrast: five fresh synthetic targets, fixed hardest-first order: `changed_action_mapping`, `competing_regions`, `reshaped_target`, `translated_variant`, `salient_noop`. Each target changes one decision-relevant property relative to the demonstration. Every target uses a fresh worker and context. The model receives the same frozen demonstration and diagnostic workflow. No baseline arm is included, so this estimates procedural success under this interface rather than demonstration causality.

Prediction: at least four of five valid targets pass, and both `changed_action_mapping` and `competing_regions` pass. A case passes only when action one occurs within two Actor calls after a real four-argument `set_experiment`, the host successor discriminates its recorded alternatives, and action two occurs within two more calls after measuring that successor. Its second conclusion and action must respond to returned evidence, make no broader claim than the pixels support, and avoid blindly copying the demonstrated translation conclusion.

Measure: per case preserve every relay receipt, forwarded request, provider response/error, worker model call, experiment, action, transition, helper, and procedure. Record calls per decision, exact Python cells, first experiment alternatives, host successor, second experiment and rationale, reported reasoning usage and cost, reservation, wall time, and cleanup state. Aggregate valid passes over all five predeclared cases. Missing provider usage is unknown and invalid, not zero.

Cap: five cases, at most four provider Actor admissions and two submitted actions per case; at most twenty provider admissions total; zero summarizer or deep-investigation requests; 240 seconds per request; 600 seconds per case and 2400 seconds total; USD 3.00 cumulative reservation. Qwen `qwen/qwen3.8-27b` via `coreweave/fp8`; temperature 0.2, top-p 0.95, top-k 20, 16384 completion tokens; reasoning explicitly disabled. Provider price ceilings: USD 0.42/million prompt and USD 3/million completion. No fallback, forwarding retry, or case retry.

Preflight: before provider contact, the final frozen runner must drive every case through the real packaged adapter and worker using local responses. For each case preserve both rendered prompts, prove exact demonstration/workflow presence, absence of mandatory helper/world-model instructions, disabled summarizer, real submit-host-transition-submit boundary, case-specific successor, and zero helper/procedure artifacts. Separately preserve passing call-limit, expired-deadline, and rejection-retirement checks. Freeze runner/card/candidate hashes only after all preparation changes. Source-only assertions do not qualify this gate.

Stop: finish each valid case after action two. A two-call decision ceiling breach is a behavioral failure for that case, followed by clean retirement and the next case. Stop the entire matrix before another case on source/treatment drift, summarizer/deep request, reasoning drift, missing usage, provider or transport failure, unresolved admitted work, total time/spend cap, or failed retirement. Preserve partial results. No automatic relaunch.

Invalid if: preflight is incomplete; card or frozen source changes after matrix start; case identity/successor provenance is ambiguous; an admitted request lacks terminal accounting; reasoning state is unknown; or worker/transport retirement is unresolved. Classify validity before behavior.

Result map: 4/5 or 5/5 with both required cases passing supports reliability at this small screen and advances the compact primitive into a fresh production comparison candidate. Three or fewer passes, or failure of either required case, stops prompt-demonstration polishing at this checkpoint. Technical failure preserves completed cases and requires a new card for any additional provider contact.

Outputs: immutable card, PRECOMMIT.json, local preflights, per-case raw evidence, aggregate RESULTS.json, and RESULT.md under `analysis/deployment-probes/20260908-procedural-transfer-reliability/`. No full ARC game, Kaggle run, upload, submission, or automatic production promotion.
