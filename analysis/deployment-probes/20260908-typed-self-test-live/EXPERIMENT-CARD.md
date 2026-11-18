# Typed self-test live discriminator

Question: Can Qwen 3.8 27B use the new host-enforced experiment interface to invent and execute a genuinely falsifiable regional test on the previously failed `competing_regions` case?

Posture: one-case live feasibility test of the materially changed interface, targeted at the earliest prior failure boundary.

Contrast: the same fresh synthetic `competing_regions` target that failed the prior reliability screen, now under the frozen typed-prediction interface. This is a before/after case comparison, not a general reliability estimate or a causal comparison isolated from all other prompt/interface changes.

Prediction: before action one, the model records two or three hypotheses whose `changed_bbox` predictions are distinct typed coordinate values; the host accepts the experiment. After the real successor, the host result exactly records the observed bbox and classifies the hypotheses. Within two further Actor calls, the model inspects that scored result, records another valid typed experiment, and submits an evidence-dependent second action.

Measure: raw relay receipts, forwarded requests, provider responses/errors, worker model calls, experiments, experiment results, actions, transitions, reasoning usage, reservation, wall time, and cleanup state. Inspect the first experiment's metric and predictions, the first host-scored result, and the second Python cell and experiment.

Cap: one case; at most four provider Actor admissions and two submitted actions; at most two Actor calls per decision; zero summarizer or deep-investigation requests; 240 seconds per request and 600 seconds total; USD 0.60 reservation. Qwen `qwen/qwen3.8-27b` via `coreweave/fp8`; temperature 0.2, top-p 0.95, top-k 20, 16384 completion tokens; reasoning explicitly disabled. No fallback, forwarding retry, or run retry.

Stop: finish after action two; stop on a two-call decision breach, source/card drift, reasoning drift, provider or transport failure, missing usage, unresolved admitted work, spend/time cap, or failed retirement. Preserve partial evidence. No automatic relaunch.

Invalid if: the final frozen runner does not first pass a real packaged-worker local preflight; candidate, runner, or card changes after live start; transition provenance is ambiguous; an admitted request lacks terminal accounting; reasoning state is unknown; or cleanup/retirement is unresolved.

Result map: full predicted sequence passes -> retain the typed interface and proceed to a fresh multi-case reliability card; valid behavioral failure -> inspect the earliest failed boundary and do not claim interface reliability; technical invalidity -> preserve evidence and require a new card before more provider contact.

Outputs: immutable card, PRECOMMIT.json, local preflight, raw live evidence, RESULTS.json, and RESULT.md under `analysis/deployment-probes/20260908-typed-self-test-live/`. No full ARC game, Kaggle run, upload, submission, or automatic production promotion.
