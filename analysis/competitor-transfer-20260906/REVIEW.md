# Competitor lessons grounded in the current solver

Reviewed 2026-09-06. Recommendation: improve action interruption and model selection before adding a larger agent architecture. These are review findings and proposed experiments; no solver source was changed or GPU run launched.

## Candidate and evidence

The current reviewed successor is `deployment/kaggle/q6-summary-boundary-20260906`, contract `24c291bc6956a96dce361a52a713c0b3c4e4208da7e464d735d2b310834976a9`. Its README and local qualification record say it has not launched. Its prediction-feedback parent did run and aborted on a summarizer integrity false positive. The successor repairs that boundary while retaining the prediction workflow. The MTP integration is a separate serving experiment, not evidence of better game judgment.

The actual import chain is packaged `nooa_worker.py` -> packaged `headless.py` -> adjacent `chassis/solver_agent.py` -> `MdArcSolverAgent`. The older `solvers/tycho_nooa` native-tool agent and the 0158 NOOA snapshot are not the actor in this chain. The current actor is one CodeAct agent with helper files, notes and a summarizer; it is not the older actor/optional-builder arrangement.

Inspected actor initialization, instructions, state access, submission gates, model feedback, host batch execution, helper persistence, and saved public transition/model-call evidence. Re-ran the successor's documented six-file qualification suite: **51 passed**, 17 existing `set_dynamic` deprecation warnings, 24.29s. Scripted-model checks establish interfaces, not gameplay value.

`reproduce.py` independently recomputes the key trace counts and a synthetic consistency counterexample. Run from repository root:

```bash
deployment/kaggle/.venv/bin/python analysis/competitor-transfer-20260906/reproduce.py
```

`EVIDENCE.json` preserves input hashes. The parent trace has 60 completed model calls, 19 executed actions, 1,256.29 seconds of model time, and zero completed levels before abort. Turns 1 and 2 required 19 and 13 calls. These are descriptive observations from an invalid full run, not a treatment effect or representative benchmark average.

## 1. Interrupt an executable plan when its prediction fails

**External lesson:** Son Pham's Duck fork reports benefits from stopping actions that only change its HUD band. Its report also documents regressions from added scaffolding. This suggests a cheap stop inside the existing execution loop, rather than another model conversation. [Primary ablations](https://github.com/sonpham-org/arc-3#where-things-stand).

**Current code:** `chassis/host_contract.py:217` stops for terminal, level change, or exactly unchanged grid. `nooa_worker.py:220` consumes this result and otherwise retains the queued actions. Executable consistency is evaluated later in `chassis/prediction_feedback.py:144`; the worker does not compare that frozen prediction before releasing the next queued action.

**Observed failure:** Parent turn 5 submitted UP, LEFT, LEFT, LEFT. The frozen model first disagreed with reality at step 7, the first LEFT. Steps 8 and 9 executed anyway. Each LEFT changed two pixels, so the host classified all three as `move`. This establishes two actions after a contradicted plan prefix, not that those two actions were provably useless under every interpretation of the game.

**Transfer:** Evaluate the committed prediction prefix after each executed action. On mismatch, cancel the remaining queue and return the observation to the actor. Keep the frozen encoder/source and public-frame provenance. Do not revive the old row-frequency HUD suppression: prior local evidence showed it could hide real object changes.

**Small test:** Replay this prefix through the worker and show that steps 8 and 9 are no longer released, while a correct multi-action prefix still runs without additional LLM calls. Then compare live score, interrupted batches and model-call cost at equal budgets. A faulty encoder can cause excessive interruptions, so action savings alone cannot establish benefit.

## 2. Separate representation adequacy from prediction agreement

**External lesson:** Full observation access matters in the Duck lineage. Tufa keeps observations available in a persistent Python REPL and uses images for perception. The transfer is to preserve access to evidence outside the current abstraction. [Tufa overview](https://tufalabs.ai/research/duck-harness/).

**Current code:** `prediction_feedback.py:170` applies the same frozen encoder to the observed successor and compares the result with its prediction. `:181` marks any matched checked prefix batch-eligible. `:189` checks source identity, not action/region coverage or objective validity. The skill explicitly asks for a compact latent state. These are useful consistency checks, but a model can omit the very variable the next decision needs.

**Reproduction:** A synthetic constant nonempty encoder receives `matched` even when every pixel changes; its model gate then accepts a 20-LEFT proposal after an UP check. This probes the model gate only, not the complete submit path. The live agent did not supply this synthetic helper. It demonstrates why consistency cannot establish representation sufficiency.

**Transfer:** Add separately committed observable predictions tied to raw public evidence for the current decision, and keep unexplained changes visible. Compare movement prediction, extraction stability, and predicted progress separately. Use the model's chosen observable and scope; do not hardcode object roles or game coordinates. Restrict plan confidence to relevant tested behavior, rather than treating one agreement as general model validation.

**Small test:** Use synthetic missing-variable and same-latent/different-outcome fixtures, plus held-out public transitions. A representation should be judged by prospective decision-relevant predictions, not rewarded for silently deleting difficult fields. Do not require a complete renderer or full-game reconstruction before allowing exploration.

## 3. Retain and score competing executable candidates

**External lesson:** Franzen's paper separates diverse candidate generation from probability-based selection, with augmentations throughout. This is confirmed prior ARC work; his current ARC-AGI-3 policy remains unknown. [Paper](https://proceedings.mlr.press/v267/franzen25a.html), [code](https://github.com/da-fr/Product-of-Experts-ARC-Paper).

**Current code:** `prediction_feedback.py:108` selects one preferred helper, and `:148` evaluates only the latest commitment. `solver_agent.py:1395` overwrites helper source. Old source survives in committed prediction logs, but the active interface has no automatic candidate scoreboard or repair-regression replay. The skill allows competing explanations in prose, so this is a missing operational capability, not an inability to express alternatives.

**Transfer:** On a persistent contradiction, retain the old hypothesis and generate at most two distinct repairs in one call. Evaluate all candidates on the same observed transition set and the same frozen observable contract; show counterexamples and runtime. If alternatives make different actionable predictions, let the actor select a short distinguishing probe. A local bounded rollout can rank short plans where the model has earned enough trust.

Do not rank different encoders by their own latent accuracy: an empty abstraction would win. Do not treat an LLM preference score as game evidence. Symmetry/augmentation is appropriate only with transformed actions and coordinates and an explicit equivalence assumption; screen directions can have distinct semantics.

**Small test:** Freeze a few real failure frontiers without successors visible to the candidate generator. Compare single repair with bounded candidate selection under equal token and action budgets. Measure whether a useful candidate was generated, whether it was selected, regressions on earlier transitions, and fresh-action success. This tests the transferred mechanism, not Daniel's undisclosed system.

## 4. Make routine turns cheap without making exploration careless

**External lesson:** Son Pham reports that state-graph queries consumed more tokens/action without triggering the intended rejection, and early mandatory bookkeeping substantially increased cost. Tufa's minimal REPL is the relevant architectural example. These results support measuring which computation changes decisions, not assuming every graph or world model is harmful. [Ablations](https://github.com/sonpham-org/arc-3#where-things-stand), [Tufa](https://tufalabs.ai/research/duck-harness/).

**Current code/evidence:** `solver_agent.py:630` already asks for repair/test within three calls after a mismatch; proposing that prompt again would add nothing. The main instructions still require per-turn memory work (`:1469` and the skill's Long-Term Memory section). Observation reads, helper reloads and note writes are actor-managed. In the retained trace, repeated trajectory interpretation and a direct `write_helper` tool-name attempt are visible among the early calls. The recorded 60 calls/19 actions is about 3.16 calls/action and 66.1 model seconds/action, with much larger early-turn outliers.

**Transfer:** Assemble a small current-state packet with latest feedback, recent before/after references, helper identity and unresolved experiment. Keep full arrays accessible in Python and permit targeted retrieval. Make notes record changed beliefs, not mandatory restatement. Combine inspect/repair/test work within an existing cell. Budget easy continuation separately from new-rule inference; expose elapsed decision budget before the outer worker timeout.

**Small test:** Compare the current loop with a cheaper routine-turn path while holding model and serving settings fixed. Track calls, generated tokens, progress and score per wall-clock budget. Do not force arbitrary actions merely to hit a call cap. Keep MTP attribution separate from this intervention.

## 5. Turn failures into a reusable learning set

**External lesson:** The public world-model traces dataset packages state/action/successor examples with varied provenance. It makes transition-level evaluation and eventual training concrete. The currently visible discussion does not verify the pasted Third Intelligence training quote or team attribution; do not depend on that claim. [Dataset card](https://huggingface.co/datasets/fredericowieser/arc-agi-3-wm-traces), [discussion](https://huggingface.co/datasets/fredericowieser/arc-agi-3-wm-traces/discussions/2).

**Current code:** `host_contract.py:183` already records before/action/after; `prediction_feedback.py:121` stores the exact candidate source/hash before acting. These are valuable assets. The online feedback path only checks the newest committed batch; it does not automatically replay a repaired model over the earlier counterexamples or create a held-out training/evaluation split.

**Transfer:** Build a small public-trace corpus of observation, hypothesis version, committed prediction, actual successor, and later correction. Use it first to evaluate candidate selection and repair regressions. If weight training is desired, train only from admissible public fields and independently supported targets. Split by game family and time; future frames must stay invisible at the evaluated decision frontier. Failed or unsupported agent explanations are not labels. CONTEXT.md excludes oracle trajectories and privileged labels from main contestant training.

**Small test:** Require a repair to improve fresh predictions without losing previously demonstrated behavior. Later compare fine-tuned and unchanged weights under the same harness and held-out game regime. Nothing in this review establishes that fine-tuning already exists in the reviewed runtime or that it would outperform a harness change.

## Order and exclusions

Start with per-step interruption, then bounded candidate repair with independent observable checks, then lower routine-turn cost. Build the reusable evaluation corpus alongside those trials. This order addresses observed failure before a major training commitment.

Full frames and executable models already exist; they are not features to add from scratch. Current visual input defaults to off and the packaged trajectory API explicitly lacks animation, despite inherited handle documentation mentioning it; these are distinct perception experiments to qualify if decision evidence points there. Do not blindly switch models to enable vision.

Mostik latent bridges, Kyutai encoders and Jinbo rollback remain inspiration without enough submission evidence or a demonstrated local bottleneck to outrank these changes. This pass provides no estimate of leaderboard score gain. No score-sensitive choices were based on privileged game mechanics.
