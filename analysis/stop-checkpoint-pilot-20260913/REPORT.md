# Stop-checkpoint pilot: no observed improvement

The completed pilot does not support adding this checkpoint to the solver. Three paired short continuations on one archived LS20 state produced no level completions in either arm. The checkpoint arm executed fewer actions in every pair and voluntarily stopped once. More actions are not inherently better, but neither the primary progress measure nor early-stop measure improved.

## Observed results

| Seed | Baseline actions | Checkpoint actions | Baseline levels gained | Checkpoint levels gained | Voluntary stop |
|---|---:|---:|---:|---:|---|
| 41 | 6 | 4 | 0 | 0 | Neither |
| 42 | 6 | 5 | 0 | 0 | Neither |
| 43 | 5 | 3 | 0 | 0 | Checkpoint |
| Total | 17 | 12 | 0 | 0 | Baseline 0; checkpoint 1 |

Each arm used 12 model calls across its three continuations. All 24 calls completed through DekaLLM using `qwen/qwen3.8-27b`, with reasoning disabled. Provider receipts total $0.0307324, 177,931 prompt tokens and 7,382 completion tokens. Live execution took 63.015 seconds. There were no retries, fallback calls, Kaggle runs or competition submissions.

Primary evidence: `RESULT.json`, `trajectories.jsonl`, `attempts.jsonl`, and the complete request/response files in `raw/`.

## What the traces show

The checkpoint elicited explicit assumptions and alternative outcomes, but these were not consistently discriminating experiments. Several continuations treated another blocked UP action as confirmation that vertical movement works ONLY at columns 34–38. Testing one blocked position cannot establish that universal restriction. For example, the first checkpoint continuation, request 05, says its UP probe validated the corridor-only assumption.

There is an observed counterexample in the baseline continuation at seed 43. After LEFT, LEFT, the color-12 block component occupied rows 45–46 and columns 19–23. UP moved it to rows 40–41 at the same columns, then subsequent UP actions moved it farther. Vertical movement therefore occurred outside columns 34–38. This is an observation from executed actions, not a source-code-derived mechanic supplied to the model. It does not establish the full movement rule or win condition.

The checkpoint continuation at seed 43 stopped with three actions still available. Its final rationale recognized that it could move upward, then miscounted required actions and reported the revised strategy instead of executing it. The final request explicitly said one call and three actions remained. See `raw/23-request.json` and `raw/23-response.json`.

Both arms also had discrepancies between prose and executed actions. In baseline request 01, the rationale says it will move RIGHT, while the tool submits UP, UP. In checkpoint request 05, the rationale and prediction describe RIGHT, while the tool submits LEFT. All submitted actions were syntactically valid. These are model decision inconsistencies, not host direction remapping: the audit re-executed every recorded action against the local environment.

These examples support a narrower diagnosis than “the agent needs a reminder to experiment.” The model can produce experiment-shaped language while overgeneralizing from evidence, misreading transitions, or issuing a different action. The qualitative review was conducted after the run and is illustrative, not a preregistered count of useful probes.

## Contrast and limitations

The starting case was a real premature stop from the historical objection-closure run. Its final reflection claimed budget exhaustion after seven actions and retained unresolved questions. Both arms received the same retained reflection, the same complete lossless initial grid and chronological transition diffs, and corrected pilot-budget telemetry. Only the checkpoint arm received the assumption-to-probe instruction.

Each continuation replayed the same seven-action prefix under seed 0 in the locally pinned `ls20-9607627b` environment. Three model sampling seeds and counterbalanced arm order supplied a small repeated contrast. The context was reconstructed, not restored exactly. Original Python variables, tools and the full model context were not restored.

This pilot used an action-only interface with four model calls and at most six new actions per continuation. It does not test the full code-capable incumbent, on-demand parser/model construction, the original stop-trigger integration, or held-out general competence. The lossless observation encoding may itself be difficult for the model to interpret; source fidelity is not model comprehension. The public historical case was already exposed to designers. No significance or general performance-regression claim is warranted from three pairs on one boundary.

Correcting budget telemetry and restarting the agent were shared changes. Baseline recovery therefore cannot be attributed to budget correction alone. Nevertheless, a special checkpoint was unnecessary for these baseline continuations to resume acting.

## Verification

Nine offline tests passed before launch. They covered both treatment routes at the actual HTTP boundary, action execution and caps, successful-response accounting, first-request HTTP 429 accounting, invalid/missing usage and cost, unexpected reasoning/model identity, replay divergence and lossless grid encoding. The synthetic HTTP failure test emitted a ResourceWarning when cleaning up its fake HTTPError; it passed and is not a live provider failure.

The post-run `audit_final.py` passed 114 checks. It reconciled the 24 admitted and completed calls, provider receipts, budget limits, matching initial contexts within each pair, treatment presence/absence, source/card hashes and all recorded action trajectories. It replayed each recorded continuation and verified before/after grids, state and level counts. Matching visible states plus pinned deterministic prefix replay supports this local comparison; it is not a general proof of hidden-state equivalence.

The frozen experiment implementation/card remain unchanged. `audit_final.py` is explicitly a post-run audit, separate from the frozen runner. The current deployment and other pre-existing worktree changes were not edited for this pilot.

## Decision

Keep the incumbent unchanged. Do not promote this checkpoint on these results. Before spending on a larger trial, the evidence favors testing exact observation interpretation and agreement between the proposed probe and submitted action. That is a proposed next question, not an implemented fix or an authorized second run.

Re-run the offline audit with:

    uv run python analysis/stop-checkpoint-pilot-20260913/audit_final.py

Do not rerun `pilot.py run`; the one-shot launch guard intentionally refuses it.
