John Tukey — The strongest overlooked explanation is the shared batching gate: **on never earned permission to batch its existing helper; off earned it immediately before each batch.** The evidence does not establish that the fallback sentence suppressed batching, and it does not justify retiring the broader “stuck/repeating an unsuccessful route” explanation. The first off batch was interrupted by a prediction mismatch after four moving actions; the unchanged UP was a subsequent single submission. Those are material corrections to RESULT-v3.md.

The clean negative remains: **no contingent proposal or continuation was used in either recorded trajectory.** This is no efficacy test of the controller’s continuation mechanism. It is one observed failure of uptake under the exposed interface, plus a descriptive comparison of two diverging solver trajectories. Literal “the controller was never invoked” needs qualification: the on dispatcher calls its continuation-check method even when no plan is pending.

I used the research workflow: reconstructed IPC before inspecting controller source, formed the gate/trajectory explanation before reading RESULT-v3.md or PRECOMMIT-censoring.md, then compared those documents against the reconstruction. The supplied request already disclosed their hypotheses, so this is independent recomputation, not a blinded analysis. No new Kaggle run, package change, or promotion was performed. The shared-board read failed with “unable to open database file”; no board conclusions were available. I used the report-work guidance to preserve findings and their limits here.

**Raw reconstruction.** Paths below are relative to this report. Off’s workspace is [ls20-jvu9mu52](v3-output/contingent-off/nooa-join-public3/solver-workspaces/ls20-jvu9mu52/ipc/); on’s is [ls20-qr97mem0](v3-output/contingent-on/nooa-join-public3/solver-workspaces/ls20-qr97mem0/ipc/). Counts come from parsed JSONL rows, not the assessor’s summary. Stops are in the parent solver-workspaces/game-stops.jsonl files. Initial before-states are equal across arms, all 20 steps are present in order, and each arm’s after-state equals its next before-state.

| Observed quantity | off | on |
|---|---:|---:|
| Stop reason | step_budget | step_budget |
| Executed solver actions / limit | 20 / 20 | 20 / 20 |
| Admission seconds | 3000 | 3000 |
| Recorded gameplay seconds | 1134.5446 | 878.8917 |
| Final state | NOT_FINISHED | NOT_FINISHED |
| Final levels completed | 0 | 1 (at step 20) |
| RESET actions | 1 (step 14) | 0 |
| Submission rows | 13 | 20 |
| Submitted primitives | 21 | 20 |
| Model-call rows | 71 | 88 |
| Actor / summarizer calls | 61 / 10 | 75 / 13 |
| Tool-call rows (all execute_python) | 87 | 115 |
| Grid-changing transitions | 19 | 20 |
| Step checks: matched / mismatch / unresolved | 8 / 11 / 0 | 0 / 18 / 1 |

Off submission sizes, in row order: `[1,1,1,1,1,1,1,5,1,1,1,1,5]`. On: twenty `[1]` rows. Off rows 8 and 13 each submitted five UPs; row 8 executed steps 8–11 and canceled one queued UP. Row 9 separately submitted the no-impact UP at step 12. Row 13 executed all five UPs at steps 16–20. Thus off obtained **20 executed actions from 13 accepted submissions**, not 21 executed actions. Its executed mix is UP 14, DOWN 2, LEFT 2, RIGHT 1, RESET 1; UP 15 describes the submitted mix.

The following table gives the complete executed sequences and exact unequal-cell counts from before.grid versus after.grid. Step and submission row are one-based. A positive count establishes only a changed observation, not movement or progress.

| Step | off action | off submission | off changed cells | on action | on submission | on changed cells |
|---:|---|---:|---:|---|---:|---:|
| 1 | DOWN | 1 | 2 | LEFT | 1 | 52 |
| 2 | UP | 2 | 52 | RIGHT | 2 | 52 |
| 3 | DOWN | 3 | 52 | UP | 3 | 52 |
| 4 | RIGHT | 4 | 52 | DOWN | 4 | 52 |
| 5 | LEFT | 5 | 52 | UP | 5 | 52 |
| 6 | UP | 6 | 52 | UP | 6 | 52 |
| 7 | UP | 7 | 52 | UP | 7 | 52 |
| 8 | UP | 8 | 52 | LEFT | 8 | 2 |
| 9 | UP | 8 | 52 | UP | 9 | 52 |
| 10 | UP | 8 | 52 | LEFT | 10 | 52 |
| 11 | UP | 8 | 52 | LEFT | 11 | 52 |
| 12 | UP | 9 | 0 | LEFT | 12 | 52 |
| 13 | LEFT | 10 | 2 | DOWN | 13 | 58 |
| 14 | RESET | 11 | 74 | UP | 14 | 52 |
| 15 | UP | 12 | 52 | RIGHT | 15 | 52 |
| 16 | UP | 13 | 52 | RIGHT | 16 | 52 |
| 17 | UP | 13 | 52 | RIGHT | 17 | 52 |
| 18 | UP | 13 | 52 | UP | 18 | 52 |
| 19 | UP | 13 | 52 | UP | 19 | 52 |
| 20 | UP | 13 | 52 | UP | 20 | 1473 |

Off’s step 12 has no step_check row because it stopped as no_impact. On’s step 20 stops as level_up and also has no step_check row. Each therefore has 19 checks for 20 transitions; these absences agree with batch_stops.jsonl. Off’s step-11 mismatch is specifically predicted background counts versus observed counts: the expected player position `[15,34]` matched, but colors 5 and 3 differed. Describing this interruption as “stopped on a wall” substitutes a game interpretation for the logged stop condition.

**What non-use can actually be verified.** A case-insensitive search finds zero occurrences of `contingent` anywhere in either model_calls.jsonl, and no contingent_events.jsonl exists under either arm. All accepted action rationales and all other IPC JSONL files likewise lack the string. Model-call records include returned tool names and their argument strings, including the Actor’s Python code; they do not include complete input prompts. Therefore absence here means no explicit reference in recorded responses/tool requests, not that the model never received the documentation or considered the concept internally.

The relevant implementation is [solver_agent.py](../../../deployment/kaggle/contingent-controller-20260909/chassis/solver_agent.py), lines 1231–1367. I compared the preserved per-arm nooa-join copies of solver_agent.py, headless.py and prediction_feedback.py to the supplied source: all three are byte-identical in both arms. Both worker launch.json files record the requested arm. This supports applying the inspected source to these artifacts; it is not a fresh comprehensive runtime-integrity audit.

The observability distinctions matter:

- `compare_contingent_plans` computes a comparison without writing an event. `submit_contingent_plan` can return or raise before logging: disabled arm, invalid proposal, absent observation, or an already pending plan. Thus missing events alone cannot prove zero attempted API calls.
- A valid submission writes pending state and a `proposed` event **before** submitting the root. It labels the action rationale `predict: contingent root ...`. A branch selected for execution logs `branch_selected`, and a successful continuation logs `continuation_submitted` and labels its action rationale. `write_row` opens/appends/closes the JSONL file; it does not silently discard events. No such evidence exists here.
- solver_agent.py lines 785–805 record model response content and tool-call arguments. Ordinary explicit compare/submit calls would consequently be visible even when proposal validation failed. tool_calls.jsonl, by contrast, records execute_python timing, not Python method names or arguments; searching that file alone would be insufficient.
- [headless.py](../../../deployment/kaggle/contingent-controller-20260909/headless.py), lines 90–97, invokes `_continue_contingent_plan(notification)` in the on arm before Actor handling. That method returns False without an event when no pending file exists. So “no dispatcher method invocation” is not supported and conflicts with the normal source path. “No Actor proposal, branch selection, or saved-crossing continuation in the recorded run” is the warranted substantive conclusion.

These channels agree on non-use; I found no evidence of a hidden accepted plan or a logging defect that would conceal an executed continuation. The logs are not a universal Python call tracer, so I do not promote a string search into proof against every hypothetical indirect invocation.

**The competing explanations.** The single-action fallback sentence is a plausible behavioral influence. Its wording applies when estimates lack a useful basis, however, and non-use of the API does not prove the Actor evaluated that condition. The result’s “clause fires ... always” is an inference, not an observed event. Other treatment changes accompany it: the API documentation surface, the plan instructions, the decision-packet key, dispatcher behavior, and the phrase “outside a contingent plan” in the experiment instruction. Even a reproducible whole-treatment effect would not identify that one sentence without a separate contrast.

The four on-arm runs are real: UP at steps 5–7, LEFT at 10–12, RIGHT at 15–17, UP at 18–20. They are more than a retrospectively repeated sequence: for example model-call row 57 describes LEFT three times, row 80 describes RIGHT three times then UP three times, and row 85 describes UP three times, while submitting singles. These are prospective plans at the prose/code-comment level. They support investigating why plans were not submitted together. They do **not** establish four executable batching opportunities.

The omitted constraint is decisive for that narrower claim. On model_feedback.jsonl has batch_allowed=False in every one of its 19 rows. Its first check is unresolved; every subsequent preterminal check is a mismatch under the same helper hash `a2e699fa...`. For example its frozen helper predicts one-cell shifts of multiple patches and the small blob while observations show five-cell movement of the 12/9 cluster. The ordinary [batch_error implementation](../../../deployment/kaggle/contingent-controller-20260909/chassis/prediction_feedback.py), lines 219–225, requires a committed prediction AND batch_allowed AND the same helper hash. [submit_actions](../../../deployment/kaggle/contingent-controller-20260909/chassis/solver_agent.py), lines 1730–1752, enforces that gate and then checks predicted_path and retrodiction.

Off, in contrast, has batch_allowed=True following steps 7 and 15, precisely before its two accepted batches. I executed the actual batch_error function extracted from source against every recorded feedback row, granting an otherwise committed same-version proposal: only off feedback start_step 6 and 14 pass; **none of on’s feedback rows pass**. This isolates the gate without rerunning an agent. It does not assert that the Actor consciously chose singles because of that gate. No on multi-action attempt is recorded in policy_attempts.jsonl, which has twenty single-action rows. The established fact is that merely replacing the four runs with batches under the extant helper/feedback would have failed. Repairing and requalifying the helper costs work that the proposed “12 submissions” counterfactual omits.

For the stuck explanation, distinguish immobility from failure to escape an unsuccessful strategy. All nine executed actions inside off’s two batches move the 12/9 cluster upward; the first batch precedes the explicit “stuck” rationale. That contradicts a narrow story that both batches are repetitions of physically ineffective moves. But off step 1 DOWN and step 13 LEFT each change only two cells in rows 61–62; the cluster does not move. Step 12 is fully unchanged. Thus “19 changed grids” does not imply “19 moving actions,” much less rule out being stuck. After RESET, off returns along the same upward route to the same cluster location reached before getting blocked. That is consistent with strategically unproductive repetition, though it does not prove every action was useless or that stuckness caused batching. Its first batch can be locally sensible corridor traversal while the overall strategy fails.

A third explanation already has concrete support: different generated helpers and repair behavior created different batching eligibility. Off repaired its executable prediction enough to earn batches; on retained a contradicted executable model while its informal navigation improved and eventually completed a level. Sampling/early exploration and ensuing observations can produce this divergence without the fallback sentence being its cause. The identical initial observation and immediate action divergence (DOWN versus LEFT) do not tell us which upstream cause produced it. Prompt suppression, gate eligibility, and unsuccessful route repetition can also coexist. With one trajectory per arm, neither proposed causal reading is established; I would not rank sentence suppression above the gate explanation from this evidence.

**The measure and the card.** The [Kaggle card](../../../deployment/kaggle/contingent-controller-20260909/KAGGLE-EXPERIMENT-CARD.md) asks whether exposing the interface changes *useful work per model crossing*. The primary rate is an accurate descriptive quantity: 20/71 = 0.281690 for off, 20/88 = 0.227273 for on, a 19.3% lower on rate. Actor-only rates are 20/61 = 0.327869 and 20/75 = 0.266667, so including summarizers does not reverse the ordering. Neither is a direct count of contingent crossings saved: that count is zero in both arms.

The rate factors exactly into executed actions per accepted submission × accepted submissions per recorded call. Off: `(20/13) × (13/71)`; on: `(20/20) × (20/88)`. On uses fewer calls per submission (4.40 versus 5.46 total; 3.75 versus 4.69 Actor), while off amortizes submissions across batches. This is descriptive accounting, not proof that batching caused all 17 additional on calls. Model work is uneven: before the first action there are 11 off calls versus 19 on calls; at on step 13 there are 17 calls before its next submission. Equalizing submission counts would also change observations, model repair, summaries and calls per submission. RESULT-v3’s “whole of the crossings result” overstates causal attribution.

Actions are effort, not achieved work. Replaying a failed route, resetting, or issuing ineffective actions can all increase this numerator. This run demonstrates the mismatch plainly: the higher-rate arm finishes zero levels and resets; the lower-rate arm completes one level. That does not establish on is generally better, but it rules out treating the primary rate alone as evidence of useful-work improvement. Nor is this merely a measure of confidence: enforced prediction gates and model-maintenance costs also determine it. A successful frozen check establishes consistency under a chosen encoder, not strategic correctness.

The measure can support a narrow question about action-dispatch amortization at a fixed horizon, and can be a secondary efficiency measure when task progress is matched or jointly reported. It cannot alone answer the card’s useful-work question, or isolate controller efficacy when there is no plan use. For an actual mechanism test, retain separate counts of proposals, accepted roots, eligible successors, selected branches, submitted continuations and rejection reasons; compare model work at matched decision states/tasks and report progress, resets, tokens and wall time alongside it. A forced invocation diagnostic would establish reachability/mechanics, not spontaneous uptake or gameplay benefit. Repeated randomized comparisons would be needed for the total effect of exposing this prompt/interface; a separate fallback-sentence contrast would be needed for that specific causal claim.

The on gameplay interval is shorter (878.9 versus 1134.5 seconds) despite more calls. Model crossings therefore also cannot stand in for wall-time speed. Neither favorable elapsed time nor level completion is attributable to an unused continuation mechanism.

**Assessment of the existing reports and remaining limits.** I agree with their no-promotion conclusion and the absence of asymmetric wall-budget stopping: both reached the action cap. I disagree with RESULT-v3’s placement of the unchanged action inside the first batch, its submitted/executed action mix, its inference of four available batches, and its claim that changed grids retire the stuck mechanism. The controller-non-use conclusion survives these corrections.

PRECOMMIT-censoring.md is appropriately cautious about arm asymmetry, but “readable because it is a rate” does not make actions/call immune to informative stopping: startup/repair overhead and action mix vary over a trajectory, so stopping one arm earlier changes the rate’s composition. Truncating the longer arm to the shorter action count matches a count, not states, difficulty, progress, or all call-boundary costs. Neither issue materialized as wall-budget truncation here. Its addendum’s demand for measured elapsed time is necessary for reporting a speed difference, but still insufficient for attributing one to the controller. Both action-capped trajectories remain incomplete games; on finishes its level exactly at the cap, excluding subsequent consolidation and new-level work.

The [per-arm assessor](../../../deployment/kaggle/contingent-controller-20260909/assess_candidate.py) explicitly labels its result integration-only, leaves promotion=False, counts calls/actions/checks and checks interrupted suffixes. It does not require contingent uptake, analyze batching eligibility, or establish causal benefit. That is a limit of what PASS means, not by itself a defect in an integration gate. I inspected its implementation; I did not repeat the full containment, server-request reconciliation or scheduler audit, and do not independently certify those broader claims. Likewise, a single short-versus-long run comparison does not establish a general pace law or justify a tighter future admission budget without accounting for variance.

The decision-changing uncertainty is now upstream of continuation efficacy: why no plan was proposed, and why on never repaired its batching-ineligible helper. The present artifacts establish those observed boundaries but cannot distinguish a fallback-sentence effect from other prompt effects or ordinary trajectory variation. The existing evidence is sufficient to reject the claimed batching counterfactual; it is insufficient to select a causal winner between treatments.

For reproduction from raw IPC: parse each JSONL nonblank line with json.loads; count actions.jsonl rows and their actions lengths separately from transitions.jsonl rows; join transitions to submissions by turn; compare before.grid/after.grid cellwise; group model_calls by request_purpose and step; inspect every model_feedback batch_allowed flag; search model response/tool arguments for contingent; and read the shared parent game-stops.jsonl. The full sequences and cell counts above preserve the derived facts, including the canceled primitive and the precise grid-only changes. No game-rule interpretation is needed to reproduce those counts.
