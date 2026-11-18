# Transfer assessment: Chen proof agents and Shehab's lexicon view

Research only, 2026-10-13. No solver edits, inference calls, training, GPU launches, or competition submissions. Three discussions were read live, including author replies. Public notebook/runtime source was downloaded read-only. Suggested mechanisms remain hypotheses, not score gains.

## Conclusion

The strongest transfer is a bounded pipeline from executable conjecture to observed checks, prospective probe, and executable plan. Chen's streaming candidate pool and short finalization/salvage path offer useful implementation patterns. They do not establish that more self-verifier calls improve our frozen base model. Her trained-role alignment is a substantial part of the original system.

Our next cheap discriminating test is already identified by today's continuation audit: qualify the planner on the frozen best learned model, then separately test uncovered action families. A cutoff-to-artifact recovery arm is the most concrete additional inference mechanism from Chen's source. Lexicon perception is a secondary optional view, with no demonstrated score gain.

## Sources and evidence boundaries

- [746295](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746295): Shehab Anwer, not Yi-Chia Chen. Lexicon view combines inferred scale, multicolor components, content-based names, pose relationships and change descriptions. Author reports better screen reconstruction and tool uptake, but no visible score effect. Names identify pictures, not persistent individual objects across merging/separation. Saved live discussion in `sources/discussion-746295.txt`.
- [715176](https://www.kaggle.com/competitions/ai-mathematical-olympiad-proof-pilot/discussion/715176): Yi-Chia Chen. OLMo 32B tokenizer alignment, learnable sink, SFT, full-vocab distillation, OPD, then quantized single-model prover/verifier/refiner/selector. Author replies report 3,072 H200-hours for SFT and no clean SFT/FVD/OPD or sink ablation. Main ProofBench score used BF16 evaluation, not a complete rerun of the exact final quantized/DFlash deployment. Saved discussion and linked submission source.
- [699126](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/699126): Yi-Chia Chen. Agent-Factory-3 runs GPT-OSS-120B agentic RL with attention LoRA and frozen MXFP4 experts on 4 H100s. Author reports heldout improvement but acknowledges poor inference/time allocation. These are math/RL results, not ARC results. Saved discussion.
- [Agent-Factory-3 blog](https://huggingface.co/blog/ycchen/agent-factory-3-en) and [repository](https://github.com/ycchen-tw/agent-factory-3): training-side pipeline, routing replay, token/logprob preservation, environment isolation, DFlash and rollout debugging. Routing replay and stale-policy filtering do not directly address a fixed-weight ARC inference solver.
- [Proof notebook](https://www.kaggle.com/code/threerabbits/submission-32b-fix4) and [runtime dataset](https://www.kaggle.com/datasets/threerabbits/proof-pilot-code): downloaded source confirms streaming prove/verify/refine/select with six initial provers, three verifiers per candidate, five selectors, 12 concurrent calls and a reserved selection tail. Runtime includes force-close-think salvage and streaming repetition detection. These are source observations, not reproduction of her scores. Files under `sources/proof*`.
- The dfranzen lexicon notebook link returned CLI 403 and browser 'We can't find that page'. The linked [sheetu v2 notebook](https://www.kaggle.com/code/habanwer/arcaid3-lexicon-view-v2) was downloadable. Its source implements the advertised view; it is a different base and does not validate the missing dfranzen variant. Scale inference uses thresholds and majority-color blocks and reports residue; therefore the inferred native grid is not guaranteed lossless.

## 1. Executable candidates, prospective checks, and a qualified planner

Adapt Chen's candidate-pool pattern to small candidate transition programs or rule tables. Maintain at most two meaningfully different hypotheses initially; use the same decision-time observation prefix. Replay checks report executed/matched/mismatched/abstained/failed coverage. Select a permitted, low-cost probe where predictions disagree or where the intended plan depends on an untested action family. Retain the surviving program and its evidence, then plan under explicit resource bounds.

The verifier should principally test observable consequences. Several self-verifiers sharing one base model can share its mistaken assumptions. Historical agreement alone does not establish the untested consequences needed by a route. Separate recognition, transition-model correctness, planning correctness and execution success.

Current evidence: [continuation transcript audit](../counterexample-continuations-v3-20261013/transcript-audit/findings.json) records 12/12 qualified episodes with zero completion/score gain. Explicit B1 counterexamples fix all 13 changing historical transitions but both final programs miss shrinking dynamics in other inspected traces. One reminder learns a model predicting 29/29 transitions of a winning route from its actual final state, while live BFS used horizon 24. The independent reference planner/native engine finds and clears the 29-action route. This is one diagnostic case, not prevalence or a treatment effect.

First test: CPU qualify a general planner against that frozen learned program and state, without changing the model. Report FOUND, NO_PATH_WITHIN_HORIZON, RESOURCE_LIMIT or EXHAUSTED separately. Compare with a brute-force reference on small fixtures; a resource-bounded failure cannot refute the goal. Keep the analyst's known route outside contestant inputs. Follow with prospective uncovered-action testing under a matched inference budget. This extends current counterexample work rather than proposing historical replay as new.

## 2. Salvage partial reasoning into a checkable artifact

Chen's `salvage.py` closes an unfinished thinking prefix and opens a solution prefix, continuing through native token generation. The v2 stream engine also detects repetition/time pressure and gives finalization a short 2,048–4,096-token budget. For ARC, the output should be a candidate simulator, next-test prediction, or legal tool request. Compile/check candidate code first; a truncated generation is not evidence that its implied action is justified.

Our source already handles markup recovery, malformed-call rejection, timeout yielding, helper retention and post-response repetition checks. An optional reasoning-effort ladder exists but is disabled by default. The new question is whether streaming detection plus a bounded artifact-finalization phase improves completion at the SAME total budget. Simply lowering the generation ceiling, deleting partial reasoning, or adding another critic would test different hypotheses.

Local support: the continuation audit counts eight 12,288-token no-tool cutoffs: 98,304 tokens, 25% of all generated tokens. No-tool reasoning can inform later responses, so this is an opportunity indicator, not 25% recoverable waste.

First test: matched restored checkpoints, ordinary continuation vs reserved-budget finalization, same total tokens/time and serving. Measure usable validated artifact, time to legal tool call, prospective transition agreement and completed levels; count finalization tokens/errors. Do not replay a partial action call or force arbitrary action at the deadline.

Source cautions: Chen's runtime DESIGN documents verifier priority oscillation, selection and salvage minimums that can exceed remaining time, and incomplete absolute-deadline handling. Borrow the bounded-finalization principle, not the unmodified scheduler or hard minimums. Streaming text compression is a heuristic and needs ARC-specific false-positive qualification on legitimate grids, enumerations and search output.

## 3. Alternative multicolor/temporal perception

The lexicon view can supply compact multicolor pictures and pose families alongside the existing raw grid/segmentation. Its strongest extension would use action and animation evidence to preserve uncertain correspondences when objects touch or move together. A content-derived name is not persistent object identity. Scale uncertainty, residue and contact ambiguity should stay visible.

Current overlap: dfranzen/B1 already has rotation-aware segmentation, `frame_diff`, animation histories, cached grid/cells and evidence storage. Therefore a new view needs to correct a specific observed extraction failure, not merely shorten an ASCII board. Candidate diagnostic cases include tn36 rotation missed because translation was unchanged, and cn04 relative geometry. Existing segmentation may already contain the needed signal, so compare the extra view to existing tools before assuming a perception ceiling.

Cheap first step: repair contradictory frame API descriptions in a separate candidate. `solvers/m2_b1/src/inference/agent/prompts.py` says only five fields exist and raw numeric grid is unavailable, then advertises `.grid`, `.rows`, `.counts`, `.cells`. `tool_agent.py:2737` appends this addendum. The contradiction is also present in the archived B1 system messages used by today's continuation test (`inputs/b1.json`), not only in unused source. This is an interface defect; no score benefit has been measured.

Then evaluate exact extraction and prospective action-effect predictions on observed frames, with pixels retained. Author-reported 42 vs 40 levels on a short ten-game comparison is insufficient to establish score benefit.

## Deprioritized transfers

- Training: tokenizer transplant, learnable sink, full-vocab distillation, OPD, GRPO/attention LoRA all change weights or architecture. Outside the current zero-cost/no-training campaign. Teacher hidden-state storage is an elegant future-training optimization, not an inference-time addon.
- Public proof checkpoint: available in the submission metadata but unqualified for ARC perception, Python/tool use, game-rule induction and equal-budget serving. Do not replace Flash-Next on math scores alone. The already-built endpoint screen is the appropriate path for any donated/free endpoint; offline model use still needs a separate deployment qualification.
- DFlash: model-specific trained draft and runtime modifications. We already use MTP; Chen's approximate speedups are relative to her own baselines and workloads, not evidence that DFlash improves our concurrent Flash-Next path. A serving candidate needs compatible existing weights, then matched end-to-end decode/tool-loop measurement at actual concurrency/context lengths.
- More parallel critics: six provers, three verifiers and five selectors would consume the same streams currently spread over games. Begin with a bounded candidate contrast at one stuck checkpoint and count the opportunity cost.

## Recommended sequence

1. CPU planner qualification on the already-frozen learned model; keep failure statuses scoped.
2. Prospective action-family coverage and checkpoint-level artifact-finalization tests, separately at matched budgets.
3. Frame prompt consistency repair and a bounded lexicon extraction comparison if existing perception demonstrably misses useful structure.

The actionable lesson is to improve conversion from reasoning to tested program to completed play. None of the linked work demonstrates a plug-in ARC score improvement, and none reveals Chen's current private ARC solver.
