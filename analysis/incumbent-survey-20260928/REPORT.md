# What could beat KF?

Research date: 28 September 2026. The incumbent is KF, `pranavagrawai/duck-flash-next-kf2` v1. The best next candidates are repairs to its action and memory interfaces, followed by a serving-capacity experiment. No candidate in this report has demonstrated a gain over KF in gameplay.

The review recomputed 250 public-game episodes across ten recent arms, examined KF's shipped code and all 25 trace summaries, and closely inspected selected failures. Earlier work is covered through the September 9 reconciliation, its family appendices, later construction and Operational Theory audits, and the Duck experiments. This is a survey of the accessible research families, not a claim to have read every historical model token or recovered missing archives. The supplied-route ceilings, synthetic demonstrations, public-game runs, and competition results remain separate.

## Establishing the comparison

| Arm | Public episodes | Completed levels | Interpretation |
| --- | ---: | ---: | --- |
| K, Flash-Next as released | 25 | 36 | One public-25 pass |
| **KF, current incumbent** | **25** | **44** | One public-25 pass; 21 games with a level |
| F, F2, Fc, dense-27B fixes-only | 75 | 72 | Three runs: 25, 20, 27 |
| Q, Q3, external questioner | 50 | 43 | Corresponding F/F2 controls total 45 |
| Sa, Sb, Sc, self-questioning | 75 | 74 | Corresponding three controls total 72 |

KF's local `score.json` is 10.15157; K's is 5.99. These are local public-game figures, not KF's competition submission score. The card records submission 56645750 as pending at submission time; this review did not refresh that status. The adopted KF identity comes from the card and board thread 480, not from a stale README. KF already includes yield=300 seconds, game-over retry wording, and level-up review. Reproposing those changes would add nothing. [KF card][kf-card]

The census reproduces KF's 3,484 actions, 866 analyzer attempts, 172 yielded attempts, and 1,345 model responses. Server metrics independently report 1,345 completed requests. K had 1,343 responses despite its much larger 613 yielded-attempt count. Thus fewer yields do not by themselves mean more model work or explain all eight additional levels. [Census](trace-census.json), [serving metrics](serving-metrics.json)

The old approximately four-level standard deviation was estimated from older Duck runs on five games. It is not a newly measured KF uncertainty interval. A single successor scoring 45 instead of 44 would not establish superiority.

## Candidates, in the order I would investigate them

### 1. Repair ACTION7 dispatch, by itself

This is the smallest concrete candidate. KF advertises ACTION7 in six games: ar25, bp35, lf52, sb26, sk48, and su15. Its mapper converts ACTION1 through ACTION6 correctly but returns `None` for ACTION7. The actual solver normalizer returns `Unknown action at index 1: 'ACTION7'`. Zero ACTION7 events appear in KF's executed-action logs; 22 recorded Python cells contain the name, which is a textual count rather than a count of issued engine actions. [Executable contract probes](contract-probes.json)

The defect can corrupt learning as well as remove an action. Near the end of SB26, the model's cell tries ACTION7, prints an unchanged-board comparison, and reasons that ACTION7 does nothing. There was no corresponding engine action. A rejected command cannot establish a game mechanic.

The concurrent [field survey][field] nominated this issue. This review independently executed the shipped mapping functions and normalizer. The first repair should preserve the engine action's identity and dispatch it when available. Naming it UNDO should follow the engine's actual contract; dispatch need not depend on teaching a semantic alias.

**Discriminator:** first verify an advertised ACTION7 reaches an instrumented engine exactly once and that an unavailable action remains rejected. Then compare KF against KF plus only this repair on affected games, with unaffected games as regression controls. Record executed ACTION7, subsequent state changes, levels, score, and action cost. If the action becomes reachable but is unused or does not improve decisions, keep the contract correction separate from any efficacy claim.

### 2. Recover the world-model text KF already writes

The shipped note parser recognizes `World model:` but rejects `World model (revised):` and `World model v2:`. Executing the exact parser over KF assistant blocks found 465 blocks with a world-model label; **277 fail to populate the world-model field**. This is a count of authored blocks, not independent episodes or 277 lost decisions. [Probe and per-game counts](contract-probes.json)

The text still exists in conversation history until that history is trimmed. The defect is its failure to update Duck's explicit carried-memory field. An older value may therefore remain while a newer revision is visible only in transient conversation text. This is materially different from our older memory audit, where many dense-model episodes simply authored no usable note. That audit found no positive outcome association for note-writing and prevents claiming that more notes must improve levels. [Older memory audit][memory-audit]

**Discriminator:** replay the saved assistant blocks through a tolerant, bounded label parser. Preserve each revision's content and require no false extraction from unrelated prose. For gameplay, change only parsing first. Do not bundle preserving notes across GAME_OVER or changing level resets; those change staleness behavior. Success means correct carried state after context trimming and better equal-budget gameplay, not a higher note count.

Scott's public patch offers prior implementation for this repair, but its public notebook contains several treatments. Its score does not isolate parser benefit. Use the actual parser defect and our prefixes as the grounds for testing.

### 3. Increase useful serving capacity on KF's exact workload

| Measurement | K | KF |
| --- | ---: | ---: |
| Mean request latency | 145.06 s | 145.26 s |
| Mean queue time | 126.34 s | 126.42 s |
| Mean prefill time | 1.60 s | 1.65 s |
| Mean decode time | 16.76 s | 16.80 s |
| Queue time / summed request latency | 87.09% | 87.03% |
| Median running / waiting requests, at least 20 active | 3 / 22 | 3 / 22 |

These are accumulated request times, not additive GPU wall time. The GPU was serving other games during queue waits. They do not imply an 87% idle GPU or an available eightfold speedup. [Recomputed metrics](serving-metrics.json)

The exact server identity uses 5 GiB KV cache, automatic cache dtype, max_num_seqs=8, max_model_len=32768, MTP3, and prefix caching off. The startup log reports 105,202 KV tokens and maximum full-length concurrency 3.21. Model loading consumes 81.8 GiB, and graph capture another 0.36 GiB; these figures do not establish peak-memory headroom. KF recorded 27 preemptions. Merely raising max_num_seqs cannot create KV capacity. [Frozen serving identity][server-identity], [server log][server-log]

**First candidate:** a modest larger KV allocation at unchanged precision, model, context length, MTP, and game concurrency, after measuring peak-memory headroom on the target runtime. If headroom is insufficient, FP8 KV is a separate candidate, requiring numerical and long-context qualification on this pinned hybrid model. The loader exposes both controls. vLLM documents these controls, but current documentation does not certify our pinned build. [vLLM cache configuration](https://docs.vllm.ai/en/latest/api/vllm/config/cache/), [FP8 cache discussion](https://vllm.ai/blog/2026-04-22-fp8-kvcache)

**Discriminator:** replay a representative mixture of short and late-game requests under the full concurrent workload. Require improved completed useful requests per total wall budget, stable parsing and bounded memory, rather than a single-stream tokens/second win. Then run equal-budget gameplay. Reject if extra concurrency slows decode enough to erase the gain or harms decisions.

Prefix caching is not my first isolated lever: measured prefill is small relative to decode and queueing. Halving the context to 16K while increasing concurrency is a different tradeoff that can remove evidence. Do not combine it with a cache experiment and call the result a scheduler effect. Older SGLang MTP and chunk-size decisions do not adjudicate this new NVFP4/vLLM workload.

### 4. Give the actor exact level-boundary evidence

KF already prompts for a review after leveling up. SB26 shows why the instruction is insufficient.

| Actual evidence | Later model interpretation |
| --- | --- |
| Action 8 places the final color; level=1, score=0, level_completed=false. | The final placement must have completed the level. |
| Action 9 is SPACE; level changes 1 to 2, score=1, level_completed=true. | SPACE was unnecessary; completion was automatic. |
| Actor queries transitions with `after_frame.level == 1`. | The query returns actions 1 through 8 and omits the winning transition. |

At event line 228, the model writes the false automatic-completion claim into its world model. The historical transition API reconstructs old `.result` dictionaries as empty and supplies the real result only for the latest transition. Before/after levels still contain enough information to recover the win, but the actor's query discards it. [Extracted query and raw outcome](sb26-level-boundary.json), [shipped sandbox][sandbox]

**Candidate:** expose a compact, mechanically computed completion record containing the executed action, source/destination levels, before/after frame references, and authoritative outcome. Retain it for later lookup. This supplies evidence, not a host-invented winning rule. Do not certify the actor's interpretation merely because it consumed the record.

**Discriminator:** on the preserved prefix, ensure the record includes action 9 and prevents the erroneous filter from erasing the boundary. A later model replay must change the conclusion and resulting plan; then test other games and level changes. The hypothesis fails if the actor sees the exact event and still ignores it or if the new context costs more than it saves.

Related prior evidence remains relevant: the failed SB26 run from September 24 generalized a no-op before selection into a permanent prohibition, while the same click after selection later worked. Current KF solved SB26 L1 in nine actions, so the old failure should not be presented as KF's current L1 bottleneck. The general lesson is to bind negative evidence to actual execution and tested conditions. [Earlier trace contrast][sb-old]

### 5. Reuse small measurement and bounded-search helpers

KF has 70 detected tool errors across 1,335 Python calls, including three tool timeouts. Eight cells attempt the unavailable `difflib` module, and several confuse history-entry fields or frame dimensions. 915 cells define functions; this establishes repeated code generation, not how much of it is redundant or removable. The 70-error count excludes host rejection dictionaries that the model neglects to print, such as the ACTION7 case.

The strongest concrete case is LS20, event line 61. A generated search returns charcoal for coordinates outside the board, then tests a model where charcoal is traversable. It explores an unbounded plane. The actor attributes the timeout to segmentation cost and spends subsequent work around that explanation.

I replayed the exact saved cell in the shipped sandbox over archived public observations. The original times out at 30.08 seconds. Adding only `if not (0 <= r < 64 and 0 <= c < 64): return False` to its walk predicate finishes in 0.66 seconds and returns the four model comparisons. No model call or game action occurs. The first local adapter attempt used ACTION1-style names where history expects UP-style names; that failed before the relevant code, was corrected, and remains archived. [Successful reproduction](ls20-reproduction.json), [initial adapter failure](ls20-reproduction-initial-adapter-error.json), [reproduction command](reproduce_ls20.py)

This proves the cause and repair of one computation failure. It does not prove LS20 L2 would complete or establish a GPU speedup. The useful candidate is a small optional helper for finite-grid search or exact frame differences with explicit incomplete/error results. We already built planners and persistent interpreters. Rebuilding either wholesale would require stronger evidence. The old latent-planner review already caught false exhaustion, unbounded callbacks, and mutation bugs; reuse that knowledge. [Planner review][planner]

### Reserve candidates

- **Image-history trimming and token accounting.** KF estimates request tokens from serialized JSON length, including base64 image URLs. Scott's F1 patch trims old images and changes the estimate. The code identifies a plausible mismatch, but this review did not measure actual image-token cost under KF's processor. Qualify the estimator and evidence retention; do not copy a flat 120-token allowance without checking it. Keep the parser treatment separate.
- **A fresh attempt after a persistent false premise.** Our older repeated passes show alternate trajectories can win. The existing second-attempt calculation assumes independent attempts and counts whether either reaches a level; it is not an equal-total-budget deployable policy. KF already benefits from in-game retries, and resetting reasoning or a game can destroy progress. Defer until a concrete selection and retention rule survives an equal-budget test.
- **Training experimental decisions.** Training on measured inspections and useful belief revisions is more specific than successful-trajectory imitation. We do not have a demonstrated local training gain over KF. A higher-variance training project should follow the cheaper contract and serving tests.

## What the accumulated work rules out, and what it leaves open

| Research family | Retained evidence | Consequence for beating KF |
| --- | --- | --- |
| Supplied routes and early typed planning | Typed serialization repaired bounded plan selection; supplied LS20 route completed a level in 13 actions. Grounded online policies still failed routing. | Preserve the interface lesson; no autonomous discovery result to port. |
| Early native acquisition and G2 commitments | Hundreds of calls could close evidence cycles without solving a level. | More bookkeeping is not the next default. |
| Retention and Builder continuation | A retained candidate remained 8/16 while canonical was 0/16; continuation improved neither. Budget pressure degraded a candidate. | Retain verified gains, but do not buy generic Builder calls or force urgency. |
| Symmetry and scratch computation | Real narrow ar25 completions, without establishing broad or matched efficacy. | Keep cheap relational computation available; no mandatory simulator prerequisite. |
| Retrieval and compaction | Structured retrieval improved old-evidence answers and cost; q6 truly ran 100 actions with 153 Actor calls and two summaries. | Retrieval mechanisms exist. Test the actual missing field in KF. |
| Typed goals, procedural demonstrations, self-tests | Narrow uptake; five-case procedural reliability failed its mandatory competing-regions case. Some validators certified the wrong narrative. | A typed test result must not be equated with semantic support. |
| Contingent controller | Supplied plans removed a crossing locally; live actor proposed none. A level win in its on-arm did not exercise it. | Efficacy remains untested, not positive. Duck already executes code loops and batches. |
| Epistemic critic and objection closure | Cheap grounded questions could be delivered, but no level benefit; closure rules added rejected submissions. | Do not port an always-on critic. |
| Duck periodic critic | 21 versus 21 levels; an earlier critic comparison was 11 versus 14. | Same formulation has not earned another run. |
| Dense Duck questioner Q/Q3 | 43 versus 45 levels; measured exploration did not improve. | More answers to questions are not progress. |
| Actor self-question S | Initial 52 versus 45 did not replicate; final 74 versus 72. | Do not promote the first favorable pair. |
| FT09 inspection instruction | Exact-prefix replay: control 1/6 successes, prompt treatment 0/6. | Another reminder to inspect is not the proposed repair; structural evidence delivery remains open. |
| Construction R8/R9 | Valid joins and some use; LS20 actions 10/19, CD82 11/11, no levels. S5I5 produced no freeze. | Lifecycle success is not model accuracy or gameplay benefit. |
| R10, world-model utility | Source/ABI failures; later utility run stopped before control. | Unresolved or invalid comparisons cannot count as zero effect. |
| Operational Theory T4/C8 | Shared persistence/facade defects prevented intended exposure. | No clean efficacy verdict; repair qualification is separate from a reason to replace KF. |
| Serving, MTP, reasoning | Old synthetic speedups reversed on real workloads; later KF successfully uses a different Flash-Next/MTP stack. Dense reasoning-effort sweeps changed little. | Evaluate the exact model, runtime, context, and concurrency. Historical failures are not universal bans. |
| External released systems | Public systems use structured logs, perception helpers, checked plans, or optional construction, often with frontier models. | Borrow a narrow mechanism; their scores do not establish transfer to our local model. |

Sources for this table: [historical reconciliation][history], [Q/F results][fq], [S replication][s], [FT09 replay][ft-replay], [periodic critic][periodic], [R8][r8], [R9][r9], [utility retry][utility], [Operational Theory audit][c8]. The current Retrodict source also demonstrates structured log access and bounded search as engineering patterns; its stronger-model results are not an estimate for KF. [Retrodict](https://github.com/ryanbbrown/Retrodict)

## Decision

Keep KF as control. Start with independent ACTION7 and memory-parser candidates because their defects are executable and their changes are small. Prepare the serving-capacity comparison next; it has the broadest plausible upside but requires target-GPU evidence. Use the SB26 completion record and LS20 cell as discriminating fixtures for a later evidence/computation treatment.

For any claimed winner, match the total compute envelope, games, executable model settings, and actual notebook pass count. Replicate both control and treatment; report public score, levels, per-game changes, actions, and time together. A saved-prefix improvement qualifies a mechanism, not the solver. Use a withheld set of game mechanics for generalization where available, since our repeated public-game analysis is development data.

No solver file, incumbent bundle, or historical run was changed. No model, GPU, or environment run was launched. New files contain the survey, census, source identities, extracted evidence, and offline reproductions. The report does not authorize a launch or submission.

## Reproduce and inspect

Run `python3 analysis/incumbent-survey-20260928/audit.py` for the ten-arm census, serving metrics, KF reasoning digests, and SB26 boundary check. Run `python3 analysis/incumbent-survey-20260928/contract_probes.py` for ACTION7 and memory parsing. Run `python3 analysis/incumbent-survey-20260928/reproduce_ls20.py` for the two inspection-only sandbox cells; it takes about 31 seconds on this host.

[Trace and serving hashes](trace-source-hashes.json) bind the 250 event files and eight serving/score files. [Document and code sources](source-documents.json) record inspected-source identities. Digests retain complete logged KF reasoning for inspection; they are not new model-generated summaries. Numerical verification does not certify every qualitative interpretation.

[kf-card]: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/CARD.md
[field]: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/FIELD-SURVEY-20260928.md
[memory-audit]: /home/pranav/projects/arc-agi-3-reference/duck-critic-adapter/runs/memory-audit/RESULT.md
[fq]: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/qwen38-fixes/RESULT-FQ.md
[s]: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/qwen38-fixes/RESULT-S.md
[periodic]: /home/pranav/projects/arc-agi-3-reference/duck-critic-adapter/runs/periodic-v1/RESULT.md
[ft-replay]: /home/pranav/projects/arc-agi-3-reference/duck-critic-adapter/runs/trajectory-audit/replay-step14/RESULT.md
[server-identity]: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/out-kf2/vllm-server-identity.json
[server-log]: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/out-kf2/vllm-openai-server.log
[sandbox]: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/kf/source-dataset/src/ARC3-Inference/inference/agent/python_tool_sandbox.py
[history]: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/what-we-learned-20260909/REPORT.md
[planner]: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/latent-planner-proposal-review/REVIEW.md
[sb-old]: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/deployment-probes/20260924-v12-trace-comparison/COMPARISON.md
[r8]: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/actor-critic-construction-kaggle-r8-gameplay-20260915T124458Z/RESULT.md
[r9]: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/actor-critic-construction-kaggle-r9-cd82-20260915T183130Z/RESULT.md
[utility]: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/world-model-utility-retry-20260917T091830Z/post-run-result/RESULT.md
[c8]: /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/operational-theory-c8-audit-20260917/REPORT.md
