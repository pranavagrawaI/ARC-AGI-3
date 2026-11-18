# Community improvements against KF, 28 September 2026

**There are credible candidates to improve KF, but this survey found no public recipe demonstrated to beat KF under our exact model, hardware, budget, and evaluation.** First address image-context accounting and test the missing ACTION7 capability. Then test higher image resolution. The largest new serving lead is pruned Flash-Next with longer contexts. The strongest reusable execution idea is to stop a planned action sequence when an observable prediction fails.

This is a survey and artifact audit. No solver changes, model calls, GPU runs, uploads, or submissions were made.

## What the candidate must beat

The incumbent is `pranavagrawai/duck-flash-next-kf2` v1, adopted on 28 September. I recomputed its result from `out-kf2/benchmark.json`: **44 completed levels, 21/25 games with at least one clear, mean public-game score 10.15157**, one pass, 7,920 seconds per game. The four zero-level games were tn36, bp35, sk48, and g50t. This score is on the public demonstration games, not Kaggle's hidden evaluation.

The shipped environment has Flash-Next NVFP4/MTP3, a 32,768-token context, eight server sequences, 28 game workers, 5 GiB KV allocation, and prefix caching disabled. KF adds a 300-second yield and game-over/level-transition prompt fixes to Keith's serving recipe. The actual run used `MULTIMODAL_CONTEXT=current_grid` and `MULTIMODAL_UPSCALE=4`, hence 256×256 images for a 64×64 board.

[Run card](/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/CARD.md), [recomputed audit](raw/incumbent-audit.json), [audit command](audit_incumbent.py).

The source and snapshots prevent several misleading recommendations:

| Question | What KF actually does | Consequence |
|---|---|---|
| Does it have memory? | Python can inspect frame/action history; assistant reasoning is retained in conversation. | Adding generic history or reasoning retention duplicates existing behavior. End-to-end chat-template fidelity was not requalified here. |
| Are its notes empty? | 19/25 final user prompts have carried notes; 23/25 retained prompt histories contain notes. | A competitor's empty-note diagnosis does not transfer to KF. These are final snapshots, not an all-turn audit. |
| How much conversation survives? | Final snapshots contain 4–11 user messages, median 6. Code caps persistence at 30 assistant messages and also trims by estimated context size. | Selective access to older reasoning/tool results is a plausible improvement. The counts alone do not prove harmful forgetting. |
| Can Python helpers persist? | The tool is explicitly ephemeral and requires redefinition. | Persistent helpers could reduce repeated code, but must earn their complexity. |
| Are intermediate animation frames exposed? | The solver reads the settled frame; the benchmark records `record_intermediate_states=False`. | There is an information gap, but published animation results do not establish a net gain. |
| Are action batches checked against predictions? | The batch loop stops at deadlines, invalid actions/errors, level completion, game over, or completion. It has no per-step predicted-pixel check. | Prediction-checked batches would add a distinct capability. |

Source files and SHA-256 identities are recorded in the audit JSON. A source-level gap is not a measured capability deficit.

## Ranked opportunities

### 1. Borrow image-context accounting and selective history retention

Scott Le Grand's public Flash-Next notebook supplies concrete patches against the same Duck bundle: keep images in the newest two image-bearing history messages, estimate image tokens separately from base64 text, and explicitly reserve reply space. It also supplies tolerant note-label parsing and preserves notes across game-over retries. KF has the corresponding source mechanisms: JSON-length token estimation, exact-label parsing, and note clearing on game over. [Public notebook](https://www.kaggle.com/code/scottlegrand/taaf-flashnext-sheetu12b-0922), [preserved extracted source](sources/scottlegrand/notebook-source.py).

**Why it fits:** KF retains only 4–11 user messages in the final snapshots. The source establishes an accounting mismatch; it does not establish how much progress correcting it buys. KF already has notes, so this is retention and parsing work, not an empty-memory repair.

**Important source correction:** the shared-board summary attributed the notebook's indexed 5.19 score mainly to F1/F3 and said animation was disabled. Inspection of the downloaded source shows `AGENTFIX_ANIM` and `AGENTFIX_SHEET` both default **on**, their installers are called, and `ARMP_UPSCALE` defaults to **12** (768×768). It also uses sixteen server sequences and a 16K context. Its default-disabled features include ACTION7, no-impact, loop guards, result augmentation, and animation *summaries*. Comments describing older versions are inconsistent with this configuration. No current run receipt tying the indexed score to these exact bytes was recovered. Do not attribute 5.19 to F1/F3, call the larger image treatment isolated, or copy the entire notebook.

**Discriminating comparison:** first replay representative KF requests through actual image/token accounting and check retained useful history, context fit, and reserved output space. Then test image budgeting/history selection alone. Test tolerant parsing and retry-memory retention separately: a reset can invalidate positions and plans even when learned rules remain valid. Do not blindly transplant the flat 120-token image allowance, especially at scale 8 or 12. The patch changes both history and reply headroom, so account for both.

### 2. Restore executable ACTION7 as an isolated capability repair

KF's mapper advertises `ACTION7` unchanged but returns `None` when asked to execute it; the solver rejects that as an unknown action. I reproduced this by calling the incumbent's pure mapping functions, with ACTION1/MOUSE/RESET as positive controls. [Probe receipt](raw/action-mapping-probe.json). The shared-board trace census reports six public games advertise it; that count was not independently recomputed here.

Community notebooks contain the repair. Scott's downloaded source includes it but disables it by default; Muhib's recovered animation notebook enables an ACTION7/UNDO patch. This is evidence of a real interface defect, not agreement that the fix improves score. Enabling another action can also create unproductive exploration.

**Discriminating comparison:** make the engine action reachable, verify its round trip and legality, then compare gameplay independently of memory and rendering changes. Keep game-specific semantics discoverable; do not teach that ACTION7 universally means undo based on a subset of public games. No gameplay was run in this survey.

### 3. Test KF with 512×512 rendering after fixing image accounting

Vincent's public development journal reports `MULTIMODAL_UPSCALE=4` to `8` raised local mean score **3.008 to 4.984** and Kaggle score **1.53 to 2.56** on dense Qwen3.8-27B. It describes a single-setting change. However, the same post lists temperature 0.6 in one table and a 0.0 marker in another. This is author-reported evidence with an unresolved configuration inconsistency and no repeated estimate. [Pinned V24 report](https://github.com/Shih-Yu-Yeh/arc3-dev-agi-journal/blob/861a30371716121a37c4a2170556dd1c386c1d07/content/posts/2026-08-25-arc3-v24-mm-upscale8-winner.md).

**Why it fits:** KF actually uses scale 4. This is a small reversible treatment with no new agent architecture. Nearest-neighbor enlargement does not add information to the original grid; it changes how the vision processor represents that information.

**Discriminating comparison:** KF versus KF with only upscale 8, same source/model/serving/budget. Verify actual image dimensions and image-token processing first. Track grounding errors, time to first action, actions per minute, completed levels, and score. Reject if extra visual processing or earlier context eviction costs more progress than better perception buys. Do not combine this with a prompt or memory change.

### 4. Pursue pruning as a memory-and-throughput package

A September 27 report in Son/Mark's repository records Flash-Next public-suite scores **21.4 unpruned at seven concurrent games**, **19.7 pruned at seven**, and **33.9 pruned at sixteen**, with roughly 100K contexts and 132-minute game budgets. The author did not recover the kept-expert manifest or calibration method. Raw run-store evidence was not independently accessible here. This is a promising bundled result, not causal proof or a direct comparison with KF. [Pinned report](https://github.com/sonpham-org/arc-3/blob/9336c50a7a693c4e86bbd26382c4098eab5e0c4a/docs/trace-findings/2026-09-27-moe-expert-pruning-review.md).

A concrete public alternative exists: `lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448`. Its card reports 448/512 routed experts and a checkpoint reduction from 123.6 to 115.6 GiB. It preserves the MTP bank at 512 experts, requiring separate draft-model handling. I downloaded its config and compatibility hook, not weights. [Pinned model card](https://huggingface.co/lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448/blob/8d565c90add22b39cf3632943f859d7a9a411b48/README.md).

**Why it fits:** KF's 32K context and 5 GiB KV allocation constrain retained history and active requests. Freed memory could support a more useful serving configuration. Checkpoint size reduction is not an equal guarantee of GPU-memory savings because KF offloads part of the model.

**Discriminating comparison:** first hold concurrency/context fixed and check whether pruning preserves gameplay and tool/vision behavior. Then spend the released memory on context/concurrency and evaluate under the same total notebook deadline. Keep startup time and all hidden-game waves in the accounting. A per-game budget alone does not equalize a run.

**Disposition:** highest research upside; not commit-ready. The public keep-448 checkpoint is not Son's unidentified keep-384 checkpoint. Neither is a drop-in replacement for KF's pinned loader. The MLX keep-384 release is another format, not an RTX serving artifact. [Keep-384 card](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit).

### 5. Stop batches when their predictions fail

Retrodict executes predicted action queues and returns to the model when the expected cells differ. OY1 supplies a particularly small implementation: multi-action batches must predict an observable for each step and stop on a mismatch; single exploratory actions need no prediction. Its public 100-point result uses Astra and is not a component ablation. [Retrodict source](https://github.com/ryanbbrown/Retrodict), [OY1 method](https://github.com/OYLabsAI/arc-agi-3-api-harness/blob/25d2561d7709c91991376086008553210a0dbe49/docs/method.md), [runner](https://github.com/OYLabsAI/arc-agi-3-api-harness/blob/25d2561d7709c91991376086008553210a0dbe49/harness/arc_harness/runner.py).

**Why it fits:** KF already batches actions, but its execution loop does not test the predicted intermediate board. This can preserve cheap execution of understood sequences while limiting the cost of a mistaken plan. It does not require a full simulator or a second model.

**Discriminating comparison:** add only optional sparse pixel/region predictions to batches. Count incorrect actions prevented and extra model turns caused, then compare score at equal wall time. A prediction matching a few cells does not validate the whole theory. Do not impose predictions on every exploratory action or reject useful exploration merely because the model is uncertain.

### 6. Make old evidence usable after conversation trimming

PRO-LONG provides the strongest retrieved controlled evidence for programmatic history access: an average 18-point improvement over its no-log coding-agent baseline. Its workspace-persistence ablation is also a warning against overbuilding: GPT-5.5 scores 41.2 with a persistent workspace and 40.7 when it is cleared. The setting uses frontier models and public games, not KF. [Paper, including ablations](https://arxiv.org/html/2607.20064v1), [released research harness](https://github.com/alexisfox7/PRO-LONG/tree/main/research/arc-agi-3).

**Why it fits:** KF's frame history already exists. The narrower missing capability is convenient, bounded retrieval of previous reasoning and tool results after they leave the conversation. A correct old observation can also reopen a rejected hypothesis when its preconditions change.

**Discriminating comparison:** preserve the full interaction record and expose selective retrieval without injecting it all into context. At frozen prefixes, ask for evidence that is outside the retained conversation and check accuracy and cost. Only proceed to gameplay if retrieval is actually used and correct. Treat persistent helpers as a separate intervention; repeated code is a cost, not by itself a cause of failure.

KF also estimates request size by JSON character count divided by three, including image data URLs, and strips blank lines from reasoning. U4AR has concrete fixes for image-aware estimates and verbatim replay. Their consequence for KF remains unmeasured; prefix-cache benefits cannot appear while KF's prefix caching is off. [U4AR source and change inventory](https://github.com/U4AR/qwen38-arc3-rl/blob/655309aedb03df2c35957cc35afe421112437e36/README.md).

### 7. Investigate state-scoped no-impact stopping

Son/Mark report **21 versus 15 levels** and **1.62 versus 1.05 ex-ft09 score** for a statistical no-impact band on an earlier Duck stack. It stops repeated exploration when only a learned housekeeping band changes. Their broader state graph lost repeatedly; query costs consumed useful actions. The published submission wrapper still points to a private source dataset, so the README is not a complete reproduction package. [Author's results](https://github.com/sonpham-org/arc-3/blob/9336c50a7a693c4e86bbd26382c4098eab5e0c4a/README.md), [wrapper](https://github.com/sonpham-org/arc-3/blob/9336c50a7a693c4e86bbd26382c4098eab5e0c4a/kaggle/submissions/ffa7g-noimpact/README.md).

**Why it fits:** KF's raw board-change test treats any changed pixel as a change and does not interrupt a batch for no impact. A counter can therefore hide useless repeated moves.

**Discriminating comparison:** stop only the current repeated batch after an independently supported no-impact observation. Preserve the mask's provenance and relevant state. Do not make a permanent action blacklist: selection, mode, or object changes can make a previously ineffective action useful. Reject a mask that hides progress-relevant display changes. This candidate needs more implementation recovery than the prediction-check approach above.

## What the rest of the field contributes

These are research dispositions, not a common leaderboard. Public, preview, Kaggle, warm-start, and selected-run results cannot be ranked together.

| Work inspected | Evidence and usable contribution | Disposition for KF |
|---|---|---|
| [Duck / Tufa](https://tufalabs.ai/research/duck-harness/) and [Keith Flash-Next](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp) | Reproducible coding-agent and serving lineage. | Already incorporated. |
| Anim Base, FlashNext AsIs, Tuned, Hybrid REPL, FOYSAL | Earlier downloaded source comparisons distinguish copies, serving changes, and animation/no-op additions. | Reuse the [September 25 source comparison](/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/kaggle-approaches-20260925/REPORT.md); current versions could not be refreshed. |
| [Reki, milestone 1 second place](https://www.kaggle.com/code/ruichardliu/milestone1-2nd-solution) | Indexed historical best 0.86; source bundle not recovered in this session. | Historical lead, no verified new mechanism. |
| [VISTA](https://vista-research.github.io/) | Exact past-frame/crop access, concise revisable notes, direct interaction; all public levels reported solved with frontier backends. | Borrow selective observation retrieval; 100 public points do not establish transfer to Flash-Next. |
| [NVIDIA AVO](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/) | Reports all 183 public levels with Opus 5 using text grids, memory, and supervision. The authors explicitly reject a controlled-ablation interpretation. | Counters the claim that either images or an explicit simulator are universally necessary. No bounded Kaggle recipe recovered. |
| [Schema](https://schema-harness.github.io/) | Explicit world models and replay checks. Headline scores retain better per-game results under a fallback model/effort rule. | Do not compare its selected-run score with KF's one pass. |
| [Tycho](https://arxiv.org/abs/2607.28287) and [NOOA](https://arcprize.org/leaderboard/community) | Programmatic models, actor-controlled delegation, reusable code and memory. | Established research lineage already explored locally; public success is not a new reason to reinstall builders/critics. |
| [World-model ablations](https://arxiv.org/html/2607.15439v1) | Executable-only variants do not always beat textual ones; the full verification treatment performs best but spends more resources and changes the interface. | Supports selective verification. Does not isolate verification as a free improvement. |
| [Polyphony](https://github.com/Mininglamp-AI/polyphony-arc-3) | Open-model executable-policy workspace with exact transition checks; community board lists 19.8 public points. Published serving example uses eight-way parallelism and long deadlines. | Interesting alternate agent, not an established single-GPU replacement. |
| [Kepler](https://github.com/Cveinnt/kepler) | Public exact-replay receipts, resource accounting, negative results, one frozen Opus configuration. | Useful evaluation discipline; frontier-model result. |
| [arc-skill](https://github.com/pbshgthm/arc-skill) | Small prediction-first skill with a public replayed Opus result. | Supports trying a small prediction treatment before an architecture replacement. |
| [Prime Agent](https://arxiv.org/html/2608.23552v1) | Persistent REPL, retained histories, recursive calls, large test-time computation. | Useful primitives; headline performance does not establish swarm value under KF's GPU budget. |
| [TELL](https://github.com/studio-dots-ai/TELL) | One conversation and a persistent hypothesis memory file. | Supports accumulated evidence without mandatory multi-agent roles. |
| [DreamTeam](https://github.com/NVIDIA/dream-team) / [workspace optimization](https://arxiv.org/abs/2605.09650) | Six roles and shared executable artifacts; public controlled comparisons on frontier models. | Too much unqualified overhead to transplant wholesale. |
| [Continual Harness](https://github.com/feng-rrRay/Continual-Harness-ARC-AGI-3) | Revises prompts, skills, memory, and agent specifications. | Longer-term research; requires evaluation on games not used by refinement. |
| [Self-refining negative result](https://github.com/AnshChoudhary/arc-agi3-self-refining-harness) | Two plausible coach edits were rolled back; one harmed the pilot and one tied. | Accurate diagnosis does not establish an effective intervention. |
| [Agno Arcade](https://github.com/agno-agi/arc-agi-arcade) | Distinguishes cold, warm, and knowledge-sharing runs. | Keep those regimes separate when borrowing cross-run memory or training data. |
| [Crystalline](https://github.com/synchopate/arc-agi-crystalline) | Explicitly discloses reading game source and making per-game solvers. | Exclude its score as evidence of unfamiliar-game solving. |
| [TWIN](https://github.com/Alexyskoutnev/TWIN-ARC-AGI-3) | Digital-twin construction and replay checks. | Same verification family; no new matched KF evidence. |
| [StochasticGoose](https://github.com/DriesSmit/ARC3-solution), [Explore It](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore) | Action-effect learning and graph exploration; credible preview results. | Borrow cheap exploration components cautiously; preview action budgets and scoring differ. |
| [BDR-Pro](https://github.com/BDR-Pro/arc-prize-2026-arc-agi-3), [frugal graph explorer](https://github.com/shloksah/arc-agi-3-agent) | Lightweight learned transitions and return-to-frontier search. BDR reports a roughly 0.26–0.27 Kaggle floor. | No evidence they replace KF; possible bounded fallback only. |
| [FACE-OF-AGI](https://github.com/ricsinaruto/artificial-agency) | Released open-model orchestration from an earlier top-ten milestone entry. | Historical engineering reference; no current matched improvement. |
| [Community game catalog](https://github.com/sonpham-org/arc-3), [ARC-Interactive](https://github.com/theredbluepill/arc-interactive) | Many additional environments, including reviewed and unreviewed/generated sets. | Useful disjoint development tests. Audit solvability and source provenance; they are not proxies for hidden Kaggle scores. |

## Training, animation, and serving ideas to keep on the watchlist

**Trajectory SFT is not yet a reliable shortcut.** Manas Joshi reports a same-harness leaderboard lift, but no numerical replicated comparison was recovered. The earlier downloaded pilot had nine conversations. U4AR's published partial baseline table is less encouraging for terse reasoning: held-out episodes scored 0.47 without the adapter and 0 with it. This is an incomplete small sample, not a verdict on all RL. [STaR release](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739047), [U4AR measurements](https://github.com/U4AR/qwen38-arc3-rl/blob/655309aedb03df2c35957cc35afe421112437e36/results/REPORT.md).

**Train verified progress, not attractive auxiliary reward.** A separate Qwen/SkyRL project reports zero level completion despite improved shaping rewards; repeated local transitions explained the reward increase. Its training code was still announced as forthcoming. [Primary negative report](https://github.com/yuran986/arc-agi-3-agent-post-training). Any future training should demonstrate held-out progress with failed trajectories and alternatives preserved, rather than treating fluent winning transcripts as sufficient evidence.

**Animation is missing information with a demonstrated cost.** The relevant community experiment reports a nonsignificant public score change, p=0.92 over six games and four passes, and 17% more tokens per action. Only two of 181 inspections targeted genuinely informative animation. A better candidate would select transient, task-relevant changes automatically and expose them on demand. This is my proposed revision, not a published positive result. [Author's experiment](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369).

**Fewer activated experts is distinct from pruning stored experts.** The September paper changes routing normalization to avoid over-amplifying the remaining expert branch. It reports a roughly 17% end-to-end speed gain in its tested setting. It does not establish long-context ARC/tool-use behavior or KF runtime compatibility. This is a later serving experiment, not a config-only change. [Expert-halving paper](https://arxiv.org/abs/2609.04575), [ACE](https://arxiv.org/abs/2609.05228), [REAP code](https://github.com/CerebrasResearch/reap).

**A stronger released model must fit the whole job.** Dots3-note Preview is a relevant open-weight agentic model, but its primary card lists 280B total parameters and recommends an eight-GPU FP8 deployment. I found no qualified one-RTX ARC recipe. Its published training direction is research input, not an immediate KF replacement. [Model card](https://huggingface.co/dots-studio/dots3-note-prev/blob/main/README.md).

## Decision and limits

Keep KF fixed as control. First qualify image-context accounting without model calls and test ACTION7 separately. Compare note parsing/retry retention as its own treatment. Next test KF512 after accounting is sound. Recover the pruning package only when its exact model and runtime can be identified. If architecture work follows, try sparse prediction-checked batches before adding mandatory simulators or specialist agents. Scott's c16/16K profile is a later throughput candidate: halving KF's context may offset the concurrency gain, so it belongs after the retention check.

Promotion needs repeated matched gameplay at the same total resources, followed by an authorized hidden-set submission. Use mean score and level depth, paired per-game outcomes, resource use, and run validity. Do not select the best replicate per game. Report intervals and unresolved variance rather than importing a noise threshold from a different model. Keep the incumbent when the result is inconclusive.

The broad discovery pass retrieved all **423 repository records returned by GitHub's `arc-agi-3` search**, across five pages, with no duplicate records. This is a metadata inventory, not 423 code reviews. Selected repository READMEs, current papers, model cards, and pinned implementation/results files received closer inspection. The inventory is not exhaustive of differently named repos, private work, Discord, or unindexed releases.

Kaggle's authenticated leaderboard/kernel-list endpoints returned 429. The legacy public endpoint returned 401, web-rendered listing pages were empty, and no CUA browser was available. Therefore current notebook coverage and today's standings are incomplete. The September 25 standings/source survey is retained as dated evidence, not presented as current. KF's September 28 submission remains last-known PENDING; I could not refresh its receipt. `da-fr/arc-agi-3-solution` was discoverable but its tree endpoint returned 409 and no README was recoverable; this does not establish the absence of a private solution.

Another agent recovered two public notebooks during the survey; their local source was available even though my listing requests failed. The Scott notebook and extracted source are preserved here. A separate agent also reported fresh standings; I have not verified those raw receipts and do not use them to rank mechanisms. In particular, another solver's local-to-hidden score ratio is not a valid conversion for KF.

The most important remaining uncertainty is whether any candidate improves **this** incumbent under **this** compute limit. Research has narrowed the candidates; it has not run that comparison.

Evidence locations: [repository inventory](raw/repository-inventory.json), [pinned file downloads](raw/fetched-files.json), [source copies](sources), [incumbent audit](raw/incumbent-audit.json), and [file hashes](FILE-HASHES.json). Reproduce the local audit with `python3 analysis/community-survey-20260928/audit_incumbent.py`.
