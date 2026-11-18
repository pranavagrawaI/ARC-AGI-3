Investigated 2026-09-11. **The supplied contender update describes an older experiment family and overstates the certainty of its mechanisms. A later source identifies a different incumbent.** The useful transfer is to bound execution before another observation, while keeping host claims faithful to what was actually measured.

I pinned Son/Mark's repository at `c40ae7137309ae5b15981d907c37a7e4ddadedb8`, preserved 56 source files, reconstructed the integrated patch, exercised its pure detector functions, and replayed a related historical local failure. I also checked Tufa's public release, the animation write-up, Mostik's technical page, and the disputed Third Intelligence attribution. No model calls, GPU experiments, solver edits, or submissions were made.

**The later documented incumbent changes the account.**

An August 17 record names native TAAF with checkpoint-8 and Qwen3.8-27B-FP8 at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. It uses xhigh reasoning and reports a two-run official-25 mean of 4.82514. The record explicitly excludes the HUD guard, animation memory, state graph, failed-prefix guard, hypothesis lease, and plan reset. This is an author-reported restoration, not a score I independently reproduced. [Pinned restoration record](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/harnesses/taaf-plain-checkpoint8/RESTORED_CHAMPION.json).

The checkpoint patch executes at most eight actions per request, then returns the settled board and unexecuted suffix to the existing REPL. It asks the agent to reconsider that suffix. It does not determine whether an action was useful. [Inspected patch](sources/harnesses/taaf-plain-checkpoint8/patch/taaf-plain-checkpoint8.patch).

A separate manifest records a Kaggle 2.03 submission using this model and checkpoint. Its image upscale is 4x, versus 8x in the restored local record. It specifies a control removing only checkpoint-8 but publishes no control result there. The local all-25 mean, Kaggle score, and July ex-ft09 means must not be ranked as equivalent measurements. The documents do not identify the separate contributions of model, reasoning, image scale, or checkpoint. [Control manifest](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/harnesses/taaf-kaggle203-nocap-control/MANIFEST.md).

**The July no-impact result is encouraging, with a small sample.**

| Band | Pass A ex-ft09 | Pass B ex-ft09 | Reported actions/pass | Reported levels/pass |
| --- | ---: | ---: | ---: | ---: |
| Off | 0.946 | 1.145 | ~3353 | 15 |
| On | 2.061 | 1.187 | ~3476 | 21 |

The unrounded means imply +55.33%. Both on observations exceed both off observations, but there are only two passes per arm and substantial variation. This supports testing the treatment, not treating its magnitude as established. These runs finished July 18. Our [September 6 review](../competitor-transfer-20260906/REVIEW.md) already discussed this result. [Experiment manifest](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/harnesses/ffa7g/MANIFEST.md).

**The code falsifies “known-useless action” as a general description.**

The detector first executes an action. It classifies changes using a rolling row/column frequency mask, then stops the remaining exploration batch if nothing changed outside that mask. `intent="solve"` bypasses this interruption. The prompt presents the output as a gameplay judgment, although the predicate measures masked pixel change. [Integrated source patch](sources/harnesses/ffa7g/patch/ffa7g-full-stack.patch).

I extracted `_HousekeepingBand`, `_changed_axes`, and `_verify_hud_model` from the reconstructed source and ran these checks:

| Synthetic input with independently known meaning | Actual source behavior |
| --- | --- |
| An avatar moves horizontally eight times, then moves again on the same row | The row becomes masked; the ninth real move is labeled no-impact |
| The avatar leaves that row | The move remains visible |
| Reset the learned band, then move | The move remains visible |
| A genuine synthetic HUD counter advances | The detector correctly suppresses that change after warm-up |
| A proposed “HUD” model correctly predicts avatar movement | Verification accepts the predicted cells as a mask and hides the movement |

These are mechanism counterexamples, not a measured false-positive rate in Son/Mark's games. Correctly predicting a pixel does not prove that it is irrelevant to the objective. Adding predicted cells to a mask can preserve detection recall while increasing false positives.

This concern already met reality in our own system. Replaying the frozen September 5 host class over saved frames again yields `hud_only` for a 52-pixel move. Color 12's bounding box shifts five columns. The historical report records three queued actions skipped and a subsequent false wall interpretation. This is a different implementation with the same frequency-to-semantics mistake. [Fresh replay facts](LOCAL-REPLAY.json), [historical investigation](../deployment-probes/20260905-q6-independent-review/REVIEW.md).

**The graph evidence has an unresolved attribution conflict.**

The root README attributes regression to heavy tool use and query cost. The detailed manifest says the initial graph runs made zero graph-tool calls and instead carried up to 200 nodes every turn. The lean comparison reports 1.533 versus 2.061, with higher generated tokens/action and fewer actions. It also records individual games benefiting. The net regression is the consistent report; the heavy-tool-use explanation is not. [README](sources/README.md), [manifest](sources/harnesses/ffa7g/MANIFEST.md).

A separate reproducibility conflict is resolvable: the text-grid manifest says the graph was never committed as a patch. At the pinned revision, the integrated patch contains `_StateGraph`, query support, and an opt-in flag. It applies to the frozen baseline with zero fuzz. That establishes present source availability, not byte identity with the historical scored bundle. [Text-grid manifest](sources/harnesses/ffa7g-textgrid/MANIFEST.md), [reconstruction facts](FACTS.json).

**The auxiliary-model result has two distinct failure stages.**

The HUD-pixel manifest reports zero authored `advance_hud` functions in the original 50 transcripts. That result concerns uptake, not predictor quality. Its later dedicated-call prototype registered code, but no game reached trusted `active` status in seven subset attempts or the later full-25 pass. Removing voluntary tool selection therefore did not suffice. The generated predictor outputs were missing from the inspected telemetry, leaving the precise accuracy failure unresolved. [HUD-pixel investigation](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/harnesses/ffa7g-hudpixel/MANIFEST.md).

**Observation access and animation efficacy are not the same result.**

In the actual `frame-full` patch, “full” chiefly means retaining every engine animation frame in the sandbox as `last_animation`; the settled image remains available as before. This is not a clean uncropped-image versus cropped-image experiment. Its manifest documents one 1.439 ex-ft09 run, despite the root table introducing its configurations as two-pass means. [Frame-full manifest and patch](sources/harnesses/frame-full/).

ACTION7 is a separately verified interface defect: the frozen action mapper returns `None`; the patched mapper returns `ACTION7`. The accompanying variant changes animation metadata too, so that package is not an animation-only ablation. [Action7 manifest](sources/harnesses/action7-anim/MANIFEST.md), [executed check](FACTS.json).

Jakob Brüggen's animation study reports a six-game, four-pass comparison with a null mean effect, increased tokens/action, and very poor timing of animation inspection. It tests metadata, tool access, and hints together. It does not isolate passive metadata's effect. Its repeat-no-op guard also differs from Son/Mark's batch interruption: it rejects an already-attempted state/action pair and still consumes an LLM turn. [Animation write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369).

**Training and model selection remain partly unresolved.**

The SFT extractor exists. It selects solved-level transcript segments and regenerates images. Its own fidelity note says original multimodal settings were not captured and requests a pinned capture pass. An extractor and a game catalog do not establish a trained checkpoint, a valid training corpus, or leaderboard attribution. The later incumbent record names official pretrained Qwen weights. [Extractor](sources/ARC3-Inference/distill/extract_sft.py).

The text-grid manifest diagnoses earlier text-only failures as images sent to unsupported templates. The model launcher confirms that image mode was unconditional. This does not explain every reported model-swap failure, but it prevents treating the entire score table as an architecture ranking without checking request validity, parsers, serving, and reasoning controls. [Text-grid diagnosis](sources/harnesses/ffa7g-textgrid/MANIFEST.md), [launcher](sources/gcp/v12model_startup.sh).

**The other contenders still lack an attributable recipe in the sources checked.**

| Contender | Evidence inspected | Licensed conclusion |
| --- | --- | --- |
| Tufa | Official research index, July Duck article, public repo commit API | Public Duck repository still has only two July 1 commits. No inspected disclosure explains 11.04. |
| Third Intelligence | HF discussion search snapshot, current page, live discussion API | An older indexed version attributes world-model training to the team; current text omits it. API records an edit on September 5. Treat this as historical attribution, not current confirmation or a submission recipe. |
| Mostik | Company technical page | Latent bridging is described, but the inspected page provides no ARC submission configuration tying it to 7.51. |
| Daniel Franzen | Targeted ARC-AGI-3 author/method searches | No current submission recipe recovered. Prior ARC methods do not identify this entry. |

Sources: [Tufa research](https://tufalabs.ai/research/), [Duck release](https://github.com/Tufalabs/duck-harness), [HF discussion](https://huggingface.co/datasets/fredericowieser/arc-agi-3-wm-traces/discussions/2), [Mostik technical page](https://mostik.ai/read-more). The preserved API responses are `tufa-commits.response` and `hf-discussion.response`. Search indexes and leaderboard mirrors disagreed in freshness, so I did not claim a freshly verified leaderboard order.

**Judgment for our work.**

The later Son/Mark record favors a simpler harness with bounded action batches, while retaining substantial reasoning. It does not establish “less cognition wins.” The no-impact counterexample shows why cheap automatic judgments need scrutiny too: an inexpensive false assertion can steer every later decision.

The justified priority is to measure whether evidence changes the next action correctly, and whether execution returns to observation before a mistaken plan consumes the budget. We already have prediction interruption and observation-access work in our history; this investigation is not a reason to restart those designs or restore row-frequency masking. A particular checkpoint length, reasoning effort, or observation format still needs a matched experiment before adoption.

**Verification and limits.**

Run from the repository root:

```bash
python3 analysis/contender-investigation-20260911/reproduce.py
```

The command checks all pinned file hashes, applies the patch in a temporary directory with zero fuzz, runs the synthetic cases and ACTION7 round-trip, and replays the historical local HUD failure without rewriting its source evidence. The recorded output is [FACTS.json](FACTS.json). [SOURCES.json](SOURCES.json) records URLs, hashes, and access results.

The public run catalog and unauthenticated GCS bundle HEAD returned HTTP 403. The repository's logs and viewer-data documents say their canonical artifacts live outside Git. Consequently, I have not recomputed the competitor scores, verified run-to-bundle identity, or resolved conflicting telemetry summaries. Source reconstruction and a source manifest are valuable evidence, but they do not supply that missing join.
