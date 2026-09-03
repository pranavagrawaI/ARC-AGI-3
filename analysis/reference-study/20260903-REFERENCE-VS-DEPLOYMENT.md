# What the Tycho and NOOA sources say, and where our deployment diverges

Sources read: `projects/arc-agi-3-reference/papers/tycho-2607.28287.pdf` (52 pp),
`papers/nooa-2607.20709.pdf` (49 pp), the `tycho` reference implementation (17,922 lines),
and `nooa/examples/arc_agi_3` (6,711 lines). Compared against `solvers/tycho_nooa`
(40,393 lines) and the frozen V15 Kaggle candidate.

Every number below is quoted from those artifacts or from our own V15 evidence
(`analysis/deployment-probes/20260903T-public1-action-demand-v15/`).

---

## 1. What "above 10" actually means

RHAE is defined in Tycho §3.1. Per level: `e = min(115, 100·(h/a)²)` when completed, `0` when
not; level `i` of `n` carries weight `i`; game RHAE is the weighted mean **capped by the weighted
share of completed levels**. So RHAE ≤ 100 × (weighted completion fraction). A score above 10
across 25 games is not an efficiency problem — it is a *completion* problem. Nothing else moves it.

Calibration from the two papers:

| System | Model | Public-25 RHAE |
|---|---|---:|
| ARC Prize's own eval, no harness | GPT-5.6 Sol (max effort) | **13.3** |
| NOOA, world-model skill + markdown files | GPT-5.5 | 38.4 |
| NOOA, baseline skill + memory | GPT-5.5 | 41.7 |
| NOOA, world-model skill + memory | GPT-5.5 | **50.2** |
| NOOA, world-model skill + memory | GPT-5.6 Sol | 85.1 |
| Tycho, no world model | Opus 4.8 | 79.07 |
| Tycho, single (our mode) | Opus 4.8 | 85.36 |
| Tycho, orchestrator | Opus 4.8 | 88.49 |
| Tycho, orchestrator | GPT-5.6 Sol / Opus 5 | 100.00 |

Two readings matter. First, **13.3 is what a frontier model scores with no harness at all** — so
"above 10" is roughly "a good harness must rescue a much weaker model to where a raw frontier model
lands." Second, **neither paper evaluates an open-weight model.** Tycho's future work names this as
untested. Our 27B local model sits in a regime nobody in these sources measured.

The one external datapoint that does cover it: ARC Prize's milestone-1 winner (Tufa Labs, "The
Duck") ran **Qwen 3.6 27B FP8 locally** and reported that *"gains came from multimodality and
better base models, not hand-built tools."* (`analysis/deployment-route-understanding.md:67`)

---

## 2. The finding: we already proved the multimodal path works, then played the game blind

### The reference singles out Qwen as the one model that sees the grid exactly

`tycho/agent/vision.py` is the reference's per-model vision table. It assigns
`lossless_cells=False` to Claude, GPT, Gemini and Gemma — all of them downscale, so the image is
unreliable for per-cell color. Qwen is the sole exception:

> Qwen3.x VLM: patch_size 16 × spatial_merge 2 = 32 px per merged visual token. Rendering the
> 64x64 grid at 32 px/cell → 2048x2048 → 128x128 patches → merge → **4096 visual tokens = exactly
> one per ARC cell** if the serving processor is allowed max_pixels >= 4194304 …

The prompt then truthfully tells the model the image is full-resolution
(`actor.system.j2`, the `lossless_cells` branch).

### The model we deploy is a VLM, and the vision weights are on the Kaggle machine

The pinned snapshot `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a…` ships
`preprocessor_config.json` and `video_preprocessor_config.json`
(`deployment/kaggle/PROTOTYPE-qwen3_8_fp8-manifest.json`), and our own route note records the FP8
config as `quant_method: fp8, vision blocks excluded` — i.e. the vision tower is present, in
higher precision, in the 80 files we already verify byte-for-byte on every run.

### It already answered a visual question on the exact Kaggle route

`exact_request_outer_runtime.py:1181-1204` builds the runtime warmup as a base64 PNG
`image_url` chat completion. V15 kernel 13 ran it and it **passed**:

```
warmup_facts: chat_completion_accesses 1, successful 1, elapsed_s 8.103
returned: take_actions({"actions":[{"action":"LEFT"}],"rationale":"red is left of yellow"})
```

Eight seconds into the run, SGLang-served Qwen looked at a picture, reasoned about spatial
relations, and emitted a correct native tool call.

### And then the Solver fed it ASCII for the next 26 minutes

`solvers/tycho_nooa` has **no image path at all**. `vision=False` is hardcoded at
`planner.py:118` and `tests/test_native_flow.py:716`; there is no PNG renderer, no `image_url`
assembly, nothing. Across 40,393 lines the only `image` identifiers refer to Docker images.

What that cost in V15 (`MODEL-BEHAVIOUR-ANALYSIS.md`): Actor calls 1, 2 and 5 each spent the full
**16,000 reasoning tokens hand-parsing the pipe-delimited grid and emitted no tool call** — 48,000
wasted tokens, ~20 of the 26 minutes before the first action. The traces show the model repeatedly
catching its own miscounts (*"wait, I misread earlier"*, *"I mislabeled the groups earlier"*), and
in call 2 writing *"I shouldn't keep reading from the text display — better to use numpy"* before
continuing to read from the text display.

### The economics are not close

| | tokens | rate | time |
|---|---:|---:|---:|
| 64×64 grid as image @ scale 32 | 4,096 prefill | ~6,900 tok/s | **0.6 s** |
| One wasted exploratory parse | 16,000 decode | ~41 tok/s | **390 s** |

Prefill is ~170× faster per token than decode on this box, and the image is 4× smaller than one
failed parse. Trading decode for prefill is the single largest arithmetic lever available.

**Claim ceiling.** Proven: the route serves images and Qwen answers a *trivial* visual question
(256×256, two large blocks) with a valid tool call. **Not** proven: that a 64×64 ARC frame rendered
at 2048×2048 survives SGLang's image preprocessing at 4,096 visual tokens with per-cell color
recoverable. `vision.py` itself conditions on `max_pixels >= 4194304` and untruncated tokenization.
That is one cheap probe, and it must run before anything is built on it.

---

## 3. The other four divergences

### 3.1 We run 25 games serially; both references run them as a concurrent fleet

NOOA's ARC-AGI-3 result is *"25 games per fleet, **one NOOA agent per game**"* under a two-hour
**fleet** wall-clock cap (Fig. 7). Tycho ships `tycho/harness/run_parallel.py`: *"Games are fully
independent. The coordinator supervises one subprocess per game … a semaphore limits concurrent
model workloads."*

Ours is one game at a time. V15's SGLang log shows `#running-req: 1` for the entire run, and
`#cached-token: 0` on nearly every prefill because the actor rebuilds the conversation each turn,
so the radix cache never hits.

The consequence, against the 9-hour Kaggle GPU limit
(`analysis/deployment-route-understanding.md:146`):

| | V15 measured | ×25 games serial |
|---|---:|---:|
| startup + model load | 325 s | 325 s |
| time to first action | 1,561 s | 39,025 s |
| 7 further actions | 34 s | 850 s |
| **total** | 1,921 s | **40,200 s = 11.2 h** |

Over the limit at **8 actions per game**, before any real play. Single-stream decode at 41 tok/s
yields ~1.33M output tokens in 9 hours for all 25 games. Tycho's *cheapest* frontier run (Opus 5)
spent 23.4M output tokens on the same 25 games (Table 6) — we have **5.7%** of that at 1×
concurrency. On fixed hardware, batching is the only way to buy it back: 25 concurrent streams is
what makes the GPU's aggregate throughput available at all.

### 3.2 Our prompt is home-grown doctrine, not either reference's prompt

NOOA's entire ARC-AGI-3 agent instruction is **51 lines**
(`examples/arc_agi_3/skills/interactive-game-solver/SKILL.md`), and it replaced DreamTeam's 1,821
lines of role prompts plus a 4,690-line retrodiction engine — while *advancing* the score/cost
frontier. Its content is concrete and procedural: `encode(grid) -> z` under a declared `Z_SCHEMA`,
`predict(z, action) -> z'`, **retrodict every turn** (predict-vs-observed mismatch is the sole
refinement signal), search over `predict` once trusted, recall memory before deciding, and end
every turn with `submit_actions([...], rationale="predict: ...")` so each batch is a checked
experiment.

Our V15 system prompt was ~9,842 chars of Tycho's `actor.system` **plus** two locally authored
additions found in neither reference:

- `optionality.py` — *"An option is asymmetric exposure plus the right to choose and keep… state
  doubt is paid by rebuild, goal doubt only by contact. Use optionality intelligently."*
- `treatments.py` — the identity swap to *"You are a bounded decision-maker"* plus
  `DECISION_DISCIPLINE_STRATEGY_PROMPT` (*"Investigate only for a result that could change that
  commitment"*).

Abstract epistemology in place of a procedure. The measured behaviour matches: the model did no
encoding, no prediction, no retrodiction, and wrote nothing down.

### 3.3 The call budget makes world-modeling impossible by construction

| | reference | V15 deployment |
|---|---:|---:|
| LM calls / game | 3,500 (Tycho, both stages) | **6 first decision, 1 thereafter** |
| tool steps / turn | 40 | 5 |
| answer budget / call | 24,000 | 16,000 |

Tycho's `single` policy — the mode we deploy — scores 85.36, but at 3,500 calls per game. We give
it **0.17%** of that. NOOA's skill requires encode + predict + retrodict + memory recall *every
turn*; one call per turn cannot execute it. In V15 the Actor never called `wmlib.segment`, never
wrote `world_model.py`, never ran `verify.py` or `plan.py`. The whole Tycho stack sat idle because
the budget forbade using it.

### 3.4 Memory: measured at +11.8 RHAE, and we use the ablation arm

NOOA ran the ablation directly: same skill, same model (GPT-5.5), memory subsystem vs plain
markdown files — **50.2 vs 38.4 RHAE**. Winning games make ≥1 deliberate recall per decision;
recalls per decision correlate with levels completed at Spearman ρ = +0.52.

We use markdown files, and in V15 wrote nothing into them: `notes/actor_beliefs.md` held six
entries, all *"model committed \<action\> with no model-authored beliefs update."* Every turn
started cold. A prior local replay workspace had already correctly identified ls20's colour-11 bar
as a **move budget** — and none of it reached the deployment, which seeds a fresh workspace and
ships only the generic `world_model.py` stub.

### 3.5 Half our solver is experiment scaffolding

`solvers/tycho_nooa/src/tycho_nooa` is 40,393 lines. Of that, **83 files / 19,804 lines** are
`matched_actor_pilot_launch_v2…v7`, `matched_actor_pilot_manifest_v2…v7`, `compatibility_*`,
`parity_*`, `context_*_v2/v3`, `production_shadow_*` — audit and pilot machinery, versioned in
place, inside the package we ship to Kaggle. The reference Tycho is 17,922 lines *including* a
replay viewer, the full ARC harness, and three provider transports.

---

## 4. What follows

Ordered by expected effect on completed levels, which is the only thing RHAE rewards.

1. **Probe the visual channel.** Render a real ls20 frame at 32 px/cell (2048×2048), send it on
   the exact Kaggle route, ask for specified cell values and a bounding box, score against ground
   truth. Also record whether SGLang's preprocessor admits 4,194,304 pixels without downscaling.
   Cheap: no game, no launcher, a handful of requests. This gates everything in §2.
2. **Run games concurrently.** One worker per game against one SGLang server, as both references
   do. This is what converts the GPU's aggregate throughput into usable budget, and it is the only
   way 25 games fit in 9 hours.
3. **Replace the prompt with the 51-line skill's procedure**, and give the Actor enough calls per
   turn to execute it. The two are one change: the procedure is worthless at 1 call/turn.
4. **Carry a world model and notes across levels and turns**, since we cannot ship NOOA's memory
   subsystem offline — at minimum make retrodiction the per-turn contract so a mismatch is forced
   to produce an edit.
5. **Cut the scaffolding out of the shipped package.** Not a score lever; it is what makes the
   first four legible and reviewable in the time remaining.

**The cheap-trial point.** Items 3, 4 and 5 need no Kaggle quota at all. `environment_files/ls20/`
holds the offline game, the NOOA example runs offline against any OpenAI-compatible gateway, and
`runs/` already contains local offline runs on `ar25`. Kaggle GPU is needed only for the
deployment route — which V15 already qualified — and for item 1, which is small.

**What none of this establishes.** No source read here measures an open-weight 27B model on
ARC-AGI-3. The gap between 13.3 (raw frontier, no harness) and 50.2 (good harness, mid-tier
frontier) is the space we are betting in, with a model weaker than either endpoint. The five items
above close known divergences from designs that worked; they are not a prediction of a score.
