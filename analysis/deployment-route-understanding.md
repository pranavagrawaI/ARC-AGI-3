# Deployment route understanding: offline local-model route on ARC-AGI-3 Kaggle

**Status:** living understanding artifact. Sections 1–8 preserve the initial route investigation; §9 records the same-day hardware and serving update and controls where the two differ. No deployable route is chosen until it survives the staged measurements in §9.7.
**Date of research:** 2026-08-29 (UTC). Updated 2026-08-29. All "accessed" dates below are 2026-08-29 unless marked otherwise.
**Scope:** settle the factual boundary that constrains a first offline local-model Kaggle route, from primary sources only. Secondary pages were used only to locate primary ones.

Legend: **[O]** observation (directly quoted or paraphrased from a cited source) · **[I]** inference (logical consequence of observations, labelled as such) · **[U]** unknown (not settled by any primary source available; some need a Kaggle Phase A measurement).

---

## 1. ARC Prize 2026 / ARC-AGI-3 Kaggle rules that constrain an offline local model

### 1.1 Network

- **[O]** ARC Prize's official ARC-AGI-3 competition page lists under "Submission Requirements": *"Submissions must be made through the designated Kaggle competition. No internet access during evaluation. All code and methods must be open sourced to be eligible for prizes. Hardware and compute limits will be announced with the competition launch."*
  — https://arcprize.org/competitions/2026/arc-agi-3 (accessed 2026-08-29)
- **[O]** The official Kaggle Starter docs state: *"All accelerated Kaggle sessions have internet disabled, which is already the default in this kit."* — https://docs.arcprize.org/arc-prize-2026 (source: `arc-prize-2026.mdx` in the official docs repo https://github.com/arcprize/docs, fetched from raw.githubusercontent.com 2026-08-29; same text at https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter README)
- **[I]** Therefore any LLM used at evaluation time must be attached as a Kaggle Model/dataset artifact or shipped inside the notebook payload; a hosted API endpoint is usable only as a development transport. This matches the repo's internal record at `experiments/agents/0002-llm-repl-agent/model-selection.md`.

### 1.2 Competition Mode

- **[O]** Official ARC-AGI-3 docs, "Competition Mode": it is *"REQUIRED to show up on the Unverified leaderboard"* and forces: environments interacted with via the API; scoring against all available environments; only Level Resets permitted (Game Resets become Level Resets); only one `make` call per environment; only a single Scorecard; `get_scorecard` does not work for in-flight scorecards. The docs add: *"The Kaggle Competition is forced into this mode."* — https://docs.arcprize.org/toolkit/competition-mode (source: `toolkit/competition_mode.mdx`, https://github.com/arcprize/docs, fetched 2026-08-29)
- **[I]** Constraint on the agent harness: one interaction pass per environment, no scoring probes mid-run, single scorecard. Any local model route must fit its serving + inference budget inside exactly one sequential pass over the hidden game set.

### 1.3 Notebook / runtime shape and limits

- **[O]** The competition is a **code competition**: *"you submit a notebook, Kaggle runs it twice. 1. **Phase A: Save & Run All.** Kaggle runs your notebook in their real environment and validates that the code executes without errors … 2. **Phase B: Competition Rerun.** Triggered when you click **Submit to Competition** … Your agent plays the hidden game set and your leaderboard score appears."* — https://docs.arcprize.org/arc-prize-2026 (`arc-prize-2026.mdx`, https://github.com/arcprize/docs, fetched 2026-08-29)
- **[O]** Submission format: commit the notebook, click **Submit to Competition**, pick `submission.parquet` from the Output File dropdown. — same source, and https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter README (fetched 2026-08-29)
- **[O]** Per-game wall-clock behavior: the official GPT-OSS-on-Kaggle template warns to keep `GAME_TIME_LIMIT_S` low *"If it's too high you'll hit a `scorecard not produced in time` error, which fails the run."* — https://docs.arcprize.org/partner_templates/gpt-oss-kaggle (`partner_templates/gpt-oss-kaggle.mdx`, https://github.com/arcprize/docs, fetched 2026-08-29)
- **[U]** No primary source fetched states a Kaggle notebook session-length cap (e.g. 9-hour/12-hour GPU session limit), a total Phase B wall-clock budget, a per-notebook output/disk size cap, or the number of daily submissions. The Kaggle competition rules page (`https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules`) requires accepting the rules in an authenticated session; unauthenticated fetches returned only the page shell, and the public competition-list API returned `401 Unauthenticated`. **[U]** Resolution paths: accept the rules in the browser and read the Rules/Overview tabs, or call the authenticated `kaggle competitions` API with the project token.

### 1.4 Accelerators officially specified

- **[O]** Official docs give a four-value accelerator table for the generated notebook: `"cpu"` (no GPU), `"t4"` (Nvidia T4 ×2, default), `"p100"` (Nvidia P100, single big-memory GPU), `"rtx6000"` (*"Nvidia RTX 6000 (`g4-standard-48`) — Heavy ML; **ARC-AGI-3 exclusive**, burns GPU quota faster"*, reserved for ARC-AGI-3 notebooks). — https://docs.arcprize.org/arc-prize-2026 (`arc-prize-2026.mdx`) and https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter README (both fetched 2026-08-29)
- **[O]** The GPT-OSS partner template refers to the same top accelerator as *"the competition's RTX Pro 6000 accelerator"* and instructs *"Set the accelerator to **RTX Pro 6000** in the notebook settings."* — https://docs.arcprize.org/partner_templates/gpt-oss-kaggle (fetched 2026-08-29)
- **[I]** Two official pages name the top accelerator differently ("RTX 6000, g4-standard-48" vs "RTX Pro 6000"). Both agree it is the heavy-ML option reserved for this competition; the exact product name and its VRAM figure are **[U]** from docs alone. The template also states GPT-OSS-120B *"runs **slowly** on the RTX hardware available in this environment"* — the only officially documented performance observation for a large local model on this hardware.
- **[O]** The repo's own kernel metadata (`deployment/kaggle/notebooks/kernel-metadata.json`) sets `enable_gpu: true`, `enable_internet: false`, `enable_tpu: false`, `competition_sources: ["arc-prize-2026-arc-agi-3"]`, `model_sources: []` — an adapter-only notebook with no model attached. This is the recorded starting point, not an official rule.

### 1.5 Leaderboard / hidden split

- **[O]** *"Your agent plays the hidden game set and your leaderboard score appears."* (Phase B) — https://docs.arcprize.org/arc-prize-2026 (fetched 2026-08-29)
- **[O]** Milestone-1 blog: ARC-AGI-3 is *"a set of novel, video-game-like environments … Due to a hidden hold-out set, these games can't be memorized."* — https://arcprize.org/blog/arc-prize-2026-milestone-1 (fetched 2026-08-29)
- **[U]** The exact split between public/dev games and the hidden hold-out set (counts, overlap, whether hidden games are variants of public ones) is not stated in any fetched primary source.

---

## 2. The "Qwen 3.8 27B" candidate: what officially exists

- **[O]** An official model called Qwen 3.8 27B **exists**: `Qwen/Qwen3.8-27B` on Hugging Face, author Qwen, created 2026-08-05, Apache-2.0, `image-text-to-text`, architecture `Qwen3_5ForConditionalGeneration` (model_type `qwen3_5`). — https://huggingface.co/api/models/Qwen/Qwen3.8-27B and https://huggingface.co/Qwen/Qwen3.8-27B (accessed 2026-08-29)
- **[O]** Model card overview: 27B-parameter dense language model **with vision encoder**; hidden dim 5120; 64 layers; 248,320 padded token embedding; hybrid layout of Gated DeltaNet + Gated Attention; MTP-trained; *"Context Length: 262,144 natively and extensible up to 1,000,000 tokens."* Native image **and video** understanding; thinking mode on by default with `enable_thinking` / `preserve_thinking` chat-template kwargs. — https://huggingface.co/Qwen/Qwen3.8-27B model card (accessed 2026-08-29)
- **[O]** Downloadable official checkpoints (Hugging Face API + repo trees, accessed 2026-08-29):
  - `Qwen/Qwen3.8-27B` — BF16, ~55.6 GB total, 18 safetensors shards; Apache-2.0.
  - `Qwen/Qwen3.8-27B-FP8` — FP8 quantization config (`quant_method: fp8`, vision blocks excluded), 30,866,866,928 weight bytes across 66 `.safetensors` files; Apache-2.0; immutable revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. — https://huggingface.co/Qwen/Qwen3.8-27B-FP8 and https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8?blobs=true (accessed 2026-08-29)
  - No official Qwen-published GGUF/AWQ/GPTQ 27B variants found in the Qwen author listing (third-party GGUF/MLX exist, e.g. `unsloth/Qwen3.8-27B-GGUF`, but are not official Qwen artifacts).
- **[O]** Tokenizer / chat template: both checkpoints ship `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` at repo root (file trees, accessed 2026-08-29). The card documents `chat_template_kwargs: {"enable_thinking": True, "preserve_thinking": True}`.
- **[O]** Officially documented local runtimes for Qwen3.8: the card states artifacts are *"compatible with Hugging Face Transformers, vLLM, SGLang, TokenSpeed, etc."* and links a published **SGLang Qwen3.8 Cookbook** and **vLLM Qwen3.8 Recipe**. The vLLM recipe gives an FP8 TP4 launch at 262,144 context and lower-precision single-GPU routes. The SGLang cookbook reports single-GPU boot-and-serve coverage on H200, RTX PRO 6000, RTX 5090, and DGX Spark across card-appropriate precisions; it cautions that the approximately 28.5 GB FP8 weights leave little service headroom on 32 GB cards. Recommended sampling: thinking mode `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0`; instruct mode `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5`. — https://huggingface.co/Qwen/Qwen3.8-27B, https://recipes.vllm.ai/Qwen/Qwen3.8-27B, and https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B (accessed 2026-08-29)
- **[U]** Those runtime recipes do not establish that Kaggle's ambiguously named `rtx6000` tier is the same tested RTX PRO 6000 product, that the Kaggle image contains a compatible runtime build, or that the FP8 route has enough headroom for this Solver's workload. The hosted-Model Studio pairing for Qwen3.8 is announced as *"coming soon"*, so the model card supplies no hosted/local parity guarantee. The existing OpenRouter route is a separate observed transport, not checkpoint-identity evidence.

---

## 3. The recorded `Qwen/Qwen3.6-27B-FP8` route: the same exact facts

- **[O]** `Qwen/Qwen3.6-27B-FP8` exists, author Qwen, last modified 2026-04-24, Apache-2.0, `image-text-to-text`, `base_model: Qwen/Qwen3.6-27B`, FP8 (`quant_method: fp8`), architecture `Qwen3_5ForConditionalGeneration` (`qwen3_5`). — https://huggingface.co/api/models/Qwen/Qwen3.6-27B-FP8 (accessed 2026-08-29)
- **[O]** Repo tree: 30,866,866,928 weight bytes across 66 `.safetensors` files, plus `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`; immutable revision `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`. — https://huggingface.co/api/models/Qwen/Qwen3.6-27B-FP8?blobs=true (accessed 2026-08-29)
- **[O]** Model card: causal LM with vision encoder; context *"262,144 natively and extensible up to 1,010,000 tokens"*; thinking preservation (`preserve_thinking`); OOM guidance recommending ≥128K context to preserve thinking. — https://huggingface.co/Qwen/Qwen3.6-27B-FP8 (accessed 2026-08-29)
- **[O]** Officially documented local runtimes: card recommends `vllm>=0.19.0` and `sglang>=0.5.10`, with full launch commands including `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, speculative decoding (`qwen3_next_mtp` / NEXTN), and a `--language-model-only` vLLM flag that skips the vision encoder to free KV-cache memory. Transformers serving (`transformers serve … --continuous-batching`) and KTransformers are also documented. — same model card (accessed 2026-08-29)
- **[O]** Official ARC Prize evidence of feasibility: milestone-1 recap — 1st place, Tufa Labs, "The Duck": *"Runs Qwen 3.6 27B FP8 locally."*; the harness evicts oldest messages to keep playing without exhausting context; *"gains came from multimodality and better base models, not hand-built tools."* Runners-up used locally-served Gemma-4-31B. — https://arcprize.org/blog/arc-prize-2026-milestone-1 (fetched 2026-08-29)
- **[O]** Prior internal record already contains the correct primary citations for this route: `experiments/agents/0002-llm-repl-agent/model-selection.md` (verified this session; no changes made).

---

## 4. Contradiction analysis: "Qwen 3.8 27B" vs "Qwen3.6-27B-FP8"

- **[O]** Both designations resolve to real, distinct, official artifacts. `Qwen/Qwen3.8-27B(-FP8)` (created Aug 2026) and `Qwen/Qwen3.6-27B-FP8` (created Apr 2026) are different releases of the same `qwen3_5` architecture family; the shared board name "Qwen 3.8 27B" is the colloquial form of the Qwen3.8 model ID, not a typo of Qwen3.6.
- **[O]** No contradiction in existence — the two names refer to different checkpoints. The apparent referential conflict is chronological: the older internal record (`experiments/agents/0002-llm-repl-agent/model-selection.md`) pins `Qwen/Qwen3.6-27B-FP8`, while the newer whole-system ledger, exact hosted runs, and shared-board planning explicitly designate Qwen 3.8 27B as the current intended behavioral family. The older Qwen3.6 record is therefore stale as model-family authority, while its deployment precedent remains relevant evidence.
- **[I]** The facts that differ and would matter to an offline route: Qwen3.8-27B has a published dense-27B model card with video understanding and MTP; Qwen3.6-27B-FP8 has versioned, command-level vLLM/SGLang recipes and the milestone-1 winning-submission precedent. Neither card claims bit-level equivalence with any hosted endpoint (Qwen3.8 hosting is "coming soon"; the Qwen3.6 record already flags this gap as not verified).
- **[U]** The intended family is not ambiguous: it is Qwen 3.8 27B. What remains unresolved is the exact deployable artifact and precision within that family. The official FP8 checkpoint is the first documented candidate because BF16 weights are larger and the board makes deployment fit conditional, but no artifact is selected until the accelerator/runtime measurement establishes it.

---

## 5. Facts that require a Kaggle Phase A measurement, not documentation

**[U]** Each item below cannot be settled from the fetched primary sources; a Phase A (Save & Run All) run is the instrument that answers it:

1. **Whether a ~31 GB FP8 checkpoint fits the accelerator's memory and disk** at any workable context length, and whether the `"rtx6000"`/`"RTX Pro 6000"` tier — whose exact VRAM is undocumented — is required versus T4×2/P100. (Official evidence for a large local model on this hardware is limited to the GPT-OSS-120B "runs slowly" remark.)
2. **Per-game and total wall-clock feasibility** of serving + inference under Competition Mode's single-pass constraint — i.e. whether the local server plus agent can finish the hidden set before the *"scorecard not produced in time"* failure mode. The number and duration of hidden games are undocumented.
3. **Kaggle model-attachment mechanics in practice**: whether `model_sources` accepts the Qwen checkpoint directly, or the notebook must download/vend weights inside the payload, and the resulting Phase A startup time. The official starter's metadata schema exposes the field, but no official doc on its use for a 31 GB model was found.
4. **Actual Phase A/B session limits** (max session duration, output-size caps, submission quotas) — visible only on the authenticated competition pages.
5. **Whether the documented launch recipes (8-GPU tensor-parallel examples for Qwen3.6) degrade to the single-GPU RTX tier** without the `--language-model-only`-style memory tradeoffs — a runtime question, observable only on Kaggle hardware.

---

## 6. Sources used (primary only)

| # | Source | URL | Accessed |
|---|--------|-----|----------|
| 1 | ARC Prize 2026 — ARC-AGI-3 competition page | https://arcprize.org/competitions/2026/arc-agi-3 | 2026-08-29 |
| 2 | ARC-AGI-3 docs: ARC Prize 2026 Kaggle starter (`arc-prize-2026.mdx`) | https://docs.arcprize.org/arc-prize-2026 · source https://github.com/arcprize/docs | 2026-08-29 |
| 3 | ARC-AGI-3 docs: Competition Mode (`toolkit/competition_mode.mdx`) | https://docs.arcprize.org/toolkit/competition-mode · source https://github.com/arcprize/docs | 2026-08-29 |
| 4 | ARC-AGI-3 docs: GPT-OSS on Kaggle template (`partner_templates/gpt-oss-kaggle.mdx`) | https://docs.arcprize.org/partner_templates/gpt-oss-kaggle · source https://github.com/arcprize/docs | 2026-08-29 |
| 5 | ARC-AGI-3 Kaggle Starter repo (README, metadata schema) | https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter | 2026-08-29 |
| 6 | ARC Prize 2026 Milestone 1 recap | https://arcprize.org/blog/arc-prize-2026-milestone-1 | 2026-08-29 |
| 7 | Qwen/Qwen3.8-27B model card + HF API record | https://huggingface.co/Qwen/Qwen3.8-27B | 2026-08-29 |
| 8 | Qwen/Qwen3.8-27B-FP8 model card + file tree | https://huggingface.co/Qwen/Qwen3.8-27B-FP8 | 2026-08-29 |
| 9 | Qwen/Qwen3.6-27B-FP8 model card + file tree | https://huggingface.co/Qwen/Qwen3.6-27B-FP8 | 2026-08-29 |
| 10 | HF models API (author=Qwen, search for 3.8 / 3.6 checkpoints) | https://huggingface.co/api/models | 2026-08-29 |
| 11 | Internal record (observed, not a claim source): model-selection.md | `experiments/agents/0002-llm-repl-agent/model-selection.md` (this repo) | 2026-08-29 |
| 12 | Internal kernel metadata (observed): kernel-metadata.json | `deployment/kaggle/notebooks/kernel-metadata.json` (this repo) | 2026-08-29 |

Note on fetching: docs.arcprize.org is JS-rendered; content was read from its official source repository `github.com/arcprize/docs` via raw.githubusercontent.com, which the site itself links to (footer: "https://github.com/arcprize/docs").

## 7. Verification performed

- Cross-checked every competition constraint against at least two primary pages (site page + docs repo / starter repo) where available.
- Confirmed model existence, license, architecture, size, and file inventory via the Hugging Face API JSON records and repo trees, not just the rendered cards.
- Read the internal record and kernel metadata directly; no other repo file was modified.
- Explicitly left as **[U]** anything the sources did not state (session limits, VRAM, hidden-set composition, Qwen3.8 runtime version floors, split ambiguity between the two Qwen names) rather than inferring them.

## 8. Unresolved questions (summary)

This section preserves the unresolved set at the end of the initial investigation. The update in §9 resolves or narrows items 1–3 and states the current unresolved set in §9.9.

1. Which exact Qwen 3.8 artifact, precision, runtime, and context allocation can satisfy the measured Kaggle envelope? The official FP8 checkpoint is the first documented candidate; Qwen3.6 is deployment precedent, not the current intended family.
2. Exact accelerator product/VRAM of the ARC-AGI-3-exclusive tier ("RTX 6000 g4-standard-48" vs "RTX Pro 6000"). (authenticated Kaggle UI / Phase A)
3. Session-length, disk, and submission limits on the competition. (authenticated Kaggle UI)
4. Hidden vs public game-set composition. (official pages so far silent)
5. All Phase A measurement items in §5.

---

## 9. Update: same-GPU serving evidence changes the architecture worth testing

### 9.1 Source classes and authority

The initial investigation admitted only primary sources. This update adds **deployment measurements** from three independently published benchmark reports because the question has changed from “what artifacts and hardware officially exist?” to “what operating points have survived contact with this GPU?” Each observation below names its source class:

- **Official specification/model artifact:** owns hardware or model facts.
- **Independent benchmark:** useful operational evidence, but not authority for Kaggle compatibility or Solver behavior.
- **Community deployment report:** a reproducible configuration and field observation, but neither an official Qwen benchmark nor a controlled comparison unless stated.

The reports establish candidate operating points. They do not establish ARC progress, hosted/local behavioral parity, or exact Kaggle deployment viability.

### 9.2 Hardware and time envelope now established

- **[O · official specification]** Google Cloud's G4 machine table maps `g4-standard-48` to **48 vCPUs, 180 GB instance memory, one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU, and 96 GB GDDR7 GPU memory**. Google notes that one vCPU is one hardware hyper-thread. — https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4_machine_types
- **[O · official competition page]** The ARC-AGI-3 Kaggle accelerator page names `g4-standard-48` for the competition and the GPU notebook limit as **9 hours**. — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/upgraded-accelerators
- **[I]** The earlier product/VRAM ambiguity in §§1.4, 5, and 8 is resolved at the hardware-specification level: the relevant machine type carries one 96 GB RTX PRO 6000 Blackwell Server Edition. A Phase A run remains necessary to establish the exact Kaggle image, quotas, mounted-artifact behavior, and usable memory after its driver/runtime overhead.

### 9.3 Qwen3.8-27B: three facts that must not be blended

#### A. BF16 weights + FP8 KV + EAGLE

- **[O · community deployment report]** A Featurize Cloud deployment on one 96 GB RTX PRO 6000 used Qwen3.8-27B BF16 weights, FP8 KV cache, SGLang/FlashInfer, and EAGLE speculative decoding. It reported **77–80 output tok/s** (77.6 rolling average, 80.3 peak), a **128.6 s model load**, a 356,241-token KV pool, and successful 60K+ token agent traffic. — https://huggingface.co/Qwen/Qwen3.8-27B/discussions/160
- **[O]** Its launch fixed `--max-running-requests 8`. Its memory account was approximately 51.05 GB main weights, 5.53 GB MTP draft, 9.1 GB SSM state, 11.4 GB FP8 KV, and 2 GB graphs/overhead. The SSM allocation was described as covering roughly 13K average request length at that configuration.
- **[O]** The reported **~2.2×** is against a *theoretical* ~35 tok/s non-speculative bandwidth estimate, not a measured EAGLE-on/EAGLE-off A/B benchmark. It must not be cited as an observed 2.2× treatment effect.
- **[O]** The report warns that rebuilding the image can trigger 3–5 minutes of FlashInfer JIT compilation and requires `nvcc` on `PATH`.
- **[I]** This is strong single-card fit evidence for BF16 + EAGLE and weak evidence about concurrency beyond eight real CodeAct sessions. The 262K context setting demonstrates a maximum per-request route, not 8, 16, or 32 simultaneous 262K contexts.

#### B. Official FP8 weights, no speculative decoding, short fixed workload

- **[O · independent benchmark]** Growth Systemes served the official Qwen3.8-27B FP8 checkpoint on one RTX PRO 6000 Blackwell Server Edition and swept concurrency using **1,024 input tokens, 256 output tokens, greedy decoding, thinking disabled, EOS ignored**, and OpenAI-compatible streaming. It reported aggregate output throughput of **44.84 tok/s at C1, 152.14 at C4, 280.32 at C8, 451.93 at C16, and 639.66 at C32**. All 576 benchmark requests completed. — https://www.growthsystemes.com/blog/qwen3-8-27b/
- **[O]** At C32, p95 TTFT was 5.617 s and p95 end-to-end latency was 13.27 s. Its strict interactive goodput became zero at C32, although batch throughput was maximal. At C16, p95 end-to-end latency was 9.46 s and aggregate output throughput was 451.93 tok/s.
- **[I]** This establishes strong continuous-batching scaling for short, non-thinking requests. It does not establish the same scaling under Tycho's growing CodeAct histories, images, preserved reasoning, variable output lengths, or prefix-cache behavior.

#### C. Official model capability claims

- **[O · official model card]** Qwen reports 73.0 on Terminal-Bench 2.1 and 61.7 on SWE-bench Pro for Qwen3.8-27B, along with multimodal and agentic results. — https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- **[I]** These results justify treating Qwen as the stronger serious-reasoning candidate. They do not measure ARC-AGI-3 behavior or the effect of FP8, EAGLE, concurrency, or this Solver harness.

### 9.4 Gemma 4 26B-A4B is a triage candidate, not yet a triage solution

- **[O · official model card]** Gemma 4 26B-A4B is a 25.2B-parameter mixture-of-experts model with **3.8B active parameters**, 128 experts with 8 active plus one shared expert, native text+image input, and a 256K context window. — https://huggingface.co/google/gemma-4-26B-A4B
- **[O · independent benchmark]** Millstone AI measured the FP8 model on one 96 GB RTX PRO 6000 using vLLM, FP8 KV, no prompt caching, no speculative decoding, 1–10 concurrent requests, and 1,024 output tokens. At 1K context it reported **139.0 tok/s** for a single request and **674.2 aggregate output tok/s at 10 concurrent requests**. Across its broader 1K–256K suite, all 4.7K requests succeeded. At 256K context, single-request generation was 76.9 tok/s and TTFT was 43.3 s. — https://www.millstoneai.com/inference-benchmark/gemma-4-26b-a4b-fp8-1x-rtx-pro-6000-blackwell
- **Correction:** the 139/674 throughput numbers belong to Millstone AI, not the Hugging Face model card. The model card owns the architecture and capability claims.
- **[I]** Gemma's speed makes it a plausible portfolio probe. Its value depends on ranking recall: whether it preserves the games on which later Qwen computation would produce score. Generic reasoning benchmarks and raw throughput cannot answer that.

### 9.5 Quantization boundary

- **[O · official artifact inventory]** The official Qwen3.8-27B-FP8 weights total 30,866,866,928 bytes (28.747 GiB), as recorded in §2.
- **[O · third-party artifact]** `bernhardbrieger/Qwen3.8-27B-GPTQ-Int4` reports a 19.54 GB footprint, keeps the MTP draft and other precision-critical modules in BF16, and verifies serving to 163,840 tokens with FP8 KV on vLLM 0.27.1. — https://huggingface.co/bernhardbrieger/Qwen3.8-27B-GPTQ-Int4
- **[I]** INT4 buys roughly 9 GB versus the official FP8 artifact, but no controlled evidence here establishes better RTX PRO 6000 throughput or acceptable ARC behavior. It remains a fallback only if FP8's cache/state capacity is the measured bottleneck.

### 9.6 Architecture now authorized for testing

“Live games” must be separated from GPU concurrency:

```text
B = logical portfolio breadth      # resumable game states
W = runnable CPU workers           # environment/tool execution
C = in-flight model generations    # GPU admission limit
K = games receiving serious budget

B game states
    → portfolio scheduler
    → W asynchronous Tycho workers
    → C admitted inference requests
    → one vLLM/SGLang continuously batched server
```

- **[Decision]** Test an asynchronous, work-conserving portfolio architecture. Up to all 110 games may remain logically resumable.
- **[Decision]** Do **not** equate `B`, `W`, and `C`. In particular, the evidence does not yet authorize 16 or 32 simultaneous Tycho generations merely because 16 or 32 short benchmark requests scaled.
- **[Decision]** Let the serving engine own token-level continuous batching. The Solver scheduler owns game priority and request admission.
- **[Decision]** Begin with a fair work-conserving queue. Dynamic “continuous repricing” is a later treatment; adding it to the first concurrency test would confound serving concurrency with allocation policy.
- **[Decision]** Qwen3.8-27B FP8 remains the first serious-reasoning route. Gemma 4 26B-A4B is a separately falsifiable triage candidate. INT4 is deferred.

What changed: the serial mental model—one game waits while the GPU is idle during another game's CPU/tool work—is no longer the only credible route. What did not change: only a Kaggle Phase A deployment and Tycho-shaped workload can establish the usable operating point.

### 9.7 Staged experiment that separates the live claims

#### Stage 1 — exact offline Kaggle deployment

Attach the exact model artifact and pinned runtime to a network-disabled Phase A notebook, then:

1. cold-load the server;
2. exercise one visual Tycho request and one tool-call round trip;
3. verify the reasoning parser, tool-call parser, context limit, FP8 kernels, and cache paths;
4. record model/JIT startup time, usable VRAM after load, runtime versions, and raw logs.

This stage establishes deployment feasibility only. It does not select a model or concurrency.

#### Stage 2 — trace-shaped serving sweep with Qwen fixed

Replay a frozen Tycho request corpus that preserves the observed distributions of:

- prompt and growing-history lengths;
- preserved reasoning blocks;
- image sizes;
- tool schemas;
- requested and actual output lengths;
- prefix-cache hits and misses.

Sweep `C = 1, 4, 8, 16, 32`, stopping on OOM, swapping, request failure, or a sustained throughput plateau. Measure complete Tycho inference turns per minute, input and output throughput separately, TTFT, generation latency, request age/starvation, prefix-cache hit rate, VRAM, SSM/KV occupancy, and truncations.

Select the **smallest C within 5% of the maximum trace-shaped throughput**. This retains latency and memory headroom without sacrificing material capacity. Output tok/s alone is not the selection measure: repeated long-history prefill and cache churn may dominate the real loop.

#### Stage 3 — live asynchronous effect with model and policy fixed

Compare:

```text
A: Qwen, serial admission C=1
B: same Qwen, same Solver policy, asynchronous admission C=C*
```

Freeze game set, initial states, action ceilings, model settings, and wall-clock cutoff. Preserve complete traces. Record:

1. score at fixed wall-clock cutoffs; and
2. normalized area under the retained best-score curve:

\[
\frac{1}{T}\int_0^T S_{\mathrm{best}}(t)\,dt
\]

The second measure rewards early progress only when it is retained and prevents “touching many easy games” from masquerading as durable portfolio performance. Estimate run variance before setting the threshold for “markedly better.”

#### Stage 4 — Gemma triage value

Under a fixed time and action budget:

```text
Gemma probes/ranks all games
    → Qwen works the promoted top K
    → Qwen audits a random sample of Gemma-rejected games
```

Compare against Qwen-only allocation under the same budget. Measure final score, score-time area, triage cost, productive rejected games, and estimated false-negative rate. Gemma survives only if the additional breadth produces more retained Qwen-convertible opportunity than its triage cost and missed-game regret.

### 9.8 Irreversibility test for cheap probing

The uncertainty that can kill the Gemma stage is whether a probe can be rejected without spending the option it is meant to create.

- If game states or action histories can be reset, cloned, or replayed cheaply, Gemma can buy information while preserving Qwen's future choices.
- If Gemma consumes scarce actions or damages the only live trajectory, it is not cheap triage; it is a weaker model spending the serious model's opportunities.

Gemma must not own irreversible actions until the harness proves that downside is bounded or a Qwen-controlled commitment boundary is defined.

### 9.9 Current unresolved questions and decision

1. Can the exact Qwen3.8 artifact plus pinned SGLang/vLLM runtime cold-load and complete a visual/tool Tycho turn in the network-disabled Kaggle image?
2. Where is the concurrency knee for actual Tycho histories, images, thinking, and cache behavior?
3. Does asynchronous Qwen improve retained ARC score per wall-clock minute, rather than only aggregate tokens per second?
4. Can probing be reset/replayed cheaply enough for Gemma to create rather than consume optionality?
5. Does Gemma's ranking preserve Qwen-convertible games with acceptably low false-negative regret?
6. Do INT4's extra cache/state headroom and kernels ever outweigh FP8's quality and Blackwell throughput route?

**Current route judgment:** commit to testing asynchronous portfolio serving; do not commit to C16/C32, dynamic repricing, Gemma-owned committed actions, or INT4. The first decision-reversing uncertainty is the concurrency knee under exact Tycho traffic.

### 9.10 Additional sources for this update

| # | Source | Class | URL |
|---|---|---|---|
| 13 | Google Cloud G4 machine types | Official specification | https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4_machine_types |
| 14 | Kaggle ARC-AGI-3 upgraded accelerators | Official competition page | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/upgraded-accelerators |
| 15 | Qwen3.8 BF16 + FP8 KV + EAGLE deployment report | Community deployment report | https://huggingface.co/Qwen/Qwen3.8-27B/discussions/160 |
| 16 | Qwen3.8-27B FP8 concurrency benchmark | Independent benchmark | https://www.growthsystemes.com/blog/qwen3-8-27b/ |
| 17 | Gemma 4 26B-A4B official model card | Official model artifact | https://huggingface.co/google/gemma-4-26B-A4B |
| 18 | Gemma 4 26B-A4B RTX PRO 6000 benchmark | Independent benchmark | https://www.millstoneai.com/inference-benchmark/gemma-4-26b-a4b-fp8-1x-rtx-pro-6000-blackwell |
| 19 | Qwen3.8-27B FP8 official model card | Official model artifact | https://huggingface.co/Qwen/Qwen3.8-27B-FP8 |
| 20 | Qwen3.8-27B GPTQ Int4 checkpoint | Third-party artifact | https://huggingface.co/bernhardbrieger/Qwen3.8-27B-GPTQ-Int4 |

---

## 10. R7 phase boundary: from feasibility research to a release ladder

### 10.1 What R7 resolved

- **[O · Kaggle Phase A]** The explicit batch machine shape `NvidiaRtxPro6000` allocated an NVIDIA RTX PRO 6000 Blackwell Server Edition with 97,887 MiB total and 97,250 MiB free before startup.
- **[O]** All 80 required files for `Qwen/Qwen3.8-27B-FP8` revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` matched the frozen manifest. Verification took 138.143 seconds.
- **[O]** SGLang loaded all 66 shards in 2.45 seconds. Weights consumed 28.47 GB; runtime allocation then used 23.91 GB SSM state, 0.47 GB convolution state, and 27.14 GB FP8 KV cache, leaving 14.14 GB.
- **[O]** Startup failed before readiness during FlashInfer prefill CUDA-graph capture. FlashInfer logged `Failed to get device capability: SM 12.x requires CUDA >= 12.9` and then raised `FlashInfer requires GPUs with sm75 or higher`. No request reached the model.
- **[Decision]** Qwen3.8-27B FP8 is the behavioral incumbent. The frozen T4-oriented SGLang/FlashInfer route is rejected. Capacity no longer justifies a model change or INT4 work.

Raw evidence and the bounded verdict are under `analysis/deployment-probes/20260829T140038Z-qwen38-phase-a-kaggle-v7/`.

### 10.2 Submission cadence now resolved

- **[O · authenticated official rules, accessed 2026-08-29]** Competition-specific rule 2 permits at most **one submission per day** and up to **two final submissions** for judging. — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules
- **[I]** A failed competition canary costs at least one day of submission cadence even if its GPU spend is small. The first canary must therefore be the planned baseline submission, launched as soon as an exact S0 candidate exists rather than postponed until after performance optimization.

### 10.3 Release ladder

The deployment campaign now advances through these separately falsifiable gates:

| Gate | Exact question | Required evidence | Granted next capability |
|---|---|---|---|
| R8-A | Can Qwen initialize on Blackwell without the failed FlashInfer path? | healthy server plus exact backend/runtime manifest | one contract request |
| R8-B | Can Qwen execute the model-server contract? | visual input plus valid native tool call and raw parser evidence | S0 integration |
| F0 | Can the exact smaller fallback execute the same contract? | cold install → launch → readiness → identical visual request → native tool call → parser acceptance → clean shutdown | exercisable fallback; no behavioral authority |
| S0 | Can the exact Kaggle package execute one real Tycho action loop? | cold notebook → server → Actor → native tool call → ARC action → resulting observation | construct immutable canary candidate |
| S1 | Can Kaggle score those exact candidate bytes? | Phase B submission identity, complete rerun evidence, and leaderboard/scoring result | emit an additive promotion attestation; composed QRM-1 authorizes P1 |
| P1 | Where is the trace-shaped concurrency knee? | `C=1,4,8,16,32` sweep against QRM-1 | select production `C*` |
| P2 | Does asynchronous Tycho beat serial Tycho? | fixed-work serial/async contrast using complete decisions per wall-clock second and retained score | select scheduler |
| B1 | Does the Solver compete? | scored game-set evidence from the selected route | resume behavioral selection against the incumbent |

R8-A and F0 may proceed in parallel because they use separate route identities and answer independent release risks. R8-B follows only a healthy R8-A route. S0 and S1 remain serial because the scored artifact must be the exact S0 candidate.

### 10.4 Qualified Route Manifest authority

Do not let “Qwen works” or “SGLang works” become route identity. Preserve two immutable artifacts:

1. **`QRM-CANDIDATE-1` after S0:** an immutable execution capsule binding the exact artifact graph that completed one competition-shaped action. It grants only one bounded S1 canary submission and is never rewritten.
2. **`QRM-PROMOTION-1` after S1:** a separate attestation that binds the candidate manifest digest, recomputes its closure, proves the submitted notebook/version is byte-identical to the candidate, and binds the Kaggle rerun and scoring result. It grants P1 only in composition with that candidate.

Logically, `QRM-1 := QRM-CANDIDATE-1 + QRM-PROMOTION-1`. Evidence never gains authority by mutation. A failed S1 produces failure evidence but leaves the candidate bytes unchanged and produces no promotion attestation.

A completed S0 run is not a scored route. A scored notebook with no pre-submission candidate freeze is not promotable. A diagnostic run may explain failure but may construct neither artifact.

Every route execution capsule must bind at least:

```text
route/stage identity; model revision and model-file manifest digest;
runtime-bundle, notebook, source-wheel, and dependency-lock digests;
machine shape, driver, CUDA runtime, CUDA_HOME, Python and framework versions;
attention, sampling, graph-capture, cache, and reasoning/tool-parser settings;
launch command and environment-variable allowlist;
startup, model-load, first-token, and request timings;
GPU memory before load, after load, peak, and terminal state;
server stdout/stderr and exit status;
request bytes/digest, raw response bytes/digest, and parsed tool call;
environment action, predecessor observation digest, and resulting observation digest;
all referenced evidence paths and their digests.
```

The candidate and promotion constructors consume real paths and recompute digests; they do not accept caller-asserted success labels or digest-shaped strings as authority. The schemas are separate: `deployment/kaggle/qualified-route-candidate.schema.json` describes frozen S0 evidence, and `deployment/kaggle/qualified-route-promotion.schema.json` describes the additive S1 grant.

### 10.5 Campaign separation until QRM-1

Until QRM-1 exists, the **deployment campaign** owns runtime, packaging, lifecycle, evidence, fallback, Competition Mode, and scoring-route work. The **Solver campaign** remains behaviorally frozen except for changes strictly required to cross S0. In particular, do not combine CUDA/runtime diagnosis with Builder/Actor policy, world-model, memory, planning, or exploration changes.

Performance optimization follows scoring, not merely server readiness. P1 measures complete Tycho decisions per wall-clock second, not raw tokens or requests per second. Select the smallest concurrency within 5% of maximum trace-shaped decision throughput while preserving a precommitted recovery-memory reserve and near-zero OOM/timeout incidence. The observed 33-request Mamba-state cap is a cliff sign, not a target.

### 10.6 Immediate campaign

1. R8-A: hold model, hardware, runtime bundle, sampling configuration, and offline boundary fixed; change only the attention backend from FlashInfer to Triton. Before interpretation, record the selected attention backend, resolved sampling backend, whether FlashInfer was imported and whether its rejected runtime path executed, whether graph capture occurred and under which backend, and whether readiness was reached without a CUDA/kernel exception. A failure is valid only if classified as residual FlashInfer reachability, Triton/SM120 kernel incompatibility, CUDA/toolkit discovery, or a named unrelated initialization failure. Do not vary another serving flag in the same run.
2. F0 in parallel: choose one exact smaller multimodal, native-tool-capable artifact and qualify cold install, launch, readiness, the identical visual request, native tool generation, parser acceptance, and clean shutdown without behavioral development.
3. R8-B: send the frozen visual/tool request only after R8-A reaches readiness.
4. S0: integrate the qualified primary route through one real Actor-owned environment action and successor observation.
5. Freeze `QRM-CANDIDATE-1`; submit it as the first deliberately boring Phase B canary at the next available daily slot.
6. After scoring, leave the candidate untouched and emit `QRM-PROMOTION-1` against its digest. Only the validated composition is QRM-1; then begin P1.
