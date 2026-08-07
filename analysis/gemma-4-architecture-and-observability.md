# Gemma 4 architecture and ARC-AGI-3 Solver observability

## Scope, terminology, and evidence

The model is one possible **Game Judgment** component; the complete situated system is the **Solver**. In repository terminology, Game Judgment chooses the hypothesis, experiment, computation, plan, or action with greatest leverage for the next real decision, while host machinery owns reproducible measurement and execution (`CONTEXT.md`, lines 55–64). A Solver trace therefore reflects the model, prompt/template, image and grid construction, context policy, tools, parsers, provider, validators, action-release logic, and environment—not transparent access to model internals.

Primary source: Gemma Team, *Gemma 4 Technical Report*, attached PDF `/mnt/c/Users/hipra/Downloads/gemma-4-technical-report.pdf`, arXiv:2607.02770v2 (24 July 2026), 17 pages, SHA-256 `b20b25f18d76217d0c49c89f9ed1ffe3cf1912cd8ce710e1a325977653c7bfa6`.[^report] Page references below are PDF/printed pages.

ARC evidence inspected only to make the mapping concrete:

- the B.4 frozen-state competence ladder, `experiments/agents/0007-guided-control/runs/0007-b4-gemma-competence-ladder-ls20-v2/{REPORT.md,manifest.json,result.json,trace.jsonl}`;
- the v32 long-run report and manifest under `experiments/agents/0007-guided-control/artifacts/b1/0007-b1-full-checkpointed-retention-open-routing-reasoning-off-ls20-44a-v32/`;
- the pinned `modelrun/fp4` probe manifest at `experiments/probes/nooa-gemma-epistemic-context/treatment-manifest.json`.

The four sections below intentionally separate source facts, implications, hypotheses, and prohibited inferences.

---

## 1. Report observations

### 1.1 Dense and MoE family members

Gemma 4 is a decoder-only Transformer family with pre/post RMSNorm and QKNorm. E2B, E4B, 12B, and 31B are dense; `26B-A4B` is the sole mixture-of-experts member, with 26B total and 3.8B activated parameters. E2B/E4B have larger total counts because they use per-layer embeddings but are named by effective parameters. The 31B used in the inspected ARC runs is therefore **dense**, not MoE. [Report §2 and Table 1, p. 2]

The first-party model card specifies 128 total experts, eight active plus one shared for 26B-A4B.[^card] That implementation detail is not needed to describe 31B behavior and must not be projected onto it.

### 1.2 Frozen vision encoders versus unified encoder-free 12B

The image path depends on model size:

- E2B/E4B use a 150M ViT vision encoder.
- 26B-A4B/31B use a 550M ViT vision encoder.
- 12B has no separate vision or audio encoder. [Report §2.1–2.3 and Table 1, pp. 2–3]

The encoder-bearing vision models use 16×16 patches, non-causal attention, axial 2D-RoPE, and 2D absolute position embeddings. Table 10 gives 16 layers at width 768 for the 150M encoder and 27 layers at width 1152 for the 550M encoder. The report says the vision encoder is kept frozen during pre-training. [Report §2.1–2.2, pp. 2–3; Appendix Table 10, p. 16]

The unified 12B is trained from scratch to consume 48×48×3 RGB patches. A 35M-parameter matrix multiplication projects each patch into the LLM embedding space; 2D coordinate embeddings and a final LayerNorm preserve spatial indexing. “Encoder-free” therefore means no separate ViT, not no image preprocessing or no visual bottleneck. [Report §2.3, p. 3]

The official pinned configs agree: 31B declares a 27-layer, width-1152 vision encoder with patch size 16 and pooling kernel 3; 12B declares `model_patch_size: 48`, `mm_posemb_size: 1120`, and a unified vision projection.[^31cfg][^12cfg]

### 1.3 Image resizing, resolution, patches, and soft tokens

All models support maximum visual-token budgets of 70, 140, 280, 560, or 1120. For the encoder path, a resized image is divided into 16×16 patches, processed by the ViT, then pooled in 3×3 neighborhoods. One final soft-token cell consequently spans 48×48 resized pixels and aggregates nine encoder patches. [Report §2.1, p. 2; Appendix Fig. 2 and Algorithm 1, p. 16]

Algorithm 1 uses `m = k·p = 48`, target area `T = N_max·m²`, scale `f = sqrt(T/(H·W))`, and independently rounds ideal height and width **down** to multiples of 48 before bicubic resizing. The realized token count can thus be less than the nominal budget. The report's 572×1024 example at budget 10 becomes 96×192 and yields eight soft tokens. [Report Appendix Fig. 2 and Algorithm 1, p. 16]

The official 31B and 12B processor artifacts default to `max_soft_tokens: 280`; the 31B artifact records patch size 16 and pooling kernel 3, while 12B directly uses the effective 48-pixel patch.[^31proc][^12proc] Google's image guide explicitly frames budget as a speed/detail trade-off and notes that 280 allows at most 2520 pre-pooling patches because each 3×3 group becomes one embedding.[^vision]

The report evaluates vision at 1120 tokens in Table 6 and 280 in Table 12. Resolution effects are task-dependent: for 31B, InfographicVQA drops from 92.0 to 82.8 and OmniDocBench error rises from 0.131 to 0.201 at 280, while MMMU Pro changes only from 76.9 to 75.8. These aggregate benchmarks do not measure ARC cell reconstruction. [Report Tables 6 and 12, pp. 6 and 17]

### 1.4 Hybrid attention and long context

E2B uses four local sliding-attention blocks per global block; all other sizes use five local blocks per global block. Global layers use proportional RoPE (`p = 0.25`) and frequency 1M; local layers use ordinary RoPE and frequency 10k. Except in E2B/E4B, global layers reuse keys as values. The report attributes up to a 37.5% global-KV-cache reduction to these choices. [Report Introduction and §2, pp. 1–2]

Pinned 12B and 31B configs show a 1024-token local window, one full-attention layer after each five local layers, a final full-attention layer, and `max_position_embeddings: 262144`.[^31cfg][^12cfg] The first-party card calls these medium models 256K-context models.[^card]

Long-context performance is empirical, not perfect: without thinking, Table 9 reports 31B at 96.4 on 128K RULER and 79.5 Recall@k on 128K LOFT Text Retrieval; 12B scores 91.2 and 66.4. [Report Table 9, p. 8]

### 1.5 Thinking mode and tool formatting

Instruction-tuned Gemma 4 adds a mode that emits a reasoning trace before its answer. `<|think|>` in a leading system turn activates it; `<|channel>thought ... <channel|>` delimits the trace. Tool declarations and calls use `<|tool>declaration:...<tool|>` and `<|tool_call>call:...<tool_call|>`. [Report §3, p. 5; Appendix Table 11, p. 17]

First-party documentation adds `<|tool_response>...<tool_response|>` and makes the application responsible for parsing, validation, execution, and result injection.[^format][^tools] It also says ordinary multi-turn history must strip prior generated thoughts, while thoughts must remain between function calls belonging to one model turn.[^thinking] Thus thinking history and tool continuity are partly Solver context-policy choices.

### 1.6 Quantization-aware training

The report releases raw and QAT checkpoints in two named forms: mobile mixed int2/int4 weights with int8 activations, and blockwise Q4_0. Table 3 reports 31B text-only weight footprint at 64.0 GB bf16 versus 19.2 GB Q4_0, and 12B at 24.0 versus 7.65 GB, with separate int8 KV-cache increments at 32K context. The 150M image encoder also receives W8A8 QAT. [Report §2.5 and Table 3, pp. 3–4]

### 1.7 Multi-token drafter and speculative decoding

Each model has a separately counted autoregressive multi-token-prediction drafter. It receives previous-step final-layer activations, token embeddings, and the main model's KVs; its own embedder and four-layer Transformer draft future tokens while cross-attending to the main model cache. The purpose is speculative decoding. [Report Table 1, p. 2; §2.6 and Fig. 1, pp. 3–4]

For E2B/E4B, clustered top-k projection reduces the drafter's final multiplication from `d × 262,000` to `d × 4096` with similar acceptance rate. The report does not say every serving provider enables the released drafter. [Report §2.6, p. 4]

---

## 2. Direct architectural implications

These follow from the architecture but do not identify the cause of any particular Solver action.

### 2.1 The image artifact is exact; the model-native representation is bounded

A lossless PNG does not reach the backbone as lossless pixels. It is bicubically resized to a 48-pixel lattice and represented by a bounded soft-token sequence. In 31B, a frozen ViT transforms 16×16 patches before 3×3 pooling; in 12B, one 48×48 RGB patch is directly projected. Larger source rendering alone need not preserve more information if budget is unchanged.

For the B.4 assay, the prompt supplied a 64×64 integer grid and a 472×472 PNG in which each source cell was seven rendered pixels plus rulers. If—and only if—the official square-image algorithm and open 31B default budget 280 were used, the PNG would resize to 768×768 and produce 16×16 = **256** final visual tokens (not 280). A final token width would span about 4.2 source-grid cells; a 5×5 object would be only about 1.2 token widths and a 3×3 glyph about 0.7. This is a conditional processor calculation, not a recorded fact: B.4 used five OpenRouter providers and did not record realized resize/token count.

The exact text grid provides a second channel, but it serializes 4096 integers into a long one-dimensional sequence. Recovering 2D adjacency, object boundaries, coordinate convention, and role binding is then language-side reconstruction. The two channels are complementary, not equivalent guarantees of model-native exactness.

### 2.2 12B versus 31B changes more than the image front end

The 12B unified path avoids a separate frozen encoder and may retain different raw patch information; 31B can compute nonlinear visual features before pooling. However, a 12B/31B result also changes decoder size, depth, width, and training. The family supplies no same-backbone encoder/no-encoder ablation, so behavior alone cannot isolate “encoder-free” as the cause.

### 2.3 256K is capacity, not guaranteed memory or judgment

Periodic global attention gives distant tokens a computational path to later states; the 262,144-position configuration permits long requests. Neither guarantees exact retrieval, temporal binding, contradiction response, or action change. Solver-side omission, ordering, compaction, summaries, and tool-result injection determine what reaches the model at all.

The v32 report records one reasoning-off trajectory with a largest provider input of 61,434 tokens and some continuity, but it stopped before the 96K retention checkpoint. That establishes request acceptance and limited behavior under that Solver, not use of 256K or a causal role for global attention (`...v32/REPORT.md`, lines 12–29).

### 2.4 Thinking and tool tokens are protocol surfaces

A generated thought channel is parseable output, not latent-state telemetry or proof of faithful reasoning. Native function-call tokens express a request; they do not execute code, validate arguments, enforce an allowlist, or authorize an environment action. Those are Solver/provider responsibilities.

This matters for v32: the final request exposed only `return_result`, but the returned call named `execute_python`, and the worker rejected it (`REPORT.md`, lines 5–7, 15–31). That is an end-to-end tool-boundary violation. The trace does not localize whether raw model output, provider normalization, or provider-side tool-choice enforcement caused it.

### 2.5 MoE, quantization, and MTP primarily alter serving surfaces

The current 31B observations cannot expose MoE routing because 31B is dense. Quantization can change numerical outputs but a route label alone does not identify weight format, kernels, activation precision, or KV precision. Speculative decoding's strongest expected observable is latency/throughput; network, queueing, batching, hardware, and acceptance rate are competing causes.

---

## 3. Hypotheses about behavioral traces

These are testable hypotheses, not retrospective explanations.

### H1 — Some exact visual errors will vary with visual-token budget

**Current trace.** B.4 perception passed 0/3 whole trials (5/30 exact fields). Goal color survived all three, while player box/attributes, goal box/shape, changer set, and exact reachable set did not (`B.4 REPORT.md`, lines 21–25). This is consistent with coarse visual cues surviving while exact layout/binding fails, but the exact text grid was also present.

**Probe.** On frozen observations, cross budgets 70/140/280/560/1120 with image-only, grid-only, and image+grid arms. Record source hash/dimensions, processor revision, realized resize and token grid, prompt, model/provider, and exact cell/object errors. Repeat logically identical boards at several PNG scales.

**Discriminator.** Improvement only in image arms supports a preprocessing bottleneck. It still would not prove which encoder representation contained or lost a particular detail.

### H2 — Perception failure is not the only B.4 bottleneck

**Current trace.** Verified-state planning also passed 0/3; every route was illegal or missed the goal. Supplying the verified route rescued execution to 2/3 and 17/18 exact checkpoint actions (`B.4 REPORT.md`, lines 9–17 and 27–33). Thus deterministic perception alone cannot explain the whole result.

**Probe.** Factor the planning input into coordinate list versus explicit adjacency graph versus host-computed candidate routes, holding state, objective, schema, sampling, and no-action policy fixed. This separates 1D coordinate interpretation, graph construction, search, and action serialization.

### H3 — Explicit thinking may change B.4 planning, but visible thoughts will not diagnose mechanism

**Current trace.** B.4 used `reasoning_effort: null`, no retained conversation, temperature 0, and open routing. Its planning result is therefore a **non-thinking, open-routed** failure under the recorded API contract. The report's main static Table 5 uses thinking unless noted.

**Probe.** Repeat only verified-state planning with official thinking off/on templates, one pinned provider, identical prompt/schema/output cap, and raw token/control-token capture. Follow the official thought-stripping rule. Score route legality and outcome, not thought plausibility.

**Discriminator.** A legal route would show elicitation dependence; continued failure would reject “omitted thinking explains this assay” within that serving envelope. Neither result would make the generated trace a faithful causal explanation.

### H4 — Long-context failures will depend on evidence distance and Solver assembly

**Probe.** Place one decisive contradiction at controlled distances (8K, 32K, 64K, 96K, 128K, and where feasible 256K). Cross raw history, exact retrieval, and compact host summary. Measure exact recall, correct binding, theory revision, and next-action change. Keep payload ordering and tokenizer counts explicit.

**Discriminator.** Distance effects under identical assembly may implicate effective long-context use; differences across assembly arms implicate Solver retention/retrieval. Repetition or theory persistence alone does not identify local-attention “myopia.”

### H5 — Some tool failures are provider/template/parser boundary failures

**Probe.** Replay the same credential-free tool request through local official `AutoProcessor` templating and the hosted Responses surface. Record declared tool names, raw generated tokens, normalized tool objects, `tool_choice`, parser output, and allowlist verdict. Vary only schema complexity and thinking.

**Discriminator.** A raw allowed call transformed into an undeclared normalized call localizes provider translation; a forbidden raw call localizes generation/constraint failure; parser disagreement localizes the Solver boundary.

### H6 — `modelrun/fp4` may affect behavior, but the label does not identify the report's QAT format

The epistemic-context manifest pins model `google/gemma-4-31b-it`, provider-only `modelrun/fp4`, no fallback, and `png_scale: 8` (lines 24–42). This proves the requested route label only. “FP4” is not defined by the report and is not evidence of Q4_0 or mobile QAT.

**Probe.** Compare hash-pinned bf16 and named official QAT checkpoints with fixed preprocessing, kernels, KV policy, prompts, and repeated trials. Hosted comparisons additionally require serving metadata. Score exact perception, structured-call validity, and decisions separately.

### H7 — Dense/MoE differences require model-level comparison, not prose inspection

**Probe.** Compare 31B dense and 26B-A4B on a predeclared suite with equal front-end budgets and provider telemetry. Treat model identity as the intervention; do not claim per-token expert choices without router instrumentation. Size/training differences prevent a pure routing ablation.

### H8 — MTP use should be tested with drafter telemetry, chiefly for speed

**Probe.** On one local checkpoint/hardware stack, alternate drafter off/on and record proposed/accepted tokens, verification calls, time to first token, total throughput, and final deterministic token sequence. Hosted latency alone cannot identify speculative decoding, and ARC success/failure is not a drafter-use diagnostic.

---

## 4. Claims the report does not license

The report plus inspected traces do **not** license these claims:

1. **“Gemma sees the original PNG at full resolution.”** It sees a resized, token-bounded representation.
2. **“Budget 280 means exactly 280 visual tokens.”** Aspect-ratio/lattice rounding can yield fewer.
3. **“One visual token equals one ARC cell.”** Mapping depends on source scale, resize, aspect ratio, and board geometry.
4. **“Encoder-free 12B is lossless.”** It still resizes, patches, projects, positions, and obeys a token budget.
5. **“The frozen encoder failed to encode this cell/glyph.”** Existing outputs conflate preprocessing, encoder, decoder use, prompt, binding, and provider; exact grid text is also present.
6. **“Frozen during pre-training” means no later adaptation of any kind.** The report does not fully specify every post-training optimization detail needed for that universal claim.
7. **“31B traces reveal MoE expert routing.”** 31B is dense; MoE belongs to 26B-A4B.
8. **“256K context guarantees 256K memory, retrieval, or contradiction handling.”** It is a position/capacity bound, not an epistemic guarantee.
9. **“Repeated wrong actions prove local-attention myopia.”** Periodic global layers exist and Solver context assembly is an upstream confound.
10. **“A thought trace faithfully reveals causal reasoning.”** It is generated output under a formatting protocol.
11. **“Zero reported reasoning tokens means no internal computation occurred.”** It describes the API's reported channel, not latent processing.
12. **“Native function calling means Gemma executes tools or must obey the schema.”** Execution and enforcement belong to the surrounding application/provider.
13. **“The v32 undeclared call is a model-architecture defect.”** The trace identifies an end-to-end boundary violation, not its internal location.
14. **“`modelrun/fp4` is Q4_0 or an official Gemma QAT checkpoint.”** No inspected source bridges those labels.
15. **“Quantization caused a wrong action.”** A matched precision intervention is required.
16. **“Fast/slow output proves MTP was enabled/disabled.”** Provider and hardware confounds dominate without drafter telemetry.
17. **“The MTP drafter caused a semantic error.”** The report describes verified speculative drafting and does not establish hosted use.
18. **“Thinking would necessarily repair planning.”** B.4 omitted report-native thinking, which motivates a probe, not a predicted pass.
19. **“Gemma 4 benchmark scores predict ARC-AGI-3 Solver competence.”** The report does not evaluate this interactive evidence/action system.
20. **“A model output explains the Solver.”** The 0007 charter assigns truthful representations/evidence to the host and semantic interpretation/action choice to models (`experiments/agents/0007-guided-control/README.md`, lines 17–39). The complete causal unit is their coupled Solver.

## Minimum observability record for future probes

Record model/checkpoint revision; dense/MoE identity; provider/fallback; weight, activation, and KV precision; source image hash/dimensions; requested budget and realized resize/token grid; exact serialized control tokens and thinking policy; declared tools, raw output, normalized calls, and parser/validator decisions; ordered context inclusion/omission with token counts; environment action-release events; and, for MTP, explicit enablement plus proposal/acceptance telemetry.

Without those fields, architecture can nominate plausible bottlenecks and discriminating interventions. It cannot supply a hidden-mechanism story from behavioral traces.

## Sources

[^report]: Gemma Team, *Gemma 4 Technical Report*, arXiv:2607.02770v2, <https://arxiv.org/abs/2607.02770>.
[^card]: Google AI for Developers, “Gemma 4 model card,” <https://ai.google.dev/gemma/docs/core/model_card_4>.
[^vision]: Google AI for Developers, “Image understanding,” <https://ai.google.dev/gemma/docs/capabilities/vision/image>.
[^thinking]: Google AI for Developers, “Thinking mode in Gemma,” <https://ai.google.dev/gemma/docs/capabilities/thinking>.
[^format]: Google AI for Developers, “Gemma 4 Prompt Formatting,” <https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4>.
[^tools]: Google AI for Developers, “Function calling with Gemma 4,” <https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4>.
[^31cfg]: Google, `google/gemma-4-31B-it`, `config.json` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, <https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json>.
[^31proc]: Google, `google/gemma-4-31B-it`, `processor_config.json` at the same revision, <https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/processor_config.json>.
[^12cfg]: Google, `google/gemma-4-12B-it`, `config.json` at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, <https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/config.json>.
[^12proc]: Google, `google/gemma-4-12B-it`, `processor_config.json` at the same revision, <https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/processor_config.json>.
