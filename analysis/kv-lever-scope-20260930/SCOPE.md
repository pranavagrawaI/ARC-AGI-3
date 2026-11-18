# KV lever scope — 2026-09-30

**Result:** KF spends 87 % of every request waiting for one of ~3 KV slots. The cheapest real lever is **pruning
KF's own checkpoint at load time** with the published REAP-k448 expert mask. I recovered that mask exactly from
router rows (49/49 layers, bit-exact against KF's RadixArk revision). It frees 7.91 GiB of expert weights, which
takes the KV pool from 5 GiB to ~12.9 GiB (2.6× tokens). No checkpoint upload, no identity-check bypass. The open
work is one vLLM loader patch against Keith's pinned runtime, one GPU probe, then a two-pass arm.

## Why this lever (evidence, KF run 2026-09-26, `duck-qwen38/runs/flash-next/out-kf2`)

| per request, mean of 1,345 | seconds | share |
|---|---|---|
| queue | 126.4 | **87 %** |
| prefill (≈21K tokens) | 1.6 | 1 % |
| decode (≈1.5K tokens) | 16.8 | 12 % |

- Running = 3 in 709 of 792 busy samples, waiting ≈ 22. max_num_seqs 8 never binds. vLLM itself reports
  "Maximum concurrency for 32,768 tokens per request: 3.21x".
- Request size (arm G timing): prompt+completion median 22.5K, p90 26.3K. Completion median 1.2K, max 13K.
- Prefill is cheap, so prefix caching (0 % hit, disabled) would buy ≤ ~9 %. Not the lever.
- The only thing that has moved levels this month is tempo: yield fix (K 36 → KF 44) and the serving swap
  (dense F 25 → K 36). Memory repairs (R) and visual info (G) were null. The outside evidence is Son/Mark's
  pruned + 7→16-way run, 19.7 → 33.9 public (their harness, ~100K contexts; bundled, not causal).

## VRAM accounting (95 GiB card)

| item | GiB | source |
|---|---|---|
| model incl. MTP draft | 81.8 | "Model loading took 81.8 GiB" |
| KV pool (fixed) | 5.0 | `--kv-cache-memory-bytes 5368709120` → 105,202 tokens |
| CUDA graphs | 0.36 | "Graph capturing finished … took 0.36 GiB" |
| everything else (activations at 8,192 batched tokens, workspaces, context) | ≈ 7 | teardown used 96,450 MiB; FlashInfer autotuner already OOMs (42 MB free) |
| PLE n-gram table | 47.75 | **already on CPU** (PleOffload worker) |

## Levers

| lever | frees / changes | work | risk | verdict |
|---|---|---|---|---|
| **B. prune-at-load, k448 mask** | −7.91 GiB experts → KV ≈ 12.9 GiB, ~10 slots | vLLM loader patch + main/MTP expert-count split | quality loss (REAP mass kept 92.65 %); MTP acceptance | **do this** |
| A. reclaim activation reserve | max_num_batched_tokens 8192→4096, KV +1–2 GiB | config only | runtime OOM (autotuner already starved) | stack on B in the probe if cheap |
| C. analyzer context 32K→16–24K | fewer tokens per request | config (set after setup; Scott's attempt never applied) | less history; confounds agent behaviour | fallback only |
| D. prefix caching | ≤ 9 % of GPU time | config; hybrid GDN support unknown | low | skip |
| E. serve `boristown/…-reap-448e` as-is | same −7.9 GiB | bypass pinned identity; BF16/other PLE format; nvidia base, not RadixArk | unknown loader compat | superseded by B |
| F. fp8 KV | — | — | **impossible** (QSA needs BF16 KV, arm S) | closed |

## Settled offline (reproducible)

- `duck-qwen38/runs/flash-next/prune/recover_mask.py`: range-reads only router tensors. The k448 routers match RadixArk
  `7b719225…` rows bit-exactly for **49/49** routers: 48 layers keep 448/512, and the MTP router keeps 512/512.
  Mask saved as `prune/k448_mask_vs_rx.json`.
- KF's `serving_setup.py` pins `MODEL_HF_REVISION = 7b719225…` and config sha `e765305d…`. The HF config hashes
  to the same value, so the mask applies to KF's exact checkpoint.
- `prune/freed_bytes.py`: routed experts 63.28 GiB, pruned 7.91 GiB, MTP 4.86 GiB, PLE 47.75 GiB.
- The k448 card's `sitecustomize.py` shows the MTP split pattern (override draft `num_experts` to 512), but it
  targets `vllm.models.qwen4_exp`. Keith's build names the package `qwen3_8_flash_next`, so it must be ported.

## Loader work — DONE offline (2026-09-30)

Keith's vLLM source (pulled by the CPU kernel `flash-next-vllm-src-probe`, 0 GPU) gave the seams:
- `Qwen3NextSparseMoeBlock.__init__` sizes the router (`ReplicatedLinear(…, config.num_experts)`) and
  `FusedMoE(num_experts=config.num_experts)` at construction. Forward never re-reads the config.
- `Qwen3_8FlashNextModel.load_weights` fuses the shared expert at index `config.num_experts`.
- Keith's PLE patch identity gate pins `num_experts: 512`, so 448 must NOT be global.
- The MTP draft builds from `speculative_config.draft_model_config`, a separate config object, so it keeps 512
  without porting k448's sitecustomize. The MTP bank is the only fused-format expert tensor set, and it passes through.
- This build loads `vllm.general_plugins` entry points in every process when `VLLM_PLUGINS` is unset.
  serving_setup never sets it, and it appends the notebook's PYTHONPATH.

Implementation: `duck-qwen38/runs/flash-next/prune/plugin/flashnext_prune` (vLLM general plugin). num_experts is 448
only while target MoE blocks are built (inside `Qwen3_8FlashNextModel.__init__`) and while `load_weights` runs.
The weight filter drops/renumbers experts, slices routers, and skips `mtp.`. Offline tests (`prune/test_plugin.py`):
over all 296,475 real RadixArk names it drops 36,864, renumbers 258,048 and slices 48 routers. Slicing KF's real
routers with the mask reproduces k448's routers byte for byte (layers 0/15/47). Stub wiring: PLE gate 512,
target blocks 448, MTP 512, fuse at 448, configs restored.
Notebook builder: `prune/build_p.py` (KV 12 GiB, max_num_seqs 12, cudagraph 48; guard cell stops the run
unless the worker logged the plugin's load_weights line and model memory < 76 GiB).

## Plan and gates

| step | cost | gate to continue |
|---|---|---|
| 1. extract Keith's vLLM model/loader source (CPU kernel) | 0 GPU | **done** |
| 2. write + offline-qualify the loader patch | 0 GPU | **done**, all pass |
| 3. GPU probe `duck-flash-next-p-probe` (launched 2026-09-30), public-25 at 1,200 s/game | ~0.6 GPU h | loads; "Model loading" ≈ 73.9 GiB; KV ≈ 2.5× tokens; median running ≥ 7; MTP acceptance ≥ 50 % (KF 60 %); tool-call parse errors not up; levels at 1,200 s not collapsed vs KF's first 20 min |
| 4. arm P, two passes, full 7,920 s | ~6 GPU h | pre-registered: mean(P) − 43 ≥ +8.5 levels |

GPU budget: roughly 11 h left until the Sat 2026-10-03 00:00Z reset (estimate — check the Kaggle UI). Steps 1–3
fit today; step 4 fits before the reset only if step 3 passes by Thursday.

## What could reverse this

- Pruning costs more quality than the tempo gains (the probe's level count at 1,200 s is the early read; the
  arm settles it).
- MTP acceptance drops when the 512-expert draft predicts for a 448-expert target, eating the decode gain.
- Tempo elasticity is weaker than assumed: arm G lost ~30 % of actions in 17 games but only ~7 % of levels
  there (confounded). If P runs ~2× the requests and levels don't move, tempo is not the binding constraint
  either, and the premise itself needs attacking.

## After retention: remaining speed experiments (2026-09-30)

P is retained: 51/58 levels, mean 54.5 versus KF/C 43. See
[reviewed full-run evidence](../arm-p-full-20260930/REVIEW.md). P is now the
comparison configuration; the old 51.5-level threshold cannot qualify a new
improvement over P. No experiment below has been launched.

P still has capacity pressure: busy median running 8, waiting 16, KV occupancy
about 92%, generation 384/385 tokens/s. Mean queue time is 63.30/61.09 seconds,
about 64% of end-to-end request time. This is not a GPU idle-time fraction.

Ranked candidates, untried on this retained Flash-Next configuration in the
local run records inspected:

1. **MTP sparse-index reuse (first choice).** Change only
   `TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION=1`. Both P identities record false;
   their command lines omit the option. The exact RadixArk config was fetched
   at revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`; its SHA256 matches
   the launcher's `e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624`,
   and neither root nor text config specifies index sharing. Pinned
   `vllm/config/speculative.py:539` defaults it to false. In
   `vllm/v1/spec_decode/llm_base_proposer.py:580`, draft step 0 computes indices;
   later draft steps reuse them. Flash-Next's `nvidia/mtp.py:244` implements the
   required methods. This changes the drafter's computation, so acceptance can
   change; speed and correctness on this workload remain unmeasured. It does
   not intentionally change target weights, agent inputs, or KV allocation.
2. **Static MTP length 3 -> 2**, then 1 only if evidence warrants it. P's
   eight active requests make the old low-concurrency optimum uncertain.
   Keep KV and all other knobs fixed. Measure aggregate verified output
   throughput and completed requests, not acceptance alone. Dynamic batch
   scheduling exists in the launcher but adds graph-path/compatibility risk;
   establish the static comparison first.
3. **Smaller prefill batches, then a larger KV pool.** P still uses 8,192
   batched tokens and an explicit 12 GiB KV pool. Test 4,096 and measure actual
   peak memory under long prompts before increasing KV. A smaller batch limit
   alone does not enlarge an explicitly sized pool. The previous estimate of
   1-2 GiB reclaimed is a hypothesis, not measured headroom. P's logs do not
   show the old KF autotuner OOM, so that KF observation must not be carried
   forward as if it happened in P.
4. **Prefix caching, conditional on useful shared prefixes.** Correct the
   earlier blanket dismissal above: small prefill time bounds the direct
   recomputation benefit, not every possible benefit from sharing live KV
   blocks. First measure exact token-prefix overlap across concurrently active
   requests. Pinned Flash-Next target and MTP reject `mamba_cache_mode=all`
   and explicitly direct users to `align`; this is source-level feasibility,
   not runtime qualification. Hit rate and net KV-capacity benefit on Duck
   remain unknown, particularly as game-specific history diverges.

MTP-off is a secondary capacity tradeoff: the MTP bank is about 4.86 GiB, but
whether disabling it actually releases that much live memory and whether extra
KV beats lost speculation need measurement. Backend replacement, further
pruning, and shorter context carry more compatibility or quality risk. Raising
max_num_seqs alone does not address the observed KV limit. FP8 main KV remains
explicitly rejected by the pinned QSA code (`nvidia/qsa.py:109,188`).

**Recommended next decision:** a bounded matched P versus P+index-sharing
serving probe, with fixed inputs/output budgets, representative long contexts,
and sustained demand comparable to the 25-game workload. Record verified
tokens/s, completed requests/s, acceptance, latency, KV occupancy, graph mode,
errors, and peak memory. Prove the worker enabled the flag. A proposed practical
screen is at least 10% throughput improvement across repeated measurement
windows with no integrity failures; this is a screening threshold, not proof
of a gameplay gain. Follow a positive screen with full public-25 validation
against P. Prepare a frozen card before any launch. Live quota checked during
this investigation: 8.48 GPU hours remaining, reset 2026-10-03 00:00 UTC. Reserve
roughly 6 hours for full validation; do not spend the remainder on a broad sweep.

External primary evidence checked:
- [vLLM index-sharing mechanism, GLM case study](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd):
  supports the mechanism, not a transferable Flash-Next speedup estimate.
- [Dynamic speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/):
  explains batch-size-dependent draft cost; current docs limit MRv1 dynamic
  scheduling to piecewise CUDA graphs and do not claim MTP is tested.
- [Automatic prefix caching](https://github.com/vllm-project/vllm/blob/643bbef55c37edb5c3fd341aae1e20e026d0e730/docs/features/automatic_prefix_caching.md):
  prefix reuse and hybrid `align` constraints; pinned local source takes
  precedence over current upstream behavior.

**Superseded by [NEXT-PLAN.md](NEXT-PLAN.md) (2026-09-30):** index sharing, prefix caching and MTP length are screened in one benchmark session against a matched P baseline. The P logs show prefill costs 25% of decode.
