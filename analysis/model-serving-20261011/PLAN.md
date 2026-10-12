# Better model, better serving: survey and plan (2026-10-11)

Currency: **solves per GPU-hour = (solves per token) × (tokens per GPU-hour)**. Model levers move the first factor,
serving levers the second. Evidence for why this is the frontier: `analysis/oracle-hint-20261010/FAILURES.md`. The
binding quantity is rule induction per token, which decays slowly on a level (3.6 → 1.1 solves per 100K tokens).

## Incumbent (dfranzen M2, from runs/out-control logs)
- Model: **Qwen3.8-Flash-Next**, 125B MoE plus a 51B n-gram embedding, 6B active, 512 experts (10 routed + 1 shared).
  Checkpoint `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (Intel AutoRound INT4 W4A16, 185 GB of files).
- Serving: SGLang 0.5.19 "pennyroyal" fork, fp8_e4m3 KV, context 139,264, **max 10 running requests**, NEXTN MTP
  3 steps / 4 draft tokens. The drafter is the base MTP layer in INT4 g32 (`albucino` drafter) with a 64K hot-token
  map; `--ple-offload-embedding`.
- Measured: accept length 2.67; gen tok/s 789 at 10 running (per-line median; time-weighted ≈ 700). The step is
  inferred to be bound by reading the expert weights (`dfranzen-serving-is-step-time-bound`). KV usage 66%, queue ≈ 0.
- Reasoning effort: server default (the Qwen3.8 template defaults to xhigh). It steps down a ladder only after a
  truncated generation.

## Model axis: what exists (web + Kaggle, 2026-10-11)
| candidate | status | evidence | fit |
|---|---|---|---|
| **Swift 1.5 Qwen3.8-Flash-Next** (UkisAI) | released; W4A16 AutoRound on Kaggle as `phuongncn/arc3-qwen38-swift-w4a16-autoround` (2 parts, 177 GB, verbatim HF copy); NVFP4 at `lordhansolo/swift-1-5-…-nvfp4`, `cihanatak/swift15-flashnext-nvfp4-nonple` | Post-train that penalises overthinking tokens, then RL/OPD on multi-turn agent data. GPQA-D 89.6 vs 89.8 with **−63% median thinking tokens**; Aider 40% of base tokens, paired n=107 net −4 (n.s.), loses more on the hardest retry cases. MTP head is the base's and is preserved. | **drop-in format** for our SGLang stack; the albucino drafter is the same MTP layer |
| reasoning_effort high/medium on base | config only | Swift page: medium/low cost −2.4–2.6 GPQA points; the 27B Duck probe found the knob inert on token work | trivial |
| Qwen3.8-27B dense / Swift-1.5-27B | released, Apache (27B) | Flash-Next wins all 9 shared benchmarks, and is faster | low priority |
| Qwen4-27B | announced 2026-09-22, no weights | — | watch; must be public before use |
| gpt-oss-120B, Gemma-4-31B | released (older) | below Flash-Next on aggregators | low |
| Ling 3.1 Flash, Kimi K3, DeepSeek-V4, Mistral Large 4 | closed weights or > 96 GB at 4-bit | — | no |
| `huikang/qwen-38-flash-next-finetune` | on Kaggle, undocumented | unknown | inspect only |

Field: all three M2 winners run base Flash-Next (W4A16/SGLang, mixed NVFP4-FP8/vLLM, W4A16/SGLang). Tufa (top, ~56) is
reported on Qwen3.6-27B FP8 with multimodal input (Tech Times, 2026-10-06; unverified, Tufa declined M2 open-sourcing).
A Swift mirror with "arc3" in its name and 82 downloads of a Swift NVFP4 dataset say others are testing Swift now.

License: the competition requires third-party code and methods to be under "at least an open source license". Swift's
licence caps free commercial use at $1M revenue; base Flash-Next's Qwen Community License is similar and every M2
winner uses it. This is a prize-eligibility question, not a scoring one.

## Serving axis: what the winners' logs already show (no GPU)
| stack | running | gen tok/s (per-line median) | accept | context | KV usage |
|---|---|---|---|---|---|
| dfranzen (ours): SGLang penny, W4A16 | 10 (cap) | 789 | 2.67 | 139,264 | 0.66 |
| sirikilohit: SGLang penny, W4A16 | 16 | **905 (+15%)** | 2.64 | 69,632 | 0.49 med, queue 9 |
| lordhansolo: vLLM main, mixed NVFP4/FP8 | 14 | **975–1,010 (+24–28%)** | 2.75 | — | — |

The workloads differ and these are observational, but aggregate decode is mostly workload-insensitive at similar
acceptance. The concurrency cap of 10 is a harness choice. At more streams per-stream speed drops (≈ 70 vs 79 tok/s)
while aggregate rises. The KV pool (1.0M fp8 tokens) fits 14–16 streams only if context per stream falls toward
~65K. Prior level elasticity to throughput is ≈ 0.35, so +25% tokens is about +9% levels; serving alone is unlikely
to be a +20% bet unless stacked.

Other serving levers, with priors:
- REAP-448 (`boristown/…-nvfp4-reap-448e`): −12.5% expert bytes, so roughly 5–10% faster per step, with a quality risk.
- NVFP4 official (`nvidia/…-NVFP4`) vs W4A16 on sm120: same bytes per weight; any gain comes from the kernel path.
- SGLang 0.5.21 (current upstream) vs 0.5.19 fork: unknown.
- MTP depth: 0–10% (`analysis/mtp-depth-retry1-readout-20261008/ADVANTAGE-ANALYSIS.md`).
- Startup ≈ 10 min of model loading per session: small.

## Plan (ranked by expected value per GPU-hour)
**E1, Swift swap (model; the decisive uncertainty).** dfranzen raw notebook; MODEL_DIR → a merged symlink view of
the two Swift W4A16 parts. Everything else, including the drafter, stays the same. One full public-25 pass (9,000 s
hard cap, ≈ 2.5 GPU-h). Compare levels and mean score with controls c1 107/52.48 and pc 96/40.11 (and B1 102/117).
Mechanism checks: generated tokens per level on levels both solve (expect ↓ ≥ 30%), accept length (a drafter
mismatch would show here), reasoning-truncation/ladder events. Rule: ≥ +15 levels → hidden draw; ≤ −10 → drop;
otherwise a second pass. CPU pre-flight first: merged view loads its config/index, and the drafter shapes match.

**E2, concurrency 14 (serving, config only).** Same notebook as control with max running 14, priority gate 14,
cuda-graph batch sizes to 14, and context/trim lowered so 14 × the mean context fits the pool. Mechanism: aggregate
tok/s ≥ +15% at equal acceptance, KV not thrashing (retractions ≈ 0). One pass, ≈ 2.5 GPU-h. If E1 wins, run E2 on
Swift instead, since the levers stack.

**E3, serving bench (only if E2 shows tokens/s is still worth chasing).** Replay a fixed recorded request mix at
fixed concurrency through W4A16/SGLang vs NVFP4 official vs REAP-448, measuring accepted tok/s. Reuse the
mtp-depth replay harness, fixing its two failure modes (occupancy gate, 20-min load timeout).

**Watch:** Qwen4-27B release; Swift on hidden draws by others (board movement).

Budget: E1 + E2 ≈ 5 GPU-h of the ≈ 19.8 h left before the 2026-10-17 reset (before D2 billing is known).
