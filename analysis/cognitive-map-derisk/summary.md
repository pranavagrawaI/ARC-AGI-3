# De-risking cognitive-map measures on OLMo-2-0425-1B

Question under test: do the proposal's two measures (Measure 1, map formation: held-out R² of a linear probe for the in-context HMM belief state; Measure 2, structure reuse: how many observations a probe trained in alphabet A needs before it recovers beliefs on the same HMM written in alphabet B) measure anything real on a 1B base model?

Run date 2026-11-23. Model `allenai/OLMo-2-0425-1B`, bf16, residual stream after each block (layer −1 = token embeddings, layers 0–15 = after each block). Symbols are single space-prefixed letters: A = {F, Q, V}, B = {K, M, Z}. Each 4000-token input is `<|endoftext|>` followed by symbols. The hidden state at input index t has seen x₁..x_t, predicts x_{t+1}, and is paired with the belief η(x_{1:t}); index t = 0 is the BOS token, paired with the stationary distribution π.

All R² values below are variance-weighted, `1 − ‖Y−Ŷ‖²_F / ‖Y−mean(Y)‖²_F` over test rows (higher is better; 1 is perfect, 0 means no better than the test mean). "Best layer" values are maxima over the 17 layers, which flatters every method equally. "Ridge" means λ ∈ {0, 0.01, 1, 10, 100, 1000}, chosen on an inner validation split of the training rows. "OLS" means the min-norm least-squares solution (λ = 0). KL is in nats; lower is better.

## Verdict per failure mode

| Failure mode | Verdict | Decisive numbers |
|---|---|---|
| FM1: the model never reaches the converged predictions that Measure 1 presupposes | **Confirmed** | At t ∈ [3000,4000), restricted KL to the optimal predictor is worse than the unigram (0-HMM) predictor on Mess3 (0.331 vs 0.222) and Arch (0.167 vs 0.095); Wing equals the 1-HMM (0.083 vs 0.083). Off-alphabet mass grows with context: Wing A goes from 7% to 75%, Wing B from 18% to 99%. |
| FM2: probe R² reflects split leakage and passive history encoding, not a learned map | **Confirmed** | A random-split probe for cumulative symbol frequency (a slow, nearly constant target) reaches R² 0.997–0.999 on every HMM; a contiguous split gives −0.35 to −4.6. On i.i.d. (0-HMM) sequences with no hidden structure, the full-HMM belief is still decodable at contiguous-split R² 0.82 (Wing) and 0.67 (Arch), as high or higher than on true HMM sequences (0.755 and 0.133). |
| FM4: no headroom over trivial baselines | **Confirmed for honest splits** | Under the cross-sequence split, the activation probe never beats the exact next-token-probability → belief map: Δ = −0.045 (Mess3), −0.063 (Wing), −0.24 (Arch). |
| FM3a: a probe fixed in alphabet A transfers to B by in-context learning alone | **Fails, except partially on Wing** | Fixed A-probe on B: best R² −0.03 (Mess3) and −0.13 (Arch); still negative under the best of all 6 or 24 state permutations. The A and B probe subspaces are near-orthogonal (largest principal cosine 0.02–0.14 against 0.04–0.05 for a random subspace), yet a fresh B-probe gets 0.977 and 0.734. Wing layers 9–15 transfer partially (0.38–0.54). |
| FM3b: the literal Measure 2 (LoRA on A, then probe A applied to B) | **Fails, and fine-tuning makes it worse** | The fine-tuned A-probe applied to B is negative at every layer and window after 100 steps at lr 1e-4 (best −0.26), against 0.598 for the base model. The high-lr regime also leaks 41–65% of B's probability onto F/Q. A fresh B-probe is not faster after fine-tuning, except at t ∈ [100,250) (0.81 vs 0.59, late layers, n = 5 sequences). |

What follows: Measure 1 cannot be reported as a single R² from a random split. Its value depends on the split, on a matched 0-HMM control, and on the next-token-probability baseline, and once those are applied there is little left on this model. Measure 2 as specified measures token-identity entanglement of the probe rather than structure reuse. The B geometry exists (fresh probe ≈ A-probe quality, CKA ≈ 0.99 between the A and B belief projections) but lives in a different subspace.

## Step 0 — tests and bf16 vs fp32

`tests/test_core.py` passes 21 tests. They cover: row-stochastic transition matrices; π = πT; the Bayesian filter against brute-force path enumeration; P(sequences of length 5) summing to 1; the chain rule; k-HMM KL decreasing in k toward 0; marginals of the 0-HMM and 1-HMM samplers; 1:1 tokenization of 500-symbol strings in all four alphabets; the input-index/belief alignment convention; ridge and OLS against `lstsq`/normal equations/`pinv`; variance-weighted and uniform R²; causality of the frequency and EWMA features; and the pooled-statistics ridge against the SVD ridge.

One Wing sequence, N = 2000, t ∈ [1000,2000):

| quantity | bf16 | fp32 |
|---|---|---|
| peak VRAM (GiB) | 3.28 | 5.96 |
| KL full vocabulary | 0.653 | 0.644 |
| KL restricted to alphabet | 0.0873 | 0.0849 |
| off-alphabet mass | 0.376 | 0.374 |
| peak probe R² (OLS, 200 train / 800 test random) | 0.951 | 0.956 |

Mean |ΔKL_full| is 0.019 (max 0.18 at single positions). KL between the bf16 and fp32 restricted predictions is 2e-4 nats. Relative activation difference ranges from 0.2% (embeddings) to 2.4% (mid layers). bf16 is adequate for every comparison below; the smallest effects reported are ≥ 0.05 in R² or KL. One caveat: OLS R² at layer 0 differed sharply between precisions (0.90 vs 0.26), an early sign that the min-norm solution is unstable; ridge with λ = 1 agreed to within 0.002 at every layer.

Diagnostics (`results/diag_alignment.json`): `run()` log-probs match HF `model(...).logits` to 1e-6. Dropping BOS makes late restricted KL worse on all three HMMs (Mess3 0.411 vs 0.331).

## Experiment 1 — in-context convergence (FM1)

Five sequences per HMM, N = 4000, seeds 0–4, alphabet A. Mean over t ∈ [3000,4000). Columns k = 0..64 give the analytic k-HMM KL on the same sequences (the k-HMM predicts from the last k symbols, filtered from π).

| HMM | KL full | KL restricted | off-alphabet mass | 0-HMM | 1-HMM | 2 | 3 | 4 | 6 | 8 | 16 | 64 | k_eff (restricted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mess3 | 0.665 | 0.331 | 0.280 | 0.222 | 0.176 | 0.088 | 0.042 | 0.020 | 0.0047 | 0.0011 | 0.0000 | 0.0000 | < 0 (worse than unigram) |
| Wing | 1.966 | 0.083 | 0.750 | 0.114 | 0.083 | 0.057 | 0.036 | 0.023 | 0.0081 | 0.0030 | 0.0001 | 0.0000 | < 1 (5 sequences: 1.89, <1, <1, <1, 1.28) |
| Arch | 0.387 | 0.167 | 0.192 | 0.095 | 0.075 | 0.060 | 0.050 | 0.043 | 0.031 | 0.023 | 0.0075 | 0.0002 | < 0 |

Trajectory (mean ± SEM over 5 sequences):

| HMM | metric | [0,250) | [250,500) | [500,1000) | [1000,2000) | [2000,3000) | [3000,4000) |
|---|---|---|---|---|---|---|---|
| Mess3 | KL restricted | 0.450±0.033 | 0.378±0.028 | 0.341±0.021 | 0.372±0.037 | 0.333±0.028 | 0.331±0.022 |
| Mess3 | off-alphabet mass | 0.072 | 0.071 | 0.201 | 0.163 | 0.190 | 0.280 |
| Wing | KL restricted | 0.090±0.011 | 0.085±0.017 | 0.075±0.010 | 0.173±0.035 | 0.103±0.014 | 0.083±0.009 |
| Wing | off-alphabet mass | 0.071 | 0.037 | 0.211 | 0.432 | 0.713 | 0.750 |
| Arch | KL restricted | 0.422±0.076 | 0.241±0.038 | 0.259±0.030 | 0.222±0.032 | 0.241±0.027 | 0.167±0.018 |
| Arch | off-alphabet mass | 0.119 | 0.152 | 0.146 | 0.104 | 0.137 | 0.192 |

- Full-vocabulary KL (the Balcells metric) is dominated by off-alphabet leakage (KL_full = KL_restricted − log(alphabet mass)), and it grows with context on Wing: 0.26 → 1.97.
- On Wing late in context the model abandons the letter pattern. The top tokens in t ∈ [3000,4000) are "Q" 0.22, then " ", "\n\n", "X", ".", ",", "(". On Wing-B, K and M together get < 1% (`results/diag_top_tokens.json`).
- On Mess3 and Arch the model over-predicts the dominant letter: late Mess3 gives F 0.48, while the empirical frequency is 0.31.
- Arch has long memory (T⁽⁰⁾ diagonal, α = 0.99): even the 8-HMM is 0.023 nats from optimal, and the 64-HMM is 0.0002.

## Experiment 2 — probe validity, overfitting, controls (FM2, FM4)

Activations at t ∈ [2000,4000), 5 sequences per HMM. Splits:
- **random 400**: 400 train / 1600 test positions within one sequence.
- **random 1000**: 1000/1000.
- **contiguous**: train t ∈ [2000,2800), test [2800,4000).
- **cross-sequence**: train on all of sequence i, test on sequence j ≠ i, averaged over 20 ordered pairs.

Cells are "ridge best-layer R² (layer) / OLS best-layer R²".

| HMM | target | random 400 | random 1000 | contiguous | cross-sequence |
|---|---|---|---|---|---|
| Mess3 | belief | 0.967 (L4) / 0.965 | 0.980 (L8) / 0.971 | 0.933 (L0) / 0.905 | 0.955 (L1) / 0.276 |
| Wing | belief | 0.950 (L8) / 0.948 | 0.970 (L8) / 0.956 | 0.755 (L1) / 0.372 | 0.817 (L0) / 0.282 |
| Arch | belief | 0.903 (L15) / 0.895 | 0.944 (L12) / 0.919 | 0.133 (L7) / −0.230 | 0.488 (L5) / −0.057 |
| Mess3 / Wing / Arch | shuffled belief | ≤ 0.00 | ≤ 0.00 | ≤ 0.00 | −0.02 / −0.08 / −0.18 |
| Mess3 / Wing / Arch | Dirichlet random | ≤ 0.00 | ≤ 0.00 | ≤ 0.00 | ≤ 0.00 |
| Mess3 | cumulative frequency | 0.997 | 0.999 | −1.28 | −18.1 |
| Wing | cumulative frequency | 0.997 | 0.998 | −4.57 | −9.49 |
| Arch | cumulative frequency | 0.999 | 0.999 | −0.35 | −10.9 |
| Mess3 / Wing / Arch | 50-step window frequency | 0.95 / 0.98 / 0.96 | 0.97 / 0.99 / 0.98 | 0.76 / 0.79 / 0.80 | 0.86 / 0.91 / 0.90 |
| Mess3 / Wing / Arch | 200-step window frequency | 0.97 / 0.99 / 0.99 | 0.99 / 0.99 / 0.995 | 0.11 / 0.51 / 0.40 | 0.45 / 0.67 / 0.68 |

Baselines without the LLM, with the same splits and ridge probe (columns: random 400 / contiguous / cross-sequence):

| features → belief | Mess3 | Wing | Arch |
|---|---|---|---|
| exact next-token probabilities (NTP) | 1.000 / 1.000 / 1.000 | 0.883 / 0.867 / 0.880 | 0.801 / 0.331 / 0.727 |
| log NTP | 0.990 / 0.990 / 0.990 | 0.883 / 0.868 / 0.879 | 0.753 / 0.106 / 0.657 |
| EWMA bank (6 decay rates) + one-hot of last 4 tokens | 0.700 / 0.662 / 0.687 | 0.775 / 0.705 / 0.757 | 0.503 / −0.004 / 0.188 |
| one-hot of last 4 tokens only | 0.594 / 0.556 / 0.589 | 0.635 / 0.575 / 0.622 | 0.298 / −0.071 / 0.113 |

Δ = acts − NTP:

| split | Mess3 | Wing | Arch |
|---|---|---|---|
| cross-sequence | −0.045 | −0.063 | −0.24 |
| random 400 | −0.033 | +0.067 | +0.10 |

Mess3's emission matrix M is invertible, so NTP determines the belief exactly (R² = 1.000), and Mess3 cannot separate "belief" from "next-token prediction".

0-HMM and 1-HMM control sequences (3 sequences each, generated from the matched i.i.d. and first-order symbol law, scored against the full HMM's belief function). Cells are best-layer ridge R²; random 400 / contiguous.

| HMM | sequences | belief | η, last 1 symbol | last 2 | last 5 | last 10 | last 20 |
|---|---|---|---|---|---|---|---|
| Wing | true HMM | 0.950 / 0.755 | 1.000 / 1.000 | 0.999 / 0.995 | 0.974 / 0.936 | 0.962 / 0.904 | 0.945 / 0.855 |
| Wing | 0-HMM | 0.887 / 0.824 | 1.000 / 1.000 | 0.999 / 0.997 | 0.946 / 0.922 | 0.871 / 0.833 | 0.867 / 0.813 |
| Wing | 1-HMM | 0.871 / 0.787 | 1.000 / 1.000 | 0.999 / 0.997 | 0.951 / 0.935 | 0.883 / 0.836 | 0.844 / 0.756 |
| Arch | true HMM | 0.903 / 0.133 | 1.000 / 1.000 | 0.998 / 0.996 | 0.914 / 0.844 | 0.834 / 0.604 | 0.837 / 0.335 |
| Arch | 0-HMM | 0.867 / 0.665 | 1.000 / 1.000 | 0.999 / 0.997 | 0.871 / 0.831 | 0.703 / 0.591 | 0.720 / 0.598 |
| Arch | 1-HMM | 0.821 / 0.622 | 1.000 / 1.000 | 0.999 / 0.998 | 0.899 / 0.860 | 0.674 / 0.599 | 0.699 / 0.517 |

Probe R² over context windows (best-layer ridge; random 20/80, contiguous 40/60, cross-sequence):

| HMM | split | [0,500) | [500,1000) | [1000,2000) | [2000,3000) | [3000,4000) |
|---|---|---|---|---|---|---|
| Mess3 | random | 0.912 | 0.935 | 0.956 | 0.956 | 0.961 |
| Mess3 | contiguous | 0.791 | 0.851 | 0.910 | 0.917 | 0.897 |
| Mess3 | cross-sequence | 0.842 | 0.912 | 0.930 | 0.932 | 0.938 |
| Wing | random | 0.892 | 0.944 | 0.949 | 0.952 | 0.945 |
| Wing | contiguous | 0.484 | 0.126 | 0.726 | 0.396 | 0.746 |
| Wing | cross-sequence | 0.557 | 0.525 | 0.817 | 0.734 | 0.778 |
| Arch | random | 0.841 | 0.889 | 0.894 | 0.917 | 0.869 |
| Arch | contiguous | −2.19 | −0.45 | −0.43 | −0.21 | −0.10 |
| Arch | cross-sequence | −0.74 | −0.14 | 0.09 | 0.14 | 0.28 |

Readings:
1. **Random splits are leaky.** Cumulative frequency at t ≥ 2000 has total variance 2e-4, yet a random-split probe recovers it at 0.997 by interpolating between neighbouring positions; contiguous and cross-sequence splits collapse it. Every random-split number here is inflated.
2. **OLS min-norm (Balcells) breaks off-distribution.** With 2000 training rows and 2048 dimensions, OLS interpolates the training data. Cross-sequence OLS R² falls to −2 to −42 at mid and late layers on Wing, and its best layer is the embedding layer. Ridge is required.
3. **Most decodable "belief" is recent-history decoding.** The belief over the last 1 or 2 symbols is decodable at 0.995–1.000 everywhere. The 0-HMM control matches or exceeds the true-sequence R² under contiguous splits, so a high Measure-1 score does not certify a learned map.
4. **Arch exposes a confound inside the honest split.** Even exact NTP falls to 0.33 contiguous, because Arch's long memory means a 800-token training block does not cover the belief range of the next 1200 tokens. Contiguous R² there mixes leakage removal with a shift in the target distribution.
5. Uniform-average R² is lower than variance-weighted R² on Wing and Arch (cross-sequence 0.738 vs 0.817; 0.368 vs 0.488), because low-variance belief coordinates are decoded poorly.
6. The λ chosen for the belief probe under the cross-sequence split ranges from 0.01 to 100 across layers and HMMs, and is 1000 at the embedding layer. The fixed absolute λ grid interacts with the per-layer activation scale.

## Experiment 3 — zero-training transfer across alphabets (FM3a)

The same symbol sequences (seeds 0–4) are rendered in A and B, so the beliefs are identical position by position. Probes are pooled over 4 sequences and tested on the held-out 5th (leave-one-sequence-out), at t ∈ [2000,4000). λ is chosen by sequence cross-validation in the source alphabet.

| quantity (best layer) | Mess3 | Wing | Arch |
|---|---|---|---|
| A-probe on held-out A | 0.977 (L5) | 0.840 (L0) | 0.720 (L1) |
| fresh B-probe on held-out B | 0.977 (L3) | 0.851 (L0) | 0.734 (L3) |
| fixed A-probe on B, identity state labels | −0.034 (embeddings; −1.2 to −178 at layers 0–15) | 0.538 (L11) | −0.125 (embeddings; −2.1 to −307 at layers 0–15) |
| fixed A-probe on B, best of all state permutations | −0.008 | 0.538 | −0.028 |
| single-sequence A-probe on B | −0.037 | 0.503 (L13) | −0.195 |
| largest principal cosine, A-probe vs B-probe subspace (per layer) | 0.02–0.08 | 0.02–0.14 | 0.05–0.12 |
| same, between two A-probes fitted on disjoint A sequences | 0.53–0.86 | 0.18–0.98 | 0.40–0.85 |
| random-subspace reference cosine | 0.043 | 0.037 | 0.052 |
| B belief linearly refit from B activations projected on the A-probe subspace | 0.372 (L7) | 0.760 (L9) | 0.132 (L2) |
| same, projected on a random subspace of equal dimension | 0.373 (L3) | 0.729 (L4) | 0.122 (L1) |
| same, projected on the B-probe subspace | 0.985 | 0.955 | 0.911 |
| linear CKA, A-projected A activations vs B-projected B activations (matched positions) | 0.94–0.99 | 0.96–1.00 | 0.94–0.99 |

- On Mess3 the best permutation is consistently a 3-cycle (state order [1,2,0] in 4/5 sequences at layer 10), yet R² stays negative. The S₃ relabeling ambiguity exists but is not the bottleneck; the subspace is.
- The B belief geometry is as good as A's, and CKA ≈ 0.99 shows the same geometry, but it sits in a subspace that is no more aligned with A's than a random one. The A-subspace carries no more information about B beliefs than a random subspace does on Mess3 and Arch.
- Wing, the only binary process, shows partial fixed-probe transfer at layers 9–15 (0.37–0.54), with A-subspace refit 0.74–0.76 against random-subspace 0.28–0.65 at the same layers. This is weak evidence of an alphabet-agnostic feature, such as run length.

In-context alphabet switch (Wing, 5 sequences): 2000 tokens in A, then 2000 fresh tokens in B. The B segment always starts its belief at π. "Scratch" means the same B segment without the A prefix.

| B-segment window | [0,100) | [100,250) | [250,500) | [500,1000) | [1000,2000) |
|---|---|---|---|---|---|
| KL restricted, scratch | 0.144 | 0.069 | 0.117 | 0.208 | 0.174 |
| KL restricted, after A, no separator | 0.125 | 0.147 | 0.156 | 0.209 | 0.286 |
| KL restricted, after A, "\n\n" separator | 0.145 | 0.257 | 0.261 | 0.299 | 0.204 |
| mass on A tokens (scratch / no separator / separator) | 0.002 / 0.179 / 0.119 | 0.005 / 0.181 / 0.046 | 0.007 / 0.112 / 0.010 | 0.005 / 0.109 / 0.012 | 0.003 / 0.057 / 0.012 |
| fixed A-probe best layer (scratch / no separator / separator) | −0.98 / −1.83 / −2.19 | −0.76 / −0.76 / −1.76 | −0.05 / 0.37 / −0.30 | 0.57 / 0.40 / 0.32 | 0.47 / 0.36 / 0.02 |
| fresh B-probe best layer (scratch / no separator / separator) | 0.61 / 0.64 / 0.60 | 0.59 / 0.60 / 0.60 | 0.73 / 0.69 / 0.63 | 0.90 / 0.89 / 0.89 | 0.85 / 0.82 / 0.82 |

The A prefix does not speed up convergence on B. It leaks 5–18% of mass onto A tokens and makes restricted KL worse, and it does not improve the fixed A-probe's transfer. The fresh-probe trajectory is unchanged within about 0.05.

## Experiment 4 — LoRA on A, test on B (FM3b, the literal Measure 2)

Setup: a self-contained LoRA (`cmlib/lora.py`) on q/k/v/o/gate/up/down projections of all 16 blocks; 112 modules, 6.03M trainable parameters; r = 8, α = 16, B initialised to 0. Base weights bf16, LoRA parameters fp32, gradient checkpointing, AdamW. Batch is 4 sequences × 1024 tokens (2 micro-batches × 2 accumulation steps), 5-step warmup, gradient clipping at 1.0. Training data are fresh Wing sequences in A. Peak VRAM 5.74 GiB; 100 steps take 4 minutes.

Checkpoint metrics. Columns 2–6 are means over t ∈ [2000,4000) on 3 held-out sequences (seeds 500–502). The A-probe columns come from `exp4_ckpt_probe`, leave-one-sequence-out over the same 3 sequences. For reference, the 3-HMM KL on these sequences is 0.037.

| regime, step | A KL full | B KL restricted | B mass on A tokens | B mass on B tokens | A-probe held-out R² | fixed A-probe on B, t ∈ [2000,4000) |
|---|---|---|---|---|---|---|
| base (step 0) | 1.735 | 0.178 | 0.003 | 0.008 | 0.855 | 0.143 |
| 4A lr 1e-4, step 10 | 0.048 | 0.189 | 0.012 | 0.018 | 0.860 | −0.714 |
| 4A, step 25 | 0.0106 | 0.492 | 0.242 | 0.132 | 0.891 | −0.714 |
| 4A, step 50 | 0.0062 | 1.200 | 0.319 | 0.287 | 0.886 | −0.676 |
| 4A, step 100 | 0.0043 | 1.309 | 0.432 | 0.313 | 0.885 | −0.714 |
| 4B lr 2e-5, step 25 | 0.098 | 0.178 | 0.004 | 0.011 | 0.858 | −0.124 |
| 4B, step 50 | 0.026 | 0.178 | 0.012 | 0.014 | 0.854 | −0.137 |
| 4B, step 100 | 0.0092 | 0.173 | 0.093 | 0.021 | 0.862 | 0.365 |

Criterion: A KL below the 3-HMM KL (0.037) is reached by step 25 in regime 4A and by step 100 in regime 4B. Neither regime raises the A probe above about 0.89, so a "probe accuracy criterion" in the proposal's sense is not reachable: the base model is already at 0.84–0.86, and fine-tuning adds ≤ 0.03.

Full evaluation on 5 A and 5 B sequences (seeds 0–4, N = 4000), best layer per window:

| window of B | [0,100) | [100,250) | [250,500) | [500,1000) | [1000,2000) | [2000,3000) | [3000,4000) |
|---|---|---|---|---|---|---|---|
| **fixed A-probe on B (Measure 2)**: base | −6.08 | −1.11 | −0.11 | 0.28 | 0.60 (L11) | 0.46 | 0.55 |
| 4A step 10 | −4.52 | −1.20 | −0.31 | −0.20 | 0.23 | 0.16 | 0.15 |
| 4A step 100 | −6.27 | −1.09 | −0.56 | −1.57 | −0.26 | −0.77 | −0.43 |
| 4B step 100 | −6.58 | −0.71 | −0.35 | −0.04 | 0.40 | 0.03 | 0.24 |
| same, best state permutation: base / 4A step 100 / 4B step 100 | −0.95 / −0.34 / −0.87 | −0.34 / −0.39 / −0.23 | 0.04 / −0.01 / 0.02 | 0.39 / −0.23 / 0.09 | 0.61 / −0.24 / 0.43 | 0.47 / −0.26 / 0.25 | 0.55 / −0.20 / 0.25 |
| **fresh B-probe**: base | 0.93 | 0.59 | 0.68 | 0.78 | 0.90 | 0.82 | 0.86 |
| 4A step 10 | 0.94 | 0.64 | 0.68 | 0.78 | 0.90 | 0.82 | 0.86 |
| 4A step 100 | 0.95 | 0.81 | 0.73 | 0.78 | 0.90 | 0.82 | 0.87 |
| 4B step 100 | 0.95 | 0.66 | 0.68 | 0.78 | 0.90 | 0.82 | 0.86 |
| refit in A-probe subspace: base / 4A step 100 / 4B step 100 | −1.26 / −0.34 / −1.84 | −0.05 / 0.40 / −0.02 | 0.34 / 0.34 / 0.35 | 0.56 / 0.50 / 0.60 | 0.76 / 0.75 / 0.76 | 0.75 / 0.72 / 0.75 | 0.74 / 0.70 / 0.75 |
| B KL restricted: base / 4A step 100 / 4B step 100 | 0.089 / 0.074 / 0.073 | 0.100 / 0.123 / 0.125 | 0.170 / 0.241 / 0.299 | 0.170 / 0.385 / 0.215 | 0.189 / 1.228 / 0.226 | 0.195 / 1.421 / 0.194 | 0.190 / 1.151 / 0.174 |
| B mass on A tokens: base / 4A step 100 / 4B step 100 | 0.002 / 0.499 / 0.082 | 0.005 / 0.523 / 0.123 | 0.006 / 0.550 / 0.134 | 0.004 / 0.655 / 0.112 | 0.002 / 0.451 / 0.075 | 0.003 / 0.512 / 0.124 | 0.002 / 0.414 / 0.076 |
| B off-alphabet mass: base / 4A step 100 / 4B step 100 | 0.15 / 0.53 / 0.16 | 0.19 / 0.54 / 0.19 | 0.47 / 0.57 / 0.38 | 0.93 / 0.73 / 0.85 | 0.97 / 0.53 / 0.93 | 0.99 / 0.71 / 0.98 | 0.99 / 0.75 / 0.98 |

Readings:
1. **Fine-tuning moves A into the weights, not into context.** After 4A, A KL is 0.006 already at t ∈ [0,100) (base 0.47), so the fine-tuned model predicts A near-optimally without any context. "Observations needed on B" then asks whether weight-stored knowledge crosses alphabets. Behaviourally it does not: B restricted KL rises from 0.19 to 1.15–1.42, and 4A predicts F at the start of B sequences, a BOS-conditioned prior.
2. **Measure 2 gets worse with fine-tuning.** The base model's partial Wing transfer (0.60) is destroyed by 4A (negative everywhere) and halved by 4B (0.40). Permutation search does not rescue it.
3. **Re-formation is not faster after fine-tuning**, except one window: 4A step 100 at [100,250) has fresh-probe R² 0.81 against base 0.59 (late layers, 5 leave-one-sequence-out folds, about 150 positions per sequence; noise not quantified). From t ≥ 500 all four model states agree to within 0.01.
4. The window [0,100) gives fresh-probe R² 0.93–0.95 for every model state, already 0.85 after one block. This comes from the shared initial transient starting at π, which is identical across sequences, not from fast learning. Exclude t < 100 when timing re-formation.
5. Mess3 check: done zero-shot in Experiment 3 (best-permutation R² still negative; the permutation is consistently a 3-cycle). No Mess3 LoRA run was made.

## Caveats and deviations

- One model, one seed per sequence index, and 3–5 sequences per cell. Best-over-layer maxima are optimistic, and the fresh-probe "best λ" in Experiment 4 is selected on the test folds (6 candidates). The cross-sequence tables use inner-validation λ.
- In Experiment 3 and the Experiment 4 evaluation, A and B share symbol sequences. Probes are always applied to a held-out sequence (leave-one-sequence-out), except the "single-sequence A-probe" row, which tests on sequences 1–4.
- Exp 4 trains at 1024 tokens and evaluates at 4000. The quick-checkpoint A-probe uses only 3 sequences.
- `"\n\n"` (token 271) is the only separator tested.
- The λ grid is absolute while activation scale varies by layer, so the selected λ drifts with depth.
- The first parallel launch thrashed the CPU (three BLAS-heavy jobs on 20 cores). I killed the Exp 3 and Exp 4 runs and reran them one at a time; no results come from the killed runs.
- RAM: every model or PyTorch job ran under `systemd-run … MemoryMax=10G MemorySwapMax=0`. Peak VRAM was 5.96 GiB (fp32 check) and 5.74 GiB (LoRA training).
- Wall time: forward passes for all 57 base-model runs took 52 s. Exp 2 took about 18 min per HMM for the main splits, about 9 min per HMM for the trajectory, and a few minutes per HMM for the 0/1-HMM controls. Exp 3 transfer took 12 min and the switch 17 min. Each LoRA regime trained in 4 min, and each Exp 4 evaluation took 5 min.

## Files

Code: `cmlib/hmm.py` (processes, filter, k-HMM, samplers), `cmlib/probe.py` (SVD ridge/OLS, R²), `cmlib/pooled.py` (pooled sufficient-statistic ridge, permutations, subspaces, CKA, Procrustes), `cmlib/protocols.py` (splits), `cmlib/features.py`, `cmlib/lm.py`, `cmlib/lora.py`, `cmlib/common.py`. Scripts: `run_all.sh`, `exp0_sanity.py`, `forward_all.py`, `exp1_convergence.py`, `exp2_probes.py`, `exp3_cross_symbol.py`, `exp4_train.py`, `exp4_ckpt_probe.py`, `exp4_eval.py`, `diag_alignment.py`, `diag_top_tokens.py`, `merge_results.py`, `tables.py`. Tests: `tests/test_core.py`.

Results: `results/exp0_bf16_vs_fp32.json`, `results/exp1_convergence.json`, `results/exp2_probes_controls.json`, `results/exp3_cross_symbol_icl.json`, `results/exp4_finetune_reuse.json`, `results/diag_alignment.json`, `results/diag_top_tokens.json`. Temporary activation caches and LoRA checkpoints live under git-ignored `cache/` (~12 GiB) and regenerate via `forward_all.py` and `exp4_train.py`.
