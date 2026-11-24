# De-Risking Cognitive-Map Measures in LLMs (`OLMo-2-0425-1B`)

Empirical stress test of a proposed predictor of fluid intelligence (ARC-AGI-3) borrowed from human cognitive-map neuroscience (Tenderra & Theves, 2025) and computational-mechanics belief-state geometry in transformers (Shai et al., 2024; Balcells et al., 2026).

## Research Question Under Test

Can two benchmark-independent linear-probe measures taken on hidden Markov model (HMM) token streams cleanly operationalize **map formation** and **structure reuse** in open-weight LLMs?

- **Proposed Measure 1 (Map Formation):** Held-out linear-probe $R^2$ decoding the exact Bayesian belief state $\eta(x_{1:t})$ from residual-stream activations after in-context predictions converge, controlled against a probe for running symbol frequency.
- **Proposed Measure 2 (Structure Reuse):** Fine-tune a model on one HMM written in alphabet $A$ (`{F, Q}` or `{F, Q, V}`) until probe accuracy reaches a shared criterion, then present the same HMM in a novel alphabet $B$ (`{K, M}` or `{K, M, Z}`) without telling the model, and measure how many observations are needed before the fixed probe $W_A$ trained on $A$ recovers beliefs on $B$.

## Headline Verdicts

Tested on `allenai/OLMo-2-0425-1B` ($L = 16$, $d = 2048$, context length $4096$) across three contrasting HMM families from Balcells et al. (2026): `Mess3` (3 states, 3 symbols, $S_3$-symmetric, invertible emission matrix), `Wing` (3 states, 2 symbols, asymmetric, non-invertible), and `Arch` (4 states, 3 symbols, asymmetric, non-invertible).

All $R^2$ values are variance-weighted held-out $R^2$ (`1` = perfect, `0` = predicting the test mean, negative = worse than the mean; higher is better). KL divergence is in nats against the exact Bayesian HMM predictor (`0` = optimal; lower is better).

| Failure mode | Scope | Verdict | Decisive evidence |
|---|---|---|---|
| **FM1: Non-convergence & off-alphabet collapse** | 1B / 4K-context limit | **Confirmed** | Over $t \in [3000, 4000)$, within-alphabet KL is worse than a unigram (`0-HMM`) baseline on `Mess3` (`0.331` vs `0.222`) and `Arch` (`0.167` vs `0.095`), and ties `1-HMM` on `Wing` (`0.083` vs `0.083`). Off-alphabet probability mass rises from `7% -> 75%` (`Wing-A`) and `18% -> 99%` (`Wing-B`). |
| **FM2: Split leakage, passive history decoding, and broken control** | **Structural (all scales)** | **Confirmed** | Random within-sequence splits recover near-constant cumulative symbol frequency (variance $2\times 10^{-4}$) at $R^2 = 0.997\text{–}0.999$, then collapse to `-0.35` to `-18.1` on contiguous or cross-sequence splits. On structureless i.i.d. (`0-HMM`) sequences, contiguous-split belief $R^2$ is `0.824` (`Wing`) and `0.665` (`Arch`), matching or beating true HMM sequences (`0.755` and `0.133`). |
| **FM4: OLS instability & no surplus over next-token probabilities** | **Structural (all scales)** | **Confirmed** | Minimum-norm OLS blows up to $R^2 = -2$ to `-42` cross-sequence; nested-CV Ridge is required. Under cross-sequence Ridge, activation probes never beat the exact next-token-probability baseline ($\Delta R^2_{\text{acts}-\text{NTP}} = -0.045$ `Mess3`, $-0.063$ `Wing`, $-0.24$ `Arch`). |
| **FM3a: Cross-symbol subspace non-alignment (zero-training ICL)** | **Structural (all scales)** | **Fails on 3-symbol HMMs; partial on binary `Wing`** | Fresh $B$-probes match $A$-probes (`Mess3`: `0.977` vs `0.977`; `Arch`: `0.734` vs `0.720`) with linear CKA `0.94–0.99`, yet fixed $A$-probes on $B$ score negative $R^2$ (`-0.03`, `-0.13`) even over all state permutations. Principal cosines between $A$ and $B$ subspaces (`0.02–0.14`) match random subspaces (`0.04–0.05`). |
| **FM3b: Literal Measure 2 (LoRA on $A$, fixed $W_A$ on $B$)** | **Structural (all scales)** | **Fails — fine-tuning hurts transfer** | LoRA on `Wing-A` (`lr = 1e-4`, 100 steps) reaches `KL = 0.0043` on $A$ but leaks `41–65%` probability mass onto `{F, Q}` during $B$ prompts, raises $B$ restricted KL `6x` (`0.19 -> 1.15–1.42`), and drops fixed-probe $W_A \to B$ from `+0.60` (base) to negative everywhere (best `-0.26`). Gentle LoRA (`lr = 2e-5`) still cuts transfer (`0.60 -> 0.40`). |

For full per-layer tables, methodological caveats, and control breakdowns, see [summary.md](summary.md).

## Key Implications for Redesign

1. **Measure 1 (Map Formation):** Drop `Mess3` (invertible emission matrix; $R^2(\text{NTP} \to \eta) = 1.0$). Require behavioral convergence (`k_eff >= 2`) as a pre-gate, use cross-sequence splits with scale-normalized nested-CV Ridge, and report **surplus** belief decodability over both the next-token-probability baseline ($\Delta R^2_{\text{NTP}}$) and a matched i.i.d. `0-HMM` sequence control ($\Delta R^2_{\text{0-HMM}}$). Replace cumulative frequency with windowed symbol frequency (e.g., 50-step window).
2. **Measure 2 (Structure Reuse):** Do not evaluate a fixed probe $W_A$ after single-alphabet fine-tuning — transformers encode the same belief geometry in token-anchored, near-orthogonal subspaces across alphabets, and single-alphabet LoRA worsens symbol entanglement. Either (a) train/prompt across **multiple** alphabets to create inductive pressure for a shared subspace before testing zero-shot transfer to a held-out alphabet, or (b) measure pure in-context transfer using behavioral convergence speedup plus alignment-invariant geometry (Procrustes / CKA) excluding $t < 100$.

## Repository Layout

```text
analysis/cognitive-map-derisk/
├── README.md              # Overview, verdicts, and reproduction guide
├── summary.md             # Full quantitative report, tables, and caveats
├── requirements.txt       # Minimal Python dependencies (torch, transformers, numpy, pytest)
├── run_all.sh             # End-to-end reproduction script
├── cmlib/                 # HMMs, exact Bayesian filter, k-HMM baselines, SVD Ridge, CKA/Procrustes, LoRA
├── tests/test_core.py     # 21 unit tests (HMM math, tokenization alignment, Ridge/OLS, causal features)
├── exp0_sanity.py         # bf16 vs fp32 numerical agreement check
├── forward_all.py         # Base-model forward passes and activation caching
├── exp1_convergence.py    # In-context KL convergence vs analytical k-HMM baselines
├── exp2_probes.py         # Belief probes, split protocols, and all 6 controls
├── exp3_cross_symbol.py   # Zero-training cross-alphabet transfer, subspace angles, and in-context switch
├── exp4_train.py          # LoRA fine-tuning on Wing alphabet A (regimes 4A and 4B)
├── exp4_ckpt_probe.py     # Checkpoint probe trajectory
├── exp4_eval.py           # Full evaluation on alphabets A and B
├── diag_alignment.py      # Next-token shift alignment & BOS diagnostics
├── diag_top_tokens.py     # Top-token probability leakage diagnostics
├── merge_results.py       # Merges per-part outputs into results/*.json
├── tables.py              # Reprints markdown summary tables from results/*.json
└── results/               # Compact JSON outputs for every experiment and diagnostic
```

## Reproduction

Requires Python 3.11+ and a single CUDA GPU with at least 8 GiB VRAM (peak measured: 5.96 GiB for the fp32 check, 5.74 GiB for LoRA training) and ~12 GiB temporary disk space for cached bf16 activations.

```bash
cd analysis/cognitive-map-derisk
pip install -r requirements.txt

# 1. Verify math, tokenization, and probe implementations (21 tests)
pytest -q tests/test_core.py

# 2. Reprint all tables directly from committed results/ JSONs (no GPU needed)
python3 tables.py exp0 exp1 exp2 exp3 exp4

# 3. Re-run the full GPU + probing suite end to end
./run_all.sh
```

Set `HF_HUB_OFFLINE=1` if running from an existing local Hugging Face cache.
