# Q7 audit: is speculative depth distribution-preserving? Verdict: yes, for M in {1..4}

Source: Pennyroyal `jpezzulli/sglang-rtxpro6000` at d00d88efc8d6281b12be4f4073126aec95038c55 (dfranzen's PENNY_SHA). Clone at `~/projects/arc-agi-3-reference/pennyroyal/sglang-rtxpro6000`.
Config: `server_args` from the control run's serve.log (`analysis/m2-winners-20261001/runs/out-control/serve.log`).

## Production configuration (observed)

- `speculative_algorithm=EAGLE`, `speculative_eagle_topk=1`, `speculative_num_steps=3`, `speculative_num_draft_tokens=4`.
- `speculative_accept_threshold_single=1.0`, `speculative_accept_threshold_acc=1.0`, `speculative_use_rejection_sampling=False`.
- `speculative_token_map=hot_tokens_64k.pt`, `enable_deterministic_inference=False`, `sampling_defaults=model`.
- Harness sampling: temperature 0.7, top-p 0.95, top-k 20.
- Model generation_config (Intel W4A16): temperature 1.0, top_k 20, top_p 0.95, no penalties. The harness sends no presence, frequency or repetition penalty (grep of ARC3-Inference).

## Chain

1. **Verify path.** Sampling is non-greedy, so `eagle_utils.eagle_sample` (lines ~755-850) takes the sampling branch. With `use_rejection_sampling=False` it calls `tree_speculative_sampling_target_only` with `draft_probs = zeros`.
2. **Target distribution.** `target_probs = softmax(logits / T)`, then `top_k_renorm_prob`, then `top_p_renorm_prob`, over the full target vocabulary. These are the same transforms as ordinary sampling. Penalty and logit-bias branches are inactive because none are set.
3. **Kernel** (`kernels/aot/csrc/speculative/speculative_sampling.cuh`, `TreeSpeculativeSamplingTargetOnly`):
   - With topk=1, each depth has one candidate x. It is accepted iff `coin <= p(x) / threshold_acc` (threshold 1.0), or `p(x) >= threshold_single` (1.0, i.e. only p = 1).
   - On rejection, `draft_probs[x] = p(x)`, and the final token is sampled from `relu(p - draft_probs)`, which is p with x removed, renormalized.
   - If every draft is accepted, the bonus token is sampled from the full p.
   - The coins are fresh `torch.rand` per position, plus a separate coin for the final sample.
4. **Exactness for any proposal.** With proposal x' ~ q, accept x' with probability p(x'), else sample from p(. | . != x'). Then
   P(y) = q(y) p(y) + sum over x' != y of q(x') (1 - p(x')) p(y) / (1 - p(x')) = p(y).
   Chained depths condition on the accepted prefix, and each depth's target probabilities are computed on that prefix. The emitted sequence is therefore distributed exactly as target-only sampling, whatever the depth M.
5. **Token map.** `hot_vocab.py` (FR-Spec) selects hot rows of the lm_head *for the draft*, and `eagle_worker_v2` loads it into `hot_token_id` for the draft worker. The target logits keep full width. The map changes only the proposal q, so step 4 still holds.
6. **What depth changes.** M changes `num_draft_tokens` and the depth of the chain loop, which is proposal and verification work only. The target sampler is the same code path for every M >= 1.

## Residual (not distributional)

- Batch shape changes with M, so logits can differ at floating-point level (kernels are not batch-invariant). This is numeric noise, not a different distribution.
- M=0 (speculation off) uses the ordinary sampler, a different code path. It is not part of this sweep. Its empirical comparison is deferred.
- The kernel carries a "relaxed penalties" path. It is inactive here; if penalties are ever added, re-audit.

Classification: M in {1, 2, 3, 4} is lossless relative to M=3 under the production config. A winning M follows the lossless graduation rule in CARD.md.
