# Serving alternatives: ceiling, trial-1 bar, suffix speculation (2026-10-21)

Companion to the Codex survey in `analysis/serving-alternatives-20261021/` (not edited here).

## Engine-swap ceiling
- F1x+REAP448 at 14 streams: 823 tok/s, acceptance 2.67 → one verify step ≈ 14×2.67/823 = 45 ms.
- Expert reads: 56 tokens/step × top-10 of 512 touch ~67% of experts ≈ 0.67 × 62 GB / 1.79 TB/s ≈ 25 ms.
  The 10-05 fit implies the expert GEMMs already run at ~1.63 of 1.79 TB/s (~91% of peak), so an engine can't shave this part.
- The remaining ~20 ms is attention/KV reads, dense layers, launch and scheduler. A better engine can win only here.
  Halving it gives 1.28×. Eliminating it gives 1.8×, which is impossible because KV reads are real bytes.
- Independent check: lordhansolo's vLLM (NVFP4/FP8) logs 975–1,010 tok/s at 14 streams vs our 815–823 → 1.2×.
- Elasticity 0.35: 1.2× → +6.6% levels, 1.3× → +9.6%. That's about 2 hidden points vs draw sd 3.2. A small win, not a ≥20% bet.
- Breaking the ceiling needs fewer bytes per accepted token: lower-bit experts (~3 bpw, EXL3 only, quality risk) or more tokens
  per step (streams: KV-bound; MTP depth: killed 10-16; suffix speculation: below).

## Trial-1 bar is stale
`serving-vllm-smoke-1` compares against 477.6 (10-08 SGLang M3A, unpruned base model, S=10). The stack we'd ship is REAP448 at S=14,
1.18× faster in game. A "GO ≥1.30×" result therefore means ~1.10× over the shipped stack, which is a KILL under the same rule.
Equivalent bar vs 477.6: ≥1.53×. Better: replay the REAP448 SGLang stack at S=14 on the same harness, and measure the candidate at its own best S.
The candidate also changes quantization, so even a speed GO needs a quality screen, and sd ~9 levels can't see +7%.

## Suffix / prompt-lookup speculation (offline, $0)
`simulate.py` on every logged request history (16 runs, 224 distinct assistant messages, 534K generated tokens; Flash-Next tokenizer):
- 16.6% of generated tokens fall in copy runs where a suffix proposal accepts ≥4 tokens (beats MTP3's per-step max of 4). Per run: 9–21%.
- Copy steps average 8.5 tokens.
- MTP+suffix hybrid decode gain: 1.06× (MTP assumed perfect on copy spans) to 1.13× (MTP at its mean there), before verify-cost overhead.
  That is +2–4% levels. It is lossless (same weights, same distribution), so it qualifies as a no-harm small win, but only if the engine
  supports MTP+suffix together. Unverified for SGLang Pennyroyal.

## Corrections after coordinator review (2026-10-21 ~14:30Z)
- Objection accepted: REAP14's 1.18 is a gameplay rate; 477.6/518.2 are replay estimators. The 1.53 figure is a planning
  estimate, not an acceptance baseline. REAP14 is not a proven incumbent (queue blocked on D12's 0.00).
- 477.6/518.2 itself comes from 10-08 MTP retry1, which the frozen analyzer graded INVALID WORKLOAD: mean running 8.05 vs band 8.57–9.37.
  It is a screening number, not a reference.
- Cheapest direct reference: in the same Kaggle session as the candidate, run the proven serving stack (dfranzen SGLang Pennyroyal v2.5.3,
  NEXTN 3/4, S=10, which is what control and F1x run) on the identical replay block and occupancy gate. Report U_resp ratio at matched achieved occupancy.
  It costs ~10 min load + one replay block. Same GPU instance, so no cross-session hardware variance (second-kernel T4 risk).
- Suffix verify cost (step model, S=14, 6% of stream-steps are copy steps): K=12 → step +3.6%, net 1.02–1.08×. K=32 → step +10%, net 0.97–1.03×.
  ≤ +3% levels at best.
- Engine support: Pennyroyal SpeculativeAlgorithm is a single enum (NEXTN xor NGRAM). Local vLLM dev source SpeculativeConfig.method is single
  ("mtp" xor "suffix"/"ngram"). Neither engine can run the hybrid. NGRAM alone = 1.36 tok/step vs MTP 2.67, a loss.
  So the suffix estimate supports no GPU action. It would need a new combined proposer (engine development), which can't land before the 10-30 cutoff. Parked.
- Scope (coordinator, accepted): <1.20x in the matched same-session comparison closes only the tested runtime/configuration, not serving swaps in general.
  477.6/518.2 remain descriptive only.
