# marlin-reduction-replay-v3-20261010 readout (2026-10-10)

Kernel 138024107 v1 COMPLETE in ~17 min (v2 code fix confirmed: apply_patch.py ran). Raw audit (`readout/AUDIT.json`,
nooa venv numpy) qualified=true, 231 raw hashes verified, advance_to_serving_preparation=false.

Benchmark stopped after case 1/9 (layer-00-replay-08, 40 rows, TARGET_VERIFY, graph key size 10): numerical gate failed on
the `changed` variant (hidden*0.5, topk_weights*0.5, topk_ids+1). No timing was measured for any case.

| variant | ref RMS | baseline nmax | treatment nmax | drift nmax | gate |
|---|---|---|---|---|---|
| original | 0.0224 | 0.068 | 0.068 | 0.044 | pass |
| changed | 0.00107 | 0.502 | 0.502 | 0.114 | FAIL (abs nmax <=0.25; drift nmax <=0.05) |
| restored | 0.0224 | 0.068 | 0.068 | 0.044 | pass |

Reading: the unchanged control kernel fails the absolute gate exactly as the treatment does (identical max error 5.35e-4
at ref -0.0254); treatment RMS error is lower than control on every variant. Drift max = 2^-13 = one BF16 ULP at that
magnitude. BF16 output rounding alone explains nmax 0.053, so the shared 0.50 comes from the production pipeline
(likely BF16 intermediates), not from the treatment. Poisoned-scratch outputs equal treatment exactly (scratch fully
overwritten); graph allocation identical (205,824 B); treatment flags reached the GEMM (12.3 MB scratch expected).
By the card's letter this is a numerical rejection; by the evidence the gate is miscalibrated for the changed input
and the treatment is not shown to be numerically worse. The speed question is unanswered.

Collector defect: in-process audit imports numpy under the Kaggle SDK interpreter -> ModuleNotFoundError loop after a
complete download; collector was stopped manually and the audit run separately.
