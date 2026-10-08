# MTP retry1 status and readout, 2026-10-08

Kaggle reports COMPLETE with no notebook-level failure. The experiment is **INVALID WORKLOAD**, so it supports no change in MTP depth.

The competition attachment correction worked operationally: actual hardware was one NVIDIA RTX PRO 6000 Blackwell Server Edition with 97887MiB. Both M3A and M1 reached health, completed 240-second measured blocks and had zero request errors. The prior PLE initialization failure did not recur on these launches.

| Launch | Startup | Observed completion tokens/s | Interpretation |
|---|---|---:|---|
| M3A | Healthy after579s | 518.21 | Workload validity failed |
| M1 | Healthy after1175s | 481.03 | Observational rate only; no qualified comparison |
| M2, M3C, M4, M3B | Each health timeout at20min | Unmeasured | Logs show incomplete model-weight loading |

M3A mean running requests was **8.051**, below the frozen band **8.57–9.37**. Fraction of samples with >=9 running requests was **0.4298**, below **0.62–0.82**. The frozen analyzer therefore stops before candidate comparison. Additionally, M3C and M3B are absent, so the mandatory three-anchor contrast could not be completed. M3A subwindow variation was21.83%; cache fraction0.9293. Neither cache fidelity nor successful replay overrides the occupancy gate. No alpha refit was performed.

The later four logs terminate during safetensors shard loading, rather than after a model initialization exception. Startup loading slowed enough that the20-minute health deadline stopped each attempt. Why loading slowed remains unresolved; these timeouts do not demonstrate an MTP-depth incompatibility. nsys was absent, so the profile was deferred. Last launch finished2026-10-07T18:27:09.956Z. This is session evidence, not accelerator-quota accounting.

All25 output files were collected under raw/. All37 runtime payload hashes match the frozen manifest. M3A/M1 summaries were reproduced from their raw events.jsonl using the frozen summarizer, and the saved verdict was reproduced with the frozen analyzer. Runtime provenance records retry1, parent code28ef795f16f5165dc6c71426569bf7a16968d970, both freeze tags and the expected manifest SHA. Frozen source/cards and launch receipts remain unchanged.

Retain M=3. The next useful work is an offline diagnosis of occupancy mismatch and repeated model reload latency before authorizing another GPU attempt. No new launch, refit, adoption or scored submission was performed for this status request.
