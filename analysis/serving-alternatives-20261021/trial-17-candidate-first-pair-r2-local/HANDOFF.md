# Trial17 r2 exact local handoff

Separate narrow repair of the two findings in [the blocked review](../coordination-20261024/trial17-independent-review-blocked.json). Original submitted freeze3387af46 and all21028 bindings, root qualification and both witness directories remain preserved. No assignment, API/GPU/slot/launch/staging/commit authority.

Active freeze: `34befabf21e93d7753e251ae6aa3e1131ab224f4395779f60f4845bf9a1f32aa`. All **22075 bindings** verified after freeze: 584 local and 21491 references. [HANDOFF.json](HANDOFF.json) and this note are excluded only to avoid digest cycles.

Review [REPAIR.md](REPAIR.md), [exact production diff](qualification/R2-PRODUCTION.diff), [delta inventory](qualification/R2-DELTA.json), [owner review](qualification/R2-REVIEW.json), and [CARD.md](CARD.md). Ordinary command exit0 remains mandatory; health semantics stay in the unchanged exact raw readiness checker and mandatory eventual200, with generic lifecycle clocks/IDs/cleanup independently enforced. Primary collection checks one absolute120s through listing, download, verification and actual receipt-write return. Late/error completion raises and any positive receipt is quarantined; no hard syscall or arbitrary-storage durability guarantee.

**77 focused checks PASS, 34.720s**, actual CLI exit0/stderr empty, network0/GPU0. [Final proof](qualification/final-exact/PROOF.json) SHA `c781e68b209873b2c910fce6da5b6268a6c9127632a1c7092defd0c6ee13c0a9`. The real retained candidate75→200 and control75/pinned503→200 sequences run readiness and phase-deadline checking together. Negative malformed503/unexpectedHTTP/missing200/ordinary failure/late cleanup/PID fixtures remain rejected. Real codec+identity verification reproduces both121s witnesses RED before GREEN after, plus expired listing/no GET, persisted-then-write-error and timely path. [R1 RED](qualification/R1-RED.txt), [R1 GREEN](qualification/R1-GREEN.txt), [R2 RED](qualification/R2-RED.txt), [R2 GREEN](qualification/R2-GREEN.txt); final suite reruns all77. Original68 checks retained; inherited65 not rerun. Two existing isolated source mutants detected,1369 exact wire bodies rebuilt, installed SDK serialization intercepted SaveKernel1/network0 and new outer bootstrap contexts exercised. No full scientific pair pass is demonstrated.

Exact source digest `a9ceaa8441a4d77a3a8aa3e65a08844bbec8a2bd41df8e93b8b6c8a1e8cbc15d`; config `3cbd412dd6d2bf7b213f7c7b93a6b6bc837aca5aae92f7c0101d8278ca48397b`. Config and metadata byte semantics identical to submitted snapshot. Only runtime payload difference is pair_evidence.py; collection helper is local-only. **57 payloads /941085B unassigned outer**. New source pins are in HANDOFF.json. [review-packet.json](review-packet.json) names the executable payload; review-kernel remains unassigned. [RISK-BASELINE.json](qualification/RISK-BASELINE.json) and [final RISK.json](qualification/final-exact/RISK.json) use the same AST rule. No baseline coverage or comparable CRAP improvement claim; source review exposes unexecuted GPU/operational/storage-failure branches.

Caps remain10500platform/10200cooperative/9900work/300cleanup; phase9650/slack250, two1500 windows,2738 maximum workload POSTs,10 concurrent per arm. Fixed candidate-first total-package estimand, original140 independently report-only, valid host/OOM/resource/ownership/final180/publication/scientific gates and scoped1.20 mapping remain unchanged. No historical comparison/adoption/reliability inference. Inherited64/65, remoteFD/stack/3.12patch/live-code, delayed-precutoffEOF and sampled-memory limitations remain.

Reproduce locally with installed interpreter (fresh output required):

```bash
/home/pranav/.local/share/uv/tools/kaggle/bin/python \
  /home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-17-candidate-first-pair-r2-local/verify.py \
  /ABSOLUTE/FRESH/QUALIFICATION_DIRECTORY
```

The completed command, stdout/stderr hashes and observed exit are retained in [FINAL-EXIT.json](qualification/FINAL-EXIT.json). No live API was called and no GPU process, scheduler, collection monitor, stage or commit was created. Sole implementation remains Richard Hamming; root review and new explicit assignment are separate future gates.
