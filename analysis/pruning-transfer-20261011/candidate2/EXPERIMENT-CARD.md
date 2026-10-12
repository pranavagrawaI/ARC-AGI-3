# prune-retain-candidate2-20261011
Question: Does REAP448 with 18 streams play more levels than c14 (512 experts, 14 streams) on the current Intel/SGLang stack?
Why a rerun: candidate 138302732 never played. It drew a slow disk (185 GB precache in 504 s), was still loading at dfranzen's 722 s deadline, and the health assertion build.py appended made that fatal (readout/README.md). The pruned model itself loads cleanly (probe2: accept 3.10, 32/32, KV 1.37M vs 0.91M tokens).
Change from candidate: only notebook cell 12. The fatal health assert is replaced by a wait for READY up to 1300 s from notebook start. It still raises if the server dies or never comes up. Play starts READY, so the 7254 s per-game horizon is untouched; 1300 + 7254 s stays inside the 9000 s cap. The control passed the old check at 616 s, so it would have behaved identically under the new one.
Control: today's collected c14 run 138302697 (103 levels, mean score 44.10, 829 tok/s). Reference only: E2 c14 107 levels. Replicate sd is about 9 levels; a single pair resolves only large effects.
Decision rule (unchanged, evaluate.decision): RETAIN_CANDIDATE needs levels >= 1.10x control, mean score >= control, aggregate tok/s >= 1.10x control and 0 retractions. REJECT if levels <= control - 10 or score <= 0.90x control. Otherwise INCONCLUSIVE, retain c14.
Prediction, registered before launch: aggregate tok/s above c14's 829, and per-stream decode below c14's ~61 tok/s. The c14 readout showed early levels fall when per-stream speed drops, so expect a deeper early deficit and a late catch-up. Final levels within about ±10 of 103.
Also record: READY time ("SERVER READY at"), levels-over-time against the two c14 runs (readout/levels_time.py), and KV usage.
Cap: one RTX PRO 6000, one launch, 9000 s. No retry, no submission.
Authority: Pranav, 2026-10-12: "do the REAP rerun." See AUTHORITY-R3.json.
