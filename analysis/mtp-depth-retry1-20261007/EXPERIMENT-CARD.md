# mtp-depth-retry1-20261007
Question: On the intended RTX PRO 6000, which MTP depth maximizes emitted completion tokens/s under the frozen production replay?
Posture: Corrected execution after hardware eligibility failure, followed by validity evaluation.
Authority: Pranav explicitly requested "do a retry" in this thread on 2026-10-07. Exactly one new push attempt. Original run remains immutable.
Contrast: M3A -> M1 -> M2 -> M3C -> M4 -> M3B, then separate optional PROFILE_M3. Same replay dataset v2, incumbent flags, alpha=.523, workload and analyzer as tagged original.
Prediction: Attaching arc-prize-2026-arc-agi-3 enables the requested RTX PRO 6000; M3A may then start and pass occupancy. PLE health and comparative gain remain unknown.
Measure: /kaggle/working/mtp/hardware.json, per-label server.log/launch.json/events.jsonl, sweep_state.json, readback.json and verdict.json. U, drift D and variation W use original frozen definitions.
Cap: One private Kaggle kernel push attempt, one RTX PRO 6000, 4h notebook sweep deadline and 16200s hard timeout. Admit only when typed used+reserved+16200s <=30h and a slot is free. Local watcher waits at most 15min for safe admission; no second save attempt. No scored submission.
Stop: Abort before installation/loading on wrong or unknown GPU inventory. Preserve each startup failure and proceed according to frozen six-launch orchestration. Stop at deadline; optional profile cut first. Any ambiguous push acceptance stops without retry.
Invalid if: Actual hardware differs from one RTX PRO 6000 >=90000MiB; runtime readback differs; M3A missing or occupancy gate fails. Missing M3C/M3B is unresolved. All three usable anchors mandatory.
Result map: Valid clean gain -> propose short live qualification before adoption; valid null/loss -> retain M3; invalid/startup failure -> retain M3 and diagnose separately, no automatic retry; ambiguous drift -> retain M3 pending a separately authorized replicate.
Outputs: Runtime /kaggle/working/mtp; local analysis/mtp-depth-retry1-20261007/evidence and output. Frozen parent card analysis/mtp-depth-20261006/CARD.md, tag mtp-depth-sweep-prelaunch-20261007. New freeze tag mtp-depth-retry1-prelaunch-20261007.
