# gpu-q4 result — 2026-10-04 inspection

Remote status is COMPLETE, verified again on 2026-10-04. The notebook ran on one RTX PRO 6000 Blackwell GPU. Offline install and FlashNext serving succeeded. The notebook log reports:

| Game | Levels completed | Actions | Score | Completed output tokens |
| --- | ---: | ---: | ---: | ---: |
| AR25 | 1/8 | 16 | 2.78 | 28,185 |
| SB26 | 1/8 | 16 | 2.78 | 33,582 |
| LS20 | 0/7 | 16 | 0.00 | 34,479 |

Mean score 1.85; 48 actions, 96,246 reported output tokens. All three ended gave_up at the configured 16-action cap; no game was fully solved. These are smoke results, not a matched comparison. The log reports 7m44s for the gameplay phase; notebook preflight started 11:27 UTC and the final log event was at about 20m10s. These timestamps are not an authoritative GPU billing receipt; the budget conservatively charges the entire one-hour cap, leaving nine hours.

The read-only collector observed completion at 11:50 UTC on October 3, then timed out after ten minutes downloading Python environment files. It did not inspect the results or launch a follow-on experiment. No donor/hybrid comparison or ablation ran.

Recovery: retrieved the notebook log directly from the API and started a filtered output download. The collector now filters for gameplay artifacts and uses 200 files per page. Detailed role/tool/action reconciliation remains pending until those artifacts arrive. The original failed-download receipt is preserved.

Evidence: output/notebook-readable.log; output/serve.log; output/qualification-start.json; collection-receipt.json; status-history.jsonl.

## Additional verified evidence

Downloaded benchmark.json confirms the three outcomes and candidate-config.json confirms orchestrator/control, FlashNext, 600 seconds and 16 actions per game. Directly retrieved AR25 compute/tool/exit receipts show 13 Actor calls and 3 Scribe calls, zero Builder calls and zero invoke_builder attempts. AR25 executed 16 actions, reported no host failure, and its worker exited 0. Thus AR25 success does not qualify Builder execution or repair on the live model. Other game role traces remain under retrieval.

## Complete role/action summary audit

The selected top-level traces for all three games were recovered into `audit-selected/`; their paths and SHA256s are in `audit-selected/download-receipt.json`. `../audit_smoke.py` reproduces the action, token and role reconciliation in `AUDIT.json`.

| Game | Actor calls | Builder calls | Scribe calls | Builder request outcome |
| --- | ---: | ---: | ---: | --- |
| AR25 | 13 | 0 | 3 | Never requested |
| LS20 | 17 | 0 | 0 | Never requested |
| SB26 | 19 | 0 | 3 | Requested once; deadline policy rejected it |

SB26 requested construction at sequence 15, level 1 turn 3. The deadline trace records 164.423 seconds of decision timeout, a 300-second Actor reserve, and a 5-second host-return reserve. Builder allocation was therefore zero; the tool returned BuilderDeadlineUnavailable before any Builder model call. This is a mismatch between the short smoke budget and the fixed reservation policy, not evidence that the model never wanted a Builder. It was visible in local qualification under a shorter timeout; increasing the local fixture budget did not remove it from the 600-second live smoke.

All 48 host action receipts have contiguous sequences, exact successor-to-predecessor observation continuity, and action IDs/CLICK coordinates matching benchmark history. Level totals and all 96,246 completed output tokens reconcile. All three workers exited 0 with no host failure. These checks establish host/transport execution; they do not establish correct game inference.

Tool results expose two recoverable Python mistakes: LS20 imported nonexistent `np`; SB26 referenced undefined `g`. Both traces label tool execution complete despite a traceback in the result. In each case the next Python tool call corrected the error. Auditing only status labels would miss these failures. This is code-error recovery, not a measured game-mechanism recovery rate.

Both level-ending Scribes spent all three calls on run_python inspection, with no write_file/edit_file tool call. The Actor later wrote the level-insight and world-model notes in both games. The Scribe implementation also has a deterministic fallback note; role-call counts alone do not demonstrate useful consolidation. None of the three tool traces contains an explicit dfr helper call, so availability of donor segmentation does not show its use in this smoke.

Judgment: real GPU execution is demonstrated, but the intended executable-model construction/repair mechanism was not exercised. The smoke cannot answer whether that mechanism improves score. Before a comparative run, align the action horizon and Builder reservation with the experiment and qualify actual Builder execution. Do not silently change policy and call it the same frozen candidate.
